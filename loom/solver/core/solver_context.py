"""Manage CPMpy model state, symbol table, and constraint accumulation."""

from __future__ import annotations

from typing import Optional, Union

import cpmpy as cp

from .cpmpy_expr_resolver import ExprResolver
from ...loom_utils.ast import Constraint, Expr, Div, parse_expr


class SolverContext:
    """Holds the CPMpy model, symbol table, and all accumulated constraints."""

    def __init__(self) -> None:
        self.model = cp.Model()
        self.symbol_map: dict[str, cp.Expression] = {}
        self.boolean_names: set[str] = set()
        self._tracked_constraints: list[tuple[str, cp.Expression]] = []

    def load_symbols(
        self,
        metadata_symbols: dict[str, Union[str, dict]],
        domains: dict[str, list[int]],
    ) -> None:
        """Create CPMpy integer variables for each symbol with InDomain constraints."""
        for name, info in metadata_symbols.items():
            typ = info["type"] if isinstance(info, dict) else info
            if typ != "int":
                raise ValueError(f"Unsupported symbol type '{typ}' for symbol '{name}'")

            if name in domains:
                vals = domains[name]
                var = cp.intvar(min(vals), max(vals), name=name)
                self.model += cp.InDomain(var, vals)
            else:
                ub = info.get("natural_ub", 10000) if isinstance(info, dict) else 10000
                var = cp.intvar(1, ub, name=name)

            self.symbol_map[name] = var

    def load_booleans(self, bool_names: list[str]) -> dict[str, list[int]]:
        """Create integer variables in {0, 1} for each boolean symbol."""
        bool_domains: dict[str, list[int]] = {}
        for name in bool_names:
            var = cp.intvar(0, 1, name=name)
            self.symbol_map[name] = var
            self.boolean_names.add(name)
            bool_domains[name] = [0, 1]
        return bool_domains

    def add_hard_constraints(self, constraints_ast: list[Constraint]) -> None:
        """Resolve and add every hard constraint from AST nodes."""
        resolver = ExprResolver(self.symbol_map)
        for i, c in enumerate(constraints_ast):
            cp_c = resolver.resolve(c)
            if cp_c is not True:
                self.model += cp_c
                self._tracked_constraints.append((f"hard[{i}]", cp_c))
        # Add any auxiliary constraints generated during resolution
        for j, aux in enumerate(resolver.aux_constraints):
            self.model += aux
            self._tracked_constraints.append((f"hard_aux[{j}]", aux))

    def add_iter_num_constraints(self, iter_num: dict) -> None:
        """Add divisibility and positivity constraints for each iter_num expression.

        Each iter expression has the form C_0 / (C_1 * V) (parsed as Div(num, denom)).
        We enforce:
          - C_0 % (C_1 * V) == 0   (iteration count is an integer)
          - C_1 * V <= C_0          (iteration count is at least 1)
        """
        raw_iters = [iter_num["seq_iter"]] + list(iter_num.get("temp_iter", []))
        resolver = ExprResolver(self.symbol_map)

        for i, raw in enumerate(raw_iters):
            node = parse_expr(raw)
            if not isinstance(node, Div):
                raise ValueError(f"Expected Div node in iter_num, got {type(node)}: {node}")

            num_cp = resolver.resolve(node.left)
            denom_cp = resolver.resolve(node.right)
            div_c = num_cp % denom_cp == 0
            pos_c = denom_cp <= num_cp
            self.model += div_c   # divisible
            self.model += pos_c   # positive (iter >= 1)
            self._tracked_constraints.append((f"iter_div[{i}]", div_c))
            self._tracked_constraints.append((f"iter_pos[{i}]", pos_c))

    def find_optimum(
        self,
        objective_ast: Expr,
    ) -> tuple[str, Optional[int], Optional[dict[str, int]]]:
        """Minimize the objective and return (status, min_val, assignments).

        status is one of: "OPTIMAL", "INFEASIBLE", "UNKNOWN", "MODEL_INVALID".
        min_val and assignments are None when status != "OPTIMAL".
        """
        from cpmpy.solvers.solver_interface import ExitStatus  # noqa: PLC0415

        resolver = ExprResolver(self.symbol_map)
        cp_obj = resolver.resolve(objective_ast)

        # Add auxiliary constraints from Switch indicators in objective
        for aux in resolver.aux_constraints:
            self.model += aux

        self.model.minimize(cp_obj)
        try:
            self.model.solve(solver="ortools")
        except Exception:
            return "MODEL_INVALID", None, None

        exit_status = self.model.status().exitstatus
        if exit_status == ExitStatus.OPTIMAL:
            assignments = {
                name: int(var.value()) for name, var in self.symbol_map.items()
            }
            obj_val = int(self.model.objective_value())
            return "OPTIMAL", obj_val, assignments
        elif exit_status == ExitStatus.UNSATISFIABLE:
            return "INFEASIBLE", None, None
        elif exit_status == ExitStatus.UNKNOWN:
            return "UNKNOWN", None, None
        else:
            return "MODEL_INVALID", None, None

    def find_mus(self) -> list[tuple[int, str, str]]:
        """Return MUS as list of (original_index, label, expr_str).

        Uses cpmpy.tools.mus.mus() with domain/structural constraints as hard
        background and all tracked hard/iter constraints as soft candidates.
        """
        from cpmpy.tools.mus import mus as cpmpy_mus  # noqa: PLC0415

        soft = [expr for _, expr in self._tracked_constraints]
        labels = [label for label, _ in self._tracked_constraints]

        # Anything added to the model but not tracked (e.g. InDomain domain
        # constraints) is structural — keep it as hard background.
        soft_set = set(id(e) for e in soft)
        domain_constraints = [
            c for c in self.model.constraints
            if id(c) not in soft_set
        ]

        try:
            mus_subset = cpmpy_mus(soft, hard=domain_constraints)
        except Exception:
            return []

        result = []
        for cp_expr in mus_subset:
            try:
                idx = next(
                    i for i, e in enumerate(soft) if e is cp_expr
                )
            except StopIteration:
                continue
            result.append((idx, labels[idx], str(cp_expr)))
        return result
