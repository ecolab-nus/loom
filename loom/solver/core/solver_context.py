"""Manage CPMpy model state, symbol table, and constraint accumulation."""

from __future__ import annotations

import traceback
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

    def add_hard_constraints(
        self,
        constraints_ast: list[Constraint],
        label_prefix: str = "hard",
    ) -> None:
        """Resolve and add every hard constraint from AST nodes."""
        resolver = ExprResolver(self.symbol_map)
        for i, c in enumerate(constraints_ast):
            cp_c = resolver.resolve(c)
            if cp_c is not True:
                self.model += cp_c
                self._tracked_constraints.append((f"{label_prefix}[{i}]", cp_c))
        # Add any auxiliary constraints generated during resolution
        for j, aux in enumerate(resolver.aux_constraints):
            self.model += aux
            self._tracked_constraints.append((f"{label_prefix}_aux[{j}]", aux))

    @staticmethod
    def _flatten_div_chain(node: Div) -> tuple[Expr, list[Expr]]:
        """Traverse the left-spine of nested Div nodes.

        Returns (numerator, [d0, d1, ..., dN]) where numerator is the leftmost
        non-Div leaf and the denominators are in left-to-right evaluation order:
          Div(Div(Div(A, d0), d1), d2)  →  (A, [d0, d1, d2])
        """
        denoms: list[Expr] = []
        current: Expr = node
        while isinstance(current, Div):
            denoms.append(current.right)
            current = current.left
        denoms.reverse()
        return current, denoms

    def add_trip_count_constraints(
        self,
        trip_count_exprs: list[dict],
        labels: list[str] | None = None,
        require_divisibility: bool = False,
    ) -> None:
        """Add no-oversize constraints for each loop trip-count expression.

        Flattens nested Div chains before resolving:
          Div(Div(N, d0), d1)  ->  d0 * d1 <= N

        Calls _resolve_expr() directly on each non-Div leaf node to avoid the
        ceiling-division path that _resolve_expr() applies to Div nodes.
        """
        resolver = ExprResolver(self.symbol_map)

        if labels is not None and len(labels) != len(trip_count_exprs):
            raise ValueError("trip-count constraint labels must match expressions")

        for i, raw in enumerate(trip_count_exprs):
            label = labels[i] if labels is not None else f"trip_count[{i}]"
            node = parse_expr(raw)
            if not isinstance(node, Div):
                raise ValueError(
                    f"Expected top-level Div node in {label}, "
                    f"got {type(node).__name__}: {node}"
                )

            numerator_node, denom_nodes = self._flatten_div_chain(node)

            # Resolve numerator — guaranteed non-Div by _flatten_div_chain
            num_cp = resolver._resolve_expr(numerator_node)

            # Build denominator product by resolving each denom leaf individually.
            # This avoids calling _resolve_expr on a Div node (which would apply
            # ceiling division instead of exact integer division).
            denom_cp = resolver._resolve_expr(denom_nodes[0])
            for d in denom_nodes[1:]:
                denom_cp = denom_cp * resolver._resolve_expr(d)

            if require_divisibility:
                div_c = num_cp % denom_cp == 0
                self.model += div_c
                self._tracked_constraints.append((f"{label}.divisible", div_c))

            no_oversize_c = denom_cp <= num_cp
            self.model += no_oversize_c
            self._tracked_constraints.append((label, no_oversize_c))

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
        except Exception as exc:
            print(
                "[CPMPY][MODEL_INVALID] Exception while solving with OR-Tools:",
                f"{type(exc).__name__}: {exc}",
            )
            print(traceback.format_exc())
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
