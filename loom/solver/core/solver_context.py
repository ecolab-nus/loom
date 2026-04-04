"""Manage CPMpy model state, symbol table, and constraint accumulation."""

from __future__ import annotations

from typing import Optional, Union

import cpmpy as cp

from .cpmpy_expr_resolver import ExprResolver
from ...loom_utils.ast import Constraint, Expr


class SolverContext:
    """Holds the CPMpy model, symbol table, and all accumulated constraints."""

    def __init__(self) -> None:
        self.model = cp.Model()
        self.symbol_map: dict[str, cp.Expression] = {}
        self.boolean_names: set[str] = set()

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
        for c in constraints_ast:
            cp_c = resolver.resolve(c)
            if cp_c is not True:
                self.model += cp_c
        # Add any auxiliary constraints generated during resolution
        for aux in resolver.aux_constraints:
            self.model += aux

    def find_optimum(
        self,
        objective_ast: Expr,
    ) -> Optional[tuple[int, dict[str, int]]]:
        """Minimize the objective and return (min_val, assignments) or None."""
        resolver = ExprResolver(self.symbol_map)
        cp_obj = resolver.resolve(objective_ast)

        # Add auxiliary constraints from Switch indicators in objective
        for aux in resolver.aux_constraints:
            self.model += aux

        self.model.minimize(cp_obj)
        solved = self.model.solve(solver="ortools")

        if not solved:
            return None

        assignments = {
            name: int(var.value()) for name, var in self.symbol_map.items()
        }
        obj_val = int(self.model.objective_value())
        return obj_val, assignments
