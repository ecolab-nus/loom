"""Manage Z3 solver state, symbol table, and constraint accumulation using Pure Python AST.
"""

from __future__ import annotations
import z3
from typing import Optional, Union, Tuple
from itertools import product as iter_product

from .z3_expr_resolver import resolve_ast
from ...loom_utils.ast import Node, Expr, Constraint, Sym

class SolverContext:
    """Holds the Z3 solver, symbol table, and all accumulated constraints.
    """

    def __init__(self, debug: bool = False) -> None:
        self.solver = z3.Solver()
        self.symbol_map: dict[str, z3.ArithRef] = {}
        self.boolean_names: set[str] = set()
        self._constraints: list[z3.BoolRef] = []
        self._tracking_vars: dict[str, tuple[z3.BoolRef, z3.BoolRef, str | None]] = {}
        self.last_unsat_core_info: list[tuple[str, str]] | None = None
        self.debug = debug

    def load_symbols(self, metadata_symbols: dict[str, Union[str, dict]], add_positivity_guard: bool = True) -> None:
        """Declare Z3 integer variables for each symbol."""
        for name, info in metadata_symbols.items():
            typ = info["type"] if isinstance(info, dict) else info
            if typ != "int":
                raise ValueError(f"Unsupported symbol type '{typ}' for symbol '{name}'")
            var = z3.Int(name)
            self.symbol_map[name] = var
            if add_positivity_guard:
                guard = var > 0
                self.solver.add(guard)
                self._constraints.append(guard)

    def load_booleans(self, bool_names: list[str]) -> dict[str, list[int]]:
        """Declare symbolic boolean variables as Z3 Int vars in {0, 1}."""
        bool_domains: dict[str, list[int]] = {}
        for name in bool_names:
            var = z3.Int(name)
            self.symbol_map[name] = var
            self.boolean_names.add(name)
            domain_c = z3.Or(var == 0, var == 1)
            self.solver.add(domain_c)
            self._constraints.append(domain_c)
            bool_domains[name] = [0, 1]
        return bool_domains

    def add_hard_constraints_ast(self, constraints: list[Constraint]) -> None:
        """Resolve and add every hard constraint from a list of AST nodes."""
        for c in constraints:
            z3_c = resolve_ast(c, self.symbol_map)
            if self.debug:
                track_name = f"hc_{len(self._tracking_vars)}"
                track_var = z3.Bool(track_name)
                self.solver.assert_and_track(z3_c, track_var)
                self._tracking_vars[track_name] = (track_var, z3_c, str(c))
            else:
                self.solver.add(z3_c)
            self._constraints.append(z3_c)

    def add_domain_constraints(self, domains: dict[str, list[int]]) -> None:
        """Restrict each symbol to its finite domain."""
        for name, values in domains.items():
            if name not in self.symbol_map: continue
            sym = self.symbol_map[name]
            domain_c = z3.Or([sym == v for v in values])
            if self.debug:
                track_name = f"dom_{name}"
                track_var = z3.Bool(track_name)
                self.solver.assert_and_track(domain_c, track_var)
                self._tracking_vars[track_name] = (track_var, domain_c, f"{name} ∈ {values}")
            else:
                self.solver.add(domain_c)
            self._constraints.append(domain_c)

    def find_optimum(
        self,
        objective_ast: Expr,
        domains: dict[str, list[int]],
        force_enumerate: bool = False,
    ) -> Optional[tuple[int, dict[str, int]]]:
        """Find the global minimum of objective_ast via binary search."""
        if force_enumerate:
            # Objective must be resolved for enumeration
            z3_obj = resolve_ast(objective_ast, self.symbol_map)
            return self._enumerate_all(z3_obj, domains)

        # Objective resolved to Z3
        z3_obj = resolve_ast(objective_ast, self.symbol_map)
        
        # Initial feasibility
        result = self.solver.check()
        if result == z3.unsat:
            if self.debug: self._capture_unsat_core()
            return None
        if result == z3.unknown:
            return self._enumerate_all(z3_obj, domains)

        model = self.solver.model()
        upper = model.eval(z3_obj, model_completion=True).as_long()
        best_assignments = {n: model.eval(v, model_completion=True).as_long() for n, v in self.symbol_map.items()}

        # Binary search
        lower = 0
        while lower < upper:
            mid = (lower + upper) // 2
            self.solver.push()
            self.solver.add(z3_obj <= mid)
            res = self.solver.check()
            if res == z3.sat:
                model = self.solver.model()
                upper = model.eval(z3_obj, model_completion=True).as_long()
                best_assignments = {n: model.eval(v, model_completion=True).as_long() for n, v in self.symbol_map.items()}
                self.solver.pop()
            elif res == z3.unsat:
                if self.debug: self._capture_unsat_core()
                self.solver.pop()
                lower = mid + 1
            else:
                self.solver.pop()
                return self._enumerate_all(z3_obj, domains)

        return upper, best_assignments

    def find_mus(self, hard_constraints_ast: list[Constraint], domains: dict[str, list[int]]) -> list[tuple[int, str, str]]:
        """Find a Minimum Unsatisfiable Subset for infeasible context."""
        # This implementation follows the legacy logic but works with AST nodes.
        labeled: list[tuple[str, z3.BoolRef]] = []
        for name, var in self.symbol_map.items():
            labeled.append((f"positivity({name})", var > 0))
        for name, values in domains.items():
            if name not in self.symbol_map: continue
            sym = self.symbol_map[name]
            labeled.append((f"domain({name})", z3.Or([sym == v for v in values])))
        for i, c in enumerate(hard_constraints_ast):
            labeled.append((f"hard[{i}]", resolve_ast(c, self.symbol_map)))

        remaining = list(range(len(labeled)))
        for idx in list(remaining):
            candidate = [i for i in remaining if i != idx]
            s = z3.Solver()
            for i in candidate: s.add(labeled[i][1])
            if s.check() != z3.sat:
                remaining.remove(idx)
        return [(i, labeled[i][0], str(labeled[i][1])) for i in remaining]

    def _capture_unsat_core(self) -> None:
        core = self.solver.unsat_core()
        info = []
        for v in core:
            name = str(v)
            if name in self._tracking_vars:
                _, _, display = self._tracking_vars[name]
                info.append((name, display))
            else:
                info.append((name, "(bound constraint)"))
        self.last_unsat_core_info = info

    def _enumerate_all(self, objective: z3.ArithRef, domains: dict[str, list[int]]) -> Optional[tuple[int, dict[str, int]]]:
        sym_names = list(self.symbol_map.keys())
        active = [(n, domains[n]) for n in sym_names if n in domains]
        if not active: return None

        names, value_lists = zip(*active)
        best = None
        s = z3.Solver()
        s.add(self._constraints)

        for combo in iter_product(*value_lists):
            s.push()
            for name, val in zip(names, combo):
                s.add(self.symbol_map[name] == val)
            if s.check() == z3.sat:
                model = s.model()
                val = model.eval(objective, model_completion=True).as_long()
                if best is None or val < best[0]:
                    best = (val, dict(zip(names, combo)))
            s.pop()
        return best
