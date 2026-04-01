"""Manage Z3 solver state, symbol table, and constraint accumulation.

SolverContext acts as the 'virtual symbol table' for the SMT problem.
It owns a Z3 Solver and the symbol_map, and provides methods for loading
symbols, adding constraints, and finding the optimum via binary search.

NOTE: We deliberately use z3.Solver (not z3.Optimize) because the T_total
objective is non-linear (nested If/max, symbolic multiplication, integer
division).  z3.Optimize (nu-Z) is incomplete for NIA and can return
sub-optimal solutions without reporting UNKNOWN.  z3.Solver.check() on
bounded integer domains is decidable and complete, so a binary search on
the objective value guarantees the true global minimum.
"""

from dataclasses import dataclass, field
from enum import Enum
from itertools import product as iter_product

import z3

from .expr_resolver import resolve_constraint, resolve_expr
from ..utils.expr_printer import constraint_to_str


class ConstraintStatus(Enum):
    """Classification of a constraint at the optimal assignment."""

    TIGHT = "tight"              # slack == 0
    DISCRETE_WALL = "wall"       # slack > 0 but no symbol can step up without violating
    ACTIVE_SLACK = "slack"       # slack > 0 and some symbol CAN step
    SATISFIED = "satisfied"      # binary constraint satisfied (Divisible, Eq, Ne)
    VIOLATED = "violated"        # constraint violated (should not happen at feasible point)


@dataclass
class SymbolStepInfo:
    """Per-symbol headroom analysis for an inequality constraint."""

    symbol: str
    current_value: int
    next_value: int | None       # None if at max domain value
    step_cost: int | None        # Decrease in slack when symbol steps; None if no next value
    would_violate: bool          # True if stepping would violate the constraint


@dataclass
class ConstraintAnalysis:
    """Full analysis result for a single hard constraint."""

    index: int                   # Index in hard_constraints list
    constraint_json: dict
    tag: str                     # "Ge", "Le", "Divisible", etc.
    status: ConstraintStatus
    description: str             # Human-readable concrete summary (e.g. "1 >= 1")
    symbolic: str = ""           # Human-readable symbolic form (e.g. "32/tile_k >= 1")
    slack: int | None = None
    lhs_value: int | None = None
    rhs_value: int | None = None
    symbol_steps: list[SymbolStepInfo] = field(default_factory=list)
    sub_analyses: list['ConstraintAnalysis'] = field(default_factory=list)


def _collect_symbols(expr: dict | str | list) -> set[str]:
    """Recursively collect all symbol names from a JSON expression or constraint."""
    if isinstance(expr, (str, int, float)):
        return set()
    if isinstance(expr, list):
        result: set[str] = set()
        for item in expr:
            result |= _collect_symbols(item)
        return result
    if not isinstance(expr, dict):
        return set()
    tag, payload = next(iter(expr.items()))
    if tag == "Sym":
        return {payload}
    if tag == "Const":
        return set()
    if tag == "Divisible":
        return _collect_symbols(payload["x"]) | _collect_symbols(payload["by"])
    if tag in ("Ge", "Le", "Gt", "Lt", "Eq", "Ne", "Mul", "Div"):
        return _collect_symbols(payload[0]) | _collect_symbols(payload[1])
    if tag in ("Add", "Min", "And"):
        result = set()
        for item in payload:
            result |= _collect_symbols(item)
        return result
    if tag == "IfElse":
        return (
            _collect_symbols(payload["cond"])
            | _collect_symbols(payload["then_expr"])
            | _collect_symbols(payload["else_expr"])
        )
    return set()


class SolverContext:
    """Holds the Z3 solver, symbol table, and all accumulated constraints.

    Usage:
        ctx = SolverContext()
        ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
        ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])
        ctx.add_domain_constraints(symbol_domains)
        result = ctx.find_optimum(t_total_expr, domains)
    """

    def __init__(self, debug: bool = False) -> None:
        self.solver = z3.Solver()
        self.symbol_map: dict[str, z3.ArithRef] = {}
        self.boolean_names: set[str] = set()
        self._constraints: list[z3.BoolRef] = []
        # Each entry: (tracker_bool, z3_constraint, compact_display_str_or_None)
        self._tracking_vars: dict[str, tuple[z3.BoolRef, z3.BoolRef, str | None]] = {}
        self.last_unsat_core_info: list[tuple[str, str]] | None = None
        self.debug = debug

    def load_symbols(self, metadata_symbols: dict[str, str | dict]) -> None:
        """Declare Z3 integer variables for each symbol in the metadata.

        Args:
            metadata_symbols: maps symbol name to either a plain type string
                (e.g. ``"int"``) or a dict ``{"type": "int", "natural_ub": 4096}``.

        Raises:
            ValueError: If a symbol type other than "int" is encountered.
        """
        for name, info in metadata_symbols.items():
            typ = info["type"] if isinstance(info, dict) else info
            if typ != "int":
                raise ValueError(f"Unsupported symbol type '{typ}' for symbol '{name}'")
            var = z3.Int(name)
            self.symbol_map[name] = var
            guard = var > 0
            self.solver.add(guard)
            self._constraints.append(guard)

    def load_booleans(self, bool_names: list[str]) -> dict[str, list[int]]:
        """Declare symbolic boolean variables as Z3 Int vars in {0, 1}.

        Boolean variables participate in optimization just like integer symbols.
        The solver determines True (1) or False (0) to minimize cost.

        Args:
            bool_names: list of boolean variable names from metadata.

        Returns:
            Domain dict mapping each boolean name to [0, 1], suitable for
            merging into the main domains dict.
        """
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

    def add_hard_constraints(self, constraints: list[dict]) -> None:
        """Resolve and add every hard constraint from the ETG JSON.

        Args:
            constraints: List of constraint dicts (Ge, Divisible, …).
        """
        for c in constraints:
            z3_c = resolve_constraint(c, self.symbol_map)
            self.solver.add(z3_c)
            self._constraints.append(z3_c)

    def add_domain_constraints(self, domains: dict[str, list[int]]) -> None:
        """Restrict each symbol to a finite set of allowed values.

        Args:
            domains: e.g. {"BM": [32, 64, 128, …], "BK": [32, 64, …, 512]}
        """
        for name, values in domains.items():
            if name not in self.symbol_map:
                continue
            sym = self.symbol_map[name]
            domain_c = z3.Or([sym == v for v in values])
            self.solver.add(domain_c)
            self._constraints.append(domain_c)

    def get_constraints(self) -> list[z3.BoolRef]:
        """Return all accumulated constraints (for use in enumeration mode)."""
        return list(self._constraints)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def find_optimum(
        self,
        objective: z3.ArithRef,
        domains: dict[str, list[int]],
        force_enumerate: bool = False,
    ) -> tuple[int, dict[str, int]] | None:
        """Find the minimum value of *objective* via binary search.

        Phase 0 – simplify constraints and objective via Z3 tactics.
        Phase 1 – obtain an initial feasible upper bound by calling
        ``solver.check()`` with no objective constraint.
        Phase 2 – binary-search: add ``obj_var <= mid``, check SAT/UNSAT.
        Fallback – if Z3 ever returns UNKNOWN in Phase 1 or Phase 2,
        fall back to full enumeration.

        When *force_enumerate* is True, skip Phases 0-2 entirely and go
        straight to brute-force enumeration over the domain.

        Returns:
            ``(min_value, {sym: value, …})`` on success, or ``None`` when the
            constraint system is infeasible (UNSAT).
        """
        if force_enumerate:
            return self._enumerate_all(objective, domains)

        # Phase 0: canonicalize constraints + objective
        obj_var = self._simplify(objective)

        # Phase 1: any feasible point → upper bound
        status, upper, best_assignments = self._find_initial_bound(obj_var)
        if status == z3.unsat:
            if self.debug:
                self._capture_unsat_core()
            return None
        if status == z3.unknown:
            return self._enumerate_all(objective, domains)

        # Phase 2: binary search
        lower = 0
        while lower < upper:
            mid = (lower + upper) // 2
            self.solver.push()
            if self.debug:
                bound_label = f"__obj_le_{mid}"
                bound_var = z3.Bool(bound_label)
                self.solver.assert_and_track(obj_var <= mid, bound_var)
            else:
                self.solver.add(obj_var <= mid)
            result = self.solver.check()

            if result == z3.sat:
                model = self.solver.model()
                val = model.eval(obj_var, model_completion=True).as_long()
                upper = val
                best_assignments = {
                    name: model.eval(var, model_completion=True).as_long()
                    for name, var in self.symbol_map.items()
                }
                self.solver.pop()
            elif result == z3.unsat:
                if self.debug:
                    self._capture_unsat_core()
                self.solver.pop()
                lower = mid + 1
            else:
                self.solver.pop()
                return self._enumerate_all(objective, domains)

        return upper, best_assignments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simplify(self, objective: z3.ArithRef) -> z3.ArithRef:
        """Run Z3 tactic pipeline to canonicalize constraints and objective.

        Introduces an auxiliary variable ``__obj == objective``, runs the
        tactic chain, and rebuilds the solver with simplified constraints.
        When ``self.debug`` is True, constraints are added via
        ``assert_and_track()`` so that UNSAT cores can be extracted later.

        Returns:
            The ``__obj`` auxiliary variable (bound to the simplified
            objective inside the solver).
        """
        tactic = z3.Then(
            'simplify',           # constant folding, flatten, basic arith
            'propagate-values',   # substitute known equalities
            'ctx-simplify',       # context-dependent simplification
            'elim-term-ite',      # lift if-then-else out of terms
        )

        obj_var = z3.Int('__obj')

        goal = z3.Goal()
        for c in self.solver.assertions():
            goal.add(c)
        goal.add(obj_var == objective)

        simplified = tactic(goal)

        self.solver.reset()
        self._constraints = []
        self._tracking_vars = {}
        for i, subgoal in enumerate(simplified):
            for j, c in enumerate(subgoal):
                if self.debug:
                    track_name = f"sc_{i}_{j}"
                    track_var = z3.Bool(track_name)
                    self.solver.assert_and_track(c, track_var)
                    compact = self._detect_domain_pattern(c)
                    self._tracking_vars[track_name] = (track_var, c, compact)
                else:
                    self.solver.add(c)
                self._constraints.append(c)

        return obj_var

    def _find_initial_bound(
        self, objective: z3.ArithRef,
    ) -> tuple[z3.CheckSatResult, int | None, dict[str, int] | None]:
        """Check feasibility and return an initial upper bound if SAT.

        Returns:
            ``(z3.sat, upper_bound, assignments)`` on success,
            ``(z3.unsat, None, None)`` when infeasible, or
            ``(z3.unknown, None, None)`` when the solver cannot decide.
        """
        result = self.solver.check()
        if result == z3.sat:
            model = self.solver.model()
            val = model.eval(objective, model_completion=True).as_long()
            assignments = {
                name: model.eval(var, model_completion=True).as_long()
                for name, var in self.symbol_map.items()
            }
            return z3.sat, val, assignments
        if result == z3.unsat:
            return z3.unsat, None, None
        return z3.unknown, None, None

    def _capture_unsat_core(self) -> None:
        """Store formatted UNSAT core from the most recent UNSAT check."""
        core = self.solver.unsat_core()
        info = []
        for v in core:
            name = str(v)
            if name in self._tracking_vars:
                _track_var, z3_constraint, compact = self._tracking_vars[name]
                display = compact if compact is not None else str(z3_constraint)
                info.append((name, display))
            else:
                info.append((name, "(bound constraint)"))
        self.last_unsat_core_info = info

    @staticmethod
    def _detect_domain_pattern(constraint: z3.BoolRef) -> str | None:
        """Detect Or(sym == v1, sym == v2, ...) and return a compact display string.

        Returns a compact string like ``'tile_m ∈ {32, 64, ..., 4096} [128 values, step=32]'``
        if the constraint matches the pattern, or ``None`` otherwise.
        """
        if not z3.is_or(constraint):
            return None

        children = constraint.children()
        if not children:
            return None

        # All children must be (sym == IntVal)
        sym_name: str | None = None
        values: list[int] = []
        for child in children:
            if not (z3.is_eq(child) and len(child.children()) == 2):
                return None
            lhs, rhs = child.children()
            # Normalise: ensure lhs is a bare variable, rhs is a constant
            if z3.is_int_value(lhs) and z3.is_const(rhs):
                lhs, rhs = rhs, lhs
            if not (z3.is_const(lhs) and z3.is_int_value(rhs)):
                return None
            name = str(lhs)
            if sym_name is None:
                sym_name = name
            elif sym_name != name:
                return None  # different symbols — not a domain block
            values.append(rhs.as_long())

        if sym_name is None or len(values) < 2:
            return None

        values.sort()
        n = len(values)
        lo, hi = values[0], values[-1]

        # Detect uniform step
        steps = {values[i + 1] - values[i] for i in range(n - 1)}
        step_str = f", step={next(iter(steps))}" if len(steps) == 1 else ""

        # Show first 2 and last value with ellipsis, or all values if 5 or fewer
        if n <= 5:
            vals_str = ", ".join(str(v) for v in values)
        else:
            vals_str = f"{values[0]}, {values[1]}, ..., {values[-1]}"

        return f"{sym_name} ∈ {{{vals_str}}}  [{n} values{step_str}]"

    def _enumerate_all(
        self,
        objective: z3.ArithRef,
        domains: dict[str, list[int]],
    ) -> tuple[int, dict[str, int]] | None:
        """Brute-force all domain combinations (UNKNOWN fallback)."""
        sym_names = list(self.symbol_map.keys())
        active = [(n, domains[n]) for n in sym_names if n in domains]
        if not active:
            return None

        names, value_lists = zip(*active)
        best: tuple[int, dict[str, int]] | None = None

        for combo in iter_product(*value_lists):
            s = z3.Solver()
            s.add(self._constraints)
            for name, val in zip(names, combo):
                s.add(self.symbol_map[name] == val)

            if s.check() != z3.sat:
                continue

            model = s.model()
            val = model.eval(objective, model_completion=True).as_long()
            if best is None or val < best[0]:
                best = (val, dict(zip(names, combo)))

        if best is None:
            self.last_unsat_core_info = [
                ("enumerate_fallback", "no feasible combination found in domain")
            ]

        return best

    # ------------------------------------------------------------------
    # Debug analysis
    # ------------------------------------------------------------------

    def find_active_constraints(
        self,
        hard_constraints: list[dict],
        assignments: dict[str, int],
        domains: dict[str, list[int]],
    ) -> list[ConstraintAnalysis]:
        """Analyze all hard constraints at the given assignments.

        For inequality constraints (Ge, Le, Gt, Lt), computes slack and
        per-symbol step costs to detect "discrete walls" — constraints that
        are not tight but block every upward step in the domain.

        For binary constraints (Divisible, Eq, Ne), reports satisfied/violated.

        Returns:
            List of ``ConstraintAnalysis`` for every hard constraint.
        """
        concrete_map = {name: z3.IntVal(val) for name, val in assignments.items()}
        sorted_domains = {
            name: sorted(vals) for name, vals in domains.items()
        }
        results: list[ConstraintAnalysis] = []
        for i, c in enumerate(hard_constraints):
            results.append(
                self._analyze_single_constraint(
                    i, c, concrete_map, assignments, sorted_domains,
                )
            )
        return results

    def _analyze_single_constraint(
        self,
        index: int,
        constraint: dict,
        concrete_map: dict[str, z3.ArithRef],
        assignments: dict[str, int],
        sorted_domains: dict[str, list[int]],
    ) -> ConstraintAnalysis:
        """Analyze one constraint, dispatching by tag."""
        tag, payload = next(iter(constraint.items()))

        if tag in ("Ge", "Le", "Gt", "Lt"):
            return self._analyze_inequality(
                index, constraint, tag, payload,
                concrete_map, assignments, sorted_domains,
            )
        if tag == "Divisible":
            return self._analyze_divisible(index, constraint, payload, concrete_map)
        if tag in ("Eq", "Ne"):
            return self._analyze_equality(index, constraint, tag, payload, concrete_map)
        if tag == "And":
            sub = [
                self._analyze_single_constraint(
                    index, sc, concrete_map, assignments, sorted_domains,
                )
                for sc in payload
            ]
            has_wall = any(s.status == ConstraintStatus.DISCRETE_WALL for s in sub)
            has_tight = any(s.status == ConstraintStatus.TIGHT for s in sub)
            status = (
                ConstraintStatus.DISCRETE_WALL if has_wall
                else ConstraintStatus.TIGHT if has_tight
                else ConstraintStatus.SATISFIED
            )
            return ConstraintAnalysis(
                index=index, constraint_json=constraint, tag=tag,
                status=status,
                description=f"And: {len(payload)} sub-constraints",
                sub_analyses=sub,
            )
        # Unknown tag — report as satisfied
        return ConstraintAnalysis(
            index=index, constraint_json=constraint, tag=tag,
            status=ConstraintStatus.SATISFIED,
            description=f"{tag}: (unhandled constraint type)",
        )

    def _analyze_inequality(
        self,
        index: int,
        constraint: dict,
        tag: str,
        payload: list,
        concrete_map: dict[str, z3.ArithRef],
        assignments: dict[str, int],
        sorted_domains: dict[str, list[int]],
    ) -> ConstraintAnalysis:
        """Analyze a Ge/Le/Gt/Lt constraint with per-symbol headroom."""
        lhs_val = z3.simplify(resolve_expr(payload[0], concrete_map)).as_long()
        rhs_val = z3.simplify(resolve_expr(payload[1], concrete_map)).as_long()

        # Compute slack (positive = satisfied)
        if tag in ("Le", "Lt"):
            slack = rhs_val - lhs_val
        else:
            slack = lhs_val - rhs_val

        # Per-symbol step analysis
        symbols = _collect_symbols(constraint)
        symbol_steps: list[SymbolStepInfo] = []
        for sym_name in sorted(symbols):
            if sym_name not in sorted_domains or sym_name not in assignments:
                continue
            dom = sorted_domains[sym_name]
            curr_val = assignments[sym_name]
            try:
                curr_idx = dom.index(curr_val)
            except ValueError:
                continue
            next_val = dom[curr_idx + 1] if curr_idx + 1 < len(dom) else None

            if next_val is not None:
                modified_map = {**concrete_map, sym_name: z3.IntVal(next_val)}
                new_lhs = z3.simplify(resolve_expr(payload[0], modified_map)).as_long()
                new_rhs = z3.simplify(resolve_expr(payload[1], modified_map)).as_long()
                if tag in ("Le", "Lt"):
                    new_slack = new_rhs - new_lhs
                    step_cost = slack - new_slack  # how much slack decreased
                else:
                    new_slack = new_lhs - new_rhs
                    step_cost = slack - new_slack
                if tag == "Le":
                    would_violate = new_lhs > new_rhs
                elif tag == "Lt":
                    would_violate = new_lhs >= new_rhs
                elif tag == "Ge":
                    would_violate = new_lhs < new_rhs
                else:  # Gt
                    would_violate = new_lhs <= new_rhs
            else:
                step_cost = None
                would_violate = False

            symbol_steps.append(SymbolStepInfo(
                symbol=sym_name, current_value=curr_val,
                next_value=next_val, step_cost=step_cost,
                would_violate=would_violate,
            ))

        # Classify
        steppable = [s for s in symbol_steps if s.next_value is not None]
        if slack == 0:
            status = ConstraintStatus.TIGHT
        elif slack < 0:
            status = ConstraintStatus.VIOLATED
        elif steppable and all(s.would_violate for s in steppable):
            status = ConstraintStatus.DISCRETE_WALL
        else:
            status = ConstraintStatus.ACTIVE_SLACK

        op = "<=" if tag in ("Le", "Lt") else ">="
        desc = f"{tag}: {lhs_val} {op} {rhs_val} (slack={slack})"
        if status == ConstraintStatus.DISCRETE_WALL:
            blocked = [s.symbol for s in steppable if s.would_violate]
            desc += f" [DISCRETE WALL -- blocks: {', '.join(blocked)}]"

        symbolic = constraint_to_str(constraint)

        return ConstraintAnalysis(
            index=index, constraint_json=constraint, tag=tag,
            status=status, description=desc, symbolic=symbolic,
            slack=slack, lhs_value=lhs_val, rhs_value=rhs_val,
            symbol_steps=symbol_steps,
        )

    @staticmethod
    def _analyze_divisible(
        index: int,
        constraint: dict,
        payload: dict,
        concrete_map: dict[str, z3.ArithRef],
    ) -> ConstraintAnalysis:
        """Analyze a Divisible constraint (binary: satisfied or violated)."""
        x_val = z3.simplify(resolve_expr(payload["x"], concrete_map)).as_long()
        by_val = z3.simplify(resolve_expr(payload["by"], concrete_map)).as_long()
        satisfied = (by_val != 0) and (x_val % by_val == 0)
        status = ConstraintStatus.SATISFIED if satisfied else ConstraintStatus.VIOLATED
        symbolic = constraint_to_str(constraint)
        return ConstraintAnalysis(
            index=index, constraint_json=constraint, tag="Divisible",
            status=status,
            description=f"Divisible: {x_val} % {by_val} == 0 ({'yes' if satisfied else 'NO'})",
            symbolic=symbolic,
            lhs_value=x_val, rhs_value=by_val,
        )

    @staticmethod
    def _analyze_equality(
        index: int,
        constraint: dict,
        tag: str,
        payload: list,
        concrete_map: dict[str, z3.ArithRef],
    ) -> ConstraintAnalysis:
        """Analyze an Eq or Ne constraint (binary)."""
        lhs_val = z3.simplify(resolve_expr(payload[0], concrete_map)).as_long()
        rhs_val = z3.simplify(resolve_expr(payload[1], concrete_map)).as_long()
        if tag == "Eq":
            satisfied = lhs_val == rhs_val
            desc = f"Eq: {lhs_val} == {rhs_val}"
        else:
            satisfied = lhs_val != rhs_val
            desc = f"Ne: {lhs_val} != {rhs_val}"
        status = ConstraintStatus.SATISFIED if satisfied else ConstraintStatus.VIOLATED
        symbolic = constraint_to_str(constraint)
        return ConstraintAnalysis(
            index=index, constraint_json=constraint, tag=tag,
            status=status, description=desc, symbolic=symbolic,
            lhs_value=lhs_val, rhs_value=rhs_val,
        )

    def find_mus(
        self,
        hard_constraints: list[dict],
        domains: dict[str, list[int]],
    ) -> list[tuple[int, str, str]]:
        """Find a Minimum Unsatisfiable Subset via deletion-based algorithm.

        Builds a fresh solver for each removal attempt so the main solver
        state is not disturbed.  Operates on original (pre-simplification)
        constraints for interpretability.

        Returns:
            List of ``(index, label, z3_expr_str)`` for each MUS member.
        """
        labeled: list[tuple[str, z3.BoolRef]] = []

        # Positivity guards
        for name, var in self.symbol_map.items():
            labeled.append((f"positivity({name})", var > 0))

        # Domain constraints
        for name, values in domains.items():
            if name not in self.symbol_map:
                continue
            sym = self.symbol_map[name]
            labeled.append((f"domain({name})", z3.Or([sym == v for v in values])))

        # Hard constraints
        for i, c in enumerate(hard_constraints):
            tag = next(iter(c.keys()))
            z3_c = resolve_constraint(c, self.symbol_map)
            labeled.append((f"hard[{i}]({tag})", z3_c))

        # Deletion-based MUS
        remaining = list(range(len(labeled)))
        for idx in list(remaining):
            candidate = [i for i in remaining if i != idx]
            s = z3.Solver()
            for i in candidate:
                s.add(labeled[i][1])
            if s.check() != z3.sat:
                # Still UNSAT without this constraint → not needed
                remaining.remove(idx)

        return [(i, labeled[i][0], str(labeled[i][1])) for i in remaining]
