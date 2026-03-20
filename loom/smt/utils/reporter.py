"""Console reporting utilities for SMT solver results.

Responsible for formatting and printing the per-scope / per-stage / per-queue
timing breakdown after optimal symbol values have been determined.
"""

import sys
from typing import TextIO

import z3

from ..core.expr_resolver import resolve_expr, resolve_constraint


def eval_expr(expr_dict: dict, concrete_map: dict[str, z3.ArithRef]) -> int:
    """Evaluate a JSON Expr to a concrete integer given a concrete symbol map."""
    # Since concrete_map contains only IntVal constants, resolve_expr returns 
    # an expression where all symbols are replaced by constants. 
    # simplify() should then reduce it to a single IntNumRef.
    res = resolve_expr(expr_dict, concrete_map)
    simplified = z3.simplify(res)
    
    if z3.is_int_value(simplified):
        return simplified.as_long()
        
    # Fallback for extremely complex expressions or non-linear division that 
    # simplify() might not fully reduce.
    s = z3.Solver()
    s.add(z3.Int("tmp_eval") == simplified)
    if s.check() == z3.sat:
        return s.model().eval(simplified).as_long()
        
    raise RuntimeError(f"Could not evaluate expression to a constant: {expr_dict}")


def print_breakdown(
    variant: dict,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
) -> None:
    """Write a hierarchical timing breakdown for the given symbol assignments.

    Substitutes concrete integer values for all symbols, then writes every
    scope → stage → queue resolved_time, scope totals, and the final T_total
    derivation to *file* (defaults to sys.stdout).

    Args:
        variant:     A single variant dict from the staged ETG JSON.
        assignments: Mapping of symbol name → concrete integer value.
        min_val:     The optimal T_total value.
        index:       0-based index of this variant in the full list.
        total:       Total number of variants.
        file:        Output stream. Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    variant_name = variant.get("variant_name", "unknown")
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"Optimal T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        p(f"  {sym} = {val}")
    p()

    concrete_map: dict[str, z3.ArithRef] = {
        name: z3.IntVal(val) for name, val in assignments.items()
    }

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_val = eval_expr(iter_num["seq_iter"], concrete_map)
    temp_vals = [eval_expr(t, concrete_map) for t in iter_num["temp_iter"]]
    temp_product = 1
    for v in temp_vals:
        temp_product *= v

    temp_str = " × ".join(str(v) for v in temp_vals)
    p(f"Iteration factors:  seq_iter={seq_val},  temp_iter=[{temp_str}]  (product={temp_product})")
    p()

    scope_totals: list[tuple[str, int]] = []

    for scope_key in ("compute_scope", "memory_scope"):
        scope = variant[scope_key]
        scope_name = scope.get("scope_name", scope_key)
        p(f"  [{scope_name}]")

        scope_total = 0
        for stage in scope["stages"]:
            stage_id = stage.get("stage_id", "?")
            p(f"    Stage {stage_id}:")

            seq = stage["Parallel"]["Sequential"]
            scenarios = seq["scenarios"]

            matched_idx = None
            matched_cost = None
            for si, scenario in enumerate(scenarios):
                cond_z3 = resolve_constraint(scenario["constraints"], concrete_map)
                if z3.is_true(z3.simplify(cond_z3)):
                    matched_idx = si
                    matched_cost = eval_expr(scenario["time_cost"], concrete_map)
                    break

            if matched_cost is not None:
                p(f"      scenario[{matched_idx}]  {matched_cost:>10,} cycles")
                scope_total += matched_cost
            else:
                p(f"      (no scenario matched)")

        p(f"  {'→ scope total':16s}  {scope_total:>10,} cycles")
        p()
        scope_totals.append((scope_name, scope_total))

    t_comp = next(v for n, v in scope_totals if "compute" in n.lower())
    t_mem  = next(v for n, v in scope_totals if "memory"  in n.lower())
    t_stage = max(t_comp, t_mem)

    p(f"  T_comp  = {t_comp:>10,} cycles")
    p(f"  T_mem   = {t_mem:>10,} cycles")
    p(f"  T_stage = max(T_comp, T_mem) = {t_stage:,} cycles")
    p(f"  T_total = {t_stage:,} × {seq_val} × {temp_product} = {t_stage * seq_val * temp_product:,} cycles")
    p()


def print_unsat_core(
    variant_name: str,
    unsat_core_info: list[tuple[str, str]],
    context: str = "",
    file: TextIO = None,
) -> None:
    """Write UNSAT core details for a variant.

    Args:
        variant_name: Name of the variant.
        unsat_core_info: List of (constraint_name, constraint_expr_str) tuples
                         as produced by ``SolverContext.last_unsat_core_info``.
        context: Additional context string (e.g., "Optimum bound", "Infeasible").
        file: Output stream. Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    p(f"[UNSAT Core] {variant_name}" + (f" ({context})" if context else ""))
    for name, expr in unsat_core_info:
        p(f"  {name}: {expr}")
    p()


def print_active_constraints(
    variant_name: str,
    active: list[tuple[int, dict, str]],
    file: TextIO = None,
) -> None:
    """Print active (tight) constraints at the optimum.

    Args:
        variant_name: Name of the variant.
        active: List of (index, constraint_json, description_str) tuples
                as produced by ``SolverContext.find_active_constraints()``.
        file: Output stream. Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    p(f"[Active Constraints] {variant_name}")
    if not active:
        p("  (no tight inequality constraints)")
    for idx, c_json, desc in active:
        p(f"  hard[{idx}]: {desc}")
        p(f"    JSON: {c_json}")
    p()


def print_mus(
    variant_name: str,
    mus: list[tuple[int, str, str]],
    file: TextIO = None,
) -> None:
    """Print Minimum Unsatisfiable Subset for an UNSAT variant.

    Args:
        variant_name: Name of the variant.
        mus: List of (index, label, z3_expr_str) tuples as produced by
             ``SolverContext.find_mus()``.
        file: Output stream. Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    p(f"[MUS] {variant_name}  ({len(mus)} constraints)")
    for idx, label, expr_str in mus:
        p(f"  [{idx}] {label}: {expr_str}")
    p()


def print_result_summary(
    variant_name: str,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
) -> None:
    """Print compact result: optimal value and block size assignments.

    Used in non-debug mode as a lightweight alternative to ``print_breakdown()``.

    Args:
        variant_name: Name of the variant.
        assignments: Mapping of symbol name → concrete integer value.
        min_val:     The optimal T_total value.
        index:       0-based index of this variant in the full list.
        total:       Total number of variants.
        file:        Output stream. Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"  T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        p(f"  {sym} = {val}")
    p()
