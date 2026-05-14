"""Console reporting utilities for Loom solver results.
"""

from __future__ import annotations
import sys
from typing import TextIO

from ..ast import parse_expr, parse_constraint
from .analysis import ConstraintAnalysis, ConstraintStatus


def print_breakdown(
    variant: dict,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
) -> None:
    """Write a hierarchical timing breakdown for the given symbol assignments."""
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    variant_name = variant.get("variant_name", "unknown")
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"Optimal T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        if not sym.startswith("__"):
            p(f"  {sym} = {val}")
    p()

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_val = parse_expr(iter_num["seq_iter"]).eval(assignments)
    temp_vals = [parse_expr(t).eval(assignments) for t in iter_num["temp_iter"]]
    temp_product = 1
    for v in temp_vals:
        temp_product *= v

    temp_str = " × ".join(str(v) for v in temp_vals)
    p(f"Iteration factors:  seq_iter={seq_val},  temp_iter=[{temp_str}]  (product={temp_product})")
    p()

    root = variant.get("kernel_block", variant)
    _print_block(root, assignments, depth=1, p=p)

    p(f"  T_total = {min_val:,} cycles")
    p()


def _eval_scenario(scenarios: list[dict], assignments: dict) -> tuple[int | None, int | None]:
    for si, scenario in enumerate(scenarios):
        if parse_constraint(scenario["constraints"]).eval(assignments):
            return si, parse_expr(scenario["time_cost"]).eval(assignments)
    return None, None


def _print_block(block: dict, assignments: dict, depth: int, p) -> int:
    """Recursively print a kernel_block or for_loop_block; return evaluated total cost."""
    indent = "  " * depth

    t_load = _print_scope(block["load_scope"], assignments, depth, p)
    t_comp = _print_scope(block["compute_scope"], assignments, depth, p)
    t_store = _print_scope(block["store_scope"], assignments, depth, p)
    t_body = t_comp + t_store

    is_db = assignments.get("is_double_buffer", 0)
    t_self = max(t_load, t_body) if is_db else (t_load + t_body)
    p(f"{indent}T_load={t_load:,}  T_compute={t_comp:,}  "
      f"T_store={t_store:,}  T_compute_store={t_body:,}  "
      f"{'max' if is_db else 'sum'}={t_self:,} cycles")
    return t_self


def _print_scope(scope: dict, assignments: dict, depth: int, p) -> int:
    """Print stages within one scope; return sum of stage costs."""
    stages = scope.get("stages", [])
    if not stages:
        return 0
    indent = "  " * depth
    scope_name = scope.get("scope_name", "scope")
    p(f"{indent}[{scope_name}]")
    total = 0
    for stage in stages:
        total += _print_stage(stage, assignments, depth + 1, p)
    p(f"{indent}  → scope total: {total:,} cycles")
    return total


def _print_stage(stage: dict, assignments: dict, depth: int, p) -> int:
    """Print one stage; return its evaluated cost."""
    indent = "  " * depth
    stage_id = stage.get("stage_id", "?")

    if "for_loop_block" in stage:
        flb = stage["for_loop_block"]
        trip = parse_expr(flb["trip_count"]).eval(assignments)
        block_sym = flb.get("block_sym") or "loop"
        p(f"{indent}Stage {stage_id} [for_loop_block block_sym={block_sym!r} trip={trip}]")
        inner = _print_block(flb, assignments, depth + 1, p)
        cost = inner * trip
        p(f"{indent}  → {inner:,} × {trip} = {cost:,} cycles")
        return cost

    if "Parallel" in stage:
        parallel = stage["Parallel"]
        p(f"{indent}Stage {stage_id} [Parallel × {len(parallel)}]")
        branch_costs = []
        for i, item in enumerate(parallel):
            scenarios = item["Sequential"]["scenarios"]
            si, cost = _eval_scenario(scenarios, assignments)
            label = f"Parallel[{i}] scenario[{si}]" if si is not None else f"Parallel[{i}] (no match)"
            p(f"{indent}  {label:<30s}  {cost if cost is not None else '?':>10} cycles")
            branch_costs.append(cost or 0)
        stage_cost = max(branch_costs)
        if len(branch_costs) > 1:
            p(f"{indent}  → stage max: {stage_cost:,} cycles")
        return stage_cost

    return 0


def print_active_constraints(
    variant_name: str,
    analyses: list[ConstraintAnalysis],
    file: TextIO = None,
) -> None:
    """Print constraint analysis at the optimum."""
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    p(f"[Constraint Analysis] {variant_name}")

    tight = [a for a in analyses if a.status == ConstraintStatus.TIGHT]
    walls = [a for a in analyses if a.status == ConstraintStatus.DISCRETE_WALL]
    slacked = [a for a in analyses if a.status == ConstraintStatus.ACTIVE_SLACK]
    satisfied = [a for a in analyses if a.status == ConstraintStatus.SATISFIED]
    violated = [a for a in analyses if a.status == ConstraintStatus.VIOLATED]

    if violated:
        p("  VIOLATED (should not happen at feasible optimum):")
        for a in violated: _print_constraint_line(p, a)

    if tight:
        p("  TIGHT constraints (slack=0):")
        for a in tight: _print_constraint_line(p, a)

    if walls:
        p("  DISCRETE WALLS (next step would violate):")
        for a in walls:
            _print_constraint_line(p, a)
            for s in a.symbol_steps:
                if s.next_value is not None and s.would_violate:
                    p(f"      {s.symbol}: {s.current_value} -> {s.next_value}"
                      f"  would cost +{s.step_cost} (VIOLATES)")

    if slacked:
        p("  Inequality constraints with slack:")
        for a in slacked: _print_constraint_line(p, a)

    sat_tags: dict[str, int] = {}
    for a in satisfied:
        sat_tags[a.tag] = sat_tags.get(a.tag, 0) + 1
    if sat_tags:
        parts = [f"{count} {tag}" for tag, count in sorted(sat_tags.items())]
        p(f"  Other satisfied: {', '.join(parts)}")
    p()


def _print_constraint_line(p, a: ConstraintAnalysis) -> None:
    prefix = f"    hard[{a.index}]:"
    if a.symbolic:
        p(f"{prefix} {a.symbolic}")
        p(f"    {' ' * len(f'hard[{a.index}]:')}  → {a.description}")
    else:
        p(f"{prefix} {a.description}")


def print_unsat_core(
    variant_name: str,
    unsat_core_info: list[tuple[str, str]],
    context: str = "",
    file: TextIO = None,
) -> None:
    if file is None: file = sys.stdout
    def p(*args, **kwargs): print(*args, **kwargs, file=file)
    p(f"[UNSAT Core] {variant_name}" + (f" ({context})" if context else ""))
    for name, expr in unsat_core_info:
        p(f"  {name}: {expr}")
    p()


def print_mus(
    variant_name: str,
    mus: list[tuple[int, str, str]],
    file: TextIO = None,
) -> None:
    if file is None: file = sys.stdout
    def p(*args, **kwargs): print(*args, **kwargs, file=file)
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
    if file is None: file = sys.stdout
    def p(*args, **kwargs): print(*args, **kwargs, file=file)
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"  T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        if not sym.startswith("__"):
            p(f"  {sym} = {val}")
    p()
