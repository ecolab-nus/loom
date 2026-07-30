"""Console reporting utilities for Loom solver results.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any, Callable, TextIO

from ..ast import parse_expr, parse_constraint
from .analysis import ConstraintAnalysis, ConstraintStatus

CostParser = Callable[[Any], Any]


@dataclass(frozen=True)
class _BlockTiming:
    total: int
    load: int
    compute: int
    store: int
    body: int


def print_breakdown(
    variant: dict,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
    cost_parser: CostParser = parse_expr,
    unit: str = "cycles",
) -> None:
    """Write a hierarchical timing breakdown for the given symbol assignments."""
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    variant_name = variant.get("variant_name", "unknown")
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"Optimal T_total: {min_val:,} {unit}")
    for sym, val in sorted(assignments.items()):
        if not sym.startswith("__"):
            p(f"  {sym} = {val}")
    p()

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_expr, seq_divisible = iter_num["seq_iter"]
    seq_val = parse_expr(seq_expr).eval(assignments)
    temp_pairs = iter_num["temp_iter"]
    temp_vals = [parse_expr(t[0]).eval(assignments) for t in temp_pairs]
    temp_product = 1
    for v in temp_vals:
        temp_product *= v

    temp_str = " × ".join(str(v) for v in temp_vals)
    p(
        f"Iteration factors:  seq_iter={seq_val} ({seq_divisible=}),  "
        f"temp_iter=[{temp_str}]  (product={temp_product})"
    )
    p()

    root = variant.get("kernel_block", variant)
    _print_block(root, assignments, depth=1, p=p, cost_parser=cost_parser, unit=unit)

    p(f"  T_total = {min_val:,} {unit}")
    p()


def _eval_scenario(
    scenarios: list[dict],
    assignments: dict,
    cost_parser: CostParser,
) -> tuple[int | None, int | None]:
    for si, scenario in enumerate(scenarios):
        if parse_constraint(scenario["constraints"]).eval(assignments):
            return si, cost_parser(scenario["time_cost"]).eval(assignments)
    return None, None


def _print_block(
    block: dict,
    assignments: dict,
    depth: int,
    p,
    cost_parser: CostParser,
    unit: str,
) -> int:
    """Recursively print a kernel_block or for_loop_block; return evaluated total cost."""
    return _print_block_timing(block, assignments, depth, p, cost_parser, unit).total


def _print_block_timing(
    block: dict,
    assignments: dict,
    depth: int,
    p,
    cost_parser: CostParser,
    unit: str,
) -> _BlockTiming:
    """Print a block and return the evaluated timing components."""
    indent = "  " * depth

    t_load = _print_scope(block["load_scope"], assignments, depth, p, cost_parser, unit)
    t_comp = _print_scope(block["compute_scope"], assignments, depth, p, cost_parser, unit)
    t_store = _print_scope(block["store_scope"], assignments, depth, p, cost_parser, unit)
    t_body = t_comp + t_store

    is_db = assignments.get("is_double_buffer", 0)
    t_self = max(t_load, t_body) if is_db else (t_load + t_body)
    p(f"{indent}T_load={t_load:,}  T_compute={t_comp:,}  "
      f"T_store={t_store:,}  T_compute_store={t_body:,}  "
      f"{'max' if is_db else 'sum'}={t_self:,} {unit}")
    return _BlockTiming(
        total=t_self,
        load=t_load,
        compute=t_comp,
        store=t_store,
        body=t_body,
    )


def _print_scope(
    scope: dict,
    assignments: dict,
    depth: int,
    p,
    cost_parser: CostParser,
    unit: str,
) -> int:
    """Print stages within one scope; return sum of stage costs."""
    stages = scope.get("stages", [])
    if not stages:
        return 0
    indent = "  " * depth
    scope_name = scope.get("scope_name", "scope")
    p(f"{indent}[{scope_name}]")
    total = 0
    for stage in stages:
        total += _print_stage(stage, assignments, depth + 1, p, cost_parser, unit)
    p(f"{indent}  → scope total: {total:,} {unit}")
    return total


def _print_stage(
    stage: dict,
    assignments: dict,
    depth: int,
    p,
    cost_parser: CostParser,
    unit: str,
) -> int:
    """Print one stage; return its evaluated cost."""
    indent = "  " * depth
    stage_id = stage.get("stage_id", "?")

    if "for_loop_block" in stage:
        flb = stage["for_loop_block"]
        trip = parse_expr(flb["trip_count"]).eval(assignments)
        block_sym = flb.get("block_sym") or "loop"
        p(
            f"{indent}Stage {stage_id} "
            f"[for_loop_block block_sym={block_sym!r} trip={trip}]"
        )
        timing = _print_block_timing(flb, assignments, depth + 1, p, cost_parser, unit)
        inner = timing.total
        base_cost = inner * trip
        db_tail = min(timing.load, timing.body) if assignments.get("is_double_buffer", 0) else 0
        cost = base_cost + db_tail
        if db_tail:
            p(
                f"{indent}  → {inner:,} × {trip} + "
                f"db_tail(min(load={timing.load:,}, body={timing.body:,})="
                f"{db_tail:,}) = {cost:,} {unit}"
            )
        else:
            p(f"{indent}  → {inner:,} × {trip} = {cost:,} {unit}")
        return cost

    if "Parallel" in stage:
        parallel = stage["Parallel"]
        p(f"{indent}Stage {stage_id} [Parallel × {len(parallel)}]")
        branch_costs = []
        for i, item in enumerate(parallel):
            sequential = item["Sequential"]
            scenarios = sequential["scenarios"]
            si, cost = _eval_scenario(scenarios, assignments, cost_parser)
            label = (
                f"Parallel[{i}] scenario[{si}]"
                if si is not None
                else f"Parallel[{i}] (no match)"
            )
            cost_display = cost if cost is not None else "?"
            p(f"{indent}  {label:<30s}  {cost_display:>10} {unit}")
            _print_func_breakdown(sequential, assignments, depth + 1, p, cost_parser, unit)
            branch_costs.append(cost or 0)
        stage_cost = max(branch_costs)
        if len(branch_costs) > 1:
            p(f"{indent}  → stage max: {stage_cost:,} {unit}")
        return stage_cost

    return 0


def _print_func_breakdown(
    sequential: dict,
    assignments: dict,
    depth: int,
    p,
    cost_parser: CostParser,
    unit: str,
) -> None:
    """Print Func-level timing for the schedules inside one Sequential branch."""
    indent = "  " * depth
    for schedule in sequential.get("schedules", []):
        func_node = schedule.get("Func") if isinstance(schedule, dict) else None
        if not func_node:
            continue

        func = func_node.get("func", {})
        label = func.get("op_label") or func.get("name") or "<unknown func>"
        scenarios = func_node.get("scenarios") or []
        _, cost = (
            _eval_scenario(scenarios, assignments, cost_parser)
            if scenarios
            else (None, None)
        )
        cost_display = f"{cost:,}" if cost is not None else "?"
        p(f"{indent}  {cost_display:>10} {unit}  {label}")


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
