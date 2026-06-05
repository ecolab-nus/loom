"""Aggregate scenario-based timing expressions across the ETG hierarchy using Pure Python AST.

The ETG uses a nested for_loop_block tree structure:

    variant.kernel_block
      load_scope.stages[]     (DRAM -> L1 copies)
      compute_scope.stages[]
        stage = { for_loop_block: {load_scope, compute_scope, store_scope, trip_count, ...} }  # sub-loop
              | { Parallel: [{Sequential: {scenarios}}] }                            # leaf op
      store_scope.stages[]    (L1 -> DRAM copies)
    variant.constraint_scope  (unchanged — drives hard constraints only)

Cost model:
    t_body(block) = t_compute + t_store
    t_self(block) = IF(is_double_buffer==1, MAX(t_load, t_body), t_load + t_body)
    stage cost when the stage is a for_loop_block:
        t_self(block) * trip_count
        + is_double_buffer * t_load  # DB boundary compensation
    stages within one scope are sequential → summed
    siblings within one Parallel node execute concurrently → MAX'd
"""

from __future__ import annotations

from ..ast import (
    Expr, Const, Sym, CommonExpr, Add, Mul, Div, Max, Min, IfElse, Switch, Eq, Top,
    parse_expr, parse_constraint,
)

TIME_COST_SCALE = 64

_INF_CYCLES = 2**31
_SCALED_INF = Const((_INF_CYCLES + TIME_COST_SCALE - 1) // TIME_COST_SCALE)


class _CommonExprBuilder:
    def __init__(self) -> None:
        self._counter = 0

    def wrap(self, prefix: str, expr: Expr) -> Expr:
        if isinstance(expr, (Const, Sym, CommonExpr)):
            return expr

        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return CommonExpr(name, expr)


def compute_total_time_ast(
    variant: dict,
    *,
    scale_time_costs: bool = True,
    use_common_expr: bool = False,
) -> Expr:
    """Build the Pure Python AST for the total pipeline execution time.

    Reads variant["kernel_block"] and recurses through the for_loop_block tree.
    variant["constraint_scope"] is consumed only for the is_double_buffer flag;
    trip-count feasibility constraints are handled separately by SolverContext.
    """
    meta = variant["constraint_scope"]["metadata"]
    booleans_meta = meta.get("booleans", [])
    symbols_meta  = meta.get("symbols", {})
    has_db = ("is_double_buffer" in booleans_meta) or ("is_double_buffer" in symbols_meta)

    common = _CommonExprBuilder() if use_common_expr else None
    return _block_time_ast(variant["kernel_block"], has_db, scale_time_costs, common)


def _block_time_ast(
    block: dict,
    has_db: bool,
    scale_time_costs: bool,
    common: _CommonExprBuilder | None,
) -> Expr:
    """Cost of one block (kernel_block or for_loop_block), excluding its own trip_count.

    The caller is responsible for multiplying by trip_count when this block is
    consumed as a stage of a parent scope.
    """
    t_load = _scope_time_ast(block["load_scope"], has_db, scale_time_costs, common)
    t_comp = _scope_time_ast(block["compute_scope"], has_db, scale_time_costs, common)
    t_store = _scope_time_ast(block["store_scope"], has_db, scale_time_costs, common)
    t_body = Add([t_comp, t_store])
    if has_db and common is not None:
        t_load = common.wrap("block_load", t_load)
        t_body = common.wrap("block_body", t_body)
    return _combine_double_buffer(t_load, t_body, has_db)


def _scope_time_ast(
    scope: dict,
    has_db: bool,
    scale_time_costs: bool,
    common: _CommonExprBuilder | None,
) -> Expr:
    """Sum of stage times within one scope. Returns Const(0) for an empty scope."""
    stages = scope.get("stages", [])
    if not stages:
        return Const(0)
    stage_times = [_stage_time_ast(s, has_db, scale_time_costs, common) for s in stages]
    return stage_times[0] if len(stage_times) == 1 else Add(stage_times)


def _stage_time_ast(
    stage: dict,
    has_db: bool,
    scale_time_costs: bool,
    common: _CommonExprBuilder | None,
) -> Expr:
    """Cost contribution of one stage entry."""
    if "for_loop_block" in stage:
        flb = stage["for_loop_block"]
        t_load = _scope_time_ast(flb["load_scope"], has_db, scale_time_costs, common)
        t_comp = _scope_time_ast(flb["compute_scope"], has_db, scale_time_costs, common)
        t_store = _scope_time_ast(flb["store_scope"], has_db, scale_time_costs, common)
        t_body = Add([t_comp, t_store])
        if has_db and common is not None:
            t_load = common.wrap("loop_load", t_load)
            t_body = common.wrap("loop_body", t_body)
        trip = parse_expr(flb["trip_count"])
        base = Mul(trip, _combine_double_buffer(t_load, t_body, has_db))
        if has_db:
            return Add([base, Mul(Sym("is_double_buffer"), Min([t_load, t_body]))])
        return base

    if "Parallel" in stage:
        par_times = [
            _fold_scenarios(item["Sequential"]["scenarios"], scale_time_costs)
            for item in stage["Parallel"]
        ]
        return par_times[0] if len(par_times) == 1 else Max(par_times)

    raise ValueError(f"Unknown stage shape: keys={list(stage.keys())}")


def _combine_double_buffer(t_load: Expr, t_body: Expr, has_db: bool) -> Expr:
    t_sum = Add([t_load, t_body])
    if has_db:
        t_max = Max([t_load, t_body])
        return IfElse(Eq(Sym("is_double_buffer"), Const(1)), t_max, t_sum)
    return t_sum


def _fold_scenarios(scenarios: list[dict], scale_time_costs: bool) -> Expr:
    """Fold a scenarios list into a Switch (or bare expression for single-case)."""
    assert scenarios, "scenarios list must be non-empty"

    if len(scenarios) == 1:
        cond = parse_constraint(scenarios[0]["constraints"])
        cost = _parse_time_cost(scenarios[0]["time_cost"], scale_time_costs)
        if isinstance(cond, Top):
            return cost
        return Switch(cases=[(cond, cost)], default=_switch_default(scale_time_costs))

    cases = [
        (
            parse_constraint(s["constraints"]),
            _parse_time_cost(s["time_cost"], scale_time_costs),
        )
        for s in scenarios
    ]
    return Switch(cases=cases, default=_switch_default(scale_time_costs))


def _parse_time_cost(time_cost: object, scale_time_costs: bool) -> Expr:
    """Parse scenario time cost in solver units while leaving ETG data untouched."""
    parsed = parse_expr(time_cost)
    if not scale_time_costs:
        return parsed
    return Div(parsed, Const(TIME_COST_SCALE))


def _switch_default(scale_time_costs: bool) -> Expr:
    """Large fallback value in the same units as the scenario cases."""
    if scale_time_costs:
        return _SCALED_INF
    return Const(_INF_CYCLES)
