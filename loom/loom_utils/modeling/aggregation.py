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
    stage cost when the stage is a for_loop_block = t_self(block) * trip_count
    stages within one scope are sequential → summed
    siblings within one Parallel node execute concurrently → MAX'd
"""

from __future__ import annotations

from ..ast import Expr, Const, Sym, Add, Mul, Max, IfElse, Switch, Eq, Top, parse_expr, parse_constraint

# Large enough to never be chosen as the minimum.
_INF = Const(2**31)


def compute_total_time_ast(variant: dict) -> Expr:
    """Build the Pure Python AST for the total pipeline execution time.

    Reads variant["kernel_block"] and recurses through the for_loop_block tree.
    variant["constraint_scope"] is consumed only for the is_double_buffer flag;
    iter_num divisibility constraints are handled separately by SolverContext.
    """
    meta = variant["constraint_scope"]["metadata"]
    booleans_meta = meta.get("booleans", [])
    symbols_meta  = meta.get("symbols", {})
    has_db = ("is_double_buffer" in booleans_meta) or ("is_double_buffer" in symbols_meta)

    return _block_time_ast(variant["kernel_block"], has_db)


def _block_time_ast(block: dict, has_db: bool) -> Expr:
    """Cost of one block (kernel_block or for_loop_block), excluding its own trip_count.

    The caller is responsible for multiplying by trip_count when this block is
    consumed as a stage of a parent scope.
    """
    t_load = _scope_time_ast(block["load_scope"], has_db)
    t_comp = _scope_time_ast(block["compute_scope"], has_db)
    t_store = _scope_time_ast(block["store_scope"], has_db)
    t_body = Add([t_comp, t_store])
    return _combine_double_buffer(t_load, t_body, has_db)


def _scope_time_ast(scope: dict, has_db: bool) -> Expr:
    """Sum of stage times within one scope. Returns Const(0) for an empty scope."""
    stages = scope.get("stages", [])
    if not stages:
        return Const(0)
    return Add([_stage_time_ast(s, has_db) for s in stages])


def _stage_time_ast(stage: dict, has_db: bool) -> Expr:
    """Cost contribution of one stage entry."""
    if "for_loop_block" in stage:
        flb        = stage["for_loop_block"]
        inner_cost = _block_time_ast(flb, has_db)
        trip       = parse_expr(flb["trip_count"])
        return Mul(trip, inner_cost)

    if "Parallel" in stage:
        par_times = [
            _fold_scenarios(item["Sequential"]["scenarios"])
            for item in stage["Parallel"]
        ]
        return par_times[0] if len(par_times) == 1 else Max(par_times)

    raise ValueError(f"Unknown stage shape: keys={list(stage.keys())}")


def _combine_double_buffer(t_load: Expr, t_body: Expr, has_db: bool) -> Expr:
    t_sum = Add([t_load, t_body])
    t_max = Max([t_load, t_body])
    if has_db:
        return IfElse(Eq(Sym("is_double_buffer"), Const(1)), t_max, t_sum)
    return t_sum


def _fold_scenarios(scenarios: list[dict]) -> Expr:
    """Fold a scenarios list into a Switch (or bare expression for single-case)."""
    assert scenarios, "scenarios list must be non-empty"

    if len(scenarios) == 1:
        cond = parse_constraint(scenarios[0]["constraints"])
        cost = parse_expr(scenarios[0]["time_cost"])
        if isinstance(cond, Top):
            return cost
        return Switch(cases=[(cond, cost)], default=_INF)

    cases = [
        (parse_constraint(s["constraints"]), parse_expr(s["time_cost"]))
        for s in scenarios
    ]
    return Switch(cases=cases, default=_INF)
