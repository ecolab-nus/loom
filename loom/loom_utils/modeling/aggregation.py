"""Aggregate scenario-based timing expressions across the ETG hierarchy using Pure Python AST.
"""

from __future__ import annotations

from ..ast import Expr, Const, Sym, Add, Mul, Max, IfElse, Eq, parse_expr, parse_constraint

# Large enough to never be chosen as the minimum.
_INF = Const(2**31)


def compute_total_time_ast(
    variant: dict,
) -> Expr:
    """Build the Pure Python AST expression for the total pipeline execution time."""
    t_comp = _aggregate_scope(variant["compute_scope"])
    t_mem = _aggregate_scope(variant["memory_scope"])

    symbols_meta = variant["constraint_scope"]["metadata"].get("symbols", {})
    booleans_meta = variant["constraint_scope"]["metadata"].get("booleans", [])
    
    t_max = Max([t_comp, t_mem])
    if "is_double_buffer" in symbols_meta or "is_double_buffer" in booleans_meta:
        is_db = Sym("is_double_buffer")
        t_stage = IfElse(Eq(is_db, Const(1)), t_max, Add([t_comp, t_mem]))
    else:
        t_stage = t_max

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_iter = parse_expr(iter_num["seq_iter"])
    temp_iter_exprs = [parse_expr(t) for t in iter_num["temp_iter"]]
    
    p = Const(1)
    for t in temp_iter_exprs: p = Mul(p, t)
    
    return Mul(Mul(t_stage, seq_iter), p)


def _fold_scenarios(
    scenarios: list[dict],
) -> Expr:
    """Fold a scenarios list into a nested If-tree (right-fold)."""
    assert scenarios, "scenarios list must be non-empty"

    result = _INF
    for scenario in reversed(scenarios):
        cond = parse_constraint(scenario["constraints"])
        cost = parse_expr(scenario["time_cost"])
        result = IfElse(cond, cost, result)
    return result


def _aggregate_scope(
    scope: dict,
) -> Expr:
    """Compute the AST for a single scope (compute or memory)."""
    stage_times: list[Expr] = []

    for stage in scope["stages"]:
        parallel = stage["Parallel"]
        par_times: list[Expr] = []
        for item in parallel:
            seq = item["Sequential"]
            scenarios = seq["scenarios"]
            par_times.append(_fold_scenarios(scenarios))

        if len(par_times) == 1:
            stage_time = par_times[0]
        else:
            stage_time = Max(par_times)
        
        stage_times.append(stage_time)

    return Add(stage_times)
