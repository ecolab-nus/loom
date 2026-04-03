"""Aggregate scenario-based timing expressions across the ETG hierarchy using Pure Python AST.

Aggregation rules (from spec):
  - Within a stage:  Parallel Sequential(s) → Max of Sequential times.
  - Across stages:   stages execute in SERIAL → Sum of per-stage maxima.
  - Across scopes:   T_stage = max(T_comp, T_mem).
  - Total:           T_total = T_stage * seq_iter * product(temp_iter).

Each Sequential's time is derived from its ``scenarios`` list — a piecewise
conditional model folded into a nested If-tree.
"""

from __future__ import annotations
import math

from .ast_core import Node, Expr, Constraint, Const, Sym, Add, Mul, Div, Mod, Min, Max, IfElse, Eq, Ne, Ge, Gt, Le, Lt, And, Or, Divisible
from .ast_parser import parse_expr, parse_constraint

# Large enough to never be chosen as the minimum.
_INF = Const(2**31)


def compute_total_time_ast(
    variant: dict,
) -> Expr:
    """Build the Pure Python AST expression for the total pipeline execution time.
    """
    t_comp = _aggregate_scope(variant["compute_scope"])
    t_mem = _aggregate_scope(variant["memory_scope"])

    # Double-buffered: compute and memory overlap → max(T_comp, T_mem)
    # Not double-buffered: sequential execution   → T_comp + T_mem
    # Note: is_double_buffer is captured as a symbolic Sym if it's there.
    
    symbols_meta = variant["constraint_scope"]["metadata"].get("symbols", {})
    booleans_meta = variant["constraint_scope"]["metadata"].get("booleans", [])
    
    t_max = Max([t_comp, t_mem])
    if "is_double_buffer" in symbols_meta or "is_double_buffer" in booleans_meta:
        is_db = Sym("is_double_buffer")
        t_stage = IfElse(Eq(is_db, Const(1)), t_max, Add([t_comp, t_mem]))
    else:
        t_stage = t_max  # default to max

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_iter = parse_expr(iter_num["seq_iter"])
    temp_iter_exprs = [parse_expr(t) for t in iter_num["temp_iter"]]
    
    # temp_iter_product = functools.reduce(operator.mul, temp_iter_exprs)
    # Manual product for simplicity
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
        assert isinstance(parallel, list), (
            f"Stage {stage.get('stage_id', '?')}: Parallel must be a list"
        )

        par_times: list[Expr] = []
        for i, item in enumerate(parallel):
            assert "Sequential" in item, (
                f"Stage {stage.get('stage_id', '?')}.Parallel[{i}]: missing 'Sequential'"
            )
            seq = item["Sequential"]
            scenarios = seq["scenarios"]
            assert scenarios, (
                f"Stage {stage.get('stage_id', '?')}.Parallel[{i}].Sequential: missing scenarios"
            )
            par_times.append(_fold_scenarios(scenarios))

        # Within a stage, multiple parallel workloads are running → max execution time.
        if len(par_times) == 1:
            stage_time = par_times[0]
        else:
            stage_time = Max(par_times)
        
        stage_times.append(stage_time)

    if not stage_times:
        raise ValueError(
            f"Scope '{scope.get('scope_name', '?')}' has no stages with scenarios"
        )

    # Serial stages → sum
    return Add(stage_times)
