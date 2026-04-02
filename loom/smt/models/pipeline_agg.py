"""Aggregate scenario-based timing expressions across the ETG hierarchy.

Aggregation rules (from spec):
  - Within a stage:  Parallel Sequential(s) → Max of Sequential times.
  - Across stages:   stages execute in SERIAL → Sum of per-stage maxima.
  - Across scopes:   T_stage = max(T_comp, T_mem).
  - Total:           T_total = T_stage * seq_iter * product(temp_iter).

Each Sequential's time is derived from its ``scenarios`` list — a piecewise
conditional model folded into a Z3 nested If-tree.

This module only builds Z3 expression trees; it does not interact with the
solver directly.
"""

import functools
import operator

import z3

from ..core.expr_resolver import resolve_expr, resolve_constraint


# Sentinel value used as the default (no-match) branch in the scenario
# If-tree.  Large enough to never be chosen as the minimum by the solver.
_INF = z3.IntVal(2**31)


def compute_total_time(
    variant: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Build the Z3 expression for the total pipeline execution time.

    Args:
        variant:    A single variant dict from the ETG JSON.
        symbol_map: Z3 integer variables keyed by symbol name.

    Returns:
        A Z3 ArithRef for T_total = T_stage * seq_iter * Πtemp_iter,
        where T_stage = max(T_comp, T_mem) if double-buffered (overlap),
        or T_stage = T_comp + T_mem if not (sequential).
    """
    t_comp = _aggregate_scope(variant["compute_scope"], symbol_map)
    t_mem = _aggregate_scope(variant["memory_scope"], symbol_map)

    # is_double_buffer: symbolic Int {0,1} — Z3 decides the optimal value.
    # Double-buffered: compute and memory overlap → max(T_comp, T_mem)
    # Not double-buffered: sequential execution   → T_comp + T_mem
    is_db = symbol_map.get("is_double_buffer")
    t_max = z3.If(t_comp >= t_mem, t_comp, t_mem)
    if is_db is not None:
        t_stage = z3.If(is_db == 1, t_max, t_comp + t_mem)
    else:
        t_stage = t_max  # backward compat: default to max if no boolean declared

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_iter = resolve_expr(iter_num["seq_iter"], symbol_map)
    temp_iter_exprs = [resolve_expr(t, symbol_map) for t in iter_num["temp_iter"]]
    temp_iter_product = functools.reduce(operator.mul, temp_iter_exprs)

    return t_stage * seq_iter * temp_iter_product


def _fold_scenarios(
    scenarios: list[dict],
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Fold a scenarios list into a Z3 nested If-tree (right-fold).

    Each scenario is ``{"constraints": ..., "time_cost": ...}``.
    Produces::

        If(c_0, t_0, If(c_1, t_1, ... If(c_n, t_n, INF)))

    where INF is a large sentinel that ensures no-match is never optimal.

    Args:
        scenarios:  Non-empty list of scenario dicts.
        symbol_map: Z3 integer variables keyed by symbol name.

    Returns:
        A Z3 ArithRef representing the piecewise time expression.
    """
    assert scenarios, "scenarios list must be non-empty"

    result = _INF
    for scenario in reversed(scenarios):
        cond = resolve_constraint(scenario["constraints"], symbol_map)
        cost = resolve_expr(scenario["time_cost"], symbol_map)
        result = z3.If(cond, cost, result)
    return result


def _aggregate_scope(
    scope: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Compute the total time for a single scope (compute or memory).

    Stages are serial (sum).  Within a stage, ``Parallel`` contains one or
    more ``Sequential`` nodes whose scenario-derived times are taken as max
    (parallel execution).

    Args:
        scope:      A compute_scope or memory_scope dict.
        symbol_map: Z3 integer variables keyed by symbol name.

    Returns:
        A Z3 ArithRef for the total scope time.

    Raises:
        AssertionError: If a stage's Parallel/Sequential structure is invalid.
        ValueError:     If the scope has no stages with resolvable times.
    """
    stage_times: list[z3.ArithRef] = []

    for stage in scope["stages"]:
        parallel = stage["Parallel"]
        assert isinstance(parallel, list), (
            f"Stage {stage.get('stage_id', '?')}: Parallel must be a list"
        )

        par_times: list[z3.ArithRef] = []
        for i, item in enumerate(parallel):
            assert "Sequential" in item, (
                f"Stage {stage.get('stage_id', '?')}.Parallel[{i}]: missing 'Sequential'"
            )
            seq = item["Sequential"]
            scenarios = seq["scenarios"]
            assert scenarios, (
                f"Stage {stage.get('stage_id', '?')}.Parallel[{i}].Sequential: missing scenarios"
            )
            par_times.append(_fold_scenarios(scenarios, symbol_map))

        # Within a stage, multiple parallel workloads are running → max execution time.
        if len(par_times) == 1:
            stage_time = par_times[0]
        else:
            stage_time = par_times[0]
            for pt in par_times[1:]:
                stage_time = z3.If(pt >= stage_time, pt, stage_time)
        
        stage_times.append(stage_time)

    if not stage_times:
        raise ValueError(
            f"Scope '{scope.get('scope_name', '?')}' has no stages with scenarios"
        )

    # Serial stages → sum
    return functools.reduce(operator.add, stage_times)
