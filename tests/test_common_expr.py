from __future__ import annotations

import cpmpy as cp

from loom.loom_utils.ast import Add, CommonExpr, Const, Max, Mul, Sym
from loom.loom_utils.modeling.aggregation import compute_total_time_ast
from loom.solver.core.cpmpy_expr_resolver import ExprResolver


def _const(value: int) -> dict:
    return {"Const": value}


def _scenario(cost: int) -> dict:
    return {"constraints": "True", "time_cost": _const(cost)}


def _parallel_stage(cost: int) -> dict:
    return {
        "Parallel": [
            {
                "Sequential": {
                    "scenarios": [_scenario(cost)],
                    "schedules": [],
                }
            }
        ],
    }


def _scope(stages: list[dict] | None = None) -> dict:
    return {"scope_name": "scope", "stages": stages or []}


def _variant(load_cost: int, compute_cost: int) -> dict:
    return {
        "kernel_block": {
            "load_scope": _scope(),
            "compute_scope": _scope(
                [
                    {
                        "for_loop_block": {
                            "load_scope": _scope([_parallel_stage(load_cost)]),
                            "compute_scope": _scope([_parallel_stage(compute_cost)]),
                            "store_scope": _scope(),
                            "trip_count": _const(4),
                            "block_sym": "tile_m",
                            "iter_type": "temporal",
                        }
                    }
                ]
            ),
            "store_scope": _scope(),
        },
        "constraint_scope": {
            "metadata": {
                "symbols": {},
                "booleans": ["is_double_buffer"],
                "iter_num": {"seq_iter": _const(1), "temp_iter": []},
            },
            "hard_constraints": [],
        },
    }


def test_common_expr_eval_matches_inner_expr() -> None:
    expr = CommonExpr("shared", Add([Const(3), Mul(Sym("x"), Const(2))]))

    assert expr.eval({"x": 5}) == 13


def test_cpmpy_resolver_reuses_common_expr_aux_var() -> None:
    x = cp.intvar(1, 5, name="x")
    shared = CommonExpr("shared", Mul(Sym("x"), Const(2)))
    objective = Add([shared, shared, Max([shared, Const(1)])])
    resolver = ExprResolver({"x": x})

    resolver.resolve(objective)

    assert len(resolver.aux_constraints) == 1


def test_common_expr_objective_matches_expanded_objective() -> None:
    x_common = cp.intvar(1, 5, name="x_common")
    shared = CommonExpr("shared", Mul(Sym("x"), Const(2)))
    common_resolver = ExprResolver({"x": x_common})
    common_obj = common_resolver.resolve(Add([shared, shared]))
    common_model = cp.Model()
    common_model += common_resolver.aux_constraints
    common_model.minimize(common_obj)

    x_expanded = cp.intvar(1, 5, name="x_expanded")
    expanded_resolver = ExprResolver({"x": x_expanded})
    expanded_obj = expanded_resolver.resolve(
        Add([Mul(Sym("x"), Const(2)), Mul(Sym("x"), Const(2))])
    )
    expanded_model = cp.Model()
    expanded_model += expanded_resolver.aux_constraints
    expanded_model.minimize(expanded_obj)

    assert common_model.solve(solver="ortools")
    assert expanded_model.solve(solver="ortools")
    assert common_model.objective_value() == expanded_model.objective_value() == 4


def test_aggregation_common_expr_preserves_approx_objective() -> None:
    for variant in (_variant(load_cost=5, compute_cost=3), _variant(load_cost=3, compute_cost=5)):
        expanded = compute_total_time_ast(variant, scale_time_costs=False)
        common = compute_total_time_ast(
            variant,
            scale_time_costs=False,
            use_common_expr=True,
        )

        for is_double_buffer in (0, 1):
            assignments = {"is_double_buffer": is_double_buffer}
            assert common.eval(assignments) == expanded.eval(assignments)
