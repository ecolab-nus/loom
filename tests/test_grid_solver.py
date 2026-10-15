from __future__ import annotations

import pytest

from loom.loom_utils.ast import (
    Add, CommonExpr, Const, Div, Eq, Ge, IfElse, Le, Lt, Max, Mod, Mul, Sym, Switch,
)
from loom.solver.core.grid_solver import GridUnsupported, solve_on_grid
from loom.solver.core.solver_context import SolverContext

SYMBOLS = {"tile_m": [32, 64, 96, 128], "tile_n": [32, 64, 96, 128], "is_db": [0, 1]}


def _cpsat(constraints, objective):
    ctx = SolverContext()
    ctx.load_symbols(
        {"tile_m": {"type": "int"}, "tile_n": {"type": "int"}},
        {k: v for k, v in SYMBOLS.items() if k != "is_db"},
    )
    ctx.load_booleans(["is_db"])
    # Mention every symbol so CP-SAT assigns all of them.
    every = Ge(Add([Sym("tile_m"), Sym("tile_n"), Sym("is_db")]), Const(0))
    ctx.add_hard_constraints([*constraints, every])
    return ctx.find_optimum(objective)


def _both(constraints, objective):
    grid = solve_on_grid(SYMBOLS, constraints, [], objective)
    status, val, _ = _cpsat(constraints, objective)
    assert grid[0] == status
    assert grid[1] == val
    return grid


M, N, DB = Sym("tile_m"), Sym("tile_n"), Sym("is_db")


def test_ceildiv_mod_minmax_match_cpsat():
    obj = Add([
        Div(Const(1000), M),
        Mul(Mod(Const(200), N), Const(3)),
        Max([Div(Const(512), N), Mul(DB, Const(5))]),
    ])
    _both([Le(Add([M, N]), Const(192))], obj)


def test_switch_requires_exactly_one_case():
    # tile_m == 32 matches both cases -> infeasible there; no case at 128 -> infeasible.
    obj = Switch([(Le(M, Const(64)), Const(10)), (Lt(M, Const(100)), M)], default=Const(0))
    status, val, asg, _ = _both([], obj)
    assert status == "OPTIMAL" and val == 96 and asg["tile_m"] == 96


def test_common_expr_upper_bound_and_zero_divisor_are_infeasible():
    big = CommonExpr("big", Mul(M, Const(2**26)))  # > 2**31 for tile_m >= 64
    _both([], Add([big, N]))
    # Divisor is zero when is_db == 0, even inside the untaken IfElse branch.
    obj = IfElse(Eq(DB, Const(1)), Const(1), Div(Const(7), DB))
    status, val, asg, _ = _both([], obj)
    assert asg["is_db"] == 1


def test_ties_pick_smallest_values():
    status, val, asg, n_optima = solve_on_grid(SYMBOLS, [Ge(M, Const(64))], [], Const(5))
    assert (status, val, n_optima) == ("OPTIMAL", 5, 3 * 4 * 2)
    assert asg == {"tile_m": 64, "tile_n": 32, "is_db": 0}


def test_trip_count_divisibility():
    trips = [(Const(160), [M], True), (Const(128), [N], False)]
    status, val, asg, _ = solve_on_grid(SYMBOLS, [], trips, Add([Div(Const(160), M), N]))
    assert status == "OPTIMAL" and asg["tile_m"] == 32 and asg["tile_n"] == 32


def test_oversized_grid_is_unsupported():
    with pytest.raises(GridUnsupported):
        solve_on_grid({"a": list(range(2000)), "b": list(range(2000))}, [], [], Sym("a"))
