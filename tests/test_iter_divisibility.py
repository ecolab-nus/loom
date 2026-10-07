from __future__ import annotations

import pytest

from loom.loom_utils.ast import Const, Mul, Sym
from loom.loom_utils.modeling import derive_domains_from_etg, parse_user_block_sizes
from loom.solver.core.solver_context import SolverContext
from loom.solver.main import _trip_counts, sample_block_size_neighbors


def _div(n, *dens):
    expr = {"Const": n}
    for d in dens:
        expr = {"Div": [expr, d]}
    return expr


def test_every_tiled_trip_count_is_constrained() -> None:
    iter_num = {
        "seq_iter": [_div(64, {"Sym": "a"}), {"Const": 3}],
        "temp_iter": [_div(64, {"Sym": "b"}, {"Const": 4})],
    }
    assert _trip_counts(iter_num)[1] == ["iter_num.seq_iter[0]", "iter_num.temp_iter[0]"]


def test_grid_trip_count_divides_tile_times_cores() -> None:
    # 512 / tile / 8 must divide exactly: the largest legal tile is 64.
    ctx = SolverContext()
    ctx.load_symbols({"tile": {"type": "int", "natural_ub": 512}}, {"tile": [1, 2, 4, 5, 8, 64, 100, 512]})
    ctx.add_divisibility_constraints(*_trip_counts(
        {"seq_iter": [], "temp_iter": [_div(512, {"Sym": "tile"}, {"Const": 8})]}
    ))
    _, _, assignments = ctx.find_optimum(Mul(Sym("tile"), Const(-1)))
    assert assignments["tile"] == 64


def _etg(alignment: int) -> list[dict]:
    return [{"constraint_scope": {"metadata": {"symbols": {
        "tile": {"type": "int", "natural_ub": 128, "alignment": alignment}
    }}}}]


def test_domains_follow_symbol_alignment() -> None:
    assert derive_domains_from_etg(_etg(1))["tile"] == list(range(1, 129))
    assert derive_domains_from_etg(_etg(32))["tile"] == [32, 64, 96, 128]
    user = parse_user_block_sizes({"tile": {"lb": 40, "ub": 100}})
    assert derive_domains_from_etg(_etg(32), user)["tile"] == [64, 96]
    with pytest.raises(ValueError, match="multiple of its alignment"):
        derive_domains_from_etg(_etg(32), parse_user_block_sizes({"tile": {"lb": 33, "ub": 63}}))


def test_neighbors_are_nearest_feasible_domain_values() -> None:
    variant = {
        "kernel_block": {
            name: {"scope_name": name, "stages": []}
            for name in ("load_scope", "compute_scope", "store_scope")
        },
        "constraint_scope": {
            "hard_constraints": [],
            "metadata": {
                "symbols": {"tile": {"type": "int", "natural_ub": 128, "alignment": 32}},
                "booleans": [],
                "iter_num": {"seq_iter": [_div(128, {"Sym": "tile"})], "temp_iter": []},
                "memory_footprints": [],
            },
        },
    }
    domains = derive_domains_from_etg([variant])
    combos = sample_block_size_neighbors({"tile": 64}, variant, domains, topk_block_size=3)
    # 96 is the adjacent domain value but does not divide 128.
    assert sorted(c["tile"] for c in combos) == [32, 64, 128]
