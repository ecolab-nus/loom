from __future__ import annotations

import pytest

from loom.solver.main import prepare_manual_block_sizes


def _ge(value: int) -> dict:
    return {"Ge": [{"Sym": "tile"}, {"Const": value}]}


def _scope(stages: list[dict] | None = None) -> dict:
    return {"scope_name": "scope", "stages": stages or []}


def _variant(name: str, capacity: int, guards: list | None = None) -> dict:
    scenarios = [
        {"constraints": guard, "time_cost": {"Const": 1}}
        for guard in (guards or ["True"])
    ]
    stage = {"Parallel": [{"Sequential": {"scenarios": scenarios, "schedules": []}}]}
    return {
        "variant_name": name,
        "kernel_block": {
            "load_scope": _scope(),
            "compute_scope": _scope([stage]),
            "store_scope": _scope(),
        },
        "constraint_scope": {
            "hard_constraints": ["True"],
            "metadata": {
                "symbols": {
                    "tile": {"type": "int", "natural_ub": 64, "alignment": 1}
                },
                "booleans": ["is_double_buffer"],
                "iter_num": {
                    "seq_iter": [{"Div": [{"Const": 64}, {"Sym": "tile"}]}],
                    "temp_iter": [],
                },
                "memory_footprints": [{
                    "memory": "SRAM",
                    "capacity_bytes": capacity,
                    "load_bytes": [],
                    "compute_bytes": [
                        {"Mul": [{"Sym": "tile"}, {"Const": 4}]}
                    ],
                    "store_bytes": [],
                }],
            },
        },
    }


def _assign(variant: dict, tile: int) -> dict:
    return prepare_manual_block_sizes(
        [variant], {"ALL": {"tile": tile, "is_double_buffer": 0}}
    )


def test_manual_assignment_rejects_only_infeasible_variants() -> None:
    variants = [_variant("small", 128), _variant("large", 256)]
    result = prepare_manual_block_sizes(
        variants,
        {"ALL": {"tile": 64, "is_double_buffer": 0}},
    )
    assert result == {"large": {"tile": 64, "is_double_buffer": 0}}


def test_manual_assignment_fails_when_no_variant_is_feasible() -> None:
    with pytest.raises(ValueError, match="No feasible"):
        _assign(_variant("small", 128), 64)


def test_manual_assignment_requires_exact_trip_count() -> None:
    assert _assign(_variant("v", 1024), 32)
    with pytest.raises(ValueError, match="does not divide"):
        _assign(_variant("v", 1024), 48)


@pytest.mark.parametrize("tile, ok", [(16, False), (32, True), (64, False)])
def test_manual_assignment_requires_exactly_one_alternative(tile: int, ok: bool) -> None:
    # tile=16 matches no alternative, tile=64 matches both.
    variant = _variant("v", 1024, guards=[_ge(32), _ge(64)])
    if ok:
        assert _assign(variant, tile)
    else:
        with pytest.raises(ValueError, match="perf-model alternatives match"):
            _assign(variant, tile)
