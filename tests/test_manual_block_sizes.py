from __future__ import annotations

import pytest

from loom.solver.main import prepare_manual_block_sizes


def _variant(name: str, capacity: int) -> dict:
    return {
        "variant_name": name,
        "constraint_scope": {
            "hard_constraints": ["True"],
            "metadata": {
                "symbols": {
                    "tile": {"type": "int", "natural_ub": 64, "alignment": 32}
                },
                "booleans": ["is_double_buffer"],
                "iter_num": {
                    "seq_iter": [
                        {"Div": [{"Const": 64}, {"Sym": "tile"}]},
                        True,
                    ],
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


def test_manual_assignment_rejects_only_infeasible_variants() -> None:
    variants = [_variant("small", 128), _variant("large", 256)]
    result = prepare_manual_block_sizes(
        variants,
        {"ALL": {"tile": 64, "is_double_buffer": 0}},
    )
    assert result == {"large": {"tile": 64, "is_double_buffer": 0}}


def test_manual_assignment_fails_when_no_variant_is_feasible() -> None:
    with pytest.raises(ValueError, match="No feasible"):
        prepare_manual_block_sizes(
            [_variant("small", 128)],
            {"ALL": {"tile": 64, "is_double_buffer": 0}},
        )
