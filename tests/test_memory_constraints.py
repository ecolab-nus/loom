from __future__ import annotations

import pytest

from loom.loom_utils.ast import Const, Eq, Mul, Sym, build_memory_constraints
from loom.solver.core.solver_context import SolverContext


def _mul(*terms: object) -> dict:
    assert len(terms) >= 2
    expr = terms[0]
    for term in terms[1:]:
        expr = {"Mul": [expr, term]}
    return expr


def _sym(name: str) -> dict:
    return {"Sym": name}


def _const(value: int) -> dict:
    return {"Const": value}


def _mixed_metadata() -> dict:
    a_bytes = _mul(_const(2), _const(32), _sym("K"))
    b_bytes = _mul(_const(2), _sym("K"), _const(32))
    c_bytes = _mul(_const(4), _const(32), _const(32))
    return {
        "booleans": ["is_double_buffer"],
        "memory_footprints": [
            {
                "memory": "RRAM",
                "capacity_bytes": 4096,
                "load_bytes": [b_bytes],
                "compute_bytes": [],
                "store_bytes": [],
            },
            {
                "memory": "SRAM",
                "capacity_bytes": 8192,
                "load_bytes": [a_bytes],
                "compute_bytes": [],
                "store_bytes": [c_bytes],
            },
        ],
    }


def test_per_memory_constraints_charge_shared_double_buffer_locally() -> None:
    constraints = dict(build_memory_constraints(_mixed_metadata()))

    assert constraints["SRAM"].eval({"K": 64, "is_double_buffer": 0})
    assert constraints["RRAM"].eval({"K": 64, "is_double_buffer": 0})
    assert not constraints["SRAM"].eval({"K": 64, "is_double_buffer": 1})
    assert not constraints["RRAM"].eval({"K": 64, "is_double_buffer": 1})
    assert constraints["SRAM"].eval({"K": 32, "is_double_buffer": 1})
    assert constraints["RRAM"].eval({"K": 32, "is_double_buffer": 1})


def test_mixed_placement_makes_larger_tile_solver_feasible() -> None:
    constraints = build_memory_constraints(_mixed_metadata())
    ctx = SolverContext()
    ctx.load_symbols({"K": {"type": "int"}}, {"K": [32, 64]})
    ctx.load_booleans(["is_double_buffer"])
    for memory, constraint in constraints:
        ctx.add_hard_constraints([constraint], label_prefix=f"memory[{memory}]")
    ctx.add_hard_constraints([Eq(Sym("is_double_buffer"), Const(0))])

    status, _, assignments = ctx.find_optimum(Mul(Const(-1), Sym("K")))

    assert status == "OPTIMAL"
    assert assignments is not None
    assert assignments["K"] == 64


def test_all_sram_rejects_larger_tile() -> None:
    metadata = _mixed_metadata()
    rram, sram = metadata["memory_footprints"]
    sram["load_bytes"].extend(rram["load_bytes"])
    metadata["memory_footprints"] = [sram]
    constraint = build_memory_constraints(metadata)[0][1]

    assert constraint.eval({"K": 32, "is_double_buffer": 0})
    assert not constraint.eval({"K": 64, "is_double_buffer": 0})


def test_memory_record_order_does_not_change_constraints() -> None:
    metadata = _mixed_metadata()
    forward = dict(build_memory_constraints(metadata))
    metadata["memory_footprints"].reverse()
    reverse = dict(build_memory_constraints(metadata))

    for memory, constraint in forward.items():
        for assignment in (
            {"K": 32, "is_double_buffer": 0},
            {"K": 64, "is_double_buffer": 1},
        ):
            assert constraint.eval(assignment) == reverse[memory].eval(assignment)


def test_memory_metadata_rejects_legacy_and_duplicate_records() -> None:
    with pytest.raises(TypeError, match="memory_footprints"):
        build_memory_constraints(
            {"L1_footprint": {}, "datatype": "f16", "booleans": []}
        )

    metadata = _mixed_metadata()
    metadata["memory_footprints"].append(dict(metadata["memory_footprints"][0]))
    with pytest.raises(ValueError, match="Duplicate memory"):
        build_memory_constraints(metadata)


def test_load_bytes_require_shared_double_buffer_symbol() -> None:
    metadata = _mixed_metadata()
    metadata["booleans"] = []

    with pytest.raises(ValueError, match="is_double_buffer"):
        build_memory_constraints(metadata)


def test_empty_memory_footprints_need_no_capacity_constraint() -> None:
    assert build_memory_constraints(
        {"booleans": ["is_double_buffer"], "memory_footprints": []}
    ) == []
