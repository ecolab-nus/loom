from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from loom.loom_utils.ast import build_memory_constraints


def test_native_etg_memory_footprints_feed_solver_constraints() -> None:
    repo = Path(__file__).resolve().parents[1]
    dataflow = repo / "third_party/loom-dataflow"
    staged_etg = dataflow / "build/tool/loom-opt/single_stage/staged_etg"
    if not staged_etg.is_file():
        pytest.skip(f"native staged_etg unavailable at {staged_etg}")

    fixture_dir = dataflow / "test/lcs"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "etg.json"
        subprocess.run(
            [
                str(staged_etg),
                "--input",
                str(fixture_dir / "per_memory_capacity_input.mlir"),
                "--hw_spec",
                str(fixture_dir / "per_memory_capacity_hw.mlir"),
                "--output",
                str(output),
            ],
            check=True,
        )
        variant = json.loads(output.read_text())[0]

    constraints = dict(
        build_memory_constraints(variant["constraint_scope"]["metadata"])
    )
    assert constraints["SRAM"].eval({"K": 64, "is_double_buffer": 0})
    assert constraints["RRAM"].eval({"K": 64, "is_double_buffer": 0})
    assert not constraints["SRAM"].eval({"K": 64, "is_double_buffer": 1})
    assert not constraints["RRAM"].eval({"K": 64, "is_double_buffer": 1})
