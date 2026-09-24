from __future__ import annotations

from loom.loom_utils.mlar import core


def test_fill_func_scenarios_preserves_placed_target(monkeypatch) -> None:
    placed = {
        "PlacedFunc": {
            "func": {"name": "matmul", "symbols": [], "sym_map": {"entries": []}},
            "target": {"array": "matrix_sram", "selectors": []},
            "scenarios": [],
        }
    }

    def evaluate(schedule, *, evaluator_path=None):
        assert schedule["Sequential"]["schedules"] == [placed]
        return {
            "Sequential": {
                "scenarios": [{"constraints": "True", "time_cost": {"Const": 7}}]
            }
        }

    monkeypatch.setattr(core, "evaluate_schedule", evaluate)
    filled = core._fill_func_scenarios([placed])

    assert filled[0]["PlacedFunc"]["target"] == placed["PlacedFunc"]["target"]
    assert filled[0]["PlacedFunc"]["scenarios"][0]["time_cost"] == {"Const": 7}
