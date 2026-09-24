from __future__ import annotations

from io import StringIO

from loom.loom_utils.ast import parse_expr
from loom.loom_utils.modeling.reporter import _print_func_breakdown


def _scenario(cost: int) -> dict:
    return {"constraints": "True", "time_cost": {"Const": cost}}


def _sequential(*funcs: tuple[str, list[dict]]) -> dict:
    return {
        "schedules": [
            {
                "Func": {
                    "func": {"op_label": label},
                    "scenarios": scenarios,
                }
            }
            for label, scenarios in funcs
        ]
    }


def _render(sequential: dict) -> list[str]:
    output = StringIO()

    def emit(line: str) -> None:
        print(line, file=output)

    _print_func_breakdown(
        sequential,
        assignments={},
        depth=0,
        p=emit,
        cost_parser=parse_expr,
        unit="solver units",
    )
    return output.getvalue().splitlines()


def test_func_breakdown_prints_aligned_cost_before_label() -> None:
    lines = _render(
        _sequential(
            ("linalg.batch_matmul(%1: 1, %2, %3)", [_scenario(1234)]),
            ("arith.addf(%4, %5)", [_scenario(2)]),
        )
    )

    assert lines == [
        "       1,234 solver units  linalg.batch_matmul(%1: 1, %2, %3)",
        "           2 solver units  arith.addf(%4, %5)",
    ]
    assert lines[0].index(" solver units") == lines[1].index(" solver units")


def test_func_breakdown_keeps_unknown_cost_at_front() -> None:
    lines = _render(_sequential(("arith.mulf(%1, %2)", [])))

    assert lines == ["           ? solver units  arith.mulf(%1, %2)"]


def test_func_breakdown_prints_placed_functions() -> None:
    sequential = _sequential(("linalg.matmul(%0, %1, %2)", [_scenario(9)]))
    schedule = sequential["schedules"][0]
    schedule["PlacedFunc"] = {
        **schedule.pop("Func"),
        "target": {"array": "matrix_sram", "selectors": []},
    }

    assert _render(sequential) == [
        "           9 solver units  linalg.matmul(%0, %1, %2)"
    ]
