from __future__ import annotations

from loom.loom_utils.modeling.memory_access import (
    MemoryAccessSummary,
    format_memory_access_summary,
    summarize_memory_accesses,
    walk_workloads,
)


def _scope(stages: list[dict] | None = None) -> dict:
    return {"scope_name": "scope", "stages": stages or []}


def _block(stages: list[dict] | None = None) -> dict:
    return {
        "load_scope": _scope(stages),
        "compute_scope": _scope(),
        "store_scope": _scope(),
    }


def _func(
    label: str,
    *,
    read: str | None = None,
    write: str | None = None,
) -> dict:
    func = {"op_label": label}
    if read is not None:
        func["read"] = read
    if write is not None:
        func["write"] = write
    return {
        "Parallel": [
            {
                "Sequential": {
                    "schedules": [{"Func": {"func": func}}],
                }
            }
        ]
    }


def _loop(trip: object, stages: list[dict]) -> dict:
    return {
        "for_loop_block": {
            **_block(stages),
            "trip_count": trip,
        }
    }


def test_walk_workloads_multiplies_only_ancestor_loop_trips() -> None:
    root = _block(
        [
            _loop(
                {"Sym": "outer"},
                [
                    _loop({"Const": 3}, [_func("nested")]),
                ],
            ),
            _loop({"Const": 5}, [_func("sibling")]),
            _func("outside"),
        ]
    )

    visits = list(walk_workloads(root, {"outer": 2}))

    assert [(v.func["op_label"], v.total_trip) for v in visits] == [
        ("nested", 6),
        ("sibling", 5),
        ("outside", 1),
    ]


def test_summary_counts_operands_and_merges_matching_labels() -> None:
    root = _block(
        [
            _loop(
                {"Const": 4},
                [_func("linalg.add(%0, %1, %2)", read="%0: 0;%1: 0", write="%2: 1")],
            ),
            _loop(
                {"Const": 6},
                [_func("linalg.add(%0, %1, %2)", read="%0: 0;%1: 0", write="%2: 1")],
            ),
            _loop(
                {"Sym": "iters"},
                [
                    _func(
                        "linalg.matmul(%3, %4, %5)",
                        read="%3: 0;%4: 1",
                        write="%5: 0",
                    )
                ],
            ),
        ]
    )

    summaries = summarize_memory_accesses(
        {"kernel_block": root},
        {"iters": 8},
    )

    assert summaries == [
        MemoryAccessSummary(
            mem_kind=0,
            access="read",
            op_label="linalg.add(%0, %1, %2)",
            total_trip=20,
        ),
        MemoryAccessSummary(
            mem_kind=0,
            access="read",
            op_label="linalg.matmul(%3, %4, %5)",
            total_trip=8,
        ),
        MemoryAccessSummary(0, "write", "linalg.matmul(%3, %4, %5)", 8),
        MemoryAccessSummary(1, "read", "linalg.matmul(%3, %4, %5)", 8),
        MemoryAccessSummary(1, "write", "linalg.add(%0, %1, %2)", 10),
    ]


def test_summary_handles_old_etg_and_empty_fields() -> None:
    variant = {"kernel_block": _block([_func("old")])}

    assert summarize_memory_accesses(variant, {}) == []
    assert format_memory_access_summary([]) == [
        "Memory Access Summary",
        "  (no memory access metadata)",
    ]


def test_summary_table_contains_only_count_and_op_label() -> None:
    lines = format_memory_access_summary(
        [
            MemoryAccessSummary(0, "write", "linalg.add(%0, %1)", 512),
            MemoryAccessSummary(1, "read", "loom.copy(%2, %3)", 8192),
        ]
    )
    rendered = "\n".join(lines)

    assert "mem_kind  access  count" in rendered
    assert "0         WRITE   512    linalg.add(%0, %1)" in rendered
    assert "1         READ    8,192  loom.copy(%2, %3)" in rendered
    assert "tensor<" not in rendered
