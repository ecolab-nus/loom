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


def _func(label: str, accesses: list[dict] | None = None) -> dict:
    func = {"op_label": label}
    if accesses is not None:
        func["memory_accesses"] = accesses
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


def _access(
    direction: str,
    shape: list[object],
    *,
    mem_kind: int | None = None,
) -> dict:
    result = {
        "access": direction,
        "memory_space": "mem_L1",
        "operand_index": 1 if direction == "write" else 0,
        "shape": shape,
        "element_type": "f16",
    }
    if mem_kind is not None:
        result["mem_kind"] = mem_kind
    return result


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


def test_summary_evaluates_shapes_defaults_kind_and_merges_matches() -> None:
    write = _access(
        "write",
        [{"Sym": "tile_b"}, {"Sym": "tile_n"}, {"Const": 512}],
    )
    root = _block(
        [
            _loop({"Const": 4}, [_func("copy0", [write])]),
            _loop({"Const": 6}, [_func("copy1", [write])]),
            _loop(
                {"Sym": "iters"},
                [
                    _func(
                        "copy2",
                        [
                            _access(
                                "read",
                                [
                                    {"Const": 1},
                                    {"Const": 512},
                                    {"Sym": "tile_n"},
                                ],
                                mem_kind=1,
                            )
                        ],
                    )
                ],
            ),
        ]
    )

    summaries = summarize_memory_accesses(
        {"kernel_block": root},
        {"tile_b": 1, "tile_n": 256, "iters": 8},
    )

    assert summaries == [
        MemoryAccessSummary(
            mem_kind=0,
            access="write",
            tensor_type="tensor<1x256x512xf16>",
            total_trip=10,
        ),
        MemoryAccessSummary(
            mem_kind=1,
            access="read",
            tensor_type="tensor<1x512x256xf16>",
            total_trip=8,
        ),
    ]


def test_summary_handles_old_etg_without_access_metadata() -> None:
    variant = {"kernel_block": _block([_func("old")])}

    assert summarize_memory_accesses(variant, {}) == []
    assert format_memory_access_summary([]) == [
        "Memory Access Summary",
        "  (no memory access metadata)",
    ]


def test_summary_table_has_read_and_write_rows_per_kind() -> None:
    lines = format_memory_access_summary(
        [
            MemoryAccessSummary(0, "write", "tensor<1x32xf16>", 512),
            MemoryAccessSummary(1, "read", "tensor<1x64xf16>", 8192),
        ]
    )
    rendered = "\n".join(lines)

    assert "0         READ    -" in rendered
    assert "0         WRITE   512 * tensor<1x32xf16>" in rendered
    assert "1         READ    8,192 * tensor<1x64xf16>" in rendered
    assert "1         WRITE   -" in rendered
