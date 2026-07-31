"""Memory-access analysis for resolved ETG variants."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..ast import parse_expr


@dataclass(frozen=True)
class WorkloadVisit:
    """One workload plus the evaluated product of all enclosing loop trips."""

    func: dict
    total_trip: int


@dataclass(frozen=True, order=True)
class MemoryAccessSummary:
    mem_kind: int
    access: str
    tensor_type: str
    total_trip: int


def walk_workloads(
    root: dict,
    assignments: dict[str, int],
) -> Iterator[WorkloadVisit]:
    """Yield workloads while carrying the product of enclosing loop trips."""
    yield from _walk_block(root, assignments, total_trip=1)


def _walk_block(
    block: dict,
    assignments: dict[str, int],
    total_trip: int,
) -> Iterator[WorkloadVisit]:
    for scope_name in ("load_scope", "compute_scope", "store_scope"):
        scope = block.get(scope_name, {})
        for stage in scope.get("stages", []):
            yield from _walk_stage(stage, assignments, total_trip)


def _walk_stage(
    stage: dict,
    assignments: dict[str, int],
    total_trip: int,
) -> Iterator[WorkloadVisit]:
    if "for_loop_block" in stage:
        loop = stage["for_loop_block"]
        trip = parse_expr(loop["trip_count"]).eval(assignments)
        yield from _walk_block(loop, assignments, total_trip * trip)
        return

    for parallel_item in stage.get("Parallel", []):
        sequential = parallel_item.get("Sequential", {})
        for schedule in sequential.get("schedules", []):
            func_node = schedule.get("Func") if isinstance(schedule, dict) else None
            if not func_node:
                continue
            func = func_node.get("func")
            if isinstance(func, dict):
                yield WorkloadVisit(func=func, total_trip=total_trip)


def summarize_memory_accesses(
    variant: dict,
    assignments: dict[str, int],
) -> list[MemoryAccessSummary]:
    """Aggregate accesses by memory kind, direction, and concrete tensor type."""
    root = variant.get("kernel_block", variant)
    totals: dict[tuple[int, str, str], int] = {}

    for visit in walk_workloads(root, assignments):
        for access in visit.func.get("memory_accesses", []):
            direction = str(access.get("access", "")).lower()
            if direction not in ("read", "write"):
                continue
            shape = [
                parse_expr(dim).eval(assignments)
                for dim in access.get("shape", [])
            ]
            element_type = str(access.get("element_type", "unknown"))
            tensor_type = _format_tensor_type(shape, element_type)
            key = (int(access.get("mem_kind", 0)), direction, tensor_type)
            totals[key] = totals.get(key, 0) + visit.total_trip

    return [
        MemoryAccessSummary(
            mem_kind=key[0],
            access=key[1],
            tensor_type=key[2],
            total_trip=total_trip,
        )
        for key, total_trip in sorted(
            totals.items(),
            key=lambda item: (
                item[0][0],
                0 if item[0][1] == "read" else 1,
                item[0][2],
            ),
        )
    ]


def format_memory_access_summary(
    summaries: list[MemoryAccessSummary],
) -> list[str]:
    """Format summaries as one aligned table with explicit READ/WRITE rows."""
    lines = ["Memory Access Summary"]
    if not summaries:
        lines.append("  (no memory access metadata)")
        return lines

    by_kind_access: dict[tuple[int, str], list[MemoryAccessSummary]] = {}
    mem_kinds = sorted({summary.mem_kind for summary in summaries})
    for summary in summaries:
        by_kind_access.setdefault(
            (summary.mem_kind, summary.access), []
        ).append(summary)

    rows: list[tuple[str, str, str]] = []
    for mem_kind in mem_kinds:
        for direction in ("read", "write"):
            entries = by_kind_access.get((mem_kind, direction), [])
            if not entries:
                rows.append((str(mem_kind), direction.upper(), "-"))
                continue
            for entry in entries:
                rows.append(
                    (
                        str(mem_kind),
                        direction.upper(),
                        f"{entry.total_trip:,} * {entry.tensor_type}",
                    )
                )

    headers = ("mem_kind", "access", "breakdown")
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines.append(
        "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    )
    lines.append(
        "  ".join("-" * widths[i] for i in range(len(headers)))
    )
    lines.extend(
        "  ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
        for row in rows
    )
    return lines


def _format_tensor_type(shape: list[int], element_type: str) -> str:
    body = "x".join([*(str(dim) for dim in shape), element_type])
    return f"tensor<{body}>"
