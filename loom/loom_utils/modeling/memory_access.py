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
    op_label: str
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
    """Aggregate operand accesses by memory kind, direction, and op label."""
    root = variant.get("kernel_block", variant)
    totals: dict[tuple[int, str, str], int] = {}

    for visit in walk_workloads(root, assignments):
        op_label = str(
            visit.func.get("op_label", visit.func.get("name", "unknown"))
        )
        for direction in ("read", "write"):
            for mem_kind in _parse_operand_mem_kinds(
                visit.func.get(direction, "")
            ):
                key = (mem_kind, direction, op_label)
                totals[key] = totals.get(key, 0) + visit.total_trip

    return [
        MemoryAccessSummary(
            mem_kind=key[0],
            access=key[1],
            op_label=key[2],
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
    """Format one row per memory kind, direction, and operation label."""
    lines = ["Memory Access Summary"]
    if not summaries:
        lines.append("  (no memory access metadata)")
        return lines

    rows = [
        (
            str(entry.mem_kind),
            entry.access.upper(),
            f"{entry.total_trip:,}",
            entry.op_label,
        )
        for entry in summaries
    ]

    headers = ("mem_kind", "access", "count", "op_label")
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


def _parse_operand_mem_kinds(raw: object) -> list[int]:
    """Parse the ETG `%ssa: mem_kind;%ssa: mem_kind` representation."""
    text = str(raw).strip()
    if not text:
        return []
    kinds: list[int] = []
    for operand in text.split(";"):
        name, separator, kind = operand.rpartition(":")
        if not separator or not name.strip().startswith("%"):
            raise ValueError(f"invalid ETG operand access metadata: {operand!r}")
        try:
            kinds.append(int(kind.strip()))
        except ValueError as exc:
            raise ValueError(
                f"invalid ETG memory kind in operand metadata: {operand!r}"
            ) from exc
    return kinds
