"""Build solver-side capacity constraints from ETG memory metadata."""

from __future__ import annotations

from typing import Any

from .core import Add, Const, Le, Mul, Sym
from .parser import parse_expr


def build_memory_constraints(metadata: dict[str, Any]) -> list[tuple[str, Le]]:
    """Create one shared-double-buffer-aware constraint per physical memory."""
    footprints = metadata.get("memory_footprints")
    if not isinstance(footprints, list):
        raise TypeError("metadata.memory_footprints must be a list")

    constraints: list[tuple[str, Le]] = []
    seen_memories: set[str] = set()
    for index, footprint in enumerate(footprints):
        path = f"metadata.memory_footprints[{index}]"
        if not isinstance(footprint, dict):
            raise TypeError(f"{path} must be an object")

        memory = footprint.get("memory")
        if not isinstance(memory, str) or not memory:
            raise ValueError(f"{path}.memory must be a non-empty string")
        if memory in seen_memories:
            raise ValueError(f"Duplicate memory footprint record: {memory!r}")
        seen_memories.add(memory)

        capacity = footprint.get("capacity_bytes")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError(f"{path}.capacity_bytes must be a positive int")

        parsed: dict[str, list] = {}
        for key in ("load_bytes", "compute_bytes", "store_bytes"):
            terms = footprint.get(key)
            if not isinstance(terms, list):
                raise TypeError(f"{path}.{key} must be a list")
            parsed[key] = [parse_expr(term) for term in terms]

        load_terms = parsed["load_bytes"]
        if load_terms and "is_double_buffer" not in metadata.get("booleans", []):
            raise ValueError(
                "metadata.booleans must contain 'is_double_buffer' when "
                f"{path}.load_bytes is non-empty"
            )

        base_bytes = _sum_expr(
            load_terms + parsed["compute_bytes"] + parsed["store_bytes"]
        )
        extra_load_bytes = (
            Mul(Sym("is_double_buffer"), _sum_expr(load_terms))
            if load_terms
            else Const(0)
        )
        constraints.append(
            (memory, Le(Add([base_bytes, extra_load_bytes]), Const(capacity)))
        )

    return constraints


def _sum_expr(terms):
    return Add([Const(0), *terms])
