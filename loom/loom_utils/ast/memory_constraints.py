"""Build solver-side memory constraints from ETG metadata."""

from __future__ import annotations

from typing import Any

from .core import Add, Const, Eq, IfElse, Le, Mul, Sym
from .parser import parse_expr


_DTYPE_BYTES = {
    "f16": 2,
    "bf16": 2,
    "i16": 2,
    "f32": 4,
    "i32": 4,
}


def build_l1_memory_constraint(metadata: dict[str, Any]) -> Le:
    """Create the DB-aware L1 capacity constraint from L1 footprint metadata."""
    footprint = metadata.get("L1_footprint")
    if not isinstance(footprint, dict):
        raise ValueError("metadata.L1_footprint must use the grouped object schema")

    capacity = footprint.get("capacity")
    if type(capacity) is not int:
        raise ValueError("metadata.L1_footprint.capacity must be an int")

    for key in ("load", "compute", "store"):
        if key not in footprint or not isinstance(footprint[key], list):
            raise ValueError(f"metadata.L1_footprint.{key} must be a list")

    datatype = metadata.get("datatype")
    if datatype not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported L1 footprint datatype: {datatype!r}")

    load_terms = [parse_expr(t) for t in footprint["load"]]
    compute_terms = [parse_expr(t) for t in footprint["compute"]]
    store_terms = [parse_expr(t) for t in footprint["store"]]

    if load_terms and "is_double_buffer" not in metadata.get("booleans", []):
        raise ValueError(
            "metadata.booleans must contain 'is_double_buffer' when load "
            "L1 footprint terms are present"
        )

    base_terms = _sum_expr(load_terms + compute_terms + store_terms)
    db_extra = (
        IfElse(Eq(Sym("is_double_buffer"), Const(1)), _sum_expr(load_terms), Const(0))
        if load_terms
        else Const(0)
    )
    effective_terms = Add([base_terms, db_extra])
    effective_bytes = Mul(effective_terms, Const(_DTYPE_BYTES[datatype]))
    return Le(effective_bytes, Const(capacity))


def _sum_expr(terms):
    return Add([Const(0), *terms])
