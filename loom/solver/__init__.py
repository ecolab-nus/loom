"""CPMpy-based block-size optimizer for the Loom dataflow pipeline."""

from .main import run as cpmpy_run
from .main import (
    find_unsatisfiable_trip_count,
    prepare_manual_block_sizes,
    write_manual_breakdown_log,
)

__all__ = [
    "cpmpy_run",
    "find_unsatisfiable_trip_count",
    "prepare_manual_block_sizes",
    "write_manual_breakdown_log",
]
