"""Utility for invoking the MLAR Rust evaluator binary to fill schedule scenarios.

The evaluator binary accepts a Schedule JSON on stdin and returns the same
Schedule JSON with ``scenarios`` populated on every node (Func, Sequential,
Parallel) on stdout.

TODO: Replace subprocess-based invocation with a native Python extension built
      via pyo3 (https://pyo3.rs). The pyo3 package will expose the evaluator
      logic directly as a Python callable, eliminating the need for a compiled
      binary on PATH and the subprocess round-trip.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Re-exports for backward compatibility
from .etg_resolver import resolve_etg_variants, validate_scenarios
from .evaluator_core import (
    evaluate_schedule, mock_evaluate_schedule, resolve_schedule, 
    _fill_func_scenarios, _DEFAULT_EVALUATOR
)
from .json_formatter import smart_json_dumps
from .schedule_utils import contains_sequential as _contains_sequential


def evaluate_schedule_file(
    json_path: Path | str,
    *,
    in_place: bool = False,
    evaluator_path: Path | str | None = None,
    resolve: bool = True,
) -> dict:
    """Load a Schedule JSON file, evaluate it, and optionally write back."""
    json_path = Path(json_path)
    schedule = json.loads(json_path.read_text())
    if resolve:
        result = resolve_schedule(schedule, evaluator_path=evaluator_path)
    else:
        result = evaluate_schedule(schedule, evaluator_path=evaluator_path)
    if in_place:
        json_path.write_text(json.dumps(result, indent=2))
    return result


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate an MLAR Schedule JSON using the Rust evaluator binary.",
    )
    parser.add_argument("json_file", help="Path to the Schedule JSON file to evaluate.")
    parser.add_argument(
        "--evaluator",
        default=None,
        metavar="PATH",
        help="Path to the evaluator binary (default: built-in eval_core).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of printing to stdout.",
    )
    args = parser.parse_args()

    result = evaluate_schedule_file(
        args.json_file,
        in_place=args.in_place,
        evaluator_path=args.evaluator,
    )
    if not args.in_place:
        print(json.dumps(result, indent=2))
    else:
        print(f"Scenarios filled and written back to {args.json_file}")


if __name__ == "__main__":
    _main()
