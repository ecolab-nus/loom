"""Utility for invoking the MLAR Rust evaluator binary to fill schedule scenarios.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from .core import evaluate_schedule, resolve_schedule


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


def main() -> None:
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
    main()
