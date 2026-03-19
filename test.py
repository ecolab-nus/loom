#!/usr/bin/env python3
"""Evaluate all variants in new_etg.json: real eval for compute scope, mock for memory scope."""

from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure the loom package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "loom"))
from loom_utils.mlar_evaluator import mock_evaluate_schedule, resolve_schedule, smart_json_dumps

INPUT_PATH = (
    Path(__file__).resolve().parent
    / "third_party"
    / "loom-dataflow"
    / "test"
    / "Passes"
    / "mm_2Dmesh"
    / "constraint_space"
    / "new_etg.json"
)
OUTPUT_PATH = INPUT_PATH


def _check_all_scenarios_filled(node, path="") -> list[str]:
    """Walk *node* and return paths where ``scenarios`` is empty."""
    issues: list[str] = []
    if isinstance(node, dict):
        if "scenarios" in node and not node["scenarios"]:
            issues.append(path or "<root>")
        for k, v in node.items():
            issues.extend(_check_all_scenarios_filled(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            issues.extend(_check_all_scenarios_filled(item, f"{path}[{i}]"))
    return issues


def process_variant(variant: dict) -> dict:
    """Process a single variant: real eval for compute_scope, mock for memory_scope."""
    variant = copy.deepcopy(variant)

    # Compute scope — real Rust evaluator.
    variant["compute_scope"] = resolve_schedule(variant["compute_scope"])

    # Memory scope — mock evaluator (Rust support under development).
    variant["memory_scope"] = resolve_schedule(
        variant["memory_scope"], evaluator_fn=mock_evaluate_schedule
    )

    return variant


def main() -> None:
    data = json.loads(INPUT_PATH.read_text())
    print(f"Loaded {len(data)} variants from {INPUT_PATH.name}")

    results: list[dict | None] = [None] * len(data)
    errors = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(process_variant, v): i for i, v in enumerate(data)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            name = data[idx].get("variant_name", f"variant_{idx}")
            try:
                results[idx] = future.result()
                print(f"  [{idx:>2}] {name} — ok")
            except Exception as exc:
                print(f"  [{idx:>2}] {name} — FAILED: {exc}")
                results[idx] = data[idx]  # keep original on failure
                errors += 1

    # Verify all scenarios are filled.
    empty_count = 0
    for i, variant in enumerate(results):
        issues = _check_all_scenarios_filled(variant)
        if issues:
            name = variant.get("variant_name", f"variant_{i}")
            for p in issues:
                print(f"  WARNING: {name} has empty scenarios at {p}")
            empty_count += len(issues)

    print()
    if empty_count:
        print(f"{empty_count} scenario field(s) still empty!")
    else:
        print("All scenarios filled successfully.")

    if errors:
        print(f"{errors} variant(s) failed during evaluation.")

    OUTPUT_PATH.write_text(smart_json_dumps(results))
    print(f"Results written back to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
