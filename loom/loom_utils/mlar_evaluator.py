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

import copy
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Path to the evaluator binary relative to the repo root.
# Adjust this once the binary is installed to a stable location or when the
# pyo3 package replaces this shim.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVALUATOR = (
    _REPO_ROOT
    / "third_party"
    / "loom-mlar"
    / "tests"
    / "2d_mesh"
    / "evaluators"
    / "eval_core"
)


def evaluate_schedule(
    schedule: dict,
    evaluator_path: Path | str | None = None,
) -> dict:
    """Run the MLAR evaluator on *schedule* and return the evaluated result.

    Parameters
    ----------
    schedule:
        A Schedule dict (already parsed from JSON) to evaluate.
    evaluator_path:
        Path to the evaluator binary.  Defaults to the pre-built
        ``eval_core`` binary inside ``third_party/loom-mlar``.

    Returns
    -------
    dict
        The evaluated Schedule with ``scenarios`` filled on every node.

    Raises
    ------
    FileNotFoundError
        If the evaluator binary does not exist.
    subprocess.CalledProcessError
        If the evaluator exits with a non-zero status.
    json.JSONDecodeError
        If the evaluator output is not valid JSON.
    """
    binary = Path(evaluator_path) if evaluator_path is not None else _DEFAULT_EVALUATOR
    if not binary.exists():
        raise FileNotFoundError(f"Evaluator binary not found: {binary}")

    result = subprocess.run(
        [str(binary)],
        input=json.dumps(schedule),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def mock_evaluate_schedule(schedule: dict) -> dict:
    """Return placeholder scenarios without calling the Rust binary.

    Intended as a stand-in for scopes whose evaluation is not yet supported
    by the Rust backend (e.g. memory scope).  Each ``Func`` receives a single
    scenario with ``constraints = "True"`` and ``time_cost = {"Concrete": {"Const": 1}}``.
    The ``Sequential`` node receives a combined scenario whose cost equals the
    number of ``Func`` children.

    The return shape matches what :func:`evaluate_schedule` produces so that
    :func:`resolve_schedule` can merge the result identically.
    """
    inner = schedule["Sequential"]
    schedules = inner.get("schedules", [])

    mock_scenario = {
        "constraints": "True",
        "time_cost": {"Concrete": {"Const": 1}},
    }

    # Fill each Func's scenarios.
    new_schedules = []
    func_count = 0
    for sched in schedules:
        if "Func" in sched:
            new_func = {**sched["Func"], "scenarios": [mock_scenario]}
            new_schedules.append({"Func": new_func})
            func_count += 1
        else:
            new_schedules.append(sched)

    combined_scenario = {
        "constraints": "True",
        "time_cost": {"Concrete": {"Const": func_count}},
    }

    return {
        "scenarios": [combined_scenario],
        "schedules": new_schedules,
    }


from .schedule_utils import contains_sequential as _contains_sequential


def _fill_func_scenarios(schedules, *, evaluator_path=None, evaluator_fn=None):
    """Fill empty Func-level scenarios inside *schedules*.

    For each ``Func`` whose ``scenarios`` is empty, wrap it in a trivial
    single-child Sequential, evaluate, and assign the resulting scenarios
    back to the Func.
    """
    filled = []
    for sched in schedules:
        if "Func" in sched and not sched["Func"].get("scenarios"):
            wrapper = {"Sequential": {"schedules": [sched], "scenarios": []}}
            if evaluator_fn is not None:
                result = evaluator_fn(wrapper)
            else:
                full_result = evaluate_schedule(wrapper, evaluator_path=evaluator_path)
                # Unwrap the Schedule enum wrapper {"Sequential": {...}}.
                result = full_result["Sequential"]
            # The evaluator returns {"scenarios": [...]} for the wrapper
            # Sequential.  For a single-Func Sequential those are exactly
            # the Func's own scenarios.
            func_scenarios = result.get("scenarios", [])
            filled.append({"Func": {**sched["Func"], "scenarios": func_scenarios}})
        else:
            filled.append(sched)
    return filled


def resolve_schedule(
    node,
    evaluator_path: Path | str | None = None,
    evaluator_fn=None,
) -> dict:
    """Walk *node* and evaluate every innermost Sequential with the Rust binary.

    An "innermost Sequential" is a ``Sequential`` whose ``schedules`` children
    contain no further ``Sequential`` nodes anywhere in their subtrees.  All
    other structure in the JSON (outer wrappers, sibling nodes, etc.) is
    preserved unchanged.

    The Rust evaluator binary currently returns only the evaluated fields
    (e.g. ``{"scenarios": [...]}``) rather than a full Schedule tree.  This
    function **merges** those fields back into the original node so that all
    existing fields (``schedules``, ``mlir_ref``, ``processor``, …) are
    preserved and ``scenarios`` (and any other returned field) is added.

    Walking is generic: the function recurses into every dict value and every
    list element, so it does not need to know about ``Parallel`` or any other
    specific variant that wraps the Sequentials.

    Parameters
    ----------
    node:
        A Schedule dict (or any JSON value) already parsed from JSON.
    evaluator_path:
        Forwarded to :func:`evaluate_schedule`.
    evaluator_fn:
        Optional callable replacing the Rust binary invocation.  When set,
        ``evaluator_fn({"Sequential": inner})`` is called instead of
        :func:`evaluate_schedule`.  Useful for mocking (see
        :func:`mock_evaluate_schedule`).

    Returns
    -------
    dict | list | scalar
        The input with every innermost Sequential filled with evaluated scenarios.
    """
    if isinstance(node, list):
        return [resolve_schedule(item, evaluator_path, evaluator_fn) for item in node]

    if not isinstance(node, dict):
        return node

    if "Sequential" in node:
        inner = node["Sequential"]
        schedules = inner.get("schedules", [])
        # No nested Sequential in any child → this is the innermost one.
        if not any(_contains_sequential(child) for child in schedules):
            # Pass only the {"Sequential": inner} slice to the binary — not the
            # surrounding node, which may carry unrelated keys (e.g. when the
            # Sequential key appears inside a larger dict).
            # Binary returns only the evaluated fields (e.g. {"scenarios": [...]}).
            # Merge back so original fields (schedules, mlir_ref, …) are preserved.
            if evaluator_fn is not None:
                evaluated_fields = evaluator_fn({"Sequential": inner})
            else:
                full_result = evaluate_schedule(
                    {"Sequential": inner}, evaluator_path=evaluator_path
                )
                # The binary serializes a Schedule enum, so the output is
                # wrapped: {"Sequential": {...}}.  Unwrap to get the inner
                # content so the merge below works correctly.
                evaluated_fields = full_result["Sequential"]
            filled_inner = {**inner, **evaluated_fields}
            # Also fill individual Func scenarios if the evaluator didn't
            # include them (the Rust binary returns only Sequential-level
            # scenarios, not per-Func ones).
            filled_inner["schedules"] = _fill_func_scenarios(
                filled_inner.get("schedules", []),
                evaluator_path=evaluator_path,
                evaluator_fn=evaluator_fn,
            )
            # Preserve any other keys that existed alongside "Sequential" in node.
            return {**node, "Sequential": filled_inner}
        # Has nested Sequentials → recurse into schedules only; other inner
        # fields are left untouched.
        new_schedules = [resolve_schedule(child, evaluator_path, evaluator_fn) for child in schedules]
        return {**node, "Sequential": {**inner, "schedules": new_schedules}}

    # For any other node (Parallel, Func, custom wrapper, …) recurse into all
    # values generically — no special-casing needed.
    return {k: resolve_schedule(v, evaluator_path, evaluator_fn) for k, v in node.items()}


# ---------------------------------------------------------------------------
# ETG variant resolution
# ---------------------------------------------------------------------------


def validate_scenarios(node, path: str = "") -> list[str]:
    """Walk *node* and return JSON-paths where ``scenarios`` is empty.

    An empty list means every ``scenarios`` field in the tree is populated.
    """
    issues: list[str] = []
    if isinstance(node, dict):
        if "scenarios" in node and not node["scenarios"]:
            issues.append(path or "<root>")
        for k, v in node.items():
            issues.extend(validate_scenarios(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            issues.extend(validate_scenarios(item, f"{path}[{i}]"))
    return issues


def _resolve_single_variant(variant: dict) -> dict:
    """Resolve compute and memory scopes of a single ETG variant."""
    variant = copy.deepcopy(variant)

    # Compute scope — real Rust evaluator.
    variant["compute_scope"] = resolve_schedule(variant["compute_scope"])

    # TODO: Replace mock_evaluate_schedule with the real Rust evaluator once
    # the Rust backend supports memory scope evaluation.
    variant["memory_scope"] = resolve_schedule(
        variant["memory_scope"], evaluator_fn=mock_evaluate_schedule
    )
    return variant


def resolve_etg_variants(
    variants: list[dict],
    njobs: int = 1,
) -> list[dict]:
    """Resolve all ETG variants in parallel using the MLAR evaluator.

    For each variant, evaluates ``compute_scope`` with the Rust binary and
    ``memory_scope`` with the mock evaluator.  After resolution, validates
    that all ``scenarios`` fields are populated.

    Parameters
    ----------
    variants:
        Parsed ETG variant list (as loaded from the exploration ETG JSON).
    njobs:
        Number of parallel worker threads.  The Rust evaluator runs as a
        subprocess so the GIL is released during I/O wait — threads are
        appropriate here.

    Returns
    -------
    list[dict]
        The resolved variant list with all ``scenarios`` filled.

    Raises
    ------
    RuntimeError
        If any variant fails resolution or has empty ``scenarios`` after
        resolution.
    """
    total = len(variants)
    results: list[dict | None] = [None] * total

    with ThreadPoolExecutor(max_workers=njobs) as executor:
        future_to_idx = {
            executor.submit(_resolve_single_variant, v): i
            for i, v in enumerate(variants)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            name = variants[idx].get("variant_name", f"variant_{idx}")
            result = future.result()  # Let exceptions propagate.
            results[idx] = result
            print(f"  [{idx:>2}] {name} — resolved")

    # Validate all scenarios are filled.
    for i, variant in enumerate(results):
        issues = validate_scenarios(variant)
        if issues:
            name = variant.get("variant_name", f"variant_{i}")
            detail = "; ".join(issues[:5])
            raise RuntimeError(
                f"Variant '{name}' has empty scenarios after resolution: {detail}"
            )

    return results


# ---------------------------------------------------------------------------
# Smart JSON formatting — structural keys pretty, expressions compact
# ---------------------------------------------------------------------------

# Keys that indicate an expression node (should be serialized compactly).
_EXPR_KEYS = frozenset({
    "Sym", "Const", "Add", "Sub", "Mul", "Div", "Min", "Max", "Neg",
    "Ge", "Le", "Gt", "Lt", "Eq", "Ne", "And", "Or", "Not",
    "Divisible", "Concrete",
})


def _is_expr(obj) -> bool:
    """Return True if *obj* looks like a symbolic expression node."""
    if isinstance(obj, dict):
        if len(obj) == 1 and next(iter(obj)) in _EXPR_KEYS:
            return True
        # {"by": ..., "x": ...} inside Divisible is also expression-like.
        if set(obj.keys()) <= {"by", "x"}:
            return all(_is_expr(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_is_expr(item) for item in obj)
    if isinstance(obj, (int, float, str)):
        return True
    return False


def smart_json_dumps(obj, indent=2) -> str:
    """Serialize *obj* to JSON with hybrid formatting.

    Structural keys (schedules, stages, Func, Sequential, …) are
    pretty-printed with *indent*.  Expression-like sub-trees (Mul, Add,
    Sym, Const, constraint nodes, …) are kept compact on a single line.
    """

    def _fmt(node, depth):
        pad = " " * (indent * depth)
        pad_inner = " " * (indent * (depth + 1))

        if not isinstance(node, (dict, list)):
            return json.dumps(node)

        # Expression sub-trees → compact.
        if _is_expr(node):
            return json.dumps(node, separators=(",", ":"))

        if isinstance(node, list):
            if not node:
                return "[]"
            items = [_fmt(item, depth + 1) for item in node]
            # If all items are compact (no newlines), try single-line.
            if all("\n" not in item for item in items):
                one_line = "[" + ", ".join(items) + "]"
                if len(one_line) + len(pad) <= 120:
                    return one_line
            inner = ",\n".join(pad_inner + item for item in items)
            return "[\n" + inner + "\n" + pad + "]"

        if isinstance(node, dict):
            if not node:
                return "{}"
            parts = []
            for k, v in node.items():
                val = _fmt(v, depth + 1)
                parts.append(f"{json.dumps(k)}: {val}")
            # Try single-line for small dicts.
            if all("\n" not in p for p in parts):
                one_line = "{" + ", ".join(parts) + "}"
                if len(one_line) + len(pad) <= 120:
                    return one_line
            inner = ",\n".join(pad_inner + p for p in parts)
            return "{\n" + inner + "\n" + pad + "}"

        return json.dumps(node)

    return _fmt(obj, 0)


def evaluate_schedule_file(
    json_path: Path | str,
    *,
    in_place: bool = True,
    evaluator_path: Path | str | None = None,
    resolve: bool = True,
) -> dict:
    """Load a Schedule JSON file, evaluate it, and optionally write back.

    Parameters
    ----------
    json_path:
        Path to the Schedule JSON file.
    in_place:
        If ``True`` (default), overwrite *json_path* with the evaluated result.
    evaluator_path:
        Forwarded to :func:`evaluate_schedule` / :func:`resolve_schedule`.
    resolve:
        If ``True`` (default), use :func:`resolve_schedule` to locate and
        evaluate only the innermost Sequential node(s) in a potentially nested
        structure.  Set to ``False`` to pass the entire JSON directly to the
        evaluator (flat schedules only).

    Returns
    -------
    dict
        The evaluated Schedule dict.
    """
    json_path = Path(json_path)
    schedule = json.loads(json_path.read_text())
    if resolve:
        result = resolve_schedule(schedule, evaluator_path=evaluator_path)
    else:
        result = evaluate_schedule(schedule, evaluator_path=evaluator_path)
    if in_place:
        json_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

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
        "--no-in-place",
        dest="in_place",
        action="store_false",
        help="Print the result to stdout instead of overwriting the input file.",
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
