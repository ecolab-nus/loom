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
import subprocess
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


def _contains_sequential(node) -> bool:
    """Return True if *node* is, or contains anywhere in its subtree, a Sequential.

    Walks arbitrary JSON (dicts and lists) so the check is not tied to any
    specific Schedule variant structure.
    """
    if isinstance(node, dict):
        if "Sequential" in node:
            return True
        return any(_contains_sequential(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_sequential(item) for item in node)
    return False


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
                result = evaluate_schedule(wrapper, evaluator_path=evaluator_path)
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
                evaluated_fields = evaluate_schedule(
                    {"Sequential": inner}, evaluator_path=evaluator_path
                )
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
