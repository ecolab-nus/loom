"""Core scheduling evaluation logic for the Loom pipeline."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from .schedule_utils import contains_sequential as _contains_sequential

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_default_evaluator() -> Path | None:
    """Locate the eval_core binary using a cascading search.

    Search order:
    1. LOOM_EVAL_CORE environment variable (explicit override)
    2. $REPO_ROOT/third_party/loom-mlar/bin/eval_core (canonical build output)
    3. $REPO_ROOT/third_party/loom-mlar/tests/2d_mesh/evaluators/eval_core (legacy)
    4. 'eval_core' on $PATH (system-installed)
    """
    env_path = os.environ.get("LOOM_EVAL_CORE")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    canonical = _REPO_ROOT / "third_party" / "loom-mlar" / "bin" / "eval_core"
    if canonical.is_file():
        return canonical

    legacy = (
        _REPO_ROOT / "third_party" / "loom-mlar" / "tests"
        / "2d_mesh" / "evaluators" / "eval_core"
    )
    if legacy.is_file():
        return legacy

    system = shutil.which("eval_core")
    if system:
        return Path(system)

    return None


_DEFAULT_EVALUATOR = _find_default_evaluator()


def evaluate_schedule(
    schedule: dict,
    evaluator_path: Path | str | None = None,
) -> dict:
    """Run the MLAR evaluator on *schedule* and return the evaluated result."""
    binary = Path(evaluator_path) if evaluator_path is not None else _DEFAULT_EVALUATOR
    if binary is None:
        raise FileNotFoundError(
            "eval_core binary not found. Build it with: bash scripts/build-mlar.sh\n"
            "Or set LOOM_EVAL_CORE=/path/to/eval_core"
        )
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
    """Return placeholder scenarios without calling the Rust binary."""
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


def _fill_func_scenarios(schedules, *, evaluator_path=None, evaluator_fn=None):
    """Fill empty Func-level scenarios inside *schedules*."""
    filled = []
    for sched in schedules:
        if "Func" in sched and not sched["Func"].get("scenarios"):
            wrapper = {"Sequential": {"schedules": [sched], "scenarios": []}}
            if evaluator_fn is not None:
                result = evaluator_fn(wrapper)
            else:
                full_result = evaluate_schedule(wrapper, evaluator_path=evaluator_path)
                result = full_result["Sequential"]
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
    """Walk *node* and evaluate every innermost Sequential with the Rust binary."""
    if isinstance(node, list):
        return [resolve_schedule(item, evaluator_path, evaluator_fn) for item in node]

    if not isinstance(node, dict):
        return node

    if "Sequential" in node:
        inner = node["Sequential"]
        schedules = inner.get("schedules", [])
        if not any(_contains_sequential(child) for child in schedules):
            if evaluator_fn is not None:
                evaluated_fields = evaluator_fn({"Sequential": inner})
            else:
                full_result = evaluate_schedule(
                    {"Sequential": inner}, evaluator_path=evaluator_path
                )
                evaluated_fields = full_result["Sequential"]
            filled_inner = {**inner, **evaluated_fields}
            filled_inner["schedules"] = _fill_func_scenarios(
                filled_inner.get("schedules", []),
                evaluator_path=evaluator_path,
                evaluator_fn=evaluator_fn,
            )
            return {**node, "Sequential": filled_inner}
        new_schedules = [resolve_schedule(child, evaluator_path, evaluator_fn) for child in schedules]
        return {**node, "Sequential": {**inner, "schedules": new_schedules}}

    return {k: resolve_schedule(v, evaluator_path, evaluator_fn) for k, v in node.items()}
