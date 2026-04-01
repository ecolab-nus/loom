"""Core scheduling evaluation logic for the Loom pipeline."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from .schedule_utils import contains_sequential as _contains_sequential

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_default_evaluator() -> Path | None:
    """Locate the eval_system binary using a cascading search.

    Search order:
    1. LOOM_EVAL_SYSTEM environment variable (explicit override)
    2. $REPO_ROOT/third_party/loom-mlar/tests/2d_mesh/bin/eval_system (canonical build output)
    3. 'eval_system' on $PATH (system-installed)
    """
    env_path = os.environ.get("LOOM_EVAL_SYSTEM")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    canonical = _REPO_ROOT / "third_party" / "loom-mlar" / "tests" / "2d_mesh" / "bin" / "eval_system"
    if canonical.is_file():
        return canonical

    system = shutil.which("eval_system")
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
            "eval_system binary not found. Build it with: bash scripts/build-mlar.sh\n"
            "Or set LOOM_EVAL_SYSTEM=/path/to/eval_system"
        )
    if not binary.exists():
        raise FileNotFoundError(f"Evaluator binary not found: {binary}")

    result = subprocess.run(
        [str(binary)],
        input=json.dumps(schedule),
        # capture_output=True,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _fill_func_scenarios(schedules, *, evaluator_path=None):
    """Fill empty Func-level scenarios inside *schedules*."""
    filled = []
    for sched in schedules:
        if "Func" in sched and not sched["Func"].get("scenarios"):
            wrapper = {"Sequential": {"schedules": [sched], "scenarios": []}}
            full_result = evaluate_schedule(wrapper, evaluator_path=evaluator_path)
            func_scenarios = full_result["Sequential"].get("scenarios", [])
            filled.append({"Func": {**sched["Func"], "scenarios": func_scenarios}})
        else:
            filled.append(sched)
    return filled


def resolve_schedule(
    node,
    evaluator_path: Path | str | None = None,
) -> dict:
    """Walk *node* and evaluate every innermost Sequential with the Rust binary."""
    if isinstance(node, list):
        return [resolve_schedule(item, evaluator_path) for item in node]

    if not isinstance(node, dict):
        return node

    if "Sequential" in node:
        inner = node["Sequential"]
        schedules = inner.get("schedules", [])
        if not any(_contains_sequential(child) for child in schedules):
            full_result = evaluate_schedule(
                {"Sequential": inner}, evaluator_path=evaluator_path
            )
            evaluated_fields = full_result["Sequential"]
            filled_inner = {**inner, **evaluated_fields}
            filled_inner["schedules"] = _fill_func_scenarios(
                filled_inner.get("schedules", []),
                evaluator_path=evaluator_path,
            )
            return {**node, "Sequential": filled_inner}
        new_schedules = [resolve_schedule(child, evaluator_path) for child in schedules]
        return {**node, "Sequential": {**inner, "schedules": new_schedules}}

    return {k: resolve_schedule(v, evaluator_path) for k, v in node.items()}
