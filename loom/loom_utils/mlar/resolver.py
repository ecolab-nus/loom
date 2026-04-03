"""ETG variant resolution logic using the MLAR evaluator."""

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import resolve_schedule


def validate_scenarios(node, path: str = "") -> list[str]:
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
    return resolve_schedule(copy.deepcopy(variant))


def resolve_etg_variants(
    variants: list[dict],
    njobs: int = 1,
) -> list[dict]:
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
            result = future.result()
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
