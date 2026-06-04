"""CPMpy-based block-size optimizer for Loom using Pure Python AST."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from ..loom_utils.io import load_variants
from ..loom_utils.modeling import (
    get_variant_name, derive_domains_from_etg,
    print_breakdown, print_mus,
)
from ..loom_utils.modeling import TIME_COST_SCALE, compute_total_time_ast
from ..loom_utils.ast import (
    build_l1_memory_constraint,
    Const,
    Div,
    parse_expr,
    parse_constraint,
)
from .core.solver_context import SolverContext


def solve_variant(
    variant: dict,
    index: int,
    total: int,
    domains: dict[str, list[int]],
    debug: bool = False,
) -> dict:
    """Solve one variant and return a result dict."""
    ctx = SolverContext()
    ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"], domains)
    ctx.load_booleans(variant["constraint_scope"]["metadata"].get("booleans", []))

    # Parse hard constraints directly (no ASTTransformer needed)
    hard_constraints_ast = [
        parse_constraint(c)
        for c in variant["constraint_scope"]["hard_constraints"]
    ]
    memory_constraints_ast = [
        build_l1_memory_constraint(variant["constraint_scope"]["metadata"])
    ]
    t_total_ast = compute_total_time_ast(variant)

    # Add constraints and solve
    ctx.add_hard_constraints(hard_constraints_ast)
    ctx.add_hard_constraints(memory_constraints_ast, label_prefix="memory_l1")
    iter_exprs, iter_labels = _collect_iter_num_constraints(
        variant["constraint_scope"]["metadata"]["iter_num"]
    )
    ctx.add_trip_count_constraints(iter_exprs, labels=iter_labels)
    status, scaled_min_val, assignments = ctx.find_optimum(t_total_ast)
    min_val = scaled_min_val

    mus = None
    if debug and status == "INFEASIBLE":
        mus = ctx.find_mus()

    return {
        "variant": variant,
        "index": index,
        "total": total,
        "status": status,
        "min_val": min_val,
        "scaled_min_val": scaled_min_val,
        "assignments": assignments,
        "mus": mus,
    }


def _collect_iter_num_constraints(iter_num: dict) -> tuple[list[dict], list[str]]:
    """Collect solver feasibility expressions from constraint metadata."""
    exprs = [iter_num["seq_iter"]]
    labels = ["iter_num.seq_iter"]

    for i, temp_iter in enumerate(iter_num.get("temp_iter", [])):
        exprs.append(temp_iter)
        labels.append(f"iter_num.temp_iter[{i}]")

    return exprs, labels


def _parse_solver_time_cost(time_cost: object):
    return Div(parse_expr(time_cost), Const(TIME_COST_SCALE))


def prepare_manual_block_sizes(
    variants: list[dict],
    assigned_block_size: dict[str, Any],
) -> dict[str, Any]:
    """Expand and validate manually assigned block sizes by variant name."""
    if not assigned_block_size:
        raise ValueError("assigned_block_size must be a non-empty object")

    if "ALL" in assigned_block_size and len(assigned_block_size) > 1:
        raise ValueError("assigned_block_size cannot mix 'ALL' with candidate names")

    normalized = {
        key: _normalize_manual_assignment_value(key, assignment)
        for key, assignment in assigned_block_size.items()
    }

    variant_names = [get_variant_name(variant, i) for i, variant in enumerate(variants)]
    if "ALL" in assigned_block_size:
        assignment = normalized["ALL"]
        return {name: _copy_manual_assignment_value(assignment) for name in variant_names}

    unknown = sorted(set(assigned_block_size) - set(variant_names))
    if unknown:
        available = ", ".join(variant_names[:5])
        suffix = "..." if len(variant_names) > 5 else ""
        raise ValueError(
            "Unknown assigned_block_size candidate(s): "
            f"{', '.join(unknown)}. Available candidates include: {available}{suffix}"
        )

    return {
        name: _copy_manual_assignment_value(assignment)
        for name, assignment in normalized.items()
    }


def write_manual_breakdown_log(
    variants: list[dict],
    assigned_block_size: dict[str, Any],
    output_path: Path | str,
) -> dict[str, Any]:
    """Write reporter breakdowns for manual assignments without running the solver."""
    block_sizes = prepare_manual_block_sizes(variants, assigned_block_size)
    total = len(variants)

    with open(output_path, "w", encoding="utf-8") as log:
        for index, variant in enumerate(variants):
            vname = get_variant_name(variant, index)
            assignment = block_sizes.get(vname)
            if assignment is None:
                continue

            for combo in _iter_manual_assignment_combos(assignment):
                completed = _complete_manual_assignment(variant, combo, vname)
                min_val = compute_total_time_ast(variant).eval(completed)
                print_breakdown(
                    variant,
                    completed,
                    min_val,
                    index,
                    total,
                    file=log,
                    cost_parser=_parse_solver_time_cost,
                    unit="solver units",
                )
                print("-" * 72, file=log)

    return block_sizes


def _normalize_manual_assignment_value(key: str, assignment: Any) -> Any:
    if assignment is None:
        return None
    if isinstance(assignment, dict):
        return dict(assignment)
    if isinstance(assignment, list):
        if not assignment:
            raise ValueError(f"assigned_block_size['{key}'] must not be an empty list")
        normalized = []
        for i, combo in enumerate(assignment):
            if not isinstance(combo, dict):
                raise ValueError(
                    f"assigned_block_size['{key}'][{i}] must be an object"
                )
            normalized.append(dict(combo))
        return normalized
    raise ValueError(
        f"assigned_block_size['{key}'] must be an object, list of objects, or null"
    )


def _copy_manual_assignment_value(assignment: Any) -> Any:
    if assignment is None:
        return None
    if isinstance(assignment, list):
        return [dict(combo) for combo in assignment]
    return dict(assignment)


def _iter_manual_assignment_combos(assignment: Any) -> list[dict[str, Any]]:
    if isinstance(assignment, list):
        return assignment
    return [assignment]


def _complete_manual_assignment(
    variant: dict,
    assignment: dict[str, Any],
    variant_name: str,
) -> dict[str, int]:
    metadata = variant.get("constraint_scope", {}).get("metadata", {})
    required_symbols = set(metadata.get("symbols", {}))
    boolean_symbols = set(metadata.get("booleans", []))
    completed: dict[str, int] = {}

    for sym, value in assignment.items():
        if not isinstance(value, int):
            raise ValueError(
                f"assigned_block_size['{variant_name}']['{sym}'] must be an integer"
            )
        completed[sym] = int(value)

    missing = sorted(required_symbols - set(completed))
    if missing:
        raise ValueError(
            f"assigned_block_size['{variant_name}'] missing required symbol(s): "
            f"{', '.join(missing)}"
        )

    for sym in sorted(boolean_symbols):
        completed.setdefault(sym, 1)
        if completed[sym] not in (0, 1):
            raise ValueError(
                f"assigned_block_size['{variant_name}']['{sym}'] must be 0 or 1"
            )

    return completed


def _write_detailed_log(
    results: list[dict], output_path: Path | str, total: int, debug: bool = False,
) -> None:
    with open(output_path, "w", encoding="utf-8") as log:
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            if r["status"] != "OPTIMAL":
                print(f"Variant [{r['index']}/{total - 1}]: {vname}  {r['status']}\n", file=log)
                if debug and r.get("mus"):
                    print_mus(vname, r["mus"], file=log)
            else:
                print_breakdown(
                    r["variant"],
                    r["assignments"],
                    r["min_val"],
                    r["index"],
                    total,
                    file=log,
                    cost_parser=_parse_solver_time_cost,
                    unit="solver units",
                )
                print("-" * 72, file=log)


def run(
    input_path: Path | str,
    njobs: int = 1,
    output_path: Path | str | None = None,
    symbol_domains: dict[str, list[int]] | None = None,
    topk: int | None = None,
    debug: bool = False,
) -> dict[str, dict[str, int] | None]:
    if topk is not None and topk <= 0:
        raise ValueError("topk must be a positive integer")

    variants = load_variants(input_path)
    total = len(variants)
    etg_domains = derive_domains_from_etg(variants)
    domains = {**etg_domains, **(symbol_domains or {})}

    print(f"Solving {total} variants with {njobs} process(es) [CPMpy/CP-SAT]...")

    results: list[dict] = [None] * total
    completed = 0
    with ProcessPoolExecutor(max_workers=njobs) as pool:
        futures = {
            pool.submit(
                solve_variant,
                v,
                i,
                total,
                domains,
                debug,
            ): i
            for i, v in enumerate(variants)
        }
        for f in as_completed(futures):
            res = f.result()
            results[res["index"]] = res
            completed += 1
            vname = get_variant_name(res["variant"], res["index"])
            status = res["status"]
            if status == "OPTIMAL":
                print(f"[{completed:3d}/{total}] {vname}  T={res['min_val']:,} solver units  OPTIMAL")
            else:
                print(f"[{completed:3d}/{total}] {vname}  {status}")

    # Determine global best among OPTIMAL candidates
    feasible = [r for r in results if r["status"] == "OPTIMAL"]
    best = min(feasible, key=lambda r: r["scaled_min_val"]) if feasible else None

    topk_filter = _select_topk(feasible, topk)
    omitted_by_topk = topk_filter["omitted"]
    tied_omitted_by_topk = topk_filter["tied_omitted"]
    tied_omitted_indices = {r["index"] for r in tied_omitted_by_topk}
    if tied_omitted_by_topk:
        print("\nTOPK TIE OMITTED")
        print(
            f"  topk={topk}, cutoff T={topk_filter['cutoff_min_val']:,} solver units, "
            f"kept={topk_filter['kept_count']}, omitted_tied={len(tied_omitted_by_topk)}"
        )
        for r in sorted(tied_omitted_by_topk, key=lambda item: item["index"]):
            vname = get_variant_name(r["variant"], r["index"])
            print(f"  [{r['index']:3d}/{total - 1}] {vname}  T={r['min_val']:,} solver units")

    block_sizes = {
        get_variant_name(r["variant"], r["index"]): (
            dict(r["assignments"])
            if r["status"] == "OPTIMAL" and r["index"] not in omitted_by_topk
            else None
        )
        for r in results
    }

    # Overall omit summary (debug-only, index order)
    if debug:
        omit_lines = []
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            idx = r["index"] + 1
            if r["status"] != "OPTIMAL":
                omit_lines.append(f"  [{idx:3d}/{total}] {vname}  OMITTED: no feasible solution ({r['status']})")
            elif r["index"] in omitted_by_topk:
                if r["index"] in tied_omitted_indices:
                    omit_lines.append(f"  [{idx:3d}/{total}] {vname}  OMITTED: tied at topk cutoff (T={r['min_val']:,} solver units)")
                else:
                    omit_lines.append(f"  [{idx:3d}/{total}] {vname}  OMITTED: outside topk={topk} (T={r['min_val']:,} solver units)")
        if omit_lines:
            print("\nOMIT SUMMARY:")
            for line in omit_lines:
                print(line)

    if output_path:
        _write_detailed_log(results, output_path, total, debug=debug)

    if best:
        print("\nGLOBAL BEST")
        print_breakdown(
            best["variant"],
            best["assignments"],
            best["min_val"],
            best["index"],
            total,
            cost_parser=_parse_solver_time_cost,
            unit="solver units",
        )
    return block_sizes


def _select_topk(feasible: list[dict], topk: int | None) -> dict[str, Any]:
    if topk is None or not feasible:
        return {
            "omitted": set(),
            "tied_omitted": [],
            "cutoff_min_val": None,
            "kept_count": len(feasible),
        }

    top_results = sorted(feasible, key=lambda r: (r["scaled_min_val"], r["index"]))[:topk]
    kept_by_topk = {r["index"] for r in top_results}
    cutoff_min_val = top_results[-1]["min_val"] if top_results else None
    cutoff_scaled_min_val = top_results[-1]["scaled_min_val"] if top_results else None
    omitted: set[int] = set()
    tied_omitted = []
    for r in feasible:
        if r["index"] not in kept_by_topk:
            omitted.add(r["index"])
            if cutoff_scaled_min_val is not None and r["scaled_min_val"] == cutoff_scaled_min_val:
                tied_omitted.append(r)

    return {
        "omitted": omitted,
        "tied_omitted": tied_omitted,
        "cutoff_min_val": cutoff_min_val,
        "kept_count": len(top_results),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--topk must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Find optimal block sizes using CPMpy/CP-SAT.")
    parser.add_argument("--input", required=True, help="Path to resolved ETG JSON")
    parser.add_argument("--njobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output", help="Log file path")
    parser.add_argument("--topk", type=_positive_int, help="Output only the top K candidates")
    args = parser.parse_args()
    run(
        input_path=args.input,
        njobs=args.njobs,
        output_path=args.output,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
