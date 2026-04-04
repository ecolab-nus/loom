"""CPMpy-based block-size optimizer for Loom using Pure Python AST."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from ..loom_utils.io import load_variants
from ..loom_utils.modeling import (
    get_variant_name, derive_domains_from_etg,
    print_breakdown, print_result_summary,
)
from ..loom_utils.modeling import compute_total_time_ast
from ..loom_utils.ast import parse_constraint
from .core.solver_context import SolverContext


def solve_variant(
    variant: dict,
    index: int,
    total: int,
    domains: dict[str, list[int]],
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
    t_total_ast = compute_total_time_ast(variant)

    # Add constraints and solve
    ctx.add_hard_constraints(hard_constraints_ast)
    ctx.add_iter_num_constraints(variant["constraint_scope"]["metadata"]["iter_num"])
    result = ctx.find_optimum(t_total_ast)
    min_val, assignments = result if result is not None else (None, None)

    return {
        "variant": variant,
        "index": index,
        "total": total,
        "min_val": min_val,
        "assignments": assignments,
    }


def _write_detailed_log(
    results: list[dict], output_path: Path | str, total: int,
) -> None:
    with open(output_path, "w", encoding="utf-8") as log:
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            if r["min_val"] is None:
                print(f"Variant [{r['index']}/{total - 1}]: {vname}  UNSAT\n", file=log)
            else:
                print_result_summary(vname, r["assignments"], r["min_val"], r["index"], total, file=log)
                print("-" * 72, file=log)


def run(
    input_path: Path | str,
    njobs: int = 1,
    output_path: Path | str | None = None,
    symbol_domains: dict[str, list[int]] | None = None,
    optimal_only: bool = False,
) -> dict[str, dict[str, int] | None]:
    variants = load_variants(input_path)
    total = len(variants)
    etg_domains = derive_domains_from_etg(variants)
    domains = {**etg_domains, **(symbol_domains or {})}

    print(f"Solving {total} variants with {njobs} process(es) [CPMpy/CP-SAT]...")

    results: list[dict] = [None] * total
    with ProcessPoolExecutor(max_workers=njobs) as pool:
        futures = {
            pool.submit(solve_variant, v, i, total, domains): i
            for i, v in enumerate(variants)
        }
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            results[res["index"]] = res
            completed += 1
            vname = get_variant_name(res["variant"], res["index"])
            if res["min_val"] is None:
                print(f"[{completed:3d}/{total}] {vname}  UNSAT")
            else:
                print(f"[{completed:3d}/{total}] {vname}  T={res['min_val']:,} cycles")

    if output_path:
        _write_detailed_log(results, output_path, total)

    block_sizes = {
        get_variant_name(r["variant"], r["index"]): (
            dict(r["assignments"]) if r["min_val"] is not None else None
        )
        for r in results
    }
    feasible = [r for r in results if r["min_val"] is not None]
    if feasible:
        best = min(feasible, key=lambda r: r["min_val"])
        if optimal_only:
            for r in results:
                if r["min_val"] is not None and r["min_val"] > best["min_val"]:
                    block_sizes[get_variant_name(r["variant"], r["index"])] = None
        print("\nGLOBAL BEST")
        print_breakdown(best["variant"], best["assignments"], best["min_val"], best["index"], total)
    return block_sizes


def main():
    parser = argparse.ArgumentParser(description="Find optimal block sizes using CPMpy/CP-SAT.")
    parser.add_argument("--input", required=True, help="Path to resolved ETG JSON")
    parser.add_argument("--njobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output", help="Log file path")
    parser.add_argument("--optimal-only", action="store_true", help="Output only best")
    args = parser.parse_args()
    run(input_path=args.input, njobs=args.njobs, output_path=args.output, optimal_only=args.optimal_only)


if __name__ == "__main__":
    main()
