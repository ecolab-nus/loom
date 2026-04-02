"""SMT-based block-size optimizer for the Loom dataflow pipeline.

Solves all variants in the staged ETG JSON in parallel, writes per-variant
detailed results to a log file, and prints the global best variant to the
console.

NOTE: Z3's global AST context is not thread-safe. Workers run in separate
processes (ProcessPoolExecutor) so each gets its own Z3 context.

Usage:
    python main.py --input PATH [--njobs N] [--output PATH]

Arguments:
    --input      Path to staged_etg_dump.json (required).
    --njobs      Number of parallel worker processes (default: 1).
    --output     Path for the per-variant log file (optional).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .utils.json_loader import load_variants
from .utils.utils import get_variant_name, derive_domains_from_etg
from .utils.reporter import (
    print_breakdown, print_unsat_core,
    print_active_constraints, print_mus, print_result_summary,
)
from .core.solver_context import SolverContext
from .models.pipeline_agg import compute_total_time


# ---------------------------------------------------------------------------
# Single-variant solver (runs in a worker process)
# ---------------------------------------------------------------------------

def solve_variant(
    variant: dict,
    index: int,
    total: int,
    domains: dict[str, list[int]],
    debug: bool = False,
    force_enumerate: bool = False,
) -> dict:
    """Solve one variant and return a result dict.

    Runs in its own process, so Z3 state is fully isolated.

    Returns:
        {
            "variant":     variant dict,
            "index":       int,
            "total":       int,
            "min_val":     int | None,   # None means UNSAT
            "assignments": dict | None,
        }
    """
    ctx = SolverContext(debug=debug)
    ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
    bool_domains = ctx.load_booleans(
        variant["constraint_scope"]["metadata"].get("booleans", [])
    )
    ctx.add_hard_constraints(variant["constraint_scope"]["hard_constraints"])
    ctx.add_domain_constraints(domains)

    # Merge boolean domains so find_optimum/enumerate can iterate over them.
    domains_with_bools = {**domains, **bool_domains}

    t_total = compute_total_time(variant, ctx.symbol_map)
    result = ctx.find_optimum(t_total, domains_with_bools, force_enumerate=force_enumerate)

    min_val, assignments = result if result is not None else (None, None)

    # Debug analysis
    active_constraints = None
    mus = None
    if debug:
        hard_constraints = variant["constraint_scope"]["hard_constraints"]
        if min_val is not None:
            active_constraints = ctx.find_active_constraints(
                hard_constraints, assignments, domains_with_bools,
            )
        else:
            mus = ctx.find_mus(hard_constraints, domains_with_bools)

    return {
        "variant":     variant,
        "index":       index,
        "total":       total,
        "min_val":     min_val,
        "assignments": assignments,
        "unsat_core":  ctx.last_unsat_core_info if debug else None,
        "active_constraints": active_constraints,
        "mus":         mus,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _write_detailed_log(
    results: list[dict], 
    output_path: Path | str, 
    total: int, 
    debug: bool = False
) -> None:
    """Write per-variant details to log file, ordered by original index."""
    with open(output_path, "w", encoding="utf-8") as log:
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            if r["min_val"] is None:
                print(
                    f"Variant [{r['index']}/{total - 1}]: {vname}  UNSAT\n",
                    file=log,
                )
                if debug and r.get("mus"):
                    print_mus(vname, r["mus"], file=log)
                elif debug and r.get("unsat_core"):
                    print_unsat_core(
                        vname, r["unsat_core"],
                        context="Infeasible", file=log,
                    )
            else:
                if debug:
                    print_breakdown(
                        r["variant"], r["assignments"],
                        r["min_val"], r["index"], total,
                        file=log,
                    )
                    if r.get("active_constraints"):
                        print_active_constraints(
                            vname, r["active_constraints"], file=log,
                        )
                    if r.get("unsat_core"):
                        print_unsat_core(
                            vname, r["unsat_core"],
                            context=f"Optimum bound T>={r['min_val']}",
                            file=log,
                        )
                else:
                    print_result_summary(
                        vname, r["assignments"],
                        r["min_val"], r["index"], total,
                        file=log,
                    )
                print("-" * 72, file=log)
    print(f"\nPer-variant log written to: {output_path}")


def run(
    input_path: Path | str,
    njobs: int = 1,
    output_path: Path | str | None = None,
    debug: bool = False,
    force_enumerate: bool = False,
    symbol_domains: dict[str, list[int]] | None = None,
    optimal_only: bool = False,
) -> dict[str, dict[str, int] | None]:
    """Execute the SMT solver on the provided ETG variants."""
    variants = load_variants(input_path)
    total = len(variants)

    # Baseline: ETG-derived domains from natural_ub (lb=1)
    etg_domains = derive_domains_from_etg(variants)
    # Merge: user-supplied domains override per symbol
    domains = {**etg_domains, **(symbol_domains or {})}

    # Assert: every symbol referenced in the ETG must have a domain
    missing = sorted({
        sym
        for v in variants
        for sym in v.get("constraint_scope", {}).get("metadata", {}).get("symbols", {})
        if sym not in domains
    })
    if missing:
        raise ValueError(
            "No domain found for symbols: " + ", ".join(missing) + ". "
            "Either add 'block_sizes' in your config or ensure the ETG carries "
            "'natural_ub' for these symbols."
        )

    print(f"Solving {total} variants with {njobs} process(es)...")
    print()

    # Dispatch all variants to a process pool.
    results: list[dict] = [None] * total
    with ProcessPoolExecutor(max_workers=njobs) as pool:
        futures = {
            pool.submit(solve_variant, v, i, total, domains, debug, force_enumerate): i
            for i, v in enumerate(variants)
        }
        completed = 0
        for future in as_completed(futures):
            r = future.result()
            results[r["index"]] = r
            completed += 1
            variant_name = get_variant_name(r["variant"], r["index"])
            if r["min_val"] is None:
                print(f"[{completed:3d}/{total}] {variant_name}  UNSAT")
            else:
                print(f"[{completed:3d}/{total}] {variant_name}  T={r['min_val']:,} cycles")

    if output_path:
        _write_detailed_log(results, output_path, total, debug=debug)

    # Build the consolidated block size map (None for UNSAT variants).
    block_sizes: dict[str, dict[str, int] | None] = {}
    for r in results:
        vname = get_variant_name(r["variant"], r["index"])
        if r["min_val"] is not None:
            block_sizes[vname] = dict(r["assignments"])
        else:
            block_sizes[vname] = None

    # Find the global best (minimum T_total across all feasible variants).
    feasible = [r for r in results if r["min_val"] is not None]
    if not feasible:
        print("\nResult: UNSAT — no variant has a feasible solution.")
        return block_sizes

    best = min(feasible, key=lambda r: r["min_val"])

    if optimal_only:
        print("\nNote: --optimal-only is passed. We only materialize the optimal candidates. Other candidates' block sizes are dropped.")
        optimal_val = best["min_val"]
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            if r["min_val"] is not None and r["min_val"] > optimal_val:
                block_sizes[vname] = None

    print()
    print("=" * 72)
    print("GLOBAL BEST")
    print("=" * 72)
    print_breakdown(
        best["variant"], best["assignments"],
        best["min_val"], best["index"], total,
    )

    return block_sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find optimal block sizes across all Loom ETG variants."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="JSON",
        help="Path to staged_etg_dump.json",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Log file for per-variant results (optional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable detailed analysis: active constraints at optimum, MUS for UNSAT.",
    )
    parser.add_argument(
        "--force-enumerate",
        action="store_true",
        default=False,
        help="Skip Z3 binary search and brute-force enumerate all domain combinations.",
    )
    parser.add_argument(
        "--optimal-only",
        action="store_true",
        default=False,
        help="Keep only the optimal candidates and remove all others in the final output.",
    )
    args = parser.parse_args()
    block_sizes = run(
        input_path=args.input,
        njobs=args.njobs,
        output_path=args.output,
        debug=args.debug,
        force_enumerate=args.force_enumerate,
        optimal_only=args.optimal_only,
    )
    if not any(v is not None for v in block_sizes.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
