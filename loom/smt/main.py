"""SMT-based block-size optimizer for Loom using Pure Python AST.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from ..loom_utils.io import load_variants
from ..loom_utils.modeling import (
    get_variant_name, derive_domains_from_etg,
    print_breakdown, print_unsat_core,
    print_active_constraints, print_mus, print_result_summary,
)
from .core.solver_context import SolverContext
from ..loom_utils.modeling import compute_total_time_ast, analyze_constraints
from ..loom_utils.ast import parse_constraint, ASTTransformer


def solve_variant(
    variant: dict,
    index: int,
    total: int,
    domains: dict[str, list[int]],
    debug: bool = False,
    force_enumerate: bool = False,
) -> dict:
    """Solve one variant and return a result dict."""
    ctx = SolverContext(debug=debug)
    ctx.load_symbols(variant["constraint_scope"]["metadata"]["symbols"])
    bool_domains = ctx.load_booleans(variant["constraint_scope"]["metadata"].get("booleans", []))
    
    # 1. Build initial ASTs
    hard_constraints_ast = [parse_constraint(c) for c in variant["constraint_scope"]["hard_constraints"]]
    t_total_ast = compute_total_time_ast(variant)
    
    # 2. Transform (Division elimination)
    xf = ASTTransformer()
    t_total_clean = xf.transform(t_total_ast)
    hard_constraints_clean = [xf.transform(c) for c in hard_constraints_ast]
    
    # 2b. Load auxiliary symbols into solver context
    aux_symbols_dict = {s["name"]: s for s in xf.aux_symbols}
    ctx.load_symbols(aux_symbols_dict, add_positivity_guard=False)
    
    # Add auxiliary constraints from transformation
    hard_constraints_clean.extend(xf.aux_constraints)
    
    # 3. Solver Setup
    ctx.add_hard_constraints_ast(hard_constraints_clean)
    ctx.add_domain_constraints(domains)
    domains_with_bools = {**domains, **bool_domains}

    # 4. Optimization
    result = ctx.find_optimum(t_total_clean, domains_with_bools, force_enumerate=force_enumerate)
    min_val, assignments = result if result is not None else (None, None)

    # 5. Analysis
    active_constraints = None
    mus = None
    if debug:
        if min_val is not None:
            active_constraints = analyze_constraints(
                hard_constraints_clean, assignments, domains_with_bools
            )
        else:
            mus = ctx.find_mus(hard_constraints_clean, domains_with_bools)

    return {
        "variant": variant,
        "index": index,
        "total": total,
        "min_val": min_val,
        "assignments": assignments,
        "unsat_core": ctx.last_unsat_core_info if debug else None,
        "active_constraints": active_constraints,
        "mus": mus,
    }


def _write_detailed_log(
    results: list[dict], output_path: Path | str, total: int, debug: bool = False
) -> None:
    with open(output_path, "w", encoding="utf-8") as log:
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            if r["min_val"] is None:
                print(f"Variant [{r['index']}/{total - 1}]: {vname}  UNSAT\n", file=log)
                if debug and r.get("mus"): print_mus(vname, r["mus"], file=log)
                elif debug and r.get("unsat_core"): print_unsat_core(vname, r["unsat_core"], context="Infeasible", file=log)
            else:
                if debug:
                    print_breakdown(r["variant"], r["assignments"], r["min_val"], r["index"], total, file=log)
                    if r.get("active_constraints"): print_active_constraints(vname, r["active_constraints"], file=log)
                else:
                    print_result_summary(vname, r["assignments"], r["min_val"], r["index"], total, file=log)
                print("-" * 72, file=log)

def run(
    input_path: Path | str,
    njobs: int = 1,
    output_path: Path | str | None = None,
    debug: bool = False,
    force_enumerate: bool = False,
    symbol_domains: dict[str, list[int]] | None = None,
    optimal_only: bool = False,
) -> dict[str, dict[str, int] | None]:
    variants = load_variants(input_path)
    total = len(variants)
    etg_domains = derive_domains_from_etg(variants)
    domains = {**etg_domains, **(symbol_domains or {})}

    print(f"Solving {total} variants with {njobs} process(es)...")

    results: list[dict] = [None] * total
    with ProcessPoolExecutor(max_workers=njobs) as pool:
        futures = {pool.submit(solve_variant, v, i, total, domains, debug, force_enumerate): i for i, v in enumerate(variants)}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            results[res["index"]] = res
            completed += 1
            vname = get_variant_name(res["variant"], res["index"])
            if res["min_val"] is None: print(f"[{completed:3d}/{total}] {vname}  UNSAT")
            else: print(f"[{completed:3d}/{total}] {vname}  T={res['min_val']:,} cycles")

    if output_path: _write_detailed_log(results, output_path, total, debug=debug)
    
    block_sizes = {get_variant_name(r["variant"], r["index"]): (dict(r["assignments"]) if r["min_val"] is not None else None) for r in results}
    feasible = [r for r in results if r["min_val"] is not None]
    if feasible:
        best = min(feasible, key=lambda r: r["min_val"])
        if optimal_only:
            for r in results:
                if r["min_val"] is not None and r["min_val"] > best["min_val"]: block_sizes[get_variant_name(r["variant"], r["index"])] = None
        print("\nGLOBAL BEST")
        print_breakdown(best["variant"], best["assignments"], best["min_val"], best["index"], total)
    return block_sizes


def main():
    parser = argparse.ArgumentParser(description="Find optimal block sizes using Python AST.")
    parser.add_argument("--input", required=True, help="Path to staged_etg_dump.json")
    parser.add_argument("--njobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output", help="Log file path")
    parser.add_argument("--debug", action="store_true", help="Enable analysis")
    parser.add_argument("--force-enumerate", action="store_true", help="Brute-force")
    parser.add_argument("--optimal-only", action="store_true", help="Output only best")
    args = parser.parse_args()
    run(input_path=args.input, njobs=args.njobs, output_path=args.output, debug=args.debug, force_enumerate=args.force_enumerate, optimal_only=args.optimal_only)

if __name__ == "__main__":
    main()
