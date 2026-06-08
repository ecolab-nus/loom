"""CPMpy-based block-size optimizer for Loom using Pure Python AST."""

import argparse
import sys
from itertools import product
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
    t_total_ast = compute_total_time_ast(variant, use_common_expr=True)

    # Add constraints and solve
    ctx.add_hard_constraints(hard_constraints_ast)
    ctx.add_hard_constraints(memory_constraints_ast, label_prefix="memory_l1")
    seq_iter_exprs, seq_iter_labels, temp_iter_exprs, temp_iter_labels = (
        _collect_iter_num_constraints(
            variant["constraint_scope"]["metadata"]["iter_num"]
        )
    )
    ctx.add_trip_count_constraints(
        seq_iter_exprs,
        labels=seq_iter_labels,
        require_divisibility=True,
    )
    ctx.add_trip_count_constraints(temp_iter_exprs, labels=temp_iter_labels)
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


def _collect_iter_num_constraints(
    iter_num: dict,
) -> tuple[list[dict], list[str], list[dict], list[str]]:
    """Collect solver feasibility expressions from constraint metadata."""
    seq_iter, _ = _unpack_iter_num(iter_num["seq_iter"], "iter_num.seq_iter")
    seq_exprs = [seq_iter]
    seq_labels = ["iter_num.seq_iter"]
    temp_exprs = []
    temp_labels = []

    for i, raw_temp_iter in enumerate(iter_num.get("temp_iter", [])):
        label = f"iter_num.temp_iter[{i}]"
        temp_iter, _ = _unpack_iter_num(raw_temp_iter, label)
        if _contains_symbol(temp_iter, "tile_b"):
            seq_exprs.append(temp_iter)
            seq_labels.append(label)
        else:
            temp_exprs.append(temp_iter)
            temp_labels.append(label)

    return seq_exprs, seq_labels, temp_exprs, temp_labels


def _unpack_iter_num(raw: object, label: str) -> tuple[dict, bool]:
    """Decode an [expression, asure_divisible] iteration metadata pair."""
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not isinstance(raw[0], dict)
        or not isinstance(raw[1], bool)
    ):
        raise ValueError(f"{label} must be [expression, bool]")
    return raw[0], raw[1]


def _contains_symbol(expr: object, symbol: str) -> bool:
    """Return whether a serialized expression contains the named Sym node."""
    if isinstance(expr, dict):
        if expr == {"Sym": symbol}:
            return True
        return any(_contains_symbol(value, symbol) for value in expr.values())
    if isinstance(expr, list):
        return any(_contains_symbol(value, symbol) for value in expr)
    return False


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


def sample_block_size_neighbors(
    assignment: dict[str, int],
    metadata_symbols: dict[str, Any],
    topk_block_size: int = 1,
) -> list[dict[str, int]]:
    """Return the solved assignment plus local 32-step block-size neighbors."""
    if topk_block_size <= 0:
        raise ValueError("topk_block_size must be a positive integer")

    if topk_block_size == 1:
        return [dict(assignment)]

    radius = topk_block_size // 2
    sample_symbols = [
        sym for sym in sorted(metadata_symbols)
        if sym in assignment and not sym.startswith("__")
    ]
    fixed = {
        sym: value for sym, value in assignment.items()
        if sym not in sample_symbols
    }

    symbol_options: list[tuple[str, list[int]]] = []
    for sym in sample_symbols:
        base = assignment[sym]
        options = [base]
        for step in range(1, radius + 1):
            for candidate in (base - 32 * step, base + 32 * step):
                if candidate > 0 and candidate % 32 == 0 and candidate not in options:
                    options.append(candidate)
        symbol_options.append((sym, options))

    if not symbol_options:
        return [dict(assignment)]

    combos: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    def add_combo(values: dict[str, int]) -> None:
        key = tuple(sorted(values.items()))
        if key not in seen:
            seen.add(key)
            combos.append(values)

    add_combo(dict(assignment))
    names = [sym for sym, _ in symbol_options]
    options_product = product(*(options for _, options in symbol_options))
    for values in options_product:
        combo = dict(fixed)
        combo.update(zip(names, values))
        add_combo(combo)

    return combos


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
    topk_candidates: int | None = None,
    topk_block_size: int = 1,
    debug: bool = False,
    topk: int | None = None,
) -> dict[str, dict[str, int] | None]:
    if topk is not None:
        topk_candidates = topk
    if topk_candidates is not None and topk_candidates <= 0:
        raise ValueError("topk_candidates must be a positive integer")
    if topk_block_size <= 0:
        raise ValueError("topk_block_size must be a positive integer")

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

    # Rank solved candidates globally, then take the requested prefix directly.
    ranked_results = _rank_optimal_results(results)
    selected_results = (
        ranked_results[:topk_candidates]
        if topk_candidates is not None
        else ranked_results
    )
    selected_indices = {r["index"] for r in selected_results}
    best = ranked_results[0] if ranked_results else None

    block_sizes = {
        get_variant_name(r["variant"], r["index"]): _materialization_assignments(
            r, topk_block_size
        )
        for r in selected_results
    }

    # Overall omit summary (debug-only, original candidate order)
    if debug:
        omit_lines = []
        for r in results:
            vname = get_variant_name(r["variant"], r["index"])
            idx = r["index"] + 1
            if r["status"] != "OPTIMAL":
                omit_lines.append(f"  [{idx:3d}/{total}] {vname}  OMITTED: no feasible solution ({r['status']})")
            elif r["index"] not in selected_indices:
                omit_lines.append(f"  [{idx:3d}/{total}] {vname}  OMITTED: outside topk={topk_candidates} (T={r['min_val']:,} solver units)")
        if omit_lines:
            print("\nOMIT SUMMARY:")
            for line in omit_lines:
                print(line)

    if output_path:
        non_optimal_results = [r for r in results if r["status"] != "OPTIMAL"]
        _write_detailed_log(
            ranked_results + non_optimal_results,
            output_path,
            total,
            debug=debug,
        )

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


def _materialization_assignments(r: dict, topk_block_size: int) -> Any:
    assignments = dict(r["assignments"])
    if topk_block_size == 1:
        return assignments

    metadata_symbols = (
        r["variant"]
        .get("constraint_scope", {})
        .get("metadata", {})
        .get("symbols", {})
    )
    return sample_block_size_neighbors(assignments, metadata_symbols, topk_block_size)


def _rank_optimal_results(results: list[dict]) -> list[dict]:
    return sorted(
        (r for r in results if r["status"] == "OPTIMAL"),
        key=lambda r: (r["scaled_min_val"], r["index"]),
    )


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
