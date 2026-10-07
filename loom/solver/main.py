"""CPMpy-based block-size optimizer for Loom using Pure Python AST."""

import argparse
import json
import logging
import re
import sys
from bisect import bisect_left
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
    build_memory_constraints,
    Const,
    Div,
    Node,
    Switch,
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
    memory_constraints_ast = build_memory_constraints(
        variant["constraint_scope"]["metadata"]
    )
    t_total_ast = compute_total_time_ast(variant, use_common_expr=True)

    # Add constraints and solve
    ctx.add_hard_constraints(hard_constraints_ast)
    for memory, constraint in memory_constraints_ast:
        ctx.add_hard_constraints([constraint], label_prefix=f"memory[{memory}]")
    ctx.add_divisibility_constraints(
        *_trip_counts(variant["constraint_scope"]["metadata"]["iter_num"])
    )
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


def _trip_counts(iter_num: dict) -> tuple[list[dict], list[str]]:
    """Every loop trip count; each must divide exactly (no tail tiles)."""
    entries = [
        (f"iter_num.{kind}[{i}]", raw)
        for kind in ("seq_iter", "temp_iter")
        for i, raw in enumerate(iter_num[kind])
    ]
    tiled = [(raw, label) for label, raw in entries if "Const" not in raw]
    return [raw for raw, _ in tiled], [label for _, label in tiled]


def _switch_violation(node: Node, env: dict[str, int]) -> str | None:
    """Apply the solver's rule that exactly one Switch alternative holds."""
    seen: set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Switch):
            matches = sum(bool(cond.eval(env)) for cond, _ in n.cases)
            if matches != 1:
                return f"{matches} perf-model alternatives match ({str(n)[:200]})"
        for value in vars(n).values():
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                parts = item if isinstance(item, tuple) else (item,)
                stack.extend(x for x in parts if isinstance(x, Node))
    return None


def _parse_solver_time_cost(time_cost: object):
    if isinstance(time_cost, dict) and set(time_cost) == {"Expression"}:
        time_cost = time_cost["Expression"]
    return Div(parse_expr(time_cost), Const(TIME_COST_SCALE))


def prepare_manual_block_sizes(
    variants: list[dict],
    assigned_block_size: dict[str, Any],
    symbol_domains: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Expand manual assignments and retain only feasible variants."""
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
        expanded = {
            name: _copy_manual_assignment_value(assignment)
            for name in variant_names
        }
    else:
        unknown = sorted(set(assigned_block_size) - set(variant_names))
        if unknown:
            available = ", ".join(variant_names[:5])
            suffix = "..." if len(variant_names) > 5 else ""
            raise ValueError(
                "Unknown assigned_block_size candidate(s): "
                f"{', '.join(unknown)}. Available candidates include: "
                f"{available}{suffix}"
            )
        expanded = {
            name: _copy_manual_assignment_value(assignment)
            for name, assignment in normalized.items()
        }

    domains = derive_domains_from_etg(variants, symbol_domains)
    accepted: dict[str, Any] = {}
    rejections: list[str] = []
    for index, variant in enumerate(variants):
        name = variant_names[index]
        assignment = expanded.get(name)
        if assignment is None:
            continue
        valid = []
        for combo in _iter_manual_assignment_combos(assignment):
            try:
                completed = _complete_manual_assignment(variant, combo, name)
            except ValueError as exc:
                rejections.append(f"{name}: {exc}")
                continue
            reason = _manual_infeasibility(variant, completed, domains)
            if reason is None:
                valid.append(completed)
            else:
                rejections.append(f"{name}: {reason}")
        if valid:
            accepted[name] = valid if isinstance(assignment, list) else valid[0]

    for rejection in rejections:
        logging.warning("Rejected manual block-size assignment for %s", rejection)
    if not accepted:
        detail = "; ".join(rejections[:4])
        raise ValueError(
            "No feasible assigned_block_size variants remain"
            + (f": {detail}" if detail else "")
        )
    return accepted


def write_manual_breakdown_log(
    variants: list[dict],
    assigned_block_size: dict[str, Any],
    output_path: Path | str,
    symbol_domains: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Write reporter breakdowns for manual assignments without running the solver."""
    block_sizes = prepare_manual_block_sizes(
        variants, assigned_block_size, symbol_domains=symbol_domains
    )
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
    known_symbols = required_symbols | boolean_symbols
    completed: dict[str, int] = {}

    for sym, value in assignment.items():
        if sym not in known_symbols:
            raise ValueError(
                f"assigned_block_size['{variant_name}'] contains unknown "
                f"symbol: {sym}"
            )
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


def _manual_infeasibility(
    variant: dict,
    assignment: dict[str, int],
    domains: dict[str, list[int]],
    time_ast: Node | None = None,
) -> str | None:
    scope = variant["constraint_scope"]
    metadata = scope["metadata"]
    for sym, info in metadata.get("symbols", {}).items():
        value = assignment[sym]
        if sym in domains:
            if value not in domains[sym]:
                return f"{sym}={value} is outside its allowed domain"
        else:
            upper = info.get("natural_ub", 10000) if isinstance(info, dict) else 10000
            if value < 1 or value > upper:
                return f"{sym}={value} is outside [1, {upper}]"

    for index, raw in enumerate(scope.get("hard_constraints", [])):
        if not parse_constraint(raw).eval(assignment):
            return f"hard constraint {index} is false"
    for memory, constraint in build_memory_constraints(metadata):
        if not constraint.eval(assignment):
            return f"capacity of memory '{memory}' is exceeded"

    for raw, label in zip(*_trip_counts(metadata["iter_num"])):
        node = parse_expr(raw)
        if not isinstance(node, Div):
            raise ValueError(f"Expected top-level Div node in {label}")
        numerator, denominators = SolverContext._flatten_div_chain(node)
        numerator_value = numerator.eval(assignment)
        denominator_value = 1
        for denominator in denominators:
            denominator_value *= denominator.eval(assignment)
        if denominator_value <= 0 or numerator_value % denominator_value:
            return f"{label} does not divide its iteration extent"
    if time_ast is None:
        time_ast = compute_total_time_ast(variant)
    return _switch_violation(time_ast, assignment)


def sample_block_size_neighbors(
    assignment: dict[str, int],
    variant: dict,
    domains: dict[str, list[int]],
    topk_block_size: int = 1,
) -> list[dict[str, int]]:
    """Return the solved assignment plus its nearest feasible neighbors.

    Per symbol, up to ``topk_block_size // 2`` domain values on each side that
    are feasible with the other symbols fixed; combinations are re-checked.
    """
    if topk_block_size <= 0:
        raise ValueError("topk_block_size must be a positive integer")
    if topk_block_size == 1:
        return [dict(assignment)]

    radius = topk_block_size // 2
    time_ast = compute_total_time_ast(variant)

    def feasible(combo: dict[str, int]) -> bool:
        return _manual_infeasibility(variant, combo, domains, time_ast) is None

    symbol_options: list[tuple[str, list[int]]] = []
    for sym in sorted(variant["constraint_scope"]["metadata"]["symbols"]):
        if sym not in assignment or sym not in domains or sym.startswith("__"):
            continue
        domain = domains[sym]
        i = bisect_left(domain, assignment[sym])
        options = [assignment[sym]]
        for side in (reversed(domain[:i]), domain[i + 1:]):
            found = 0
            for value in side:
                if found == radius:
                    break
                if feasible({**assignment, sym: value}):
                    options.append(value)
                    found += 1
        symbol_options.append((sym, options))

    combos = [dict(assignment)]
    names = [sym for sym, _ in symbol_options]
    for values in product(*(options for _, options in symbol_options)):
        combo = {**assignment, **dict(zip(names, values))}
        if combo != assignment and feasible(combo):
            combos.append(combo)
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
    results_path: Path | str | None = None,
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
    domains = derive_domains_from_etg(variants, symbol_domains)

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
            r, domains, topk_block_size
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

    if results_path:
        grouped: dict[str, list[dict]] = {}
        for result in results:
            name = get_variant_name(result["variant"], result["index"])
            match = re.match(r"^(.*?__binding_\d+)(?:__|$)", name)
            group = match.group(1) if match else name
            grouped.setdefault(group, []).append(result)

        def summary(result: dict) -> dict:
            return {
                "variant": get_variant_name(result["variant"], result["index"]),
                "status": result["status"],
                "cost": result["min_val"],
                "assignments": result["assignments"],
            }

        groups = []
        for name, group_results in grouped.items():
            optimal = _rank_optimal_results(group_results)
            groups.append({
                "binding": name,
                "best": summary(optimal[0]) if optimal else None,
                "variants": [summary(result) for result in group_results],
            })
        report = {
            "complete": True,
            "binding_groups": groups,
            "overall_best": summary(best) if best else None,
        }
        Path(results_path).write_text(json.dumps(report, indent=2) + "\n")

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


def _materialization_assignments(
    r: dict, domains: dict[str, list[int]], topk_block_size: int
) -> Any:
    assignments = dict(r["assignments"])
    if topk_block_size == 1:
        return assignments
    return sample_block_size_neighbors(assignments, r["variant"], domains, topk_block_size)


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
