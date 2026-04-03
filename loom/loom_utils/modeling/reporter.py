"""Console reporting utilities for Loom solver results.
"""

from __future__ import annotations
import sys
from typing import TextIO

from ..ast import parse_expr, parse_constraint
from .analysis import ConstraintAnalysis, ConstraintStatus


def print_breakdown(
    variant: dict,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
) -> None:
    """Write a hierarchical timing breakdown for the given symbol assignments."""
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    variant_name = variant.get("variant_name", "unknown")
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"Optimal T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        if not sym.startswith("__"):
            p(f"  {sym} = {val}")
    p()

    iter_num = variant["constraint_scope"]["metadata"]["iter_num"]
    seq_val = parse_expr(iter_num["seq_iter"]).eval(assignments)
    temp_vals = [parse_expr(t).eval(assignments) for t in iter_num["temp_iter"]]
    temp_product = 1
    for v in temp_vals:
        temp_product *= v

    temp_str = " × ".join(str(v) for v in temp_vals)
    p(f"Iteration factors:  seq_iter={seq_val},  temp_iter=[{temp_str}]  (product={temp_product})")
    p()

    scope_totals: list[tuple[str, int]] = []

    for scope_key in ("compute_scope", "memory_scope"):
        scope = variant[scope_key]
        scope_name = scope.get("scope_name", scope_key)
        p(f"  [{scope_name}]")

        scope_total = 0
        for stage in scope["stages"]:
            p(f"    Stage {stage.get('stage_id', '?')}:")

            parallel = stage["Parallel"]
            stage_max_cost = 0

            for i, parallel_item in enumerate(parallel):
                seq = parallel_item["Sequential"]
                scenarios = seq["scenarios"]

                matched_idx = None
                matched_cost = None
                for si, scenario in enumerate(scenarios):
                    cond_ast = parse_constraint(scenario["constraints"])
                    if cond_ast.eval(assignments):
                        matched_idx = si
                        matched_cost = parse_expr(scenario["time_cost"]).eval(assignments)
                        break

                if matched_cost is not None:
                    label = f"scenario[{matched_idx}]"
                    if len(parallel) > 1:
                        label = f"Parallel[{i}] {label}"
                    p(f"      {label:<20s}  {matched_cost:>10,} cycles")
                    
                    if matched_cost > stage_max_cost:
                        stage_max_cost = matched_cost
                else:
                    label = "(no scenario matched)"
                    if len(parallel) > 1:
                        label = f"Parallel[{i}] {label}"
                    p(f"      {label}")

            if len(parallel) > 1:
                p(f"      {'→ stage max':20s}  {stage_max_cost:>10,} cycles")
            
            scope_total += stage_max_cost

        p(f"  {'→ scope total':16s}  {scope_total:>10,} cycles")
        p()
        scope_totals.append((scope_name, scope_total))

    comp_scopes = [v for n, v in scope_totals if "compute" in n.lower()]
    mem_scopes = [v for n, v in scope_totals if "memory" in n.lower()]
    t_comp = comp_scopes[0] if comp_scopes else 0
    t_mem = mem_scopes[0] if mem_scopes else 0
    t_stage = max(t_comp, t_mem)

    p(f"  T_comp  = {t_comp:>10,} cycles")
    p(f"  T_mem   = {t_mem:>10,} cycles")
    p(f"  T_stage = max(T_comp, T_mem) = {t_stage:,} cycles")
    p(f"  T_total = {t_stage:,} × {seq_val} × {temp_product} = {t_stage * seq_val * temp_product:,} cycles")
    p()


def print_active_constraints(
    variant_name: str,
    analyses: list[ConstraintAnalysis],
    file: TextIO = None,
) -> None:
    """Print constraint analysis at the optimum."""
    if file is None:
        file = sys.stdout

    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    p(f"[Constraint Analysis] {variant_name}")

    tight = [a for a in analyses if a.status == ConstraintStatus.TIGHT]
    walls = [a for a in analyses if a.status == ConstraintStatus.DISCRETE_WALL]
    slacked = [a for a in analyses if a.status == ConstraintStatus.ACTIVE_SLACK]
    satisfied = [a for a in analyses if a.status == ConstraintStatus.SATISFIED]
    violated = [a for a in analyses if a.status == ConstraintStatus.VIOLATED]

    if violated:
        p("  VIOLATED (should not happen at feasible optimum):")
        for a in violated: _print_constraint_line(p, a)

    if tight:
        p("  TIGHT constraints (slack=0):")
        for a in tight: _print_constraint_line(p, a)

    if walls:
        p("  DISCRETE WALLS (next step would violate):")
        for a in walls:
            _print_constraint_line(p, a)
            for s in a.symbol_steps:
                if s.next_value is not None and s.would_violate:
                    p(f"      {s.symbol}: {s.current_value} -> {s.next_value}"
                      f"  would cost +{s.step_cost} (VIOLATES)")

    if slacked:
        p("  Inequality constraints with slack:")
        for a in slacked: _print_constraint_line(p, a)

    sat_tags: dict[str, int] = {}
    for a in satisfied:
        sat_tags[a.tag] = sat_tags.get(a.tag, 0) + 1
    if sat_tags:
        parts = [f"{count} {tag}" for tag, count in sorted(sat_tags.items())]
        p(f"  Other satisfied: {', '.join(parts)}")
    p()


def _print_constraint_line(p, a: ConstraintAnalysis) -> None:
    prefix = f"    hard[{a.index}]:"
    if a.symbolic:
        p(f"{prefix} {a.symbolic}")
        p(f"    {' ' * len(f'hard[{a.index}]:')}  → {a.description}")
    else:
        p(f"{prefix} {a.description}")


def print_unsat_core(
    variant_name: str,
    unsat_core_info: list[tuple[str, str]],
    context: str = "",
    file: TextIO = None,
) -> None:
    if file is None: file = sys.stdout
    def p(*args, **kwargs): print(*args, **kwargs, file=file)
    p(f"[UNSAT Core] {variant_name}" + (f" ({context})" if context else ""))
    for name, expr in unsat_core_info:
        p(f"  {name}: {expr}")
    p()


def print_mus(
    variant_name: str,
    mus: list[tuple[int, str, str]],
    file: TextIO = None,
) -> None:
    if file is None: file = sys.stdout
    def p(*args, **kwargs): print(*args, **kwargs, file=file)
    p(f"[MUS] {variant_name}  ({len(mus)} constraints)")
    for idx, label, expr_str in mus:
        p(f"  [{idx}] {label}: {expr_str}")
    p()


def print_result_summary(
    variant_name: str,
    assignments: dict[str, int],
    min_val: int,
    index: int,
    total: int,
    file: TextIO = None,
) -> None:
    if file is None: file = sys.stdout
    def p(*args, **kwargs): print(*args, **kwargs, file=file)
    p(f"Variant [{index}/{total - 1}]: {variant_name}")
    p(f"  T_total: {min_val:,} cycles")
    for sym, val in sorted(assignments.items()):
        if not sym.startswith("__"):
            p(f"  {sym} = {val}")
    p()
