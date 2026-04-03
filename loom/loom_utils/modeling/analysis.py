"""Solver-agnostic constraint analysis for the Loom dataflow pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from ..ast import Node, Constraint, Sym, Const, Add, Mul, Div, Mod, Min, Max, IfElse, Comparison, Eq, Ne, Ge, Gt, Le, Lt, And, Or, Divisible, Top


class ConstraintStatus(Enum):
    TIGHT = "tight"
    DISCRETE_WALL = "wall"
    ACTIVE_SLACK = "slack"
    SATISFIED = "satisfied"
    VIOLATED = "violated"


@dataclass
class SymbolStepInfo:
    symbol: str
    current_value: int
    next_value: Optional[int]
    step_cost: Optional[int]
    would_violate: bool


@dataclass
class ConstraintAnalysis:
    index: int
    constraint_ast: Constraint
    tag: str
    status: ConstraintStatus
    description: str
    symbolic: str = ""
    slack: Optional[int] = None
    lhs_value: Optional[int] = None
    rhs_value: Optional[int] = None
    symbol_steps: list[SymbolStepInfo] = field(default_factory=list)
    sub_analyses: list['ConstraintAnalysis'] = field(default_factory=list)


def analyze_constraints(
    hard_constraints_ast: list[Constraint],
    assignments: dict[str, int],
    domains: dict[str, list[int]],
) -> list[ConstraintAnalysis]:
    """Analyze all hard constraints at the given assignments."""
    results: list[ConstraintAnalysis] = []
    sorted_domains = {name: sorted(vals) for name, vals in domains.items()}
    
    for i, c in enumerate(hard_constraints_ast):
        results.append(_analyze_single_constraint(i, c, assignments, sorted_domains))
        
    return results


def _analyze_single_constraint(
    index: int,
    constraint: Constraint,
    assignments: dict[str, int],
    sorted_domains: dict[str, list[int]],
) -> ConstraintAnalysis:
    if isinstance(constraint, (Ge, Gt, Le, Lt)):
        return _analyze_inequality(index, constraint, assignments, sorted_domains)
    
    if isinstance(constraint, Divisible):
        return _analyze_divisible(index, constraint, assignments)
    
    if isinstance(constraint, (Eq, Ne)):
        return _analyze_equality(index, constraint, assignments)
    
    if isinstance(constraint, And):
        sub = [_analyze_single_constraint(index, sc, assignments, sorted_domains) for sc in constraint.operands]
        has_wall = any(s.status == ConstraintStatus.DISCRETE_WALL for s in sub)
        has_tight = any(s.status == ConstraintStatus.TIGHT for s in sub)
        status = (
            ConstraintStatus.DISCRETE_WALL if has_wall
            else ConstraintStatus.TIGHT if has_tight
            else ConstraintStatus.SATISFIED
        )
        return ConstraintAnalysis(
            index=index, constraint_ast=constraint, tag="And",
            status=status,
            description=f"And: {len(constraint.operands)} sub-constraints",
            sub_analyses=sub,
        )

    return ConstraintAnalysis(
        index=index, constraint_ast=constraint, tag=type(constraint).__name__,
        status=ConstraintStatus.SATISFIED,
        description=f"{type(constraint).__name__}: (unhandled constraint type)",
    )


def _analyze_inequality(
    index: int,
    constraint: Union[Ge, Gt, Le, Lt],
    assignments: dict[str, int],
    sorted_domains: dict[str, list[int]],
) -> ConstraintAnalysis:
    lhs_val = constraint.left.eval(assignments)
    rhs_val = constraint.right.eval(assignments)
    
    tag = type(constraint).__name__
    if tag in ("Le", "Lt"):
        slack = rhs_val - lhs_val
    else:
        slack = lhs_val - rhs_val
        
    symbols = _collect_symbols(constraint)
    symbol_steps: list[SymbolStepInfo] = []
    for sym_name in sorted(symbols):
        if sym_name not in sorted_domains or sym_name not in assignments:
            continue
        dom = sorted_domains[sym_name]
        curr_val = assignments[sym_name]
        try:
            curr_idx = dom.index(curr_val)
        except ValueError:
            continue
        next_val = dom[curr_idx + 1] if curr_idx + 1 < len(dom) else None
        
        step_cost = None
        would_violate = False
        if next_val is not None:
            modified_env = {**assignments, sym_name: next_val}
            new_lhs = constraint.left.eval(modified_env)
            new_rhs = constraint.right.eval(modified_env)
            if tag in ("Le", "Lt"):
                new_slack = new_rhs - new_lhs
                step_cost = slack - new_slack
            else:
                new_slack = new_lhs - new_rhs
                step_cost = slack - new_slack
            
            if tag == "Le": would_violate = new_lhs > new_rhs
            elif tag == "Lt": would_violate = new_lhs >= new_rhs
            elif tag == "Ge": would_violate = new_lhs < new_rhs
            else: would_violate = new_lhs <= new_rhs
            
        symbol_steps.append(SymbolStepInfo(
            symbol=sym_name, current_value=curr_val,
            next_value=next_val, step_cost=step_cost,
            would_violate=would_violate,
        ))

    steppable = [s for s in symbol_steps if s.next_value is not None]
    if slack == 0:
        status = ConstraintStatus.TIGHT
    elif slack < 0:
        status = ConstraintStatus.VIOLATED
    elif steppable and all(s.would_violate for s in steppable):
        status = ConstraintStatus.DISCRETE_WALL
    else:
        status = ConstraintStatus.ACTIVE_SLACK
        
    op = "<=" if tag in ("Le", "Lt") else ">="
    desc = f"{tag}: {lhs_val} {op} {rhs_val} (slack={slack})"
    if status == ConstraintStatus.DISCRETE_WALL:
        blocked = [s.symbol for s in steppable if s.would_violate]
        desc += f" [DISCRETE WALL -- blocks: {', '.join(blocked)}]"

    return ConstraintAnalysis(
        index=index, constraint_ast=constraint, tag=tag,
        status=status, description=desc, symbolic=str(constraint),
        slack=slack, lhs_value=lhs_val, rhs_value=rhs_val,
        symbol_steps=symbol_steps,
    )


def _analyze_divisible(index: int, constraint: Divisible, assignments: dict[str, int]) -> ConstraintAnalysis:
    x_val = constraint.x.eval(assignments)
    by_val = constraint.by.eval(assignments)
    satisfied = (by_val != 0) and (x_val % by_val == 0)
    status = ConstraintStatus.SATISFIED if satisfied else ConstraintStatus.VIOLATED
    return ConstraintAnalysis(
        index=index, constraint_ast=constraint, tag="Divisible",
        status=status,
        description=f"Divisible: {x_val} % {by_val} == 0 ({'yes' if satisfied else 'NO'})",
        symbolic=str(constraint),
        lhs_value=x_val, rhs_value=by_val,
    )


def _analyze_equality(index: int, constraint: Union[Eq, Ne], assignments: dict[str, int]) -> ConstraintAnalysis:
    lhs_val = constraint.left.eval(assignments)
    rhs_val = constraint.right.eval(assignments)
    tag = type(constraint).__name__
    if tag == "Eq":
        satisfied = lhs_val == rhs_val
        desc = f"Eq: {lhs_val} == {rhs_val}"
    else:
        satisfied = lhs_val != rhs_val
        desc = f"Ne: {lhs_val} != {rhs_val}"
    status = ConstraintStatus.SATISFIED if satisfied else ConstraintStatus.VIOLATED
    return ConstraintAnalysis(
        index=index, constraint_ast=constraint, tag=tag,
        status=status, description=desc, symbolic=str(constraint),
        lhs_value=lhs_val, rhs_value=rhs_val,
    )


def _collect_symbols(node: Node) -> set[str]:
    """Recursively collect all symbol names from an AST node."""
    if isinstance(node, Sym):
        return {node.name}
    if isinstance(node, (Const, Top)):
        return set()
    
    symbols = set()
    if isinstance(node, (Add, Min, Max, And, Or)):
        for o in node.operands:
            symbols |= _collect_symbols(o)
    elif isinstance(node, (Mul, Div, Mod, Comparison, Divisible)):
        if hasattr(node, "left"): symbols |= _collect_symbols(node.left)
        if hasattr(node, "right"): symbols |= _collect_symbols(node.right)
        if hasattr(node, "x"): symbols |= _collect_symbols(node.x)
        if hasattr(node, "by"): symbols |= _collect_symbols(node.by)
    elif isinstance(node, IfElse):
        symbols |= _collect_symbols(node.cond)
        symbols |= _collect_symbols(node.then_expr)
        symbols |= _collect_symbols(node.else_expr)
        
    return symbols
