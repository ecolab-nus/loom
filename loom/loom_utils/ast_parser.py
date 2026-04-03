"""Parser for converting JSON Expr and Constraint dictionaries into Pure Python AST nodes.
"""

from __future__ import annotations
from typing import Any, Union

from .ast_core import (
    Node, Expr, Constraint, Const, Sym, Add, Mul, Div, Mod, Min, Max, IfElse,
    Eq, Ne, Ge, Gt, Le, Lt, And, Or, Divisible, Top
)


def parse_expr(expr: Any) -> Expr:
    """Recursively convert a JSON Expr dict into a Pure Python Expr AST node."""
    if isinstance(expr, (int, float)):
        return Const(int(expr))
    
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Expected a single-key dict for Expr, got: {expr!r}")

    tag, payload = next(iter(expr.items()))

    if tag == "Const":
        return Const(int(payload))
    
    if tag == "Sym":
        return Sym(payload)
    
    if tag in ("Add", "Mul", "Div", "Mod", "Min", "Max"):
        # UnaryOps not implemented in ast_core yet, handling as needed
        if tag == "Add":
            return Add([parse_expr(o) for o in payload])
        if tag == "Mul":
            return Mul(parse_expr(payload[0]), parse_expr(payload[1]))
        if tag == "Div":
            return Div(parse_expr(payload[0]), parse_expr(payload[1]))
        if tag == "Mod":
            return Mod(parse_expr(payload[0]), parse_expr(payload[1]))
        if tag == "Min":
            return Min([parse_expr(o) for o in payload])
        if tag == "Max":
            return Max([parse_expr(o) for o in payload])
    
    if tag == "IfElse":
        return IfElse(
            parse_constraint(payload["cond"]),
            parse_expr(payload["then_expr"]),
            parse_expr(payload["else_expr"])
        )
    
    if tag == "Concrete":
        return parse_expr(payload)
    
    raise ValueError(f"Unknown Expr tag: '{tag}'")


def parse_constraint(expr: Any) -> Constraint:
    """Recursively convert a JSON constraint dict into a Pure Python Constraint AST node.
    """
    if isinstance(expr, str):
        if expr == "True":
            return Top()
        raise ValueError(f"Unknown constraint string: {expr!r}")

    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Expected a single-key dict for constraint, got: {expr!r}")

    tag, payload = next(iter(expr.items()))

    if tag == "Eq":
        return Eq(parse_expr(payload[0]), parse_expr(payload[1]))
    if tag == "Ne":
        return Ne(parse_expr(payload[0]), parse_expr(payload[1]))
    if tag == "Ge":
        return Ge(parse_expr(payload[0]), parse_expr(payload[1]))
    if tag == "Gt":
        return Gt(parse_expr(payload[0]), parse_expr(payload[1]))
    if tag == "Le":
        return Le(parse_expr(payload[0]), parse_expr(payload[1]))
    if tag == "Lt":
        return Lt(parse_expr(payload[0]), parse_expr(payload[1]))
    
    if tag == "And":
        return And([parse_constraint(a) for a in payload])
    if tag == "Or":
        return Or([parse_constraint(a) for a in payload])
    
    if tag == "Divisible":
        return Divisible(parse_expr(payload["x"]), parse_expr(payload["by"]))

    raise ValueError(f"Unknown constraint tag: '{tag}'")
