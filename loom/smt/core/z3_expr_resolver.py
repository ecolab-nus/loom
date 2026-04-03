"""Translate Pure Python AST nodes into Z3 expression nodes.

This module is a pure translator bridge between the solver-agnostic AST 
and the Z3 backend. Like the original expr_resolver, it uses rational-pair 
arithmetic to preserve precision until the top-level integer division.
"""

from __future__ import annotations
import z3
from typing import Union, Tuple

from ...loom_utils.ast_core import (
    Node, Expr, Constraint, Const, Sym, Add, Mul, Div, Mod, Min, Max, IfElse,
    Comparison, Eq, Ne, Ge, Gt, Le, Lt, And, Or, Divisible, Top
)

_ONE = z3.IntVal(1)
_Rational = Tuple[z3.ArithRef, z3.ArithRef]  # (numerator, denominator)


def resolve_ast(
    node: Node,
    symbol_map: dict[str, z3.ArithRef]
) -> Union[z3.ArithRef, z3.BoolRef]:
    """Top-level entry for translating any AST node to its Z3 equivalent."""
    if isinstance(node, Expr):
        num, den = _resolve_rational(node, symbol_map)
        if z3.is_int_value(den) and den.as_long() == 1:
            return num
        return num / den
    
    if isinstance(node, Constraint):
        return _resolve_constraint(node, symbol_map)
    
    raise ValueError(f"Unknown AST node type: {type(node)}")


def _resolve_constraint(
    node: Constraint,
    symbol_map: dict[str, z3.ArithRef]
) -> z3.BoolRef:
    """Translate Constraint AST node to Z3 BoolRef."""
    if isinstance(node, Top):
        return z3.BoolVal(True)

    if isinstance(node, Eq):
        return resolve_ast(node.left, symbol_map) == resolve_ast(node.right, symbol_map)
    
    if isinstance(node, Ne):
        return resolve_ast(node.left, symbol_map) != resolve_ast(node.right, symbol_map)
    
    if isinstance(node, Ge):
        return resolve_ast(node.left, symbol_map) >= resolve_ast(node.right, symbol_map)
    
    if isinstance(node, Gt):
        return resolve_ast(node.left, symbol_map) > resolve_ast(node.right, symbol_map)
    
    if isinstance(node, Le):
        return resolve_ast(node.left, symbol_map) <= resolve_ast(node.right, symbol_map)
    
    if isinstance(node, Lt):
        return resolve_ast(node.left, symbol_map) < resolve_ast(node.right, symbol_map)
    
    if isinstance(node, And):
        return z3.And(*[_resolve_constraint(o, symbol_map) for o in node.operands])
    
    if isinstance(node, Or):
        return z3.Or(*[_resolve_constraint(o, symbol_map) for o in node.operands])
    
    if isinstance(node, Divisible):
        x_z3 = resolve_ast(node.x, symbol_map)
        by_z3 = resolve_ast(node.by, symbol_map)
        return x_z3 % by_z3 == 0

    raise ValueError(f"Unknown Constraint type: {type(node)}")


def _resolve_rational(
    node: Expr,
    symbol_map: dict[str, z3.ArithRef]
) -> _Rational:
    """Recursively convert a Python Expr AST node into a (numerator, denominator) pair."""
    if isinstance(node, Const):
        return z3.IntVal(node.value), _ONE
    
    if isinstance(node, Sym):
        if node.name not in symbol_map:
            raise KeyError(f"Symbol '{node.name}' not found in symbol_map")
        return symbol_map[node.name], _ONE

    if isinstance(node, Mul):
        a_n, a_d = _resolve_rational(node.left, symbol_map)
        b_n, b_d = _resolve_rational(node.right, symbol_map)
        return a_n * b_n, a_d * b_d

    if isinstance(node, Add):
        if not node.operands:
            return z3.IntVal(0), _ONE
        result_n, result_d = _resolve_rational(node.operands[0], symbol_map)
        for operand in node.operands[1:]:
            b_n, b_d = _resolve_rational(operand, symbol_map)
            result_n = result_n * b_d + b_n * result_d
            result_d = result_d * b_d
        return result_n, result_d

    if isinstance(node, Div):
        a_n, a_d = _resolve_rational(node.left, symbol_map)
        b_n, b_d = _resolve_rational(node.right, symbol_map)
        return a_n * b_d, a_d * b_n

    if isinstance(node, Mod):
        # Mod cannot be represented exactly by rational numerator/denominator pairs 
        # in the same way Add/Mul can. However, Div elimination is applied 
        # _before_ this step on the Python AST, so we should never see 
        # Mod here in a final solver batch.
        # Handling as top-level division for completion if it somehow slips through.
        x_z3 = resolve_ast(node.left, symbol_map)
        by_z3 = resolve_ast(node.right, symbol_map)
        return (x_z3 % by_z3), _ONE

    if isinstance(node, Min):
        if not node.operands: raise ValueError("Min requires operands")
        res_n, res_d = _resolve_rational(node.operands[0], symbol_map)
        for oper in node.operands[1:]:
            b_n, b_d = _resolve_rational(oper, symbol_map)
            lhs = res_n * b_d
            rhs = b_n * res_d
            res_n = z3.If(lhs <= rhs, lhs, rhs)
            res_d = res_d * b_d
        return res_n, res_d

    if isinstance(node, Max):
        if not node.operands: raise ValueError("Max requires operands")
        res_n, res_d = _resolve_rational(node.operands[0], symbol_map)
        for oper in node.operands[1:]:
            b_n, b_d = _resolve_rational(oper, symbol_map)
            lhs = res_n * b_d
            rhs = b_n * res_d
            res_n = z3.If(lhs >= rhs, lhs, rhs)
            res_d = res_d * b_d
        return res_n, res_d

    if isinstance(node, IfElse):
        cond_z3 = _resolve_constraint(node.cond, symbol_map)
        t_n, t_d = _resolve_rational(node.then_expr, symbol_map)
        e_n, e_d = _resolve_rational(node.else_expr, symbol_map)
        res_n = z3.If(cond_z3, t_n * e_d, e_n * t_d)
        res_d = t_d * e_d
        return res_n, res_d

    raise ValueError(f"Unknown Expr type: {type(node)}")
