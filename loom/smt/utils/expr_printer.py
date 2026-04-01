"""Convert JSON Expr trees into human-readable infix strings.

This module provides a "formatting back-end" for the tagged-union Expr grammar,
parallel to expr_resolver.py (which converts to Z3 AST) and json_formatter.py
(which serialises to compact JSON).  It has no dependencies on Z3.

Operator precedence (higher = tighter binding):
    NONE < IFELSE < ADD < MUL

Parenthesisation is minimal: a child is wrapped only when its natural
precedence is strictly lower than the context in which it appears.
For division the denominator is one level tighter than the numerator so that
``a / (b * c)`` is printed correctly (without that rule, the denominator
would not get parens because it is at the same precedence level).
"""

from __future__ import annotations

from typing import Union

# Precedence levels
_PREC_NONE = 0     # top-level / comparison operands
_PREC_ADD = 1      # additive context
_PREC_MUL = 2      # multiplicative context
_PREC_DIV_RHS = 3  # denominator of a division — one tighter than MUL


def expr_to_str(expr: dict | int | float | str, prec: int = _PREC_NONE) -> str:
    """Recursively convert a JSON Expr dict into a human-readable infix string.

    Args:
        expr:  A single-key JSON dict (e.g. ``{"Mul": [...]}``) or a bare scalar.
        prec:  The precedence of the enclosing context.  Sub-expressions whose
               natural precedence is strictly lower will be wrapped in parens.

    Returns:
        A human-readable string representation.
    """
    # Bare scalar (shouldn't appear in normal ETG, but handle gracefully)
    if not isinstance(expr, dict):
        return str(expr)
    if len(expr) != 1:
        return str(expr)

    tag, payload = next(iter(expr.items()))

    if tag == "Const":
        return str(payload)

    if tag == "Sym":
        return str(payload)

    if tag == "Concrete":
        # Transparent wrapper — delegate to the inner expression
        return expr_to_str(payload, prec)

    if tag == "Mul":
        left = expr_to_str(payload[0], _PREC_MUL)
        right = expr_to_str(payload[1], _PREC_MUL)
        res = f"{left}*{right}"
        return f"({res})" if prec >= _PREC_DIV_RHS else res

    if tag == "Add":
        # Strip leading Const(0) terms that the ETG sometimes emits
        meaningful = [p for p in payload if p != {"Const": 0}]
        if not meaningful:
            return "0"
        if len(meaningful) == 1:
            return expr_to_str(meaningful[0], prec)
        parts = [expr_to_str(p, _PREC_ADD) for p in meaningful]
        res = " + ".join(parts)
        return f"({res})" if prec > _PREC_ADD else res

    if tag == "Div":
        # Denominator needs one extra precedence level so that any product or
        # division in the denominator is automatically parenthesised.
        left = expr_to_str(payload[0], _PREC_MUL)
        right = expr_to_str(payload[1], _PREC_DIV_RHS)
        res = f"{left}/{right}"
        return f"({res})" if prec >= _PREC_DIV_RHS else res

    if tag == "Min":
        parts = [expr_to_str(p, _PREC_NONE) for p in payload]
        return f"min({', '.join(parts)})"

    if tag == "IfElse":
        cond = constraint_to_str(payload["cond"])
        then_val = expr_to_str(payload["then_expr"], _PREC_NONE)
        else_val = expr_to_str(payload["else_expr"], _PREC_NONE)
        return f"if({cond}, {then_val}, {else_val})"

    # Unknown / unhandled tag
    return f"<{tag}>"


def constraint_to_str(expr: Union[dict, str]) -> str:
    """Convert a JSON constraint/boolean Expr into a human-readable string.

    Args:
        expr: A single-key JSON dict for a boolean/comparison operator, or the
              string ``"True"`` for the unconditional constraint.

    Returns:
        A human-readable string representation.
    """
    if isinstance(expr, str):
        return expr  # e.g. "True"

    if not isinstance(expr, dict) or len(expr) != 1:
        return str(expr)

    tag, payload = next(iter(expr.items()))

    if tag in ("Eq", "Ne", "Ge", "Gt", "Le", "Lt"):
        left = expr_to_str(payload[0], _PREC_NONE)
        right = expr_to_str(payload[1], _PREC_NONE)
        op = {"Eq": "==", "Ne": "!=", "Ge": ">=", "Gt": ">", "Le": "<=", "Lt": "<"}[tag]
        return f"{left} {op} {right}"

    if tag == "And":
        parts = [constraint_to_str(p) for p in payload]
        # Wrap sub-expression only if it contains lower-precedence "or"
        return " and ".join(f"({p})" if " or " in p else p for p in parts)

    if tag == "Or":
        parts = [constraint_to_str(p) for p in payload]
        # Wrap sub-expression only if it contains lower-precedence "and"
        return " or ".join(f"({p})" if " and " in p else p for p in parts)

    if tag == "Divisible":
        x = expr_to_str(payload["x"], _PREC_NONE)
        by = expr_to_str(payload["by"], _PREC_NONE)
        return f"{x} % {by} == 0"

    # Unknown / unhandled tag
    return f"<{tag}>"
