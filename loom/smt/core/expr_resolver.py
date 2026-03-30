"""Convert JSON Expr trees (Rust-style tagged unions) into Z3 AST nodes.

This module is a pure "compiler back-end": it has no side effects, holds no
state, and does not know anything about the surrounding solver or pipeline.

Supported arithmetic tags:
    Const, Sym, Mul, Add, Div, Min, IfElse

Supported constraint/boolean tags (used in IfElse conditions and hard_constraints):
    Eq, Ge, Le, And, Divisible, "True" (string literal)

Division handling:
    To avoid precision loss from Z3 integer division truncating intermediate
    results (e.g. 4096/8192 → 0 instead of 0.5), arithmetic expressions are
    internally represented as (numerator, denominator) rational pairs.  A single
    integer division is performed only at the top level of ``resolve_expr()``,
    where the result is guaranteed to be an exact integer.
"""

import functools
from typing import Union

import z3


_ONE = z3.IntVal(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_expr(
    expr: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.ArithRef:
    """Recursively convert a JSON Expr dict into a Z3 integer expression.

    Internally uses rational-pair arithmetic to avoid intermediate truncation
    from Z3 integer division.  The final result is ``numerator / denominator``
    (a single Z3 integer division) which is exact when the mathematical model
    guarantees integer outcomes.

    Args:
        expr:       A dict with exactly one key identifying the operator.
        symbol_map: Mapping from symbol name to the corresponding z3.Int variable.

    Returns:
        A Z3 ArithRef representing the expression.

    Raises:
        ValueError: On unknown tags or malformed structure.
        KeyError:   When a Sym references a name not in symbol_map.
    """
    num, den = _resolve_rational(expr, symbol_map)
    # When denominator is trivially 1, skip the division entirely.
    if z3.is_int_value(den) and den.as_long() == 1:
        return num
    return num / den


def resolve_constraint(
    expr: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> z3.BoolRef:
    """Convert a JSON constraint/boolean Expr dict into a Z3 BoolRef.

    Args:
        expr:       A dict with exactly one key identifying the boolean operator.
        symbol_map: Mapping from symbol name to the corresponding z3.Int variable.

    Returns:
        A Z3 BoolRef representing the boolean expression.

    Raises:
        ValueError: On unknown tags or malformed structure.
    """
    if isinstance(expr, str):
        if expr == "True":
            return z3.BoolVal(True)
        raise ValueError(f"Unknown constraint string: {expr!r}")

    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Expected a single-key dict for constraint, got: {expr!r}")

    tag, payload = next(iter(expr.items()))

    if tag == "Eq":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) == resolve_expr(payload[1], symbol_map)
    
    if tag == "Ne":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) != resolve_expr(payload[1], symbol_map)

    if tag == "Ge":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) >= resolve_expr(payload[1], symbol_map)

    if tag == "Gt":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) > resolve_expr(payload[1], symbol_map)

    if tag == "Le":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) <= resolve_expr(payload[1], symbol_map)

    if tag == "Lt":
        _check_binary(tag, payload)
        return resolve_expr(payload[0], symbol_map) < resolve_expr(payload[1], symbol_map)

    if tag == "And":
        _check_variadic(tag, payload)
        args = [resolve_constraint(a, symbol_map) for a in payload]
        return z3.And(*args)

    if tag == "Divisible":
        x_z3 = resolve_expr(payload["x"], symbol_map)
        by_z3 = resolve_expr(payload["by"], symbol_map)
        return x_z3 % by_z3 == 0

    raise ValueError(f"Unknown constraint tag: '{tag}'")


# ---------------------------------------------------------------------------
# Rational-pair arithmetic (internal)
# ---------------------------------------------------------------------------

_Rational = tuple[z3.ArithRef, z3.ArithRef]  # (numerator, denominator)


def _resolve_rational(
    expr: dict,
    symbol_map: dict[str, z3.ArithRef],
) -> _Rational:
    """Recursively convert a JSON Expr into a (numerator, denominator) pair.

    No Z3 integer division is performed; divisions are represented by
    accumulating into the denominator.  This preserves precision for
    expressions like ``(M*N / 8192) * 1024`` where intermediate integer
    division would truncate.
    """
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Expected a single-key dict for Expr, got: {expr!r}")

    tag, payload = next(iter(expr.items()))

    if tag == "Const":
        return z3.IntVal(int(payload)), _ONE

    if tag == "Sym":
        if payload not in symbol_map:
            raise KeyError(f"Symbol '{payload}' not found in symbol_map")
        return symbol_map[payload], _ONE

    if tag == "Mul":
        _check_binary(tag, payload)
        a_n, a_d = _resolve_rational(payload[0], symbol_map)
        b_n, b_d = _resolve_rational(payload[1], symbol_map)
        return a_n * b_n, a_d * b_d

    if tag == "Add":
        _check_variadic(tag, payload)
        # Fold with common denominator: a/d + b/e = (a*e + b*d) / (d*e)
        result_n, result_d = _resolve_rational(payload[0], symbol_map)
        for operand in payload[1:]:
            b_n, b_d = _resolve_rational(operand, symbol_map)
            result_n = result_n * b_d + b_n * result_d
            result_d = result_d * b_d
        return result_n, result_d

    if tag == "Div":
        _check_binary(tag, payload)
        a_n, a_d = _resolve_rational(payload[0], symbol_map)
        b_n, b_d = _resolve_rational(payload[1], symbol_map)
        # (a_n/a_d) / (b_n/b_d) = (a_n * b_d) / (a_d * b_n)
        return a_n * b_d, a_d * b_n

    if tag == "Min":
        _check_variadic(tag, payload)
        # Reduce pairwise: min(a/d, b/e) with common denominator d*e.
        # Compare a*e vs b*d (safe because denominators are positive).
        result_n, result_d = _resolve_rational(payload[0], symbol_map)
        for operand in payload[1:]:
            b_n, b_d = _resolve_rational(operand, symbol_map)
            # Cross-multiply to common denominator
            lhs = result_n * b_d  # a_n * b_d
            rhs = b_n * result_d  # b_n * a_d
            common_d = result_d * b_d
            result_n = z3.If(lhs <= rhs, lhs, rhs)
            result_d = common_d
        return result_n, result_d

    if tag == "IfElse":
        cond_z3 = resolve_constraint(payload["cond"], symbol_map)
        t_n, t_d = _resolve_rational(payload["then_expr"], symbol_map)
        e_n, e_d = _resolve_rational(payload["else_expr"], symbol_map)
        # Common denominator: t_d * e_d
        common_d = t_d * e_d
        result_n = z3.If(cond_z3, t_n * e_d, e_n * t_d)
        return result_n, common_d

    if tag == "Concrete":
        return _resolve_rational(payload, symbol_map)

    raise ValueError(f"Unknown Expr tag: '{tag}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_binary(tag: str, payload) -> None:
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"'{tag}' requires exactly 2 operands, got: {payload!r}")


def _check_variadic(tag: str, payload) -> None:
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"'{tag}' requires at least 2 operands, got: {payload!r}")
