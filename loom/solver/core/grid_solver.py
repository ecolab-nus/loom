"""Exhaustive block-size search over the full symbol-domain grid.

Block-size domains are small (a few thousand to tens of thousands of points),
so evaluating the model at every point with numpy is far cheaper than building
a CP-SAT model through CPMpy. The model semantics mirror what SolverContext and
ExprResolver hand to CP-SAT, not ``Node.eval``:

* Div is ceildiv, and Div/Mod use CP-SAT's truncating division and modulo;
  a zero divisor anywhere (even in an untaken branch) makes the point invalid.
* A CommonExpr is an auxiliary variable in [0, 2**31].
* A Switch needs exactly one true case; its value is that case's cost.
* Trip counts are flattened Div chains: ``prod(denoms) <= numerator``, plus
  ``numerator % prod(denoms) == 0`` where divisibility is required.

Anything outside this (huge grids, possible int64 overflow, unknown nodes)
raises GridUnsupported so the caller can fall back to CP-SAT.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, Optional, Union

import numpy as np

from ...loom_utils.ast import (
    Node, Expr, Constraint,
    Const, Sym, CommonExpr, Add, Mul, Div, Mod, Min, Max, IfElse, Switch,
    Eq, Ne, Ge, Gt, Le, Lt, And, Or, Divisible, Top,
)

MAX_GRID_POINTS = 1 << 17  # past ~1e5 points CP-SAT is faster
_COMMON_EXPR_UB = 2**31
_INT_LIMIT = 2**62

Value = Union[int, np.ndarray]


class GridUnsupported(Exception):
    """The model cannot be searched exhaustively; use CP-SAT instead."""


def _mag(x: Value) -> int:
    if isinstance(x, np.ndarray):
        return int(np.abs(x).max()) if x.size else 0
    return abs(int(x))


def _checked(bound: int) -> None:
    if bound >= _INT_LIMIT:
        raise GridUnsupported("possible int64 overflow")


def _trunc_div(a: Value, b: Value) -> Value:
    q = np.abs(a) // np.abs(b)
    return np.where((np.sign(a) * np.sign(b)) < 0, -q, q)


class GridModel:
    """Evaluates AST expressions and constraints at every grid point."""

    def __init__(self, symbols: dict[str, list[int]]) -> None:
        sizes = [len(v) for v in symbols.values()]
        n_points = int(np.prod(sizes, dtype=object)) if sizes else 1
        if n_points > MAX_GRID_POINTS:
            raise GridUnsupported(f"{n_points} grid points")
        axes = [np.asarray(v, dtype=np.int64) for v in symbols.values()]
        grids = np.meshgrid(*axes, indexing="ij") if axes else []
        self.names = list(symbols)
        self.columns = {n: g.reshape(-1) for n, g in zip(self.names, grids)}
        self.n_points = n_points
        self.invalid = np.zeros(n_points, dtype=bool)
        self._memo: dict[int, Value] = {}
        self._common: dict[str, Value] = {}

    # -- expressions ---------------------------------------------------------

    def expr(self, node: Expr) -> Value:
        key = id(node)
        if key not in self._memo:
            self._memo[key] = self._expr(node)
        return self._memo[key]

    def _divisor(self, b: Value) -> Value:
        zero = b == 0
        if np.any(zero):
            self.invalid |= np.broadcast_to(zero, self.invalid.shape)
            b = np.where(zero, 1, b)
        return b

    def _expr(self, node: Expr) -> Value:
        if isinstance(node, Const):
            _checked(abs(node.value))
            return int(node.value)
        if isinstance(node, Sym):
            if node.name not in self.columns:
                raise GridUnsupported(f"unknown symbol {node.name}")
            return self.columns[node.name]
        if isinstance(node, CommonExpr):
            if node.name not in self._common:
                v = self.expr(node.expr)
                self.invalid |= np.broadcast_to(
                    (v < 0) | (v > _COMMON_EXPR_UB), self.invalid.shape
                )
                self._common[node.name] = v
            return self._common[node.name]
        if isinstance(node, Add):
            vals = [self.expr(o) for o in node.operands]
            _checked(sum(_mag(v) for v in vals))
            return sum(vals[1:], vals[0]) if vals else 0
        if isinstance(node, Mul):
            a, b = self.expr(node.left), self.expr(node.right)
            _checked(_mag(a) * _mag(b))
            return a * b
        if isinstance(node, Div):
            a, b = self.expr(node.left), self.expr(node.right)
            _checked(_mag(a) + _mag(b))
            b = self._divisor(b)
            return _trunc_div(a + b - 1, b)
        if isinstance(node, Mod):
            a, b = self.expr(node.left), self.expr(node.right)
            b = self._divisor(b)
            return a - b * _trunc_div(a, b)
        if isinstance(node, Min):
            return reduce(np.minimum, [self.expr(o) for o in node.operands])
        if isinstance(node, Max):
            return reduce(np.maximum, [self.expr(o) for o in node.operands])
        if isinstance(node, IfElse):
            cond = self.constraint(node.cond)
            then_val, else_val = self.expr(node.then_expr), self.expr(node.else_expr)
            return np.where(cond, then_val, else_val)
        if isinstance(node, Switch):
            conds = [self.constraint(c) for c, _ in node.cases]
            costs = [self.expr(e) for _, e in node.cases]
            _checked(sum(_mag(c) for c in costs))
            active = np.sum([np.broadcast_to(c, self.invalid.shape) for c in conds], axis=0)
            self.invalid |= active != 1
            return sum(np.where(c, v, 0) for c, v in zip(conds, costs))
        raise GridUnsupported(f"unsupported expression {type(node).__name__}")

    # -- constraints ---------------------------------------------------------

    def constraint(self, node: Constraint) -> Union[bool, np.ndarray]:
        key = id(node)
        if key not in self._memo:
            self._memo[key] = self._constraint(node)
        return self._memo[key]

    def _constraint(self, node: Constraint) -> Union[bool, np.ndarray]:
        if isinstance(node, Top):
            return True
        if isinstance(node, (Eq, Ne, Ge, Gt, Le, Lt)):
            a, b = self.expr(node.left), self.expr(node.right)
            op = {Eq: np.equal, Ne: np.not_equal, Ge: np.greater_equal,
                  Gt: np.greater, Le: np.less_equal, Lt: np.less}[type(node)]
            return op(a, b)
        if isinstance(node, And):
            return reduce(np.logical_and, [self.constraint(o) for o in node.operands])
        if isinstance(node, Or):
            return reduce(np.logical_or, [self.constraint(o) for o in node.operands])
        if isinstance(node, Divisible):
            x, by = self.expr(node.x), self.expr(node.by)
            by = self._divisor(by)
            return (x - by * _trunc_div(x, by)) == 0
        raise GridUnsupported(f"unsupported constraint {type(node).__name__}")

    def trip_count(self, numerator: Expr, denoms: list[Expr], divisible: bool):
        num = self.expr(numerator)
        den: Value = 1
        for d in denoms:
            dv = self.expr(d)
            _checked(_mag(den) * _mag(dv))
            den = den * dv
        ok = den <= num
        if divisible:
            safe = self._divisor(den)
            ok = ok & ((num - safe * _trunc_div(num, safe)) == 0)
        return ok


def solve_on_grid(
    symbols: dict[str, list[int]],
    constraints: list[Constraint],
    trip_counts: list[tuple[Expr, list[Expr], bool]],
    objective: Expr,
) -> tuple[str, Optional[int], Optional[dict[str, int]], int]:
    """Minimize *objective* subject to all constraints over the domain grid.

    Returns (status, min_val, assignments, n_optima). Among equal optima the
    lexicographically smallest assignment (in symbol order) is returned.
    """
    symbols = _filter_unary(symbols, constraints, trip_counts)
    if any(not values for values in symbols.values()):
        return "INFEASIBLE", None, None, 0
    model = GridModel(symbols)
    feasible = np.ones(model.n_points, dtype=bool)
    for c in constraints:
        feasible &= np.broadcast_to(model.constraint(c), feasible.shape)
    for numerator, denoms, divisible in trip_counts:
        feasible &= np.broadcast_to(
            model.trip_count(numerator, denoms, divisible), feasible.shape
        )
    obj = np.broadcast_to(np.asarray(model.expr(objective), dtype=np.int64), feasible.shape)
    feasible &= ~model.invalid
    if not feasible.any():
        return "INFEASIBLE", None, None, 0

    best = int(obj[feasible].min())
    optima = np.flatnonzero(feasible & (obj == best))
    first = int(optima[0])  # meshgrid "ij" order is lexicographic in symbol order
    assignments = {n: int(model.columns[n][first]) for n in model.names}
    return "OPTIMAL", best, assignments, int(optima.size)


def _node_symbols(node: Any, out: set[str]) -> set[str]:
    if isinstance(node, Sym):
        out.add(node.name)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _node_symbols(item, out)
    elif isinstance(node, Node):
        for value in vars(node).values():
            _node_symbols(value, out)
    return out


def _filter_unary(
    symbols: dict[str, list[int]],
    constraints: list[Constraint],
    trip_counts: list[tuple[Expr, list[Expr], bool]],
) -> dict[str, list[int]]:
    """Drop values that violate a constraint mentioning only that one symbol.

    All constraints are conjunctive, so such values are infeasible at every
    grid point; removing them first keeps the full grid small.
    """
    symbols = dict(symbols)
    checks = [(c, _node_symbols(c, set()), None) for c in constraints] + [
        (t, _node_symbols(list(t[:2]), set()), t) for t in trip_counts
    ]
    for node, names, trip in checks:
        if len(names) != 1:
            continue
        (name,) = names
        if name not in symbols or not symbols[name]:
            continue
        unary = GridModel({name: symbols[name]})
        ok = unary.trip_count(*trip) if trip is not None else unary.constraint(node)
        ok = np.broadcast_to(ok, unary.invalid.shape) & ~unary.invalid
        symbols[name] = [v for v, keep in zip(symbols[name], ok) if keep]
    return symbols


def grid_symbols(
    metadata_symbols: dict[str, Any],
    booleans: list[str],
    domains: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Per-symbol value lists matching SolverContext.load_symbols/load_booleans."""
    out: dict[str, list[int]] = {}
    for name, info in metadata_symbols.items():
        typ = info["type"] if isinstance(info, dict) else info
        if typ != "int":
            raise GridUnsupported(f"symbol type {typ}")
        if name in domains:
            out[name] = sorted(set(domains[name]))
        else:
            ub = info.get("natural_ub", 10000) if isinstance(info, dict) else 10000
            out[name] = list(range(1, ub + 1))
    for name in booleans:
        out[name] = [0, 1]
    return out
