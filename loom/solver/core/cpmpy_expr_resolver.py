"""Translate Pure Python AST nodes into CPMpy expressions.

All Div nodes are interpreted as ceiling division (ceildiv).
Switch nodes are translated to indicator-variable patterns.
"""

from __future__ import annotations

from typing import Union

import cpmpy as cp

from ...loom_utils.ast import (
    Node, Expr, Constraint,
    Const, Sym, CommonExpr, Add, Mul, Div, Mod, Min, Max, IfElse, Switch,
    Comparison, Eq, Ne, Ge, Gt, Le, Lt, And, Or, Divisible, Top,
)

# CPMpy expression type (intvar, comparison result, or plain int)
CpExpr = Union[cp.expressions.core.Expression, int, bool]
_COMMON_EXPR_UB = 2**31


class ExprResolver:
    """Stateful resolver that accumulates auxiliary constraints from Switch nodes."""

    def __init__(self, symbol_map: dict[str, CpExpr]) -> None:
        self.symbol_map = symbol_map
        self.aux_constraints: list[CpExpr] = []
        self._switch_counter = 0
        self._common_expr_cache: dict[str, CpExpr] = {}

    def resolve(self, node: Node) -> CpExpr:
        if isinstance(node, Expr):
            return self._resolve_expr(node)
        if isinstance(node, Constraint):
            return self._resolve_constraint(node)
        raise ValueError(f"Unknown AST node type: {type(node)}")

    # ------------------------------------------------------------------
    # Expression resolution
    # ------------------------------------------------------------------

    def _resolve_expr(self, node: Expr) -> CpExpr:
        if isinstance(node, Const):
            return node.value

        if isinstance(node, Sym):
            if node.name not in self.symbol_map:
                raise KeyError(f"Symbol '{node.name}' not found in symbol_map")
            return self.symbol_map[node.name]

        if isinstance(node, CommonExpr):
            return self._resolve_common_expr(node)

        if isinstance(node, Add):
            if not node.operands:
                return 0
            result = self._resolve_expr(node.operands[0])
            for op in node.operands[1:]:
                result = result + self._resolve_expr(op)
            return result

        if isinstance(node, Mul):
            return self._resolve_expr(node.left) * self._resolve_expr(node.right)

        if isinstance(node, Div):
            # Ceildiv: ceil(a/b) = (a + b - 1) // b  (for a >= 0, b > 0)
            a = self._resolve_expr(node.left)
            b = self._resolve_expr(node.right)
            return (a + b - 1) // b

        if isinstance(node, Mod):
            return self._resolve_expr(node.left) % self._resolve_expr(node.right)

        if isinstance(node, Min):
            resolved = [self._resolve_expr(o) for o in node.operands]
            return cp.Minimum(resolved)

        if isinstance(node, Max):
            resolved = [self._resolve_expr(o) for o in node.operands]
            return cp.Maximum(resolved)

        if isinstance(node, IfElse):
            cond = self._resolve_constraint(node.cond)
            then_val = self._resolve_expr(node.then_expr)
            else_val = self._resolve_expr(node.else_expr)
            # Numeric ITE via linearization: cond * A + (1 - cond) * B
            return cond * then_val + (1 - cond) * else_val

        if isinstance(node, Switch):
            return self._resolve_switch(node)

        raise ValueError(f"Unknown Expr type: {type(node)}")

    def _resolve_common_expr(self, node: CommonExpr) -> CpExpr:
        if node.name in self._common_expr_cache:
            return self._common_expr_cache[node.name]

        aux = cp.intvar(0, _COMMON_EXPR_UB, name=f"_common_{node.name}")
        self._common_expr_cache[node.name] = aux
        self.aux_constraints.append(aux == self._resolve_expr(node.expr))
        return aux

    # ------------------------------------------------------------------
    # Switch → indicator variables
    # ------------------------------------------------------------------

    def _resolve_switch(self, node: Switch) -> CpExpr:
        n = len(node.cases)
        tag = self._switch_counter
        self._switch_counter += 1

        indicators = [cp.boolvar(name=f"_sw{tag}_{i}") for i in range(n)]

        # Exactly one indicator active
        self.aux_constraints.append(sum(indicators) == 1)

        # Bidirectional linking: indicator ↔ condition
        for i, (cond_ast, _) in enumerate(node.cases):
            cond_cp = self._resolve_constraint(cond_ast)
            self.aux_constraints.append(indicators[i].implies(cond_cp))
            self.aux_constraints.append(cond_cp.implies(indicators[i]))

        # Weighted sum of costs
        terms = []
        for i, (_, cost_ast) in enumerate(node.cases):
            cost_cp = self._resolve_expr(cost_ast)
            terms.append(indicators[i] * cost_cp)

        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    # ------------------------------------------------------------------
    # Constraint resolution
    # ------------------------------------------------------------------

    def _resolve_constraint(self, node: Constraint) -> CpExpr:
        if isinstance(node, Top):
            return True

        if isinstance(node, Eq):
            return self._resolve_expr(node.left) == self._resolve_expr(node.right)

        if isinstance(node, Ne):
            return self._resolve_expr(node.left) != self._resolve_expr(node.right)

        if isinstance(node, Ge):
            return self._resolve_expr(node.left) >= self._resolve_expr(node.right)

        if isinstance(node, Gt):
            return self._resolve_expr(node.left) > self._resolve_expr(node.right)

        if isinstance(node, Le):
            return self._resolve_expr(node.left) <= self._resolve_expr(node.right)

        if isinstance(node, Lt):
            return self._resolve_expr(node.left) < self._resolve_expr(node.right)

        if isinstance(node, And):
            resolved = [self._resolve_constraint(o) for o in node.operands]
            result = resolved[0]
            for r in resolved[1:]:
                result = result & r
            return result

        if isinstance(node, Or):
            resolved = [self._resolve_constraint(o) for o in node.operands]
            result = resolved[0]
            for r in resolved[1:]:
                result = result | r
            return result

        if isinstance(node, Divisible):
            x = self._resolve_expr(node.x)
            by = self._resolve_expr(node.by)
            return x % by == 0

        raise ValueError(f"Unknown Constraint type: {type(node)}")
