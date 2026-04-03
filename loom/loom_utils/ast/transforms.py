"""AST transformations for the Loom dataflow pipeline.
"""

from __future__ import annotations
from typing import Any

from .core import (
    Node, Expr, Constraint, Const, Sym, Add, Mul, Div, Mod, Min, Max, IfElse,
    Comparison, Eq, Ge, Lt
)


class ASTTransformer:
    """Rewrite Pure Python AST to eliminate division and modulo.
    """

    def __init__(self, counter_start: int = 0) -> None:
        self._counter: int = counter_start
        self._aux_constraints: list[Constraint] = []
        self._aux_symbols: list[dict[str, str]] = []
        # Cache for sub-expressions
        self._cache: dict[int, Node] = {}

    @property
    def aux_constraints(self) -> list[Constraint]:
        return self._aux_constraints

    @property
    def aux_symbols(self) -> list[dict[str, str]]:
        return self._aux_symbols

    def transform(self, node: Node) -> Node:
        node_id = id(node)
        if node_id in self._cache:
            return self._cache[node_id]

        new_node = self._visit(node)
        self._cache[node_id] = new_node
        return new_node

    def _visit(self, node: Node) -> Node:
        from .core import Top, Divisible, And, Or  # Avoid circular if needed, but they are in core

        if isinstance(node, (Const, Sym, Top)):
            return node
        
        if isinstance(node, Add):
            return Add([self.transform(o) for o in node.operands])
        
        if isinstance(node, Mul):
            return Mul(self.transform(node.left), self.transform(node.right))
        
        if isinstance(node, Div):
            return self._eliminate_div(self.transform(node.left), self.transform(node.right))
        
        if isinstance(node, Mod):
            return self._eliminate_mod(self.transform(node.left), self.transform(node.right))
        
        if isinstance(node, (Min, Max)):
            cls = type(node)
            return cls([self.transform(o) for o in node.operands])
        
        if isinstance(node, IfElse):
            return IfElse(
                self.transform(node.cond),
                self.transform(node.then_expr),
                self.transform(node.else_expr)
            )

        if isinstance(node, Comparison):
            cls = type(node)
            return cls(self.transform(node.left), self.transform(node.right))
        
        if isinstance(node, (And, Or)):
            cls = type(node)
            return cls([self.transform(o) for o in node.operands])
        
        if isinstance(node, Divisible):
            return self._eliminate_divisible(self.transform(node.x), self.transform(node.by))
        
        return node

    def _eliminate_div(self, a: Expr, b: Expr) -> Expr:
        q = self._fresh_sym("dq")
        r = self._fresh_sym("dr")
        self._add_euclidean_constraints(a, b, q, r)
        return q

    def _eliminate_mod(self, a: Expr, b: Expr) -> Expr:
        q = self._fresh_sym("dq")
        r = self._fresh_sym("dr")
        self._add_euclidean_constraints(a, b, q, r)
        return r

    def _eliminate_divisible(self, x: Expr, by: Expr) -> Constraint:
        q = self._fresh_sym("dq")
        r = self._fresh_sym("dr")
        self._add_euclidean_constraints(x, by, q, r)
        return Eq(r, Const(0))

    def _add_euclidean_constraints(self, a: Expr, b: Expr, q: Sym, r: Sym) -> None:
        self._aux_constraints.append(Eq(a, Add([Mul(b, q), r])))
        self._aux_constraints.append(Ge(r, Const(0)))
        self._aux_constraints.append(Lt(r, b))

    def _fresh_sym(self, prefix: str) -> Sym:
        name = f"__{prefix}_{self._counter}"
        self._counter += 1
        self._aux_symbols.append({"name": name, "type": "int"})
        return Sym(name)
