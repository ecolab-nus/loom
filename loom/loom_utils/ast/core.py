"""Pure Python AST nodes for Loom dataflow pipeline expressions and constraints.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union, Optional


# ---------------------------------------------------------------------------
# Base Classes
# ---------------------------------------------------------------------------

class Node:
    """Base class for all AST nodes."""
    def eval(self, env: dict[str, int]) -> Any:
        """Evaluate the node in a given environment mapping symbol names to values."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Return a human-readable representation."""
        raise NotImplementedError()


class Expr(Node):
    """Base class for arithmetic expression nodes."""
    def eval(self, env: dict[str, int]) -> int:
        raise NotImplementedError()


class Constraint(Node):
    """Base class for boolean constraint nodes."""
    def eval(self, env: dict[str, int]) -> bool:
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Expression Nodes
# ---------------------------------------------------------------------------

@dataclass
class Const(Expr):
    value: int
    def eval(self, env: dict[str, int]) -> int: return self.value
    def __str__(self) -> str: return str(self.value)


@dataclass
class Sym(Expr):
    name: str
    def eval(self, env: dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in environment")
        return env[self.name]
    def __str__(self) -> str: return self.name


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class BinaryOp(Expr):
    left: Expr
    right: Expr


class Add(Expr):
    def __init__(self, operands: list[Expr]):
        self.operands = operands
    def eval(self, env: dict[str, int]) -> int:
        return sum(o.eval(env) for o in self.operands)
    def __str__(self) -> str:
        return "(" + " + ".join(str(o) for o in self.operands) + ")"


@dataclass
class Mul(BinaryOp):
    def eval(self, env: dict[str, int]) -> int:
        return self.left.eval(env) * self.right.eval(env)
    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class Div(BinaryOp):
    def eval(self, env: dict[str, int]) -> int:
        left = self.left.eval(env)
        right = self.right.eval(env)
        return (left + right - 1) // right
    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


@dataclass
class Mod(BinaryOp):
    def eval(self, env: dict[str, int]) -> int:
        return self.left.eval(env) % self.right.eval(env)
    def __str__(self) -> str:
        return f"({self.left} % {self.right})"


class Min(Expr):
    def __init__(self, operands: list[Expr]):
        self.operands = operands
    def eval(self, env: dict[str, int]) -> int:
        return min(o.eval(env) for o in self.operands)
    def __str__(self) -> str:
        return f"min({', '.join(str(o) for o in self.operands)})"


class Max(Expr):
    def __init__(self, operands: list[Expr]):
        self.operands = operands
    def eval(self, env: dict[str, int]) -> int:
        return max(o.eval(env) for o in self.operands)
    def __str__(self) -> str:
        return f"max({', '.join(str(o) for o in self.operands)})"


@dataclass
class IfElse(Expr):
    cond: Constraint
    then_expr: Expr
    else_expr: Expr
    def eval(self, env: dict[str, int]) -> int:
        return self.then_expr.eval(env) if self.cond.eval(env) else self.else_expr.eval(env)
    def __str__(self) -> str:
        return f"if({self.cond}, {self.then_expr}, {self.else_expr})"


class Switch(Expr):
    """Multi-branch conditional with mutually-exclusive cases.

    Each case is a (Constraint, Expr) pair. Evaluates to the cost of the
    first matching case, or *default* if none match.
    """
    def __init__(self, cases: list[tuple[Constraint, Expr]], default: Expr):
        self.cases = cases
        self.default = default

    def eval(self, env: dict[str, int]) -> int:
        for cond, expr in self.cases:
            if cond.eval(env):
                return expr.eval(env)
        return self.default.eval(env)

    def __str__(self) -> str:
        branches = ", ".join(f"{c} => {e}" for c, e in self.cases)
        return f"switch({branches}, default={self.default})"


# ---------------------------------------------------------------------------
# Constraint Nodes
# ---------------------------------------------------------------------------

@dataclass
class Comparison(Constraint):
    left: Expr
    right: Expr


class Eq(Comparison):
    def eval(self, env: dict[str, int]) -> bool: return self.left.eval(env) == self.right.eval(env)
    def __str__(self) -> str: return f"({self.left} == {self.right})"


class Ne(Comparison):
    def eval(self, env: dict[str, int]) -> bool: return self.left.eval(env) != self.right.eval(env)
    def __str__(self) -> str: return f"({self.left} != {self.right})"


class Ge(Comparison):
    def eval(self, env: dict[str, int]) -> bool: return self.left.eval(env) >= self.right.eval(env)
    def __str__(self) -> str: return f"({self.left} >= {self.right})"


class Gt(Comparison):
    def eval(self, env: dict[str, int]) -> bool: return self.left.eval(env) > self.right.eval(env)
    def __str__(self) -> str: return f"({self.left} > {self.right})"


class Le(Comparison):
    def eval(self, env: dict[str, int]) -> bool: return self.left.eval(env) <= self.right.eval(env)
    def __str__(self) -> str: return f"({self.left} <= {self.right})"


class Lt(Comparison):
    def eval(self, env: dict[str, int]) -> bool: return self.left.eval(env) < self.right.eval(env)
    def __str__(self) -> str: return f"({self.left} < {self.right})"


@dataclass
class And(Constraint):
    operands: list[Constraint]
    def eval(self, env: dict[str, int]) -> bool: return all(o.eval(env) for o in self.operands)
    def __str__(self) -> str: return "(" + " and ".join(str(o) for o in self.operands) + ")"


@dataclass
class Or(Constraint):
    operands: list[Constraint]
    def eval(self, env: dict[str, int]) -> bool: return any(o.eval(env) for o in self.operands)
    def __str__(self) -> str: return "(" + " or ".join(str(o) for o in self.operands) + ")"


@dataclass
class Divisible(Constraint):
    x: Expr
    by: Expr
    def eval(self, env: dict[str, int]) -> bool:
        by_val = self.by.eval(env)
        if by_val == 0: return False
        return self.x.eval(env) % by_val == 0
    def __str__(self) -> str: return f"({self.x} % {self.by} == 0)"


class Top(Constraint):
    def eval(self, env: dict[str, int]) -> bool: return True
    def __str__(self) -> str: return "True"
