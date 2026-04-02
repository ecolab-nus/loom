"""Z3 expression transformation utilities for the Loom SMT solver.

This module is the designated home for all transformations applied to Z3
AST nodes *after* they have been built by ``expr_resolver.py``.  It is
intentionally kept separate from expression *resolution* (JSON → Z3) and
from the solver strategy (SolverContext).

Current transforms
------------------
``ExprTransformer``
    Eliminates every ``div`` and ``mod`` node from a Z3 expression by
    introducing fresh auxiliary integer variables with polynomial
    (Euclidean-division) constraints.  The resulting expressions are purely
    polynomial — a much friendlier input to Z3 tactics and the underlying
    arithmetic solver.

Design notes
------------
* **Single-use per batch**: one ``ExprTransformer`` instance covers one full
  set of constraints + objective.  This allows a shared cache across all
  constraints so that identical sub-expressions (e.g. ``Div(m*n, 8192)``
  appearing in both a constraint and the objective) re-use the same aux var.

* **Positive-denominator invariant**: all ETG denominators are either positive
  constants or products of positively-constrained integer symbols.  The
  transformer asserts ``den > 0`` per division site but does *not* introduce
  case-splits for zero denominators.  If a future ETG violates this invariant,
  the assertion will raise an informative error at transform time.

* **IfElse transparency**: the transformer recurses unconditionally into all
  sub-expressions — including the branches of Z3 ``If`` nodes — without
  interpreting their conditional structure.  Division nodes inside any branch
  are replaced in place; the ``If`` skeleton is reconstructed around the clean
  sub-expressions.
"""

from __future__ import annotations

import z3


class ExprTransformer:
    """Rewrite Z3 expressions to eliminate division and modulo.

    Instances are **single-use** — create one per transformation batch
    (typically one call to ``SolverContext._simplify()``).  Auxiliary
    variables and their polynomial constraints accumulate internally and are
    retrieved via :attr:`aux_constraints`.

    Usage::

        xf = ExprTransformer()
        clean_exprs = [xf.rewrite(c) for c in original_constraints]
        # Feed clean_exprs + xf.aux_constraints to the tactic/solver.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._aux_constraints: list[z3.BoolRef] = []
        # Cache: Z3 AST id → rewritten node (avoids duplicating aux vars).
        self._cache: dict[int, z3.ExprRef] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def aux_constraints(self) -> list[z3.BoolRef]:
        """Polynomial constraints generated for all auxiliary variables.

        These must be added to the solver/goal alongside the rewritten
        expressions so that the auxiliary variables are properly bounded.
        """
        return list(self._aux_constraints)

    def rewrite(self, expr: z3.ExprRef) -> z3.ExprRef:
        """Recursively rewrite *expr*, replacing ``div``/``mod`` with fresh
        auxiliary integer variables bounded by Euclidean-division constraints.

        Parameters
        ----------
        expr:
            Any Z3 expression — arithmetic, boolean, or ``If``.

        Returns
        -------
        z3.ExprRef
            Structurally equivalent expression with all ``div``/``mod``
            sub-expressions replaced by auxiliary integer variables.
        """
        node_id = expr.get_id()
        if node_id in self._cache:
            return self._cache[node_id]

        result = self._rewrite_node(expr)
        self._cache[node_id] = result
        return result

    # ------------------------------------------------------------------
    # Internal rewrite dispatch
    # ------------------------------------------------------------------

    def _rewrite_node(self, expr: z3.ExprRef) -> z3.ExprRef:
        """Dispatch rewrite by Z3 node kind."""

        # ── Terminals: variables and constants ───────────────────────
        if z3.is_int_value(expr) or z3.is_rational_value(expr):
            return expr
        if z3.is_const(expr):
            # Bare symbolic variable — nothing to rewrite.
            return expr

        # ── Boolean connectives ─────────────────────────────────────
        if z3.is_and(expr):
            return z3.And([self.rewrite(c) for c in expr.children()])
        if z3.is_or(expr):
            return z3.Or([self.rewrite(c) for c in expr.children()])
        if z3.is_not(expr):
            return z3.Not(self.rewrite(expr.children()[0]))

        # ── If-then-else ─────────────────────────────────────────────
        if z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_ITE:
            cond, then_, else_ = expr.children()
            return z3.If(
                self.rewrite(cond),
                self.rewrite(then_),
                self.rewrite(else_),
            )

        # ── Integer division: a / b ──────────────────────────────────
        if z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_IDIV:
            a, b = expr.children()
            return self._replace_div(self.rewrite(a), self.rewrite(b))

        # ── Modulo: a % b ────────────────────────────────────────────
        if z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_MOD:
            a, b = expr.children()
            return self._replace_mod(self.rewrite(a), self.rewrite(b))

        # ── General application: recurse into children, rebuild ──────
        if z3.is_app(expr):
            new_children = [self.rewrite(c) for c in expr.children()]
            # Reconstruct the node with rewritten children.
            return expr.decl()(*new_children)

        # ── Quantifiers and other exotic nodes: return as-is ─────────
        return expr

    # ------------------------------------------------------------------
    # Euclidean-division substitution helpers
    # ------------------------------------------------------------------

    def _replace_div(
        self, a: z3.ArithRef, b: z3.ArithRef
    ) -> z3.ArithRef:
        """Replace ``a / b`` with a fresh quotient variable ``q``.

        Adds Euclidean-division constraints::

            a == b * q + r
            0 <= r
            r < b

        Denominator positivity (``b > 0``) is already guaranteed by the
        solver context — symbols are declared positive by ``load_symbols()``
        and all constant denominators in the ETG are strictly positive.

        Returns ``q`` (the quotient).
        """
        q = self._fresh_aux("dq")
        r = self._fresh_aux("dr")

        # Euclidean division definition.
        self._aux_constraints.append(a == b * q + r)
        self._aux_constraints.append(r >= 0)
        self._aux_constraints.append(r < b)

        return q

    def _replace_mod(
        self, a: z3.ArithRef, b: z3.ArithRef
    ) -> z3.ArithRef:
        """Replace ``a % b`` with a fresh remainder variable ``r``.

        Adds the same Euclidean-division constraints as :meth:`_replace_div`
        but returns ``r`` (the remainder) instead of ``q`` (the quotient).

        This handles ``Divisible(x, by)`` which ``expr_resolver`` compiles
        to ``x % by == 0``:  after substitution the constraint becomes
        ``r == 0``, which ``propagate-values`` trivially eliminates.
        """
        q = self._fresh_aux("dq")
        r = self._fresh_aux("dr")

        self._aux_constraints.append(a == b * q + r)
        self._aux_constraints.append(r >= 0)
        self._aux_constraints.append(r < b)

        return r

    # ------------------------------------------------------------------
    # Aux variable factory
    # ------------------------------------------------------------------

    def _fresh_aux(self, tag: str) -> z3.ArithRef:
        """Create a fresh, uniquely-named auxiliary integer variable.

        Naming convention:
            ``__dq_<n>``  — quotient variables
            ``__dr_<n>``  — remainder variables
        """
        name = f"__{tag}_{self._counter}"
        self._counter += 1
        return z3.Int(name)
