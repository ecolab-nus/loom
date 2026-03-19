"""Reusable helpers for navigating Schedule JSON trees.

These utilities work on arbitrary JSON structures (dicts and lists) and are
not tied to any specific Schedule variant.
"""

from __future__ import annotations


def contains_sequential(node) -> bool:
    """Return True if *node* is, or contains anywhere in its subtree, a Sequential.

    Walks arbitrary JSON (dicts and lists) so the check is not tied to any
    specific Schedule variant structure.
    """
    if isinstance(node, dict):
        if "Sequential" in node:
            return True
        return any(contains_sequential(v) for v in node.values())
    if isinstance(node, list):
        return any(contains_sequential(item) for item in node)
    return False
