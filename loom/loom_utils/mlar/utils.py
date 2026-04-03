"""Reusable helpers for navigating Schedule JSON trees.
"""

from __future__ import annotations


def contains_sequential(node) -> bool:
    if isinstance(node, dict):
        if "Sequential" in node:
            return True
        return any(contains_sequential(v) for v in node.values())
    if isinstance(node, list):
        return any(contains_sequential(item) for item in node)
    return False
