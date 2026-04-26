"""Reusable helpers for navigating Schedule JSON trees."""

from __future__ import annotations


def is_sequential_node(node) -> bool:
    """Return True when *node* is a schedule wrapper with a Sequential member."""
    return isinstance(node, dict) and "Sequential" in node


def contains_sequential(node) -> bool:
    if isinstance(node, dict):
        if is_sequential_node(node):
            return True
        return any(contains_sequential(v) for v in node.values())
    if isinstance(node, list):
        return any(contains_sequential(item) for item in node)
    return False


def is_innermost_sequential_node(node) -> bool:
    """Return True for Sequential nodes whose schedules contain no nested Sequential."""
    if not is_sequential_node(node):
        return False

    schedules = node["Sequential"].get("schedules", [])
    return not any(contains_sequential(child) for child in schedules)
