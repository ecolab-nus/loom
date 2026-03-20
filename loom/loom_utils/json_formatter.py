"""Smart JSON formatting for symbolic expressions and structural nodes."""

import json

# Keys that indicate an expression node (should be serialized compactly).
_EXPR_KEYS = frozenset({
    "Sym", "Const", "Add", "Sub", "Mul", "Div", "Min", "Max", "Neg",
    "Ge", "Le", "Gt", "Lt", "Eq", "Ne", "And", "Or", "Not",
    "Divisible", "Concrete",
})


def _is_expr(obj) -> bool:
    """Return True if *obj* looks like a symbolic expression node."""
    if isinstance(obj, dict):
        if len(obj) == 1 and next(iter(obj)) in _EXPR_KEYS:
            return True
        # {"by": ..., "x": ...} inside Divisible is also expression-like.
        if set(obj.keys()) <= {"by", "x"}:
            return all(_is_expr(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_is_expr(item) for item in obj)
    if isinstance(obj, (int, float, str)):
        return True
    return False


def smart_json_dumps(obj, indent: int = 2) -> str:
    """Serialize *obj* to JSON with hybrid formatting.

    Structural keys (schedules, stages, Func, Sequential, …) are
    pretty-printed with *indent*.  Expression-like sub-trees (Mul, Add,
    Sym, Const, constraint nodes, …) are kept compact on a single line.
    """

    def _fmt(node, depth):
        pad = " " * (indent * depth)
        pad_inner = " " * (indent * (depth + 1))

        if not isinstance(node, (dict, list)):
            return json.dumps(node)

        # Expression sub-trees → compact.
        if _is_expr(node):
            return json.dumps(node, separators=(",", ":"))

        if isinstance(node, list):
            if not node:
                return "[]"
            items = [_fmt(item, depth + 1) for item in node]
            # If all items are compact (no newlines), try single-line.
            if all("\n" not in item for item in items):
                one_line = "[" + ", ".join(items) + "]"
                if len(one_line) + len(pad) <= 120:
                    return one_line
            inner = ",\n".join(pad_inner + item for item in items)
            return "[\n" + inner + "\n" + pad + "]"

        if isinstance(node, dict):
            if not node:
                return "{}"
            parts = []
            for k, v in node.items():
                val = _fmt(v, depth + 1)
                parts.append(f"{json.dumps(k)}: {val}")
            # Try single-line for small dicts.
            if all("\n" not in p for p in parts):
                one_line = "{" + ", ".join(parts) + "}"
                if len(one_line) + len(pad) <= 120:
                    return one_line
            inner = ",\n".join(pad_inner + p for p in parts)
            return "{\n" + inner + "\n" + pad + "}"

        return json.dumps(node)

    return _fmt(obj, 0)
