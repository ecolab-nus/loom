"""Utilities for parsing kernel-size CLI overrides."""

from __future__ import annotations

import re


def parse_kernel_size(value: str) -> dict[str, int]:
    """Parse a kernel-size string like ``B1_H32_L512_D64`` into a dict."""
    if not value:
        raise ValueError("Kernel size string is empty.")

    out: dict[str, int] = {}
    for part in value.split("_"):
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)(\d+)", part)
        if not match:
            raise ValueError(
                f"Invalid kernel size segment: {part}. Expected <KEY><int>, e.g. B1 or TILE64."
            )
        key = match.group(1)
        if key in out:
            raise ValueError(f"Duplicate kernel size key: {key}.")
        out[key] = int(match.group(2))

    return out


def resolve_kernel_shape_args(argv: list[str]) -> tuple[dict[str, int] | None, list[str]]:
    """Extract optional kernel-size args and return filtered argv for downstream parsers."""
    kernel_shape: dict[str, int] | None = None
    remaining = [argv[0]]

    idx = 1
    while idx < len(argv):
        token = argv[idx]

        # Compatibility aliases for LoomKernel args.
        if token == "--output_path":
            remaining.append("--output-path")
            idx += 1
            continue
        if token == "--hw_spec":
            remaining.append("--hw-spec")
            idx += 1
            continue

        if token.startswith("-") and not token.startswith("--") and len(token) > 1:
            candidate = token[1:]
            try:
                parsed = parse_kernel_size(candidate)
            except ValueError:
                parsed = None
            if parsed is not None:
                if kernel_shape is not None:
                    raise ValueError("Multiple kernel size overrides provided.")
                kernel_shape = parsed
                idx += 1
                continue

        if token == "--kernel-size":
            if idx + 1 >= len(argv):
                raise ValueError("Missing value for --kernel-size.")
            if kernel_shape is not None:
                raise ValueError("Multiple kernel size overrides provided.")
            kernel_shape = parse_kernel_size(argv[idx + 1])
            idx += 2
            continue

        if token.startswith("--kernel-size="):
            if kernel_shape is not None:
                raise ValueError("Multiple kernel size overrides provided.")
            kernel_shape = parse_kernel_size(token.split("=", 1)[1])
            idx += 1
            continue

        remaining.append(token)
        idx += 1

    return kernel_shape, remaining
