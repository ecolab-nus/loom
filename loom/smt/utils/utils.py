"""Utility helpers for the SMT solver module."""

import math


def multiples_of_32_range(lo: int, hi: int) -> list[int]:
    """Return all multiples of 32 in the closed interval [lo, hi].

    Args:
        lo: Lower bound (rounded up to the nearest multiple of 32).
        hi: Upper bound (inclusive).

    Returns:
        Sorted list of multiples of 32 from ceil32(lo) to hi inclusive.

    Example:
        >>> multiples_of_32_range(32, 256)
        [32, 64, 96, 128, 160, 192, 224, 256]
    """
    start = math.ceil(lo / 32) * 32
    return list(range(start, hi + 1, 32))



def parse_user_block_sizes(block_sizes: dict[str, dict]) -> dict[str, list[int]]:
    """Convert user-provided lb/ub bounds to domain lists for each symbol.

    Args:
        block_sizes: Mapping of symbol name → {"lb": int, "ub": int}.
                     Both lb and ub must be multiples of 32, with lb <= ub.

    Returns:
        Mapping of symbol name → sorted list of multiples of 32 in [lb, ub].

    Example:
        >>> parse_user_block_sizes({"block_size_0": {"lb": 32, "ub": 128}})
        {'block_size_0': [32, 64, 96, 128]}
    """
    return {
        sym: multiples_of_32_range(bounds["lb"], bounds["ub"])
        for sym, bounds in block_sizes.items()
    }


def derive_domains_from_etg(variants: list[dict]) -> dict[str, list[int]]:
    """Derive symbol domains from natural_ub embedded in ETG metadata.

    Reads natural_ub from each symbol's info dict and reuses
    parse_user_block_sizes to produce power-of-2 domain lists.

    Args:
        variants: Variant list loaded from the ETG JSON.

    Returns:
        Mapping of symbol name → sorted list of multiples of 32 in [32, ub].
        Symbols without a natural_ub (or with old string-format info) are omitted.
    """
    bounds: dict[str, dict] = {}
    for variant in variants:
        for sym, info in (
            variant.get("constraint_scope", {})
            .get("metadata", {})
            .get("symbols", {})
            .items()
        ):
            if isinstance(info, dict) and "natural_ub" in info and sym not in bounds:
                bounds[sym] = {"lb": 32, "ub": int(info["natural_ub"])}
    return parse_user_block_sizes(bounds)


def get_variant_name(variant: dict, index: int) -> str:
    """Return the name of a variant, defaulting to 'variant_{index}'."""
    return variant.get("variant_name", f"variant_{index}")
