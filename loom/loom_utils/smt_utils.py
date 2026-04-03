"""General helper utilities for block-size domain construction."""

import math


def multiples_of_32_range(lo: int, hi: int) -> list[int]:
    """Return all multiples of 32 in the closed interval [lo, hi]."""
    start = math.ceil(lo / 32) * 32
    return list(range(start, hi + 1, 32))


def parse_user_block_sizes(block_sizes: dict[str, dict]) -> dict[str, list[int]]:
    """Convert user-provided lb/ub bounds to domain lists for each symbol."""
    return {
        sym: multiples_of_32_range(bounds["lb"], bounds["ub"])
        for sym, bounds in block_sizes.items()
    }


def derive_domains_from_etg(variants: list[dict]) -> dict[str, list[int]]:
    """Derive symbol domains from natural_ub embedded in ETG metadata."""
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
