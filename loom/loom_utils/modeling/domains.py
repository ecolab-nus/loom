"""General helper utilities for block-size domain construction."""

import math


def multiples_of_32_range(lo: int, hi: int) -> list[int]:
    """Return all multiples of 32 in the closed interval [lo, hi]."""
    start = math.ceil(lo / 32) * 32
    return list(range(start, hi + 1, 32))


def aligned_range(ub: int, alignment: int = 1) -> list[int]:
    """Return positive integers <= ub that are multiples of alignment."""
    alignment = max(1, alignment)
    return list(range(alignment, ub + 1, alignment))


def parse_user_block_sizes(block_sizes: dict[str, dict]) -> dict[str, list[int]]:
    """Convert user-provided lb/ub bounds to domain lists for each symbol."""
    return {
        sym: multiples_of_32_range(bounds["lb"], bounds["ub"])
        for sym, bounds in block_sizes.items()
    }


def derive_domains_from_etg(variants: list[dict]) -> dict[str, list[int]]:
    """Derive aligned symbol domains from ETG metadata."""
    domains: dict[str, list[int]] = {}
    for variant in variants:
        for sym, info in (
            variant.get("constraint_scope", {})
            .get("metadata", {})
            .get("symbols", {})
            .items()
        ):
            if not isinstance(info, dict) or "natural_ub" not in info or sym in domains:
                continue

            domains[sym] = aligned_range(
                ub=int(info["natural_ub"]),
                alignment=int(info.get("alignment", 1)),
            )
    return domains


def get_variant_name(variant: dict, index: int) -> str:
    """Return the name of a variant, defaulting to 'variant_{index}'."""
    return variant.get("variant_name", f"variant_{index}")
