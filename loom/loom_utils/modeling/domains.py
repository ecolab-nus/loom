"""General helper utilities for block-size domain construction."""

def parse_user_block_sizes(block_sizes: dict[str, dict]) -> dict[str, list[int]]:
    """Convert user-provided lb/ub bounds to every integer in [lb, ub]."""
    return {
        sym: list(range(bounds["lb"], bounds["ub"] + 1))
        for sym, bounds in block_sizes.items()
    }


def derive_domains_from_etg(
    variants: list[dict], user_domains: dict[str, list[int]] | None = None
) -> dict[str, list[int]]:
    """Symbol domains: user candidates (default [1, natural_ub]) that are
    multiples of the ETG symbol alignment (the target's storage rule)."""
    user_domains = user_domains or {}
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
            alignment = max(1, int(info.get("alignment", 1)))
            base = user_domains.get(sym, range(1, int(info["natural_ub"]) + 1))
            domains[sym] = [v for v in base if v % alignment == 0]
            if not domains[sym]:
                raise ValueError(
                    f"no candidate for {sym} is a multiple of its alignment {alignment}"
                )
    return {**user_domains, **domains}


def get_variant_name(variant: dict, index: int) -> str:
    """Return the name of a variant, defaulting to 'variant_{index}'."""
    return variant.get("variant_name", f"variant_{index}")
