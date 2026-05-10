"""Shared middleware planning helpers across adapters."""

from __future__ import annotations

from typing import Iterable


AUTO_SENTINELS = {"all", "auto", "aiwaf.all"}


def is_auto_selection(requested: Iterable[str] | None) -> bool:
    if requested is None:
        return False
    return any(str(item).strip().lower() in AUTO_SENTINELS for item in requested)


def should_enable_logging(access_log: str | None) -> bool:
    """Enable logger middleware when external access logs are not configured."""
    return not bool(str(access_log or "").strip())


def should_enable_geo(
    *,
    geo_enabled_flag: bool,
    static_block_countries: Iterable[str] | None,
    dynamic_block_countries: Iterable[str] | None,
) -> bool:
    if geo_enabled_flag:
        return True
    if static_block_countries and any(str(c).strip() for c in static_block_countries):
        return True
    if dynamic_block_countries and any(str(c).strip() for c in dynamic_block_countries):
        return True
    return False


def should_enable_uuid_tamper(*, has_uuid_routes: bool | None) -> bool:
    """
    Enable UUID tamper middleware only when UUID-capable routes are present.

    ``None`` means unknown and defaults to enabled for compatibility.
    """
    if has_uuid_routes is None:
        return True
    return bool(has_uuid_routes)


def plan_enabled_middlewares(
    *,
    ordered_available: list[str],
    requested: Iterable[str] | None,
    disabled: Iterable[str] | None,
    access_log: str | None,
    geo_enabled_flag: bool,
    static_block_countries: Iterable[str] | None,
    dynamic_block_countries: Iterable[str] | None,
    has_uuid_routes: bool | None = None,
) -> set[str]:
    """
    Plan effective middleware set.

    - Default behavior (requested is None): keep all enabled for compatibility.
    - Auto behavior (requested contains "all"/"auto"): enable canonical set but
      gate logging/geo by runtime signals.
    - Explicit list: enable only listed names.
    """
    disabled_set = {str(x) for x in (disabled or [])}
    all_set = set(ordered_available)

    if requested is None:
        enabled = set(all_set)
    elif is_auto_selection(requested):
        enabled = set(all_set)
        if not should_enable_logging(access_log):
            enabled.discard("logging")
            enabled.discard("logging_middleware")
        if "geo_block" in enabled and not should_enable_geo(
            geo_enabled_flag=geo_enabled_flag,
            static_block_countries=static_block_countries,
            dynamic_block_countries=dynamic_block_countries,
        ):
            enabled.discard("geo_block")
        if "uuid_tamper" in enabled and not should_enable_uuid_tamper(has_uuid_routes=has_uuid_routes):
            enabled.discard("uuid_tamper")
    else:
        enabled = {name for name in requested if name in all_set}

    enabled -= {name for name in disabled_set if name in all_set}
    return enabled
