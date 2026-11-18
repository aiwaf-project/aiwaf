"""AIWAF_PROFILE: one setting for sensible defaults.

- ``balanced`` (the default) keeps every default as it is.
- ``strict`` halves the rate limits and pins the protective switches on.
- ``monitor`` is a dry run: every check runs, and what it would block is logged instead, as
  "AIWAF monitor: would block ...". Nothing is blocked and no IP is added to the blocklist.

A setting given explicitly always wins over the profile.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

PROFILE_SETTING = "AIWAF_PROFILE"
PROFILES = ("strict", "balanced", "monitor")
DEFAULT_PROFILE = "balanced"

LOGGER = logging.getLogger("aiwaf.monitor")

# Set by the framework adapter from its settings; the environment is used until then.
_active: Optional[str] = None


def resolve(value: Any = None) -> str:
    """The profile named by a setting, or AIWAF_PROFILE in the environment, or balanced."""
    raw = value if value not in (None, "") else os.environ.get(PROFILE_SETTING, "")
    name = str(raw or DEFAULT_PROFILE).strip().lower()
    if name not in PROFILES:
        raise ValueError(f"{PROFILE_SETTING} must be one of {', '.join(PROFILES)}, not {raw!r}")
    return name


def activate(value: Any = None) -> str:
    """Records the profile for this process, as read from the framework's settings."""
    global _active
    _active = resolve(value)
    return _active


def active() -> str:
    return _active if _active is not None else resolve()


def monitoring() -> bool:
    try:
        return active() == "monitor"
    except ValueError:
        return False


def settings_for(profile: str, rate_max: int, rate_flood: int) -> Dict[str, Any]:
    """The settings a profile sets, given this framework's default rate limits."""
    if profile == "strict":
        return {
            "AIWAF_RATE_MAX": max(1, int(rate_max) // 2),
            "AIWAF_RATE_FLOOD": max(2, int(rate_flood) // 2),
            "AIWAF_UUID_SCORE_ENABLED": True,
            "AIWAF_HEADER_VALIDATION_ENABLED": True,
            "AIWAF_SQL_INJECTION_MODE": "block",
            "AIWAF_INJECTION_RISK_MODE": "block",
        }
    if profile == "monitor":
        # These two checks have their own log-only modes.
        return {"AIWAF_SQL_INJECTION_MODE": "monitor", "AIWAF_INJECTION_RISK_MODE": "log"}
    return {}


def apply(settings: Mapping[str, Any], set_if_missing, rate_max: int = 20, rate_flood: int = 40) -> str:
    """Sets the profile's values for every setting not given explicitly. Returns the profile."""
    profile = activate(settings.get(PROFILE_SETTING) if hasattr(settings, "get") else None)
    for name, value in settings_for(profile, rate_max, rate_flood).items():
        set_if_missing(name, value)
    if profile == "monitor":
        LOGGER.warning("AIWAF monitor: nothing is blocked; what would be blocked is logged here")
    return profile


def would_block(method: str, path: str, ip: str, status: Any, reason: str) -> None:
    LOGGER.warning("AIWAF monitor: would block %s %s from %s with %s (%s)", method or "-", path or "-",
                   ip or "-", status, reason or "blocked")


def skip_blocklist(ip: str, reason: Any) -> bool:
    """True in monitor mode, after logging the IP that would have been added to the blocklist."""
    if not monitoring():
        return False
    LOGGER.warning("AIWAF monitor: would add %s to the blocklist (%s)", ip, reason)
    return True
