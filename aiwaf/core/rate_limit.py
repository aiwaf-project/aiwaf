"""Shared rate-limiting helpers for framework adapters."""

from dataclasses import dataclass
from typing import Iterable, List


ALLOW = "allow"
THROTTLE = "throttle"
FLOOD_BLOCK = "flood_block"


@dataclass(frozen=True)
class RateLimitDecision:
    action: str
    count: int
    timestamps: List[float]


def normalize_rate_key_mode(mode: str) -> str:
    """Normalize configured key mode to a supported value."""
    mode_value = (mode or "").strip().lower()
    if mode_value in {"ip", "ip_only"}:
        return "ip"
    return "ip_path"


def build_rate_limit_key(prefix: str, ip: str, path: str, key_mode: str = "ip_path", app_key: str = "") -> str:
    """Build a consistent cache key for rate limiting."""
    mode = normalize_rate_key_mode(key_mode)
    ip_value = ip or "unknown"
    path_value = path or "unknown"
    app_part = f"{app_key}:" if app_key else ""
    if mode == "ip":
        return f"{prefix}:{app_part}{ip_value}"
    return f"{prefix}:{app_part}{ip_value}:{path_value}"


def evaluate_rate_limit(
    timestamps: Iterable[float],
    now: float,
    window_seconds: float,
    max_requests: int,
    flood_threshold: int,
) -> RateLimitDecision:
    """
    Evaluate request rate and return action + updated timestamp window.

    Actions:
    - allow: request is allowed
    - throttle: request exceeds soft limit (429)
    - flood_block: request exceeds hard limit (403 + blacklist)
    """
    window = max(float(window_seconds), 1.0)
    trimmed = [t for t in list(timestamps or []) if now - t < window]
    trimmed.append(now)
    count = len(trimmed)

    action = ALLOW
    if count > int(flood_threshold):
        action = FLOOD_BLOCK
    elif count > int(max_requests):
        action = THROTTLE

    return RateLimitDecision(action=action, count=count, timestamps=trimmed)
