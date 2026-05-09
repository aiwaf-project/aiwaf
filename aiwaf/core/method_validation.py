"""Shared request-method validation policy."""

from dataclasses import dataclass
from typing import Optional

from .honeypot import should_block_get_to_post_only_endpoint


ACTION_ALLOW = "allow"
ACTION_BLOCK = "block"


@dataclass(frozen=True)
class MethodDecision:
    action: str
    reason: Optional[str] = None
    status_code: int = 405
    message: Optional[str] = None


def evaluate_method_policy(
    *,
    method: str,
    path: str,
    accepts_get: bool,
    accepts_post: bool,
    accepts_method: bool,
) -> MethodDecision:
    method_u = (method or "").upper()
    if method_u == "GET":
        if not accepts_get and should_block_get_to_post_only_endpoint(path, accepts_get=False):
            return MethodDecision(
                action=ACTION_BLOCK,
                reason=f"GET to obvious POST-only endpoint: {path}",
                message=f"GET not allowed for {path}",
            )
        return MethodDecision(action=ACTION_ALLOW)

    if method_u == "POST":
        if not accepts_post:
            return MethodDecision(
                action=ACTION_BLOCK,
                reason=f"POST to GET-only view: {path}",
                message=f"POST not allowed for {path}",
            )
        return MethodDecision(action=ACTION_ALLOW)

    if method_u in {"HEAD", "OPTIONS"}:
        return MethodDecision(action=ACTION_ALLOW)

    if not accepts_method:
        return MethodDecision(
            action=ACTION_BLOCK,
            reason=f"{method_u} to view that doesn't support it: {path}",
            message=f"{method_u} not allowed for {path}",
        )

    return MethodDecision(action=ACTION_ALLOW)
