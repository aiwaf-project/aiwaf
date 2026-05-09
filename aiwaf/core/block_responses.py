"""Shared blocked/throttle response contract helpers."""

from typing import Dict, Optional, Tuple


def blocked_payload(message: Optional[str] = None) -> Dict[str, str]:
    payload: Dict[str, str] = {"error": "blocked"}
    if message:
        payload["message"] = str(message)
    return payload


def blocked_response(message: Optional[str] = None, status_code: int = 403) -> Tuple[Dict[str, str], int]:
    return blocked_payload(message), int(status_code or 403)


def throttle_response() -> Tuple[Dict[str, str], int]:
    return {"error": "too_many_requests"}, 429

