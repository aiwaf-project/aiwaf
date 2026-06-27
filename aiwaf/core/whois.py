"""WHOIS lookup helper for AIWAF."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import whois


def _resolve_domain(target: str) -> str:
    candidate = str(target).strip()
    if not candidate:
        raise ValueError("Target is required")

    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return candidate

    try:
        host, _, _ = socket.gethostbyaddr(candidate)
        return host
    except Exception as exc:
        raise ValueError(f"Cannot resolve reverse DNS for IP {candidate}") from exc


def run_whois_lookup(target: str) -> Any:
    domain = _resolve_domain(target)
    return whois.whois(domain)
