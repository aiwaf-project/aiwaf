"""
Shared non-framework utilities.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# Upstream peers allowed to supply X-Forwarded-For unless AIWAF_TRUSTED_PROXIES
# is set. A public peer is never trusted by default, so internet clients that
# reach the application directly can't choose their own address.
DEFAULT_TRUSTED_PROXIES = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
)
MAX_FORWARDED_FOR_LENGTH = 4096
MAX_FORWARDED_FOR_ENTRIES = 32


class RouteNode:
    def __init__(self, name, full_path):
        self.name = name
        self.full_path = full_path
        self.children = {}
        self.is_endpoint = False

def normalize_path(path, trailing_slash=True):
    path = str(path).strip()
    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    if trailing_slash and not path.endswith("/"):
        path = path + "/"
    return path

def build_tree(routes):
    root = RouteNode("/", "/")
    for route in routes:
        route = normalize_path(route)
        parts = [p for p in route.strip("/").split("/") if p]
        node = root
        current = ""
        for part in parts:
            current = normalize_path(f"{current}/{part}", trailing_slash=True)
            if part not in node.children:
                node.children[part] = RouteNode(part, current)
            node = node.children[part]
        node.is_endpoint = True
    return root

def sorted_children(node):
    return sorted(node.children.values(), key=lambda n: n.name)

def _normalize_ip(value) -> str:
    """Return a canonical IP string, or "" when ``value`` is not an IP literal."""
    text = str(value or "").strip()
    if text.startswith("[") and "]" in text:
        text = text[1:text.index("]")]
    elif text.count(":") == 1:
        # IPv4 with a port, as sent by some load balancers.
        text = text.split(":", 1)[0]
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return ""
    mapped = getattr(ip, "ipv4_mapped", None)
    return str(mapped or ip)


@lru_cache(maxsize=32)
def _parse_trusted_networks(entries: tuple) -> tuple:
    networks = []
    for entry in entries:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            # An unparseable entry must never widen trust.
            logger.warning("Ignoring invalid AIWAF_TRUSTED_PROXIES entry: %r", entry)
    return tuple(networks)


def trusted_proxy_networks(configured=None) -> tuple:
    """Parse a list or comma-separated string; None uses the env var, then defaults."""
    if configured is None:
        configured = os.environ.get("AIWAF_TRUSTED_PROXIES")
    if configured is None:
        configured = DEFAULT_TRUSTED_PROXIES
    if isinstance(configured, str):
        configured = configured.split(",")
    return _parse_trusted_networks(tuple(str(e).strip() for e in configured if str(e).strip()))


def _in_networks(ip_text: str, networks: tuple) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return any(ip.version == net.version and ip in net for net in networks)


def resolve_client_ip(peer, forwarded_for=None, real_ip=None, trusted_proxies=None) -> str:
    """Resolve the client address, honoring proxy headers only from a trusted peer.

    Proxies append to X-Forwarded-For, so only entries to the right are
    trustworthy: walk from the right and stop at the first address outside the
    trusted networks. Anything further left is client-supplied. A missing peer
    means a local Unix socket, which only a co-located proxy can reach.
    """
    networks = trusted_proxy_networks(trusted_proxies)
    peer_text = str(peer or "").strip()
    peer_ip = _normalize_ip(peer_text)
    fallback = peer_ip or peer_text
    if not networks:
        return fallback
    if peer_text and not (peer_ip and _in_networks(peer_ip, networks)):
        return fallback

    header = str(forwarded_for or "")
    if header and len(header) <= MAX_FORWARDED_FOR_LENGTH:
        rightmost = ""
        for part in reversed(header.split(",")[-MAX_FORWARDED_FOR_ENTRIES:]):
            candidate = _normalize_ip(part)
            if not candidate:
                # Stop rather than skip, so a malformed hop can't expose
                # client-supplied values to its left.
                break
            if not _in_networks(candidate, networks):
                return candidate
            rightmost = rightmost or candidate
        if rightmost:
            # Every hop is internal. Only the rightmost was written by our
            # proxy, so an internal client can't choose its own address.
            return rightmost
    return _normalize_ip(real_ip) or fallback


def get_ip_from_meta(meta: dict, trusted_proxies=None) -> str:
    return resolve_client_ip(
        meta.get("REMOTE_ADDR"),
        meta.get("HTTP_X_FORWARDED_FOR"),
        meta.get("HTTP_X_REAL_IP"),
        trusted_proxies,
    )


def _joined_header(headers, name: str) -> str:
    """Read every value of a repeated header so none can be hidden from the walk."""
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        return ",".join(str(value) for value in getlist(name))
    return str(headers.get(name) or "")


def get_ip_from_headers(headers: dict, remote_addr: str | None, trusted_proxies=None) -> str:
    return resolve_client_ip(
        remote_addr,
        _joined_header(headers, "X-Forwarded-For"),
        headers.get("X-Real-IP"),
        trusted_proxies,
    )


def ip_in_allowlist(ip: str, allowlist) -> bool:
    if not allowlist:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            if "/" in str(entry):
                if ip_obj in ipaddress.ip_network(entry, strict=False):
                    return True
            elif ip_obj == ipaddress.ip_address(entry):
                return True
        except ValueError:
            continue
    return False
