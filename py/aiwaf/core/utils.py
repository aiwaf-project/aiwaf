"""
Shared non-framework utilities.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from functools import lru_cache

from . import rust_backend

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
# Published at https://www.cloudflare.com/ips/ (retrieved 2026-11-15). Keep in step with
# CLOUDFLARE_PROXIES in js/lib/runtimeUtils.js. Opt-in only: anyone can send requests
# through Cloudflare, so these ranges don't prove a request came from your own zone.
CLOUDFLARE_PROXIES = (
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
    "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
    "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
    "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32",
    "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32",
)
# Names accepted in AIWAF_TRUSTED_PROXIES alongside addresses and ranges, for example
# "default,cloudflare" to keep the built-in ranges and add Cloudflare's.
TRUSTED_PROXY_PRESETS = {"default": DEFAULT_TRUSTED_PROXIES, "cloudflare": CLOUDFLARE_PROXIES}
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


@lru_cache(maxsize=32)
def _expand_trusted_entries(configured) -> tuple:
    if isinstance(configured, str):
        configured = configured.split(",")
    entries = []
    for entry in (str(e).strip() for e in configured):
        if entry:
            entries.extend(TRUSTED_PROXY_PRESETS.get(entry.lower(), (entry,)))
    return tuple(entries)


def _trusted_entries(configured=None) -> tuple:
    """Addresses and ranges from a list or comma-separated string, with shortcut names expanded.
    None uses the environment variable, then the defaults."""
    if configured is None:
        configured = os.environ.get("AIWAF_TRUSTED_PROXIES")
    if configured is None:
        configured = DEFAULT_TRUSTED_PROXIES
    try:
        return _expand_trusted_entries(configured if isinstance(configured, str) else tuple(configured))
    except TypeError:  # Entries that can't be cached, such as unhashable objects.
        return _expand_trusted_entries.__wrapped__(configured)


def trusted_proxy_networks(configured=None) -> tuple:
    """Parse a list or comma-separated string; None uses the env var, then defaults."""
    return _parse_trusted_networks(_trusted_entries(configured))


@lru_cache(maxsize=32)
def _native_trusted_set(entries: tuple, native_module):
    """The aiwaf-rust resolver for these entries, parsed once, or None when it isn't installed."""
    cls = getattr(native_module, "TrustedProxySet", None)
    if cls is None:
        return None
    trusted = cls(list(entries))
    for entry in trusted.invalid:
        # An unparseable entry must never widen trust.
        logger.warning("Ignoring invalid AIWAF_TRUSTED_PROXIES entry: %r", entry)
    return trusted


_untrusted_peer_warned = False


def _warn_untrusted_peer_once(peer_ip: str, header: str) -> None:
    """Point out, once per process, a proxy that may need adding to AIWAF_TRUSTED_PROXIES."""
    global _untrusted_peer_warned
    if _untrusted_peer_warned:
        return
    _untrusted_peer_warned = True
    logger.warning(
        "AIWAF is ignoring %s from untrusted peer %s and using the peer's address as the client. "
        "If %s is your proxy or CDN, add it to AIWAF_TRUSTED_PROXIES (behind Cloudflare, use "
        "AIWAF_TRUSTED_PROXIES=default,cloudflare); otherwise no action is needed. "
        "This warning is logged once.",
        header, peer_ip, peer_ip,
    )


def _in_networks(ip_text: str, networks: tuple) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return any(ip.version == net.version and ip in net for net in networks)


def resolve_client_ip(peer, forwarded_for=None, real_ip=None, trusted_proxies=None) -> str:
    """Resolve the client address, honoring proxy headers only from a trusted peer.

    Proxies append to X-Forwarded-For, so only entries to the right are
    trustworthy: walk from the right and stop at the first address outside the
    trusted networks. Anything further left is client-supplied. A missing peer
    means a local Unix socket, which only a co-located proxy can reach.

    The walk runs in aiwaf-rust (``TrustedProxySet``). The Python version below is used
    only when an older aiwaf-rust without it is installed.
    """
    entries = _trusted_entries(trusted_proxies)
    native = _native_trusted_set(entries, rust_backend.aiwaf_rust)
    if native is None:
        return _resolve_client_ip_python(peer, forwarded_for, real_ip, entries)
    ip, ignored = native.resolve(str(peer or ""), str(forwarded_for or ""), str(real_ip or ""))
    if ignored:
        _warn_untrusted_peer_once(ip, ignored)
    return ip


def _resolve_client_ip_python(peer, forwarded_for, real_ip, entries: tuple) -> str:
    networks = _parse_trusted_networks(entries)
    peer_text = str(peer or "").strip()
    peer_ip = _normalize_ip(peer_text)
    fallback = peer_ip or peer_text
    if not networks:
        return fallback
    if peer_text and not (peer_ip and _in_networks(peer_ip, networks)):
        if peer_ip and (forwarded_for or real_ip):
            _warn_untrusted_peer_once(peer_ip, "X-Forwarded-For" if forwarded_for else "X-Real-IP")
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
