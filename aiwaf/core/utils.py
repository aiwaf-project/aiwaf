"""
Shared non-framework utilities.
"""

from __future__ import annotations

import ipaddress

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

def get_ip_from_meta(meta: dict) -> str:
    xff = meta.get("HTTP_X_FORWARDED_FOR", "")
    if xff:
        return xff.split(",")[0].strip()
    return meta.get("REMOTE_ADDR", "") or ""


def get_ip_from_headers(headers: dict, remote_addr: str | None) -> str:
    xff = headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return remote_addr or ""


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
