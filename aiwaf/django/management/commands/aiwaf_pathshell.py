#!/usr/bin/env python3

from django.core.management.base import BaseCommand
from django.urls import get_resolver
from django.urls.resolvers import URLPattern, URLResolver

from aiwaf.django.storage import get_path_exemption_store
from aiwaf.django.utils import get_exempt_paths


from aiwaf.core.utils import RouteNode, normalize_path, build_tree, sorted_children


def _clean_pattern(pattern):
    pattern = str(pattern).replace("^", "").replace("$", "")
    pattern = pattern.strip()
    return pattern.lstrip("/")


def _collect_routes(resolver, prefix=""):
    routes = []
    for pattern in resolver.url_patterns:
        if isinstance(pattern, URLResolver):
            nested_prefix = prefix + _clean_pattern(pattern.pattern)
            if nested_prefix and not nested_prefix.endswith("/"):
                nested_prefix += "/"
            routes.extend(_collect_routes(pattern, nested_prefix))
        elif isinstance(pattern, URLPattern):
            route = prefix + _clean_pattern(pattern.pattern)
            routes.append(normalize_path(route))
    return routes


class Command(BaseCommand):
    help = "Interactive shell to browse URL routes and add exempt paths"

    def handle(self, *args, **options):
        resolver = get_resolver()
        routes = _collect_routes(resolver)
        root = build_tree(routes)
        stack = [root]
        store = get_path_exemption_store()

        self.stdout.write("AIWAF route shell. Type 'help' for commands.")

        while True:
            current = stack[-1]
            prompt = f"aiwaf:{current.full_path}$ "
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self.stdout.write("\nExiting.")
                break

            if not raw or raw == "ls":
                children = sorted_children(current)
                if current.is_endpoint and current is not root:
                    self.stdout.write("(endpoint) .")
                if not children:
                    self.stdout.write("(empty)")
                    continue
                for idx, child in enumerate(children, 1):
                    suffix = " (endpoint)" if child.is_endpoint else ""
                    self.stdout.write(f"{idx}. {child.name}/{suffix}")
                continue

            parts = raw.split()
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in {"quit", "exit"}:
                break
            if cmd in {"help", "?"}:
                self.stdout.write("Commands: ls, cd <n|name>, up, pwd, exempt <n|name|.>, exit")
                continue
            if cmd in {"up", ".."}:
                if len(stack) > 1:
                    stack.pop()
                continue
            if cmd == "pwd":
                self.stdout.write(current.full_path)
                continue
            if cmd == "cd":
                if not arg:
                    self.stdout.write("Usage: cd <index|name>")
                    continue
                if arg in {"..", "/"}:
                    if len(stack) > 1:
                        stack.pop()
                    continue
                target = self._resolve_target(current, arg)
                if not target:
                    self.stdout.write(f"Unknown target: {arg}")
                    continue
                stack.append(target)
                continue
            if cmd == "exempt":
                if not arg:
                    self.stdout.write("Usage: exempt <index|name|.>")
                    continue
                target = current if arg == "." else self._resolve_target(current, arg)
                if not target:
                    self.stdout.write(f"Unknown target: {arg}")
                    continue
                path = target.full_path
                existing = set(get_exempt_paths())
                if path.lower() in existing:
                    self.stdout.write(f"Already exempt: {path}")
                    continue
                reason = input("Reason (optional): ").strip()
                store.add_exemption(path, reason=reason or "Manual exemption")
                self.stdout.write(self.style.SUCCESS(f" Exempted path: {path}"))
                continue

            self.stdout.write(f"Unknown command: {cmd}")

    def _resolve_target(self, current, arg):
        children = sorted_children(current)
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(children):
                return children[idx]
            return None
        for child in children:
            if child.name == arg or f"{child.name}/" == arg:
                return child
        return None
