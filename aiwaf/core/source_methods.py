"""Best-effort HTTP method inference from Python view source."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, Optional, Set

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def literal_http_method(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        method = node.value.upper()
        return method if method in HTTP_METHODS else None
    return None


def literal_http_methods(node: ast.AST) -> Set[str]:
    method = literal_http_method(node)
    if method:
        return {method}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return {
            method
            for element in node.elts
            for method in [literal_http_method(element)]
            if method
        }
    return set()


def function_defs_from_ast(tree: ast.AST, *, top_level_only: bool = False) -> Dict[str, ast.AST]:
    if top_level_only and isinstance(tree, ast.Module):
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def decorator_methods(node: ast.AST) -> Optional[Set[str]]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        name_node = call.func if call is not None else decorator
        decorator_name = ""
        if isinstance(name_node, ast.Name):
            decorator_name = name_node.id
        elif isinstance(name_node, ast.Attribute):
            decorator_name = name_node.attr

        if decorator_name in {"require_GET", "get"}:
            return {"GET"}
        if decorator_name in {"require_POST", "post"}:
            return {"POST"}
        if decorator_name in {"put"}:
            return {"PUT"}
        if decorator_name in {"patch"}:
            return {"PATCH"}
        if decorator_name in {"delete"}:
            return {"DELETE"}
        if decorator_name in {"require_http_methods", "route", "api_route"} and call is not None:
            for keyword in call.keywords:
                if keyword.arg == "methods":
                    declared = literal_http_methods(keyword.value)
                    if declared:
                        return declared
            if decorator_name == "require_http_methods" and call.args:
                declared = literal_http_methods(call.args[0])
                if declared:
                    return declared
    return None


def is_name_attr(node: ast.AST, names: Set[str], attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id in names
    )


def node_uses_payload(node: ast.AST, request_names: Set[str], payload_names: Set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in payload_names:
        return True
    return (
        is_name_attr(node, request_names, "POST")
        or is_name_attr(node, request_names, "FILES")
        or is_name_attr(node, request_names, "form")
        or is_name_attr(node, request_names, "json")
    )


def aliases_from_assign(node: ast.AST, request_names: Set[str]) -> tuple[Set[str], Set[str]]:
    method_aliases: Set[str] = set()
    payload_aliases: Set[str] = set()
    if not isinstance(node, ast.Assign):
        return method_aliases, payload_aliases
    if is_name_attr(node.value, request_names, "method"):
        method_aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    if node_uses_payload(node.value, request_names, set()):
        payload_aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return method_aliases, payload_aliases


def helper_request_names(helper: ast.AST, call: ast.Call, request_names: Set[str]) -> Set[str]:
    if not isinstance(helper, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    names: Set[str] = set()
    for index, arg in enumerate(call.args):
        if isinstance(arg, ast.Name) and arg.id in request_names and index < len(helper.args.args):
            names.add(helper.args.args[index].arg)
    return names


def methods_from_function_node(
    node: ast.AST,
    function_defs: Dict[str, ast.AST],
    request_names: Optional[Set[str]] = None,
    *,
    depth: int = 0,
    seen: Optional[Set[str]] = None,
) -> Set[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    if depth > 3:
        return set()
    seen = set(seen or set())
    if node.name in seen:
        return set()
    seen.add(node.name)

    declared = decorator_methods(node)
    if declared is not None:
        return set(declared)

    methods: Set[str] = set()
    active_request_names = set(request_names or {"request"})
    if node.args.args and not request_names:
        active_request_names.add(node.args.args[0].arg)
    method_aliases: Set[str] = set()
    payload_aliases: Set[str] = set()

    for child in ast.walk(node):
        new_method_aliases, new_payload_aliases = aliases_from_assign(child, active_request_names)
        method_aliases.update(new_method_aliases)
        payload_aliases.update(new_payload_aliases)

        if isinstance(child, ast.Attribute) and node_uses_payload(child, active_request_names, payload_aliases):
            methods.add("POST")

        if isinstance(child, ast.Compare):
            left_is_method = is_name_attr(child.left, active_request_names, "method")
            left_is_alias = isinstance(child.left, ast.Name) and child.left.id in method_aliases
            for comparator in child.comparators:
                comparator_methods = literal_http_methods(comparator)
                if left_is_method or left_is_alias:
                    methods.update(comparator_methods)
                elif is_name_attr(comparator, active_request_names, "method"):
                    methods.update(literal_http_methods(child.left))
                elif isinstance(comparator, ast.Name) and comparator.id in method_aliases:
                    methods.update(literal_http_methods(child.left))

        if isinstance(child, ast.Call):
            function = child.func
            if isinstance(function, ast.Attribute) and function.attr in {"get", "pop", "getlist"}:
                if node_uses_payload(function.value, active_request_names, payload_aliases):
                    methods.add("POST")
            if isinstance(function, ast.Name):
                helper = function_defs.get(function.id)
                if helper is None:
                    continue
                names = helper_request_names(helper, child, active_request_names)
                if names:
                    methods.update(
                        methods_from_function_node(
                            helper,
                            function_defs,
                            names,
                            depth=depth + 1,
                            seen=seen,
                        )
                    )
                elif any(node_uses_payload(arg, active_request_names, payload_aliases) for arg in child.args):
                    methods.add("POST")

    return methods


def methods_from_ast(tree: ast.AST, view_name: Optional[str] = None, *, top_level_only: bool = False) -> list[str]:
    methods = {"GET"}
    function_defs = function_defs_from_ast(tree, top_level_only=top_level_only)
    if view_name and view_name in function_defs:
        methods.update(methods_from_function_node(function_defs[view_name], function_defs))
    else:
        for node in function_defs.values():
            methods.update(methods_from_function_node(node, function_defs))
            break
    return sorted(methods)


def infer_methods_from_source(callback: Any) -> list[str]:
    try:
        unwrapped = inspect.unwrap(callback)
    except Exception:
        unwrapped = callback
    view_name = getattr(unwrapped, "__name__", None)
    try:
        filename = inspect.getsourcefile(unwrapped)
        if filename:
            with open(filename, "r", encoding="utf-8") as source_file:
                module_tree = ast.parse(source_file.read())
            if view_name in function_defs_from_ast(module_tree, top_level_only=True):
                methods = methods_from_ast(module_tree, view_name, top_level_only=True)
                if methods:
                    return methods
    except Exception:
        pass

    try:
        source = inspect.getsource(unwrapped)
    except Exception:
        return []
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []
    return methods_from_ast(tree, view_name)
