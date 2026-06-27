"""Static API endpoint detection for path manifests."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class ApiDetection:
    is_api: bool
    response_type: str
    payload_type: str
    confidence: float
    signals: List[str]
    request_body: bool = False
    form_confidence: float = 0.0
    form_signals: List[str] = field(default_factory=list)


IMPORT_SIGNALS = {
    "django.http.JsonResponse": 50,
    "rest_framework.response.Response": 50,
    "rest_framework.views.APIView": 40,
    "rest_framework.viewsets.ViewSet": 40,
    "rest_framework.viewsets.ModelViewSet": 40,
    "flask.jsonify": 50,
    "fastapi.Body": 20,
    "pydantic.BaseModel": 30,
}

FORM_IMPORT_SIGNALS = {
    "django.shortcuts.render": 40,
    "django.shortcuts.redirect": 30,
    "django.forms.Form": 30,
    "django.forms.ModelForm": 35,
    "django.contrib.auth.forms.AuthenticationForm": 35,
    "flask.render_template": 40,
    "flask.redirect": 30,
}

CALL_SIGNALS = {
    "JsonResponse": 60,
    "Response": 55,
    "jsonify": 60,
}

FORM_CALL_SIGNALS = {
    "render": 40,
    "render_template": 40,
    "redirect": 30,
}

JSON_BODY_ATTRS = {
    "body",
    "data",
    "json",
}

FORM_BODY_ATTRS = {
    "POST",
    "FILES",
    "form",
}


def _unwrap(callback: Any) -> Any:
    try:
        return inspect.unwrap(callback)
    except Exception:
        return callback


def _module_tree(callback: Any):
    try:
        filename = inspect.getsourcefile(callback)
        if not filename:
            return None
        with open(filename, "r", encoding="utf-8") as source_file:
            return ast.parse(source_file.read())
    except Exception:
        return None


def _callback_tree(callback: Any):
    try:
        source = inspect.getsource(callback)
    except Exception:
        return None
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None


def _top_level_functions(tree: ast.AST) -> Dict[str, ast.AST]:
    if not isinstance(tree, ast.Module):
        return {}
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _imports(tree: ast.AST) -> tuple[Dict[str, str], List[str], int, List[str], int]:
    imports: Dict[str, str] = {}
    signals: List[str] = []
    form_signals: List[str] = []
    score = 0
    form_score = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                local_name = alias.asname or alias.name
                imports[local_name] = full_name
                weight = IMPORT_SIGNALS.get(full_name)
                if weight:
                    score += weight
                    signals.append(full_name)
                form_weight = FORM_IMPORT_SIGNALS.get(full_name)
                if form_weight:
                    form_score += form_weight
                    form_signals.append(full_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                imports[local_name] = alias.name
    return imports, signals, score, form_signals, form_score


def _call_name(node: ast.Call) -> Optional[str]:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _annotation_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    return ""


def _is_json_body_attr(node: ast.AST, request_names: Set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in JSON_BODY_ATTRS
        and isinstance(node.value, ast.Name)
        and node.value.id in request_names
    )


def _is_form_body_attr(node: ast.AST, request_names: Set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in FORM_BODY_ATTRS
        and isinstance(node.value, ast.Name)
        and node.value.id in request_names
    )


def _literal_content_type(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and "application/json" in node.value.lower()


def _analyze_function(
    node: ast.AST,
    functions: Dict[str, ast.AST],
    imports: Dict[str, str],
    *,
    depth: int = 0,
    seen: Optional[Set[str]] = None,
) -> tuple[int, List[str], bool, int, List[str], str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0, [], False, 0, [], ""
    if depth > 3:
        return 0, [], False, 0, [], ""
    seen = set(seen or set())
    if node.name in seen:
        return 0, [], False, 0, [], ""
    seen.add(node.name)

    score = 0
    form_score = 0
    signals: List[str] = []
    form_signals: List[str] = []
    request_body = False
    payload_type = ""
    request_names = {"request"}
    if node.args.args:
        request_names.add(node.args.args[0].arg)

    if getattr(node, "returns", None) is not None:
        annotation = _annotation_name(node.returns)
        if annotation in {"dict", "list", "Dict", "List"}:
            score += 40
            signals.append(f"return_annotation:{annotation}")

    for arg in getattr(node.args, "args", []):
        annotation = getattr(arg, "annotation", None)
        annotation_name = _annotation_name(annotation) if annotation is not None else ""
        if annotation_name and annotation_name not in {"Request", "HttpRequest"}:
            score += 25
            request_body = True
            payload_type = payload_type or "json"
            signals.append(f"body_model:{annotation_name}")

    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            if isinstance(child.value, (ast.Dict, ast.List, ast.Tuple)):
                score += 50
                signals.append("return:json_literal")

        if isinstance(child, ast.Name):
            imported = imports.get(child.id)
            if imported in IMPORT_SIGNALS:
                score += IMPORT_SIGNALS[imported]
                signals.append(imported)
            if imported in FORM_IMPORT_SIGNALS:
                form_score += FORM_IMPORT_SIGNALS[imported]
                form_signals.append(imported)

        if isinstance(child, ast.Attribute) and _is_json_body_attr(child, request_names):
            score += 20
            request_body = True
            payload_type = "json"
            signals.append(f"request.{child.attr}")
        if isinstance(child, ast.Attribute) and _is_form_body_attr(child, request_names):
            form_score += 45
            request_body = True
            payload_type = "form"
            form_signals.append(f"request.{child.attr}")

        if isinstance(child, ast.Constant) and _literal_content_type(child):
            score += 30
            payload_type = "json"
            signals.append("content-type:application/json")

        if isinstance(child, ast.Call):
            name = _call_name(child)
            function = child.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr in {"get_json", "json"}
                and isinstance(function.value, ast.Name)
                and function.value.id in request_names
            ):
                score += 20
                request_body = True
                payload_type = "json"
                signals.append(f"request.{function.attr}")
            if name in CALL_SIGNALS:
                score += CALL_SIGNALS[name]
                signals.append(name)
            if name in FORM_CALL_SIGNALS:
                form_score += FORM_CALL_SIGNALS[name]
                form_signals.append(name)
            helper = functions.get(name or "")
            if helper is not None:
                (
                    helper_score,
                    helper_signals,
                    helper_request_body,
                    helper_form_score,
                    helper_form_signals,
                    helper_payload_type,
                ) = _analyze_function(
                    helper,
                    functions,
                    imports,
                    depth=depth + 1,
                    seen=seen,
                )
                score += helper_score
                form_score += helper_form_score
                request_body = request_body or helper_request_body
                signals.extend(helper_signals)
                form_signals.extend(helper_form_signals)
                payload_type = helper_payload_type or payload_type
    return score, signals, request_body, form_score, form_signals, payload_type


def detect_api_endpoint(
    callback: Any,
    *,
    framework: str,
    path: str = "",
    methods: Iterable[str] | None = None,
    route: Any = None,
) -> ApiDetection:
    score = 0
    form_score = 0
    signals: List[str] = []
    form_signals: List[str] = []
    request_body = False
    payload_type = ""
    path_lower = (path or "").lower()

    if path_lower.startswith(("/api/", "/v1/", "/v2/")) or "/api/" in path_lower:
        score += 30
        signals.append("path:/api")
    if path_lower in {"/graphql", "/token"} or path_lower.endswith(("/graphql/", "/token/")):
        score += 30
        signals.append(f"path:{path_lower.rstrip('/') or '/'}")
    if any(token in path_lower for token in ("/webhook", "/callback")):
        score += 25
        signals.append("path:webhook")

    if route is not None:
        response_model = getattr(route, "response_model", None)
        if response_model is not None:
            score += 50
            signals.append("response_model")
            payload_type = "json"
        dependencies = getattr(getattr(route, "dependant", None), "dependencies", None)
        if dependencies:
            signals.append("dependencies")

    unwrapped = _unwrap(callback) if callback is not None else None
    view_name = getattr(unwrapped, "__name__", None)
    trees = []
    if unwrapped is not None:
        module_tree = _module_tree(unwrapped)
        if module_tree is not None:
            trees.append((module_tree, True))
        callback_tree = _callback_tree(unwrapped)
        if callback_tree is not None:
            trees.append((callback_tree, False))

    best_source_score = 0
    best_source_signals: List[str] = []
    best_request_body = False
    best_form_score = 0
    best_form_signals: List[str] = []
    best_payload_type = ""
    for tree, top_level_only in trees:
        imports, import_signals, import_score, form_import_signals, form_import_score = _imports(tree)
        functions = _top_level_functions(tree) if top_level_only else {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if view_name and view_name in functions:
            call_score, call_signals, body, source_form_score, source_form_signals, source_payload_type = _analyze_function(functions[view_name], functions, imports)
        else:
            call_score, call_signals, body, source_form_score, source_form_signals, source_payload_type = 0, [], False, 0, [], ""
        source_score = call_score + (import_score if call_score else 0)
        source_form_score = source_form_score + (form_import_score if source_form_score else 0)
        source_signals = call_signals + (import_signals if call_score else [])
        source_form_signals = source_form_signals + (form_import_signals if source_form_score else [])
        if source_score + source_form_score > best_source_score + best_form_score:
            best_source_score = source_score
            best_source_signals = source_signals
            best_request_body = body
            best_form_score = source_form_score
            best_form_signals = source_form_signals
            best_payload_type = source_payload_type

    score += best_source_score
    form_score += best_form_score
    signals.extend(best_source_signals)
    form_signals.extend(best_form_signals)
    request_body = request_body or best_request_body
    payload_type = best_payload_type or payload_type

    is_form = form_score >= 50 and form_score >= score
    is_api = bool(score >= 50 and not is_form)
    confidence = min(0.99, score / 100.0)
    form_confidence = min(0.99, form_score / 100.0)
    response_type = ""
    if is_api:
        response_type = "json"
        payload_type = payload_type or "json"
    elif is_form:
        response_type = "mixed" if score else "html"
        payload_type = "form"
    return ApiDetection(
        is_api=is_api,
        response_type=response_type,
        payload_type=payload_type,
        confidence=round(confidence, 2),
        signals=sorted(set(signals)),
        request_body=request_body,
        form_confidence=round(form_confidence, 2),
        form_signals=sorted(set(form_signals)),
    )
