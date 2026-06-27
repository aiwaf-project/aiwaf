"""Static authentication endpoint detection for path manifests."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class AuthDetection:
    is_auth: bool
    action: str
    confidence: float
    signals: List[str]


DJANGO_IMPORT_SIGNALS = {
    "django.contrib.auth.authenticate": 10,
    "django.contrib.auth.login": 10,
    "django.contrib.auth.logout": 10,
    "django.contrib.auth.get_user_model": 10,
    "django.contrib.auth.models.User": 10,
    "django.contrib.auth.forms.AuthenticationForm": 20,
    "django.contrib.auth.views.LoginView": 40,
}

FLASK_IMPORT_SIGNALS = {
    "flask_login.login_user": 10,
    "flask_login.logout_user": 10,
    "flask_login.current_user": 5,
    "flask_login.login_required": 15,
    "werkzeug.security.check_password_hash": 15,
}

FASTAPI_IMPORT_SIGNALS = {
    "fastapi.Depends": 5,
    "fastapi.Security": 5,
    "fastapi.security.OAuth2PasswordBearer": 40,
    "fastapi.security.OAuth2PasswordRequestForm": 40,
    "passlib.context.CryptContext": 20,
}

CALL_SIGNALS = {
    "authenticate": (50, "login"),
    "login": (30, "login"),
    "logout": (60, "logout"),
    "create_user": (50, "register"),
    "create_superuser": (50, "register"),
    "get_user_model": (20, "user_model"),
    "login_user": (60, "login"),
    "logout_user": (60, "logout"),
    "check_password_hash": (35, "login"),
    "verify_password": (35, "login"),
    "create_access_token": (50, "token_login"),
}

NAME_SIGNALS = {
    "AuthenticationForm": (20, "login"),
    "LoginView": (40, "login"),
    "OAuth2PasswordBearer": (40, "token_auth"),
    "OAuth2PasswordRequestForm": (40, "token_login"),
    "CryptContext": (20, "password_hashing"),
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
        import textwrap

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


def _import_signals(tree: ast.AST, framework: str) -> tuple[Dict[str, str], List[str], int]:
    imports: Dict[str, str] = {}
    signals: List[str] = []
    score = 0
    import_weights = {
        "django": DJANGO_IMPORT_SIGNALS,
        "flask": FLASK_IMPORT_SIGNALS,
        "fastapi": FASTAPI_IMPORT_SIGNALS,
    }.get(framework, {})

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                local_name = alias.asname or alias.name
                imports[local_name] = full_name
                weight = import_weights.get(full_name)
                if weight:
                    score += weight
                    signals.append(full_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                imports[local_name] = alias.name
    return imports, signals, score


def _call_name(node: ast.Call) -> Optional[str]:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _name_signal(name: str, imports: Dict[str, str]) -> tuple[int, str, str] | None:
    imported = imports.get(name)
    if imported:
        simple = imported.rsplit(".", 1)[-1]
        if simple in CALL_SIGNALS:
            score, action = CALL_SIGNALS[simple]
            return score, action, imported
        if simple in NAME_SIGNALS:
            score, action = NAME_SIGNALS[simple]
            return score, action, imported
    if name in CALL_SIGNALS:
        score, action = CALL_SIGNALS[name]
        return score, action, name
    if name in NAME_SIGNALS:
        score, action = NAME_SIGNALS[name]
        return score, action, name
    return None


def _action_priority(action: str) -> int:
    return {
        "login": 50,
        "token_login": 50,
        "logout": 40,
        "register": 30,
        "token_auth": 20,
        "password_hashing": 10,
        "user_model": 5,
    }.get(action, 0)


def _merge_action(current: str, candidate: str) -> str:
    if not current:
        return candidate
    return candidate if _action_priority(candidate) > _action_priority(current) else current


def _analyze_function(
    node: ast.AST,
    functions: Dict[str, ast.AST],
    imports: Dict[str, str],
    *,
    depth: int = 0,
    seen: Optional[Set[str]] = None,
) -> tuple[int, str, List[str]]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0, "", []
    if depth > 3:
        return 0, "", []
    seen = set(seen or set())
    if node.name in seen:
        return 0, "", []
    seen.add(node.name)

    score = 0
    action = ""
    signals: List[str] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            signal = _name_signal(child.id, imports)
            if signal:
                weight, candidate_action, signal_name = signal
                score += weight
                action = _merge_action(action, candidate_action)
                signals.append(signal_name)
        elif isinstance(child, ast.Call):
            name = _call_name(child)
            if name:
                signal = _name_signal(name, imports)
                if signal:
                    weight, candidate_action, signal_name = signal
                    score += weight
                    action = _merge_action(action, candidate_action)
                    signals.append(signal_name)
                helper = functions.get(name)
                if helper is not None:
                    helper_score, helper_action, helper_signals = _analyze_function(
                        helper,
                        functions,
                        imports,
                        depth=depth + 1,
                        seen=seen,
                    )
                    score += helper_score
                    action = _merge_action(action, helper_action)
                    signals.extend(helper_signals)
    return score, action, signals


def detect_auth_endpoint(callback: Any, *, framework: str, methods: Iterable[str] | None = None) -> AuthDetection:
    unwrapped = _unwrap(callback)
    view_name = getattr(unwrapped, "__name__", None)
    trees = []
    module_tree = _module_tree(unwrapped)
    if module_tree is not None:
        trees.append((module_tree, True))
    callback_tree = _callback_tree(unwrapped)
    if callback_tree is not None:
        trees.append((callback_tree, False))

    best_score = 0
    best_action = ""
    best_signals: List[str] = []

    for tree, top_level_only in trees:
        imports, import_signals, import_score = _import_signals(tree, framework)
        functions = _top_level_functions(tree) if top_level_only else {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if view_name and view_name in functions:
            call_score, action, call_signals = _analyze_function(functions[view_name], functions, imports)
        else:
            call_score, action, call_signals = 0, "", []

        score = call_score + (import_score if call_score else 0)
        signals = call_signals + (import_signals if call_score else [])
        if score > best_score:
            best_score = score
            best_action = action
            best_signals = signals

    method_set = {str(method).upper() for method in (methods or [])}
    if best_score and "POST" in method_set:
        best_score += 10
        best_signals.append("POST")

    confidence = min(0.99, best_score / 100.0)
    return AuthDetection(
        is_auth=best_score >= 50,
        action=best_action or "auth",
        confidence=round(confidence, 2),
        signals=sorted(set(best_signals)),
    )
