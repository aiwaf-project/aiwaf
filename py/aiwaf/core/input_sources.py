"""Where a route's inputs come from besides reading its code, and how they're combined.

``init`` reads each view's code for what it reads (see input_schema). This module completes
what that can't see, then decides each route's ``inputs_complete``:

1. The framework itself: Django generic views, Django REST framework views and serializers, and
   FastAPI's resolved parameters and dependencies list exactly what they read. Forms, serializers
   and models the code reading couldn't follow are looked up in the modules the app has already
   imported, and their fields read.
2. Declarations: ``@aiwaf.inputs(...)`` on a view, or ``.aiwaf/inputs.json``.
3. An OpenAPI spec (``AIWAF_OPENAPI_SPEC``).
4. Inputs recorded while tests or staging traffic ran (see input_recorder). These only add
   names, and a name the code read that the analysis didn't find makes the route incomplete,
   since the analysis missed something.

A declaration or a closed OpenAPI operation is authoritative: the route is complete with what it
lists plus anything the code was seen reading. Nothing else marks a route complete that the
code reading didn't, unless the framework itself lists every input.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import re
import sys
import typing
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from aiwaf.core.input_schema import REASON_CODES

_KINDS = {"int", "float", "bool", "uuid", "text"}
_BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_ATTRIBUTE = "__aiwaf_inputs__"
_MAX_REASONS = 20


def _kind(value: Any) -> str:
    """int, float, bool, uuid or text, for a type, a type name or a field class name."""
    if isinstance(value, str):
        name = value.strip()
        if name in _KINDS:
            return name
        lowered = name.lower()
        if "boolean" in lowered or lowered == "bool":
            return "bool"
        if "uuid" in lowered:
            return "uuid"
        if "integer" in lowered or lowered in {"int", "long"}:
            return "int"
        if "float" in lowered or "decimal" in lowered or lowered in {"number", "double"}:
            return "float"
        return "text"
    origin = typing.get_origin(value)
    if origin is typing.Annotated:
        return _kind(typing.get_args(value)[0])
    if origin is typing.Union or (origin is not None and getattr(origin, "__name__", "") == "UnionType"):
        kinds = {_kind(arg) for arg in typing.get_args(value) if arg is not type(None)}
        return kinds.pop() if len(kinds) == 1 else "text"
    if origin in (list, set, tuple, frozenset):
        args = typing.get_args(value)
        return _kind(args[0]) if args else "text"
    if value is bool:
        return "bool"
    if isinstance(value, type):
        if issubclass(value, bool):
            return "bool"
        if issubclass(value, int):
            return "int"
        if issubclass(value, (float, Decimal)):
            return "float"
        if issubclass(value, uuid.UUID):
            return "uuid"
    return "text"


def _merge(inputs: Dict[str, str], name: str, kind: str) -> None:
    inputs[name] = kind if inputs.get(name, kind) == kind else "text"


# 2. Declarations ---------------------------------------------------------------------------


def inputs(complete: bool = True, **fields: Any) -> Callable:
    """Declare a view's query and body inputs, for when reading its code can't find them all.

        @aiwaf.inputs(q=str, page=int, user_id=uuid.UUID)
        def search(request): ...

    Types are int, float, bool, str, uuid.UUID or Decimal, or the names "int", "float", "bool",
    "uuid" and "text". With complete=True (the default) the route's list is complete: these, plus
    anything AIWAF sees the code read. complete=False only adds the names.
    """
    declared = {"inputs": {name: _kind(kind) for name, kind in fields.items()}, "complete": bool(complete)}

    def decorate(view):
        target = getattr(view, "view_class", None) or view
        try:
            setattr(target, _ATTRIBUTE, declared)
        except (AttributeError, TypeError):
            pass
        if target is not view:
            try:
                setattr(view, _ATTRIBUTE, declared)
            except (AttributeError, TypeError):
                pass
        return view

    return decorate


def declared_on(view: Any) -> Optional[dict]:
    """The @inputs declaration on a view, its class, or what it wraps, or None."""
    seen = set()
    candidate = view
    while candidate is not None and id(candidate) not in seen and len(seen) < 8:
        seen.add(id(candidate))
        for holder in (candidate, getattr(candidate, "view_class", None), getattr(candidate, "cls", None)):
            declared = getattr(holder, _ATTRIBUTE, None) if holder is not None else None
            if isinstance(declared, dict):
                return {"inputs": dict(declared.get("inputs", {})), "complete": bool(declared.get("complete", True))}
        candidate = getattr(candidate, "__wrapped__", None)
    return None


# 1. The framework ----------------------------------------------------------------------------


def class_fields(cls: Any) -> Optional[Dict[str, str]]:
    """A form, serializer, schema or model class's input fields and types, or None.

    Django forms (including ModelForms), Django REST framework serializers, Pydantic models,
    marshmallow schemas, WTForms forms, dataclasses and TypedDicts. None for anything else, and
    for one that accepts fields it doesn't declare.
    """
    if not isinstance(cls, type):
        return None
    base_fields = getattr(cls, "base_fields", None)
    if isinstance(base_fields, dict):  # Django forms.
        return {name: _kind(type(field).__name__) for name, field in base_fields.items()}
    if any(base.__module__.startswith("rest_framework.") for base in cls.__mro__) and hasattr(cls, "get_fields"):
        try:
            fields = cls().fields
        except Exception:  # noqa: BLE001 - a serializer that needs arguments can't be read
            return None
        return {getattr(field, "field_name", None) or name: _kind(type(field).__name__)
                for name, field in fields.items() if not getattr(field, "read_only", False)}
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict) and hasattr(cls, "model_config"):  # Pydantic 2.
        config = getattr(cls, "model_config", None) or {}
        if config.get("extra") == "allow":
            return None
        found: Dict[str, str] = {}
        for name, field in model_fields.items():
            for accepted in _pydantic_names(name, field, config):
                # An AliasPath's first key holds an object, not the field's own value.
                found[accepted] = "text" if accepted in _path_heads(field) else _kind(field.annotation)
        return found
    declared = getattr(cls, "_declared_fields", None)
    if isinstance(declared, dict):  # marshmallow.
        if str(getattr(getattr(cls, "Meta", None), "unknown", "")).lower() == "include":
            return None
        return {(getattr(field, "data_key", None) or name): _kind(type(field).__name__) for name, field in declared.items()}
    unbound = {name: value for name, value in vars(cls).items() if getattr(value, "_formfield", False)}
    for base in cls.__mro__[1:]:
        for name, value in vars(base).items():
            if getattr(value, "_formfield", False):
                unbound.setdefault(name, value)
    if unbound:  # WTForms; Flask-WTF adds its CSRF field.
        found = {name: _kind(getattr(value, "field_class", type(value)).__name__) for name, value in unbound.items()}
        if any(base.__name__ == "FlaskForm" for base in cls.__mro__):
            found.setdefault("csrf_token", "text")
        return found
    dataclass_fields = getattr(cls, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, dict):
        return {name: _kind(field.type) for name, field in dataclass_fields.items()}
    if hasattr(cls, "__total__") and hasattr(cls, "__annotations__"):  # TypedDict.
        return {name: _kind(kind) for name, kind in typing.get_type_hints(cls).items()}
    return None


def _pydantic_names(name: str, field: Any, config: Mapping[str, Any]) -> List[str]:
    """Every top-level key a Pydantic 2 field accepts in input.

    Its validation_alias when set (a string, each of AliasChoices', or an AliasPath's first key),
    else its alias, else its name; and its name too when the model populates by name.
    """
    names: List[str] = []
    validation = getattr(field, "validation_alias", None)
    if isinstance(validation, str):
        names.append(validation)
    elif validation is not None:
        for choice in getattr(validation, "choices", None) or [validation]:
            path = choice if isinstance(choice, str) else (getattr(choice, "path", None) or [None])[0]
            if isinstance(path, str):
                names.append(path)
    if not names:
        names.append(getattr(field, "alias", None) or name)
    if config.get("populate_by_name") or config.get("validate_by_name"):
        names.append(name)
    return list(dict.fromkeys(names))


def _path_heads(field: Any) -> set:
    """The first keys of a field's AliasPaths that go deeper, as address in AliasPath("address", "city")."""
    validation = getattr(field, "validation_alias", None)
    heads = set()
    for choice in (getattr(validation, "choices", None) or [validation]) if validation is not None and not isinstance(validation, str) else ():
        path = getattr(choice, "path", None)
        if path and len(path) > 1 and isinstance(path[0], str):
            heads.add(path[0])
    return heads


def _object_class(annotation: Any) -> Any:
    """The class an annotation names, through Optional and Annotated, or None."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _object_class(typing.get_args(annotation)[0])
    if origin is typing.Union or (origin is not None and getattr(origin, "__name__", "") == "UnionType"):
        classes = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _object_class(classes[0]) if len(classes) == 1 else None
    return annotation if isinstance(annotation, type) else None


def nested_fields(cls: Any) -> Dict[str, List[str]]:
    """The keys inside a Pydantic model's object-valued fields, as {"customer": ["email"]}.

    Only fields whose type is itself a model, dataclass or other class with declared fields,
    and only one level down, as inputs_nested holds. A field typed as a list, a dict or a
    model that accepts undeclared fields is left out, so its whole value keeps being checked.
    """
    model_fields = getattr(cls, "model_fields", None)
    if not isinstance(model_fields, dict) or not hasattr(cls, "model_config"):
        return {}
    found: Dict[str, List[str]] = {}
    for name, field in model_fields.items():
        inner = _object_class(field.annotation)
        if inner is None or inner in (str, bytes, int, float, bool) or not hasattr(inner, "__mro__"):
            continue
        keys = class_fields(inner)
        if keys:
            for accepted in _pydantic_names(name, field, getattr(cls, "model_config", None) or {}):
                found[accepted] = sorted(keys)
    return found


def _loaded_module(name: str) -> Any:
    """An already-imported module by its name, or one whose name ends with it. Never imports."""
    if not name:
        return None
    module = sys.modules.get(name)
    if module is not None:
        return module
    matches = [loaded for key, loaded in list(sys.modules.items()) if key.endswith(f".{name}")]
    return matches[0] if len(matches) == 1 else None


def resolve_class(module_name: str, dotted: str) -> Any:
    """A class named in a module the app has imported, through its own imports, or None."""
    module = _loaded_module(module_name)
    value: Any = module
    for part in dotted.split(".") if dotted else []:
        value = getattr(value, part, None)
        if value is None:
            return None
    return value if isinstance(value, type) else None


_GENERIC_MODULES = (
    "django.views.generic.", "django.contrib.auth.views", "django.contrib.auth.mixins", "django.views.generic",
)
_DRF_MODULES = ("rest_framework.generics", "rest_framework.mixins", "rest_framework.views", "rest_framework.viewsets")


def _framework_bases(view_class: type) -> Tuple[List[type], List[type]]:
    """A view class's bases from the frameworks it's read for, and every other base."""
    known, other = [], []
    for base in view_class.__mro__[1:]:
        if base is object:
            continue
        module = base.__module__ or ""
        (known if module.startswith(_GENERIC_MODULES + _DRF_MODULES) else other).append(base)
    return known, other


def _reason(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def django_inputs(callback: Any, methods: Iterable[str] = ()) -> Optional[dict]:
    """What a Django or DRF class view reads by way of the framework, or None for others.

    The result is {"inputs", "covers", "reasons", "standalone"}: covers are the code reading's
    reasons the framework answers, and standalone means the list needs no code reading.
    """
    view_class = getattr(callback, "view_class", None) or getattr(callback, "cls", None)
    if not isinstance(view_class, type):
        return None
    known, other = _framework_bases(view_class)
    names = {base.__name__ for base in view_class.__mro__}
    found: Dict[str, str] = {}
    reasons: List[dict] = []
    covers: List[str] = []
    body = bool(set(methods) & _BODY_METHODS) or not methods
    if "APIView" in names and any(base.__module__.startswith("rest_framework.") for base in view_class.__mro__):
        try:
            from rest_framework.settings import api_settings

            override = api_settings.URL_FORMAT_OVERRIDE
        except Exception:  # noqa: BLE001 - DRF without settings
            override = "format"
        if override:
            found[override] = "text"
        if "GenericAPIView" in names:
            serializer = getattr(view_class, "serializer_class", None)
            if body and serializer is not None:
                fields = class_fields(serializer)
                if fields is None:
                    reasons.append(_reason("external_class", f"{serializer.__name__}'s fields can't be read"))
                else:
                    found.update(fields)
            elif body and "get_serializer_class" in vars(view_class):
                reasons.append(_reason("class_view", f"{view_class.__name__} chooses its serializer in code"))
            pagination = getattr(view_class, "pagination_class", None)
            for attribute, kind in (("page_query_param", "text"), ("page_size_query_param", "int"), ("limit_query_param", "int"),
                                    ("offset_query_param", "int"), ("cursor_query_param", "text")):
                value = getattr(pagination, attribute, None) if pagination is not None else None
                if isinstance(value, str) and value:
                    found[value] = kind
            for backend in getattr(view_class, "filter_backends", ()) or ():
                name = getattr(backend, "__name__", str(backend))
                if name == "SearchFilter":
                    found[getattr(backend, "search_param", None) or "search"] = "text"
                elif name == "OrderingFilter":
                    found[getattr(backend, "ordering_param", None) or "ordering"] = "text"
                elif name == "DjangoFilterBackend":
                    filterset = getattr(view_class, "filterset_class", None)
                    fields = getattr(view_class, "filterset_fields", None)
                    if filterset is not None and isinstance(getattr(filterset, "base_filters", None), dict):
                        found.update({key: "text" for key in filterset.base_filters})
                    elif fields:
                        found.update({key: "text" for key in fields})
                    else:
                        reasons.append(_reason("class_view", f"{name} on {view_class.__name__} has no filterset to read"))
                else:
                    reasons.append(_reason("class_view", f"filter backend {name} isn't understood"))
        covers = ["external_class", "model_fields", "container_whole"]
    elif "View" in names and any(base.__module__.startswith("django.views.") for base in view_class.__mro__):
        if "FormMixin" in names and body:
            form_class = getattr(view_class, "form_class", None)
            if form_class is None and getattr(view_class, "model", None) is not None and getattr(view_class, "fields", None):
                try:
                    from django.forms.models import modelform_factory

                    form_class = modelform_factory(view_class.model, fields=view_class.fields)
                except Exception:  # noqa: BLE001 - a model form that can't be built
                    form_class = None
            fields = class_fields(form_class) if form_class is not None else None
            if fields is None:
                reasons.append(_reason("class_view", f"{view_class.__name__} chooses its form in code"))
            else:
                found.update(fields)
        if "MultipleObjectMixin" in names and getattr(view_class, "paginate_by", None):
            found[getattr(view_class, "page_kwarg", "page") or "page"] = "text"
        if "RedirectView" in names and getattr(view_class, "query_string", False):
            reasons.append(_reason("raw_body", f"{view_class.__name__} passes the query string on"))
        redirect_field = getattr(view_class, "redirect_field_name", None)
        if isinstance(redirect_field, str) and redirect_field:
            found[redirect_field] = "text"
        covers = ["external_class", "model_fields", "container_whole"]
    else:
        return None
    unmodeled = [base for base in known if not base.__module__.startswith(_GENERIC_MODULES + _DRF_MODULES)]
    framework_only = not other and not unmodeled
    if framework_only:
        covers.append("class_view")
    else:
        names_other = ", ".join(sorted({base.__name__ for base in other}))
        reasons.append(_reason("class_view", f"{view_class.__name__} also inherits from {names_other}"))
    return {"inputs": found, "covers": covers, "reasons": reasons, "standalone": framework_only}


def fastapi_inputs(route: Any) -> Optional[dict]:
    """What a FastAPI route reads, from the parameters and dependencies FastAPI resolved."""
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return None
    found: Dict[str, str] = {}
    reasons: List[dict] = []
    # The keys inside object-valued body fields, from the models that declare them.
    nested: Dict[str, set] = {}
    # Every parameter in the dependency tree, as FastAPI resolves them for the endpoint.
    queries: List[Any] = []
    bodies: List[Any] = []
    stack, visited = [dependant], set()
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        queries.extend(getattr(node, "query_params", ()) or ())
        bodies.extend(param for param in getattr(node, "body_params", ()) or () if all(
            other.alias != param.alias for other in bodies))
        stack.extend(getattr(node, "dependencies", ()) or ())

    def annotation(param: Any) -> Any:
        info = getattr(param, "field_info", None)
        return getattr(info, "annotation", None) if info is not None else getattr(param, "type_", None)

    for param in queries:
        _merge(found, param.alias, _kind(annotation(param)))
    for param in bodies:
        embedded = bool(getattr(getattr(param, "field_info", None), "embed", False))
        model = annotation(param)
        origin = typing.get_origin(model) or model
        if origin in (dict, typing.Any, object) or (isinstance(origin, type) and issubclass(origin, collections.abc.Mapping)):
            # A dict (or Any) body takes whatever names the client sends.
            reasons.append(_reason("container_whole", f"the body {param.alias} is a whole {getattr(origin, '__name__', 'object')}"))
            continue
        fields = class_fields(model) if len(bodies) == 1 and not embedded else None
        if fields is not None:
            for name, kind in fields.items():
                _merge(found, name, kind)
            for name, keys in nested_fields(model).items():
                nested.setdefault(name, set()).update(keys)
        elif len(bodies) == 1 and not embedded and isinstance(model, type) and hasattr(model, "model_fields"):
            reasons.append(_reason("model_fields", f"{model.__name__} accepts fields it doesn't declare"))
        else:
            _merge(found, param.alias, _kind(model))

    def walk(node: Any, top: bool) -> None:
        reads_request = any(getattr(node, attribute, None) for attribute in (
            "request_param_name", "websocket_param_name", "http_connection_param_name"))
        if reads_request and not top:
            call = getattr(node, "call", None)
            reasons.append(_reason("dependency", f"dependency {getattr(call, '__name__', 'a dependency')} takes the request itself"))
        for child in getattr(node, "dependencies", ()) or ():
            walk(child, False)

    walk(dependant, True)
    standalone = not getattr(dependant, "request_param_name", None)
    result = {"inputs": found, "covers": ["dependency", "external_class", "model_fields"], "reasons": reasons, "standalone": standalone}
    if nested:
        result["nested"] = {name: sorted(keys) for name, keys in nested.items()}
    return result


# 3. OpenAPI -----------------------------------------------------------------------------------


_PARAMETER = re.compile(r"^(?:<[^>]+>|\{[^}]+\}|:[A-Za-z_]\w*\??|\*\w*)$")


_WEBARGS_QUERY = {"query", "querystring"}
_WEBARGS_BODY = {"json", "form", "json_or_form"}
_MARSHMALLOW_KINDS = (("Integer", "int"), ("Float", "float"), ("Decimal", "float"), ("Number", "float"),
                      ("Boolean", "bool"), ("UUID", "uuid"))


def _marshmallow_kind(field: Any) -> str:
    names = [klass.__name__ for klass in type(field).__mro__]
    if "List" in names and getattr(field, "inner", None) is not None:
        return _marshmallow_kind(field.inner)
    return next((kind for name, kind in _MARSHMALLOW_KINDS if name in names), "text")


def _closure(function: Any) -> Dict[str, Any]:
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None) or ()
    values: Dict[str, Any] = {}
    for name, cell in zip(getattr(code, "co_freevars", ()), cells):
        try:
            values[name] = cell.cell_contents
        except ValueError:  # An empty cell.
            continue
    return values


def webargs_inputs(view: Any) -> Optional[dict]:
    """What webargs' use_args and use_kwargs decorators (and flask-smorest's Blueprint.arguments,
    which uses them) load for a view, from each decorator layer's closure: {"inputs", "covers",
    "reasons", "standalone"}, or None when the view has none.

    Each layer gives its schema's loadable fields (by data_key) at its location. A location whose
    unknown setting is INCLUDE passes other names through to the view (a container_whole reason).
    When both the query and the body reject unknown names (RAISE), requests with any other name
    fail before the view runs, so the list is complete: the framework answers every reason the
    code reading has but a raw body read.
    """
    found: Dict[str, str] = {}
    reasons: List[dict] = []
    closed: Dict[str, bool] = {}
    layers = 0
    seen = set()
    current = view
    while callable(current) and id(current) not in seen and len(seen) < 32:
        seen.add(id(current))
        values = _closure(current)
        parser, argmap, location = values.get("self"), values.get("argmap"), values.get("location")
        if hasattr(parser, "DEFAULT_UNKNOWN_BY_LOCATION") and argmap is not None and isinstance(location, str):
            layers += 1
            segment = "query" if location in _WEBARGS_QUERY else "body" if location in _WEBARGS_BODY else None
            try:
                schema = parser._get_schema(argmap, None) if hasattr(parser, "_get_schema") else argmap
                if isinstance(schema, type):
                    schema = schema()
                fields = dict(getattr(schema, "load_fields", None) or {})
            except Exception as exc:  # A schema factory that needs the request.
                reasons.append(_reason("external_class", f"the {location} schema can't be built without a request ({exc.__class__.__name__})"))
                current = getattr(current, "__wrapped__", None)
                continue
            if segment is not None:
                for attribute, field in fields.items():
                    _merge(found, str(getattr(field, "data_key", None) or attribute), _marshmallow_kind(field))
                # webargs' own default is a sentinel: the location's default, which None leaves to the schema.
                unknown = values.get("unknown")
                if unknown not in {"raise", "exclude", "include", None}:
                    unknown = parser.DEFAULT_UNKNOWN_BY_LOCATION.get(location)
                if unknown is None:
                    unknown = getattr(schema, "unknown", "raise")
                if unknown == "include":
                    reasons.append(_reason("container_whole", f"the {location} schema passes names it doesn't list through to the view"))
                closed[segment] = closed.get(segment, True) and unknown == "raise"
        current = getattr(current, "__wrapped__", None)
    if not layers:
        return None
    exact = bool(closed.get("query") and closed.get("body"))
    covers = [code for code in REASON_CODES if code not in {"raw_body", "unread_view"}] if exact else []
    return {"inputs": found, "covers": covers, "reasons": reasons, "standalone": exact}


def _merged(first: Mapping[str, str], second: Mapping[str, str]) -> Dict[str, str]:
    out = dict(first)
    for name, kind in second.items():
        _merge(out, name, kind)
    return out


def combine(first: Optional[dict], second: Optional[dict]) -> Optional[dict]:
    """Two framework declarations for one route (say, its GET and POST handlers) as one."""
    if not first or not second:
        return first or second
    combined = {
        # A name both declare with different types takes the weaker, as the engine keeps one type
        # for every method: POST's integer can't hold for a GET that sends any text.
        "inputs": _merged(first["inputs"], second["inputs"]),
        "covers": [code for code in first["covers"] if code in second["covers"]],
        "reasons": [*first["reasons"], *second["reasons"]],
        "standalone": bool(first["standalone"] and second["standalone"]),
    }
    # A field's inner keys hold only when neither handler takes the field whole.
    nested: Dict[str, set] = {}
    for own, other in ((first, second), (second, first)):
        for name, keys in (own.get("nested") or {}).items():
            if name in other["inputs"] and name not in (other.get("nested") or {}):
                continue
            nested.setdefault(name, set()).update(keys)
    if nested:
        combined["nested"] = {name: sorted(keys) for name, keys in nested.items()}
    return combined


def _request_schema(value: Any) -> Optional[type]:
    """The class when value is a request schema or one's instance: a pydantic model, a
    marshmallow schema, a DRF serializer or a Django or WTForms form. Not a dataclass or
    TypedDict, which closures hold for other reasons."""
    cls = value if isinstance(value, type) else type(value)
    if dataclasses.is_dataclass(cls) or typing.is_typeddict(cls):
        return None
    markers = ("model_fields", "_declared_fields", "base_fields", "_unbound_fields")
    return cls if any(hasattr(cls, marker) for marker in markers) and class_fields(cls) is not None else None


def closure_inputs(view: Any) -> Optional[dict]:
    """What a view's own decorators validate with, as @validate(Body): a request schema in a
    decorator layer's closure, through __wrapped__. Its fields are names the view may read; the
    view's code still says what else it reads. None when no layer holds one."""
    found: Dict[str, str] = {}
    seen, current, any_schema = set(), view, False
    while callable(current) and id(current) not in seen and len(seen) < 32:
        seen.add(id(current))
        values = _closure(current)
        if not hasattr(values.get("self"), "DEFAULT_UNKNOWN_BY_LOCATION"):  # webargs has its own reader.
            for value in values.values():
                schema = _request_schema(value)
                if schema is not None:
                    any_schema = True
                    for name, kind in (class_fields(schema) or {}).items():
                        _merge(found, name, kind)
        current = getattr(current, "__wrapped__", None)
    return {"inputs": found, "covers": [], "reasons": [], "standalone": False} if any_schema else None


def view_arguments(view: Any, methods: Iterable[str] = ()) -> Optional[dict]:
    """webargs arguments (and flask-smorest's) on a view function, or on each method of a class
    view (Flask's MethodView, Django's View): any webargs parser, as their decorators are alike."""
    view_class = getattr(view, "view_class", None)
    if view_class is None:
        return combine(webargs_inputs(view), closure_inputs(view))
    declared = None
    for method in methods:
        handler = getattr(view_class, str(method).lower(), None)
        if handler is not None:
            declared = combine(declared, combine(webargs_inputs(handler), closure_inputs(handler)))
    return declared


flask_inputs = view_arguments


_PATH_PARAMETER = re.compile(r"<(?:[^:>]+:)?([^>]+)>|\{([^}:]+)(?::[^}]*)?\}|:([A-Za-z_]\w*)\??|\[(?:\.\.\.)?([^\]]+)\]")


def path_parameters(route: str) -> List[str]:
    """The parameter names in a route, as id in /a/<int:id>, /a/{id}, /a/:id or /a/[id]."""
    return [next(group for group in match.groups() if group) for match in _PATH_PARAMETER.finditer(str(route))]


def canonical(path: str) -> str:
    """A route template with every parameter written {}, so /a/<int:id>/ and /a/{id} match."""
    parts = [part for part in str(path or "").split("?")[0].strip().strip("/").split("/") if part]
    return "/" + "/".join("{}" if _PARAMETER.match(part) else part for part in parts)


def _openapi_kind(schema: Mapping[str, Any]) -> str:
    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [item for item in kind if item != "null"]
        kind = kinds[0] if len(kinds) == 1 else None
    if kind == "integer":
        return "int"
    if kind == "number":
        return "float"
    if kind == "boolean":
        return "bool"
    if kind == "string" and schema.get("format") == "uuid":
        return "uuid"
    if kind == "array" and isinstance(schema.get("items"), Mapping):
        return _openapi_kind(schema["items"])
    return "text"


def openapi_inputs(spec: Mapping[str, Any]) -> Dict[str, Dict[str, dict]]:
    """Each operation's query and body inputs in an OpenAPI 3 spec, by canonical path and method.

    An operation is complete unless something in it can carry inputs it doesn't name: a body
    schema that allows other properties, a body that isn't an object, a free-form query object,
    or a reference that can't be resolved.
    """

    def resolve(node: Any, depth: int = 0) -> Optional[Mapping[str, Any]]:
        while isinstance(node, Mapping) and "$ref" in node and depth < 16:
            ref = str(node["$ref"])
            if not ref.startswith("#/"):
                return None
            target: Any = spec
            for part in ref[2:].split("/"):
                target = target.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(target, Mapping) else None
            node, depth = target, depth + 1
        return node if isinstance(node, Mapping) else None

    def properties(schema: Any, reasons: List[dict], depth: int = 0) -> Dict[str, str]:
        schema = resolve(schema)
        if schema is None or depth > 8:
            reasons.append(_reason("external_class", "the body schema can't be resolved"))
            return {}
        found: Dict[str, str] = {}
        for key in ("allOf", "oneOf", "anyOf"):
            for part in schema.get(key, ()) or ():
                found.update(properties(part, reasons, depth + 1))
        if any(key in schema for key in ("allOf", "oneOf", "anyOf")) and "properties" not in schema:
            return found
        if schema.get("type", "object") != "object":
            reasons.append(_reason("container_whole", "the body isn't an object with named properties"))
            return found
        extra = schema.get("additionalProperties", False)
        if extra not in (False, None):
            reasons.append(_reason("container_whole", "the body schema allows properties it doesn't name"))
        for name, value in (schema.get("properties") or {}).items():
            found[name] = _openapi_kind(resolve(value) or {})
        return found

    result: Dict[str, Dict[str, dict]] = {}
    for path, item in (spec.get("paths") or {}).items():
        item = resolve(item) or {}
        common = item.get("parameters") or []
        for method, operation in item.items():
            if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"} or not isinstance(operation, Mapping):
                continue
            found: Dict[str, str] = {}
            reasons: List[dict] = []
            for parameter in [*common, *(operation.get("parameters") or [])]:
                parameter = resolve(parameter)
                if parameter is None:
                    reasons.append(_reason("external_class", "a parameter reference can't be resolved"))
                    continue
                if parameter.get("in") != "query":
                    continue
                schema = resolve(parameter.get("schema")) or {}
                if schema.get("type") == "object" and parameter.get("explode", True) and not schema.get("properties"):
                    reasons.append(_reason("container_whole", f"query parameter {parameter.get('name')} is a free-form object"))
                    continue
                found[str(parameter.get("name"))] = _openapi_kind(schema)
            body = resolve(operation.get("requestBody"))
            for media, content in ((body or {}).get("content") or {}).items():
                if media.split(";")[0].strip() in {"application/json", "application/x-www-form-urlencoded", "multipart/form-data"} \
                        or media.endswith("+json"):
                    for name, kind in properties((content or {}).get("schema"), reasons).items():
                        _merge(found, name, kind)
                else:
                    reasons.append(_reason("raw_body", f"the body is {media}"))
            result.setdefault(canonical(path), {})[method.upper()] = {"inputs": found, "complete": not reasons, "reasons": reasons}
    return result


def load_spec(path: str) -> dict:
    """An OpenAPI spec from a JSON file, or YAML when PyYAML is installed."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise ValueError(f"{path} is YAML; install PyYAML, or give a JSON spec") from error
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict) or "paths" not in loaded:
        raise ValueError(f"{path} isn't an OpenAPI spec: it has no paths")
    return loaded


# 4. Recorded inputs ---------------------------------------------------------------------------


WHOLE = "*"


def load_observations(path: str) -> Dict[str, Dict[str, str]]:
    """Recorded reads (see input_recorder), by canonical route to {name: type}; * is a whole read."""
    found: Dict[str, Dict[str, str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if not isinstance(item, dict) or not item.get("route") or not item.get("key"):
            continue
        route = found.setdefault(canonical(str(item["route"])), {})
        key = str(item["key"])
        _merge(route, key, _kind(str(item.get("type", "text"))) if key != WHOLE else WHOLE)
    return found


def load_declarations(path: str) -> Dict[str, dict]:
    """.aiwaf/inputs.json: {"/route": {"inputs": {...}, "complete": true}}, optionally under "routes"."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    routes = loaded.get("routes", loaded) if isinstance(loaded, dict) else {}
    declared: Dict[str, dict] = {}
    for route, spec in routes.items():
        if isinstance(spec, dict) and isinstance(spec.get("inputs", {}), dict):
            declared[canonical(route)] = {"inputs": {name: _kind(kind) for name, kind in spec.get("inputs", {}).items()},
                                          "complete": bool(spec.get("complete", True))}
    return declared


# Combining ------------------------------------------------------------------------------------


def _resolve_classes(data: dict, inputs: Dict[str, str], reasons: List[dict]) -> List[dict]:
    """Drops the reasons for classes the app has loaded and whose fields can be read."""
    view_module = str(data.get("view", "")).rsplit(".", 1)[0]
    kept = []
    for reason in reasons:
        if reason.get("code") in {"external_class", "model_fields"} and reason.get("name"):
            cls = resolve_class(str(reason.get("module") or view_module), str(reason["name"])) \
                or resolve_class(view_module, str(reason["name"]))
            fields = class_fields(cls) if cls is not None else None
            if fields is not None:
                for name, kind in fields.items():
                    _merge(inputs, name, kind)
                continue
        kept.append(reason)
    return kept


def _declare(inputs: Dict[str, str], declared: Mapping[str, Any]) -> bool:
    for name, kind in (declared.get("inputs") or {}).items():
        inputs[name] = kind
    return bool(declared.get("complete", True))


def finalize_without_code(routes: Dict[str, Any], **sources: Any) -> Dict[str, Any]:
    """finalize without the injection audit (--no-audit, or a scan that failed).

    The framework's and declared inputs still go into the manifest, but never as complete: a
    complete list with no saved risks tells the engine the route reaches no risky call, which
    only the audit can say. So the types are checked, and strict inputs and the relaxed SQL
    rules stay off for these routes.
    """
    # A framework list that isn't complete on its own still goes in (finalize skips a route it
    # can't tell was read otherwise).
    for data in routes.values():
        framework = data.get("declared_inputs") if isinstance(data, dict) else None
        if isinstance(framework, dict) and not framework.get("standalone"):
            data["inputs_complete"] = False
            data["inputs_incomplete"] = [_reason("not_audited", "init ran without the injection audit")]
    counts = finalize(routes, **sources)
    for data in routes.values():
        if not isinstance(data, dict) or "inputs" not in data:
            continue
        data["inputs_complete"] = False
        reasons = [reason for reason in data.get("inputs_incomplete") or [] if reason.get("code") != "not_audited"]
        data["inputs_incomplete"] = [_reason(
            "not_audited", "init ran without the injection audit (--no-audit, or the scan failed), so the route's risks "
            "aren't known; its types are checked, but the list isn't treated as complete"), *reasons][:_MAX_REASONS]
        data.pop("inputs_by_method", None)
        data.pop("inputs_nested", None)
    return counts


def finalize(
    routes: Dict[str, Any],
    *,
    declarations: Optional[Mapping[str, dict]] = None,
    openapi: Optional[Mapping[str, Dict[str, dict]]] = None,
    observations: Optional[Mapping[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Combine every source into each route's inputs, inputs_complete and inputs_incomplete."""
    counts: Dict[str, Any] = {"complete": 0, "incomplete": 0, "declared": 0, "openapi": 0, "observed": 0, "reasons": {}}
    for path, data in routes.items():
        if not isinstance(data, dict):
            continue
        framework = data.pop("declared_inputs", None)
        declaration = data.pop("input_declaration", None)
        # Which method reads what: the code reading's and the framework's, per method. Names
        # from anywhere else are shared by every method.
        declared_types = data.pop("declared_types", None) or {}
        code_methods = {str(method).upper(): set(names) for method, names in (data.pop("_input_methods", None) or {}).items()}
        declared_methods = {str(method).upper(): set(names) for method, names in (data.pop("declared_methods", None) or {}).items()}
        claims = {method: set(names) for method, names in code_methods.items()}
        for method, names in declared_methods.items():
            claims.setdefault(method, set()).update(names)
        all_methods = {str(method).upper() for method in data.get("methods") or ()}
        code_read = set(data.get("inputs") or {})

        def type_holds(name: str) -> bool:
            """A framework type only holds over the code's reading of the methods it's declared for."""
            if name not in code_read:
                return True
            declared = {method for method, names in declared_methods.items() if name in names}
            if not declared:
                return len(all_methods) <= 1
            reads = {method for method, names in code_methods.items() if name in names}
            return bool(reads) and reads <= declared
        shared: set = set()
        nested = data.pop("_nested", None) or {}
        flat = set(data.pop("_flat", None) or ())
        read = "inputs_complete" in data
        found: Dict[str, str] = dict(data.get("inputs") or {})
        reasons: List[dict] = list(data.get("inputs_incomplete") or [])
        if isinstance(framework, dict):
            covered = set(framework.get("covers") or ())
            reasons = [reason for reason in reasons if reason.get("code") not in covered]
            reasons.extend(framework.get("reasons") or ())
            framework_nested = framework.get("nested") or {}
            for name, kind in (framework.get("inputs") or {}).items():
                # A type the framework validates (a Pydantic int, a schema's integer) holds whatever
                # the handler's own read says: the value reaching it is checked, or coerced, already.
                if kind and kind != "text" and type_holds(name):
                    found[name] = kind
                else:
                    _merge(found, name, kind)
                if name not in framework_nested:
                    flat.add(name)
            # The framework parses these fields into models that drop undeclared keys, so their
            # declared keys stand even where reading the code found only the whole field.
            for name, keys in framework_nested.items():
                nested.setdefault(name, set()).update(keys)
                flat.discard(name)
            read = read or bool(framework.get("standalone"))
        before = set(found)
        reasons = _resolve_classes(data, found, reasons)
        shared |= set(found) - before
        key = canonical(path)
        authoritative = False
        for declared in (declaration, (declarations or {}).get(key)):
            if isinstance(declared, Mapping):
                authoritative = _declare(found, declared) or authoritative
                shared |= set(declared.get("inputs") or {})
                flat |= set(declared.get("inputs") or {})
                read = True
                counts["declared"] += 1
        operations = (openapi or {}).get(key)
        if operations:
            methods = [method for method in data.get("methods") or operations.keys()]
            listed = [operations[method] for method in methods if method in operations]
            for method in methods:
                for name, kind in (operations.get(method) or {"inputs": {}})["inputs"].items():
                    _merge(found, name, kind)
                    claims.setdefault(str(method).upper(), set()).add(name)
            if listed and len(listed) == len(methods) and all(operation["complete"] for operation in listed):
                authoritative = True
            read = True
            counts["openapi"] += 1
        if authoritative:
            reasons = []
        seen = (observations or {}).get(key) or {}
        if seen:
            counts["observed"] += 1
        for name, kind in seen.items():
            if name == WHOLE:
                reasons.append(_reason("container_whole", "the code was seen reading a query or body container whole"))
                continue
            if name not in found and read:
                reasons.append(_reason("observed", f"{name} was read while recording, but the analysis didn't find it"))
            _merge(found, name, kind)
            shared.add(name)
            read = True
        if not read:
            continue
        # A route's own parameters come from the path, not the query or body.
        for name in path_parameters(path):
            found.pop(name, None)
        data["inputs"] = found
        data["inputs_complete"] = not reasons
        if reasons:
            data["inputs_incomplete"] = reasons[:_MAX_REASONS]
            counts["incomplete"] += 1
            for reason in reasons:
                code = str(reason.get("code"))
                counts["reasons"][code] = counts["reasons"].get(code, 0) + 1
        else:
            data.pop("inputs_incomplete", None)
            counts["complete"] += 1
        # Keys inside an input, when every reading of it reads constant keys.
        inner = {name: sorted(keys) for name, keys in nested.items() if name in found and name not in flat}
        if inner and not reasons:
            data["inputs_nested"] = inner
        else:
            data.pop("inputs_nested", None)
        # A method's own type for a name, where it isn't the route's (a POST's integer next to a
        # GET's text): engines that read inputs_types_by_method check it for that method alone.
        own_types: Dict[str, Dict[str, str]] = {}
        for method, types in declared_types.items():
            for name, kind in types.items():
                if kind != "text" and name in found and found[name] != kind:
                    own_types.setdefault(str(method).upper(), {})[name] = kind
        if own_types:
            data["inputs_types_by_method"] = own_types
        else:
            data.pop("inputs_types_by_method", None)
        by_method = _by_method(found, data.get("methods") or (), claims, shared) if not reasons else None
        if by_method:
            data["inputs_by_method"] = by_method
        else:
            data.pop("inputs_by_method", None)
    return counts


def _by_method(found: Mapping[str, str], methods: Iterable[str], claims: Mapping[str, set], shared: set) -> Optional[dict]:
    """Each method's names, when the route's complete list is read by different methods' code: a
    method takes the names its own code reads, and every name no method's code claims. None when
    every method reads them all."""
    methods = sorted({str(method).upper() for method in methods})
    if len(methods) < 2 or not claims:
        return None
    claimed = set().union(*claims.values())
    by_method = {method: sorted(name for name in found if name in claims.get(method, ()) or name in shared or name not in claimed)
                 for method in methods}
    return by_method if any(len(names) < len(found) for names in by_method.values()) else None


def describe(counts: Mapping[str, Any]) -> str:
    """One line for init's summary: how many routes have complete inputs, and the top reasons."""
    total = int(counts.get("complete", 0)) + int(counts.get("incomplete", 0))
    if not total:
        return ""
    text = f"complete inputs for {counts.get('complete', 0)} of {total} routes"
    reasons = sorted((counts.get("reasons") or {}).items(), key=lambda item: (-item[1], item[0]))
    if reasons:
        text += "; incomplete because of " + ", ".join(f"{code} ({count})" for code, count in reasons[:4])
    return text


def sources_from_environment(root: str = ".") -> Dict[str, Any]:
    """The declaration file, OpenAPI spec and recorded inputs init should read, where present.

    AIWAF_INPUTS_FILE (default .aiwaf/inputs.json), AIWAF_OPENAPI_SPEC (no default) and
    AIWAF_OBSERVED_INPUTS (default .aiwaf/observed_inputs.jsonl, which the recorder writes).
    """
    import os

    base = Path(root)
    sources: Dict[str, Any] = {}
    declarations = os.environ.get("AIWAF_INPUTS_FILE") or str(base / ".aiwaf" / "inputs.json")
    if Path(declarations).is_file():
        sources["declarations"] = load_declarations(declarations)
    spec = os.environ.get("AIWAF_OPENAPI_SPEC")
    if spec:
        sources["openapi"] = openapi_inputs(load_spec(spec))
    observed = os.environ.get("AIWAF_OBSERVED_INPUTS") or str(base / ".aiwaf" / "observed_inputs.jsonl")
    if Path(observed).is_file():
        sources["observations"] = load_observations(observed)
    return sources


__all__ = [
    "inputs", "declared_on", "class_fields", "finalize_without_code", "resolve_class", "django_inputs", "fastapi_inputs", "canonical",
    "openapi_inputs", "load_spec", "load_observations", "load_declarations", "finalize", "describe",
    "sources_from_environment", "WHOLE",
]

