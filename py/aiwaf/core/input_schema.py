"""What each view reads from the query and body, with types, for the route manifest.

``init`` saves these as each route's ``inputs`` (name to int, float, bool, uuid or text) and
``inputs_complete``: true only when every read was understood, so the request can't carry an
input the view uses that isn't listed. Requests are then reviewed against them: see
``AIWAF_ROUTE_SCHEMA_MODE`` in route_risks.

Anything not understood makes the list incomplete rather than guessed: the request passed to a
helper, a query or body container used whole (``Form(request.POST)`` with a form class that
isn't found), a key that isn't a constant, or a class view that inherits behaviour.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

# Request attributes that hold the query or body. Called ones, as get_json(), are read by calling.
_CONTAINERS = {"GET", "POST", "args", "form", "json", "values", "data", "query_params", "body", "get_json", "get_data"}
_CALLED = {"get_json", "get_data", "json", "form", "body"}
# Request attributes that read the query string or body some other way.
_RAW = {"query_string", "full_path", "stream", "get_full_path", "receive", "content"}
_REQUEST_NAMES = {"request", "req"}
_REQUEST_ANNOTATIONS = {"HttpRequest", "Request", "WSGIRequest"}
# Calls that take the request and read nothing from its query or body.
_SAFE_REQUEST_CALLS = {
    "render", "render_to_response", "render_to_string", "TemplateResponse", "redirect", "login", "logout", "authenticate",
    "add_message", "debug", "info", "success", "warning", "error", "get_token", "csrf", "get_current_site",
}
_CONVERTERS = {"int": "int", "float": "float", "Decimal": "float", "bool": "bool", "UUID": "uuid", "uuid.UUID": "uuid"}
_ANNOTATIONS = {
    "int": "int", "float": "float", "Decimal": "float", "bool": "bool", "UUID": "uuid", "conint": "int", "confloat": "float",
    "PositiveInt": "int", "NegativeInt": "int", "NonNegativeInt": "int", "StrictInt": "int", "StrictFloat": "float",
    "StrictBool": "bool", "UUID4": "uuid", "UUID1": "uuid",
}
_FORM_FIELDS = {
    "IntegerField": "int", "PositiveIntegerField": "int", "FloatField": "float", "DecimalField": "float",
    "BooleanField": "bool", "NullBooleanField": "bool", "UUIDField": "uuid",
}
# Classes whose fields are all declared on them.
_DECLARED_BASES = {"Form", "Serializer", "BaseModel", "Schema", "TypedDict"}
# Class views whose handlers read nothing the class doesn't show.
_PLAIN_VIEWS = {"View", "APIView", "MethodView", "HTTPEndpoint"}
_ROUTE_DECORATORS = {"api_route", "delete", "get", "patch", "post", "put", "route"}
_NON_INPUT_ANNOTATIONS = {"AsyncSession", "BackgroundTasks", "Connection", "Request", "Response", "Session", "WebSocket"}
_NON_INPUT_PARAMETERS = {"self", "cls", "request", "req", "db", "session", "response", "background_tasks"}


def _dotted(node: Optional[ast.AST]) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _last(node: Optional[ast.AST]) -> str:
    return _dotted(node).split(".")[-1]


def _annotation_type(node: Optional[ast.AST]) -> str:
    """int, float, bool or uuid for a scalar annotation, through Optional and lists; else text."""
    if node is None:
        return "text"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _annotation_type(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return "text"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        sides = [side for side in (node.left, node.right) if _dotted(side) != "None" and not (
            isinstance(side, ast.Constant) and side.value is None)]
        kinds = {_annotation_type(side) for side in sides}
        return kinds.pop() if len(kinds) == 1 else "text"
    if isinstance(node, ast.Subscript):
        base = _last(node.value)
        inner = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base in {"Optional", "Annotated", "List", "list", "Set", "set", "Sequence", "Tuple", "tuple"}:
            return _annotation_type(inner[0])
        return "text"
    if isinstance(node, ast.Call):
        return _ANNOTATIONS.get(_last(node.func), "text")
    return _ANNOTATIONS.get(_last(node), "text")


def _merge(inputs: Dict[str, str], name: str, kind: str) -> None:
    # A name read as two types, or as text anywhere, is text: the weaker reading wins.
    inputs[name] = kind if inputs.get(name, kind) == kind else "text"


class _Classes:
    """Classes in a module whose fields can be read: forms, serializers and models."""

    def __init__(self, tree: ast.Module):
        self.nodes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

    def fields(self, name: str, seen: Tuple[str, ...] = ()) -> Optional[Dict[str, str]]:
        """A class's fields and their types, or None when they can't all be known."""
        node = self.nodes.get(name)
        if node is None or name in seen or len(seen) > 8:
            return None
        fields: Dict[str, str] = {}
        for base in node.bases:
            base_name = _last(base)
            if base_name in _DECLARED_BASES:
                continue
            inherited = self.fields(base_name, (*seen, name))
            if inherited is None:
                return None
            fields.update(inherited)
        for item in node.body:
            if isinstance(item, ast.ClassDef) and item.name in {"Meta", "Config"} and any(
                    isinstance(entry, ast.Assign) and any(_dotted(t) in {"model", "fields", "extra"} for t in entry.targets)
                    for entry in item.body):
                return None  # ModelForm and ModelSerializer fields, or extra fields, come from elsewhere.
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fields[item.target.id] = _annotation_type(item.annotation)
            elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call) and _last(item.value.func).endswith("Field"):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        fields[target.id] = _FORM_FIELDS.get(_last(item.value.func), "text")
        return fields

    def model(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """The class an annotation names when it's a model in this module."""
        if isinstance(annotation, ast.Subscript) and _last(annotation.value) in {"Optional", "Annotated"}:
            inner = annotation.slice.elts[0] if isinstance(annotation.slice, ast.Tuple) else annotation.slice
            return self.model(inner)
        name = _last(annotation)
        return name if name in self.nodes else None


class _Reads:
    """Walks one function, or one class's methods, for what it reads from the query and body."""

    def __init__(self, classes: _Classes, flask_request: bool, functions: Optional[Dict[str, ast.AST]] = None):
        self.classes = classes
        self.flask_request = flask_request
        self.inputs: Dict[str, str] = {}
        self.complete = True
        # Module functions the request can be passed to and followed into, and, for a class
        # view read whole, its own methods and whether its base only routes the request.
        self.functions = functions or {}
        self.methods: set = set()
        self.plain_class = False
        self.active: set = set()
        # Module-level string constants, so a key held in one, as request.GET[PAGE_PARAM], is known.
        self.constants: Dict[str, str] = {}

    def request_names(self, node) -> set:
        names = {"request"} if self.flask_request else set()
        for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if parameter.arg in _REQUEST_NAMES or _last(parameter.annotation) in _REQUEST_ANNOTATIONS:
                names.add(parameter.arg)
        return names

    def walk(self, node, requests: set) -> None:
        parents: Dict[int, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        aliases: Dict[str, ast.AST] = {}

        def is_request(expr) -> bool:
            return (isinstance(expr, ast.Name) and expr.id in requests) or _dotted(expr) == "self.request"

        def up(expr):
            parent = parents.get(id(expr))
            # Through await, as in await request.json().
            while isinstance(parent, ast.Await):
                expr, parent = parent, parents.get(id(parent))
            return expr, parent

        def converted(expr) -> str:
            _, parent = up(expr)
            if isinstance(parent, ast.Call) and parent.args and parent.args[0] is expr:
                return _CONVERTERS.get(_dotted(parent.func), "text")
            return "text"

        def container(expr) -> None:
            """A query or body container, as request.GET, and what's done with it."""
            expr, parent = up(expr)
            if isinstance(parent, ast.Subscript) and parent.value is expr:
                key = self.key(parent.slice)
                if key is not None:
                    _merge(self.inputs, key, converted(parent))
                else:
                    self.complete = False
                return
            if isinstance(parent, ast.Attribute) and parent.value is expr:
                call = parents.get(id(parent))
                if parent.attr in {"get", "getlist", "get_list", "getall"} and isinstance(call, ast.Call) and call.func is parent:
                    key = self.key(call.args[0]) if call.args else None
                    if key is not None:
                        kind = next((_CONVERTERS.get(_dotted(k.value), "text") for k in call.keywords if k.arg == "type"), None)
                        _merge(self.inputs, key, kind or converted(call))
                    else:
                        self.complete = False
                    return
                self.complete = False
                return
            if isinstance(parent, ast.Compare) and expr is not parent.left:
                return  # "q" in request.GET reads nothing.
            if isinstance(parent, (ast.Assign, ast.AnnAssign)) and getattr(parent, "value", None) is expr:
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    aliases[targets[0].id] = expr
                    return
            if isinstance(parent, ast.Call) and _last(parent.func) == "loads" and parent.args and parent.args[0] is expr:
                container(parent)  # json.loads(request.body)
                return
            if isinstance(parent, (ast.Call, ast.keyword)):
                call = parents.get(id(parent)) if isinstance(parent, ast.keyword) else parent
                fields = self.classes.fields(_last(call.func)) if isinstance(call, ast.Call) else None
                if fields is not None:
                    for name, kind in fields.items():
                        _merge(self.inputs, name, kind)
                    return
            self.complete = False

        for expr in ast.walk(node):
            if isinstance(expr, ast.Attribute) and is_request(expr.value):
                if expr.attr in _CONTAINERS:
                    _, parent = up(expr)
                    if expr.attr in _CALLED and isinstance(parent, ast.Call) and parent.func is expr:
                        container(parent)
                    elif expr.attr not in {"get_json", "get_data"}:
                        container(expr)
                    else:
                        self.complete = False
                elif expr.attr in _RAW:
                    self.complete = False
            elif isinstance(expr, ast.Name) and expr.id in requests and isinstance(expr.ctx, ast.Load):
                parent = parents.get(id(expr))
                if isinstance(parent, ast.Attribute) and parent.value is expr:
                    continue
                call = parents.get(id(parent)) if isinstance(parent, ast.keyword) else parent
                if not (isinstance(call, ast.Call) and (_last(call.func) in _SAFE_REQUEST_CALLS or self.follow(call, expr, parent))):
                    self.complete = False  # The request goes somewhere that may read it.
        # What's read from an alias, as data = request.json, then data["q"].
        for expr in ast.walk(node):
            if isinstance(expr, ast.Name) and expr.id in aliases and isinstance(expr.ctx, ast.Load):
                container(expr)

    def key(self, node: ast.AST) -> Optional[str]:
        """A constant key, written out or held in a module-level string constant, or None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return self.constants.get(node.id) if isinstance(node, ast.Name) else None

    def follow(self, call: ast.Call, request: ast.AST, holder: ast.AST) -> bool:
        """Whether a call the request is passed to is one these reads cover, following it if needed."""
        callee = _dotted(call.func)
        method = callee.split(".")[-1]
        # A class view is read whole, so its own methods are covered; a plain View's dispatch and
        # setup only route the request to them.
        if callee.startswith("self.") and callee.count(".") == 1 and method in self.methods:
            return True
        if callee.startswith("super.") and self.plain_class and method in {"dispatch", "setup", "get", "post", "put", "patch", "delete", "head", "options"}:
            return True
        target = self.functions.get(callee)
        if target is None or id(target) in self.active or len(self.active) >= 3:
            return False
        # The parameter the request lands in: by position, or by keyword.
        parameters = [*target.args.posonlyargs, *target.args.args]
        name = None
        if isinstance(holder, ast.keyword):
            name = holder.arg
        elif request in call.args:
            index = call.args.index(request)
            name = parameters[index].arg if index < len(parameters) else None
        if not name:
            return False
        self.active.add(id(target))
        try:
            self.walk(target, {name})
        finally:
            self.active.discard(id(target))
        return True

    def parameters(self, node) -> None:
        """FastAPI and similar route parameters: each one is an input, a model its fields."""
        defaults = dict(zip(reversed(node.args.args), reversed(node.args.defaults)))
        defaults.update({arg: default for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default})
        models = []
        for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            default = defaults.get(parameter)
            annotation = parameter.annotation
            if parameter.arg in _NON_INPUT_PARAMETERS or _last(annotation) in _NON_INPUT_ANNOTATIONS:
                continue
            if (isinstance(default, ast.Call) and _last(default.func) in {"Depends", "Security"}) or "Depends" in ast.dump(
                    annotation or ast.Constant(None)):
                self.complete = False  # A dependency may read anything.
                continue
            name = parameter.arg
            if isinstance(default, ast.Call):
                alias = next((k.value.value for k in default.keywords if k.arg == "alias" and isinstance(k.value, ast.Constant)), None)
                name = alias if isinstance(alias, str) else name
            model = self.classes.model(annotation)
            if model is not None:
                models.append((name, model))
            elif _annotation_type(annotation) == "text" and _last(annotation) not in {"", "str", "Optional", "List", "list"} \
                    and _last(annotation) not in _ANNOTATIONS:
                self.complete = False  # A class from elsewhere: its fields aren't known.
            else:
                _merge(self.inputs, name, _annotation_type(annotation))
        # One model is the whole body; several are each under their parameter's name.
        for name, model in models:
            fields = self.classes.fields(model)
            if fields is None:
                self.complete = False
            elif len(models) == 1:
                for field, kind in fields.items():
                    _merge(self.inputs, field, kind)
            else:
                _merge(self.inputs, name, "text")


def _uses_request(node, requests: set) -> bool:
    """Whether a function takes or reads the request, as a view does."""
    return any((isinstance(item, ast.Name) and item.id in requests) or _dotted(item) == "self.request" for item in ast.walk(node))


def _route_decorated(node) -> bool:
    return any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr in _ROUTE_DECORATORS
               for item in node.decorator_list)


def _flask_request(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"flask", "quart"}
               and any(alias.name == "request" for alias in node.names) for node in ast.walk(tree))


def module_inputs(tree: ast.Module, filename: str) -> List[dict]:
    """Each function's (or class view's) inputs: {function, file, inputs, complete}."""
    classes = _Classes(tree)
    flask_request = _flask_request(tree)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    constants = {
        target.id: node.value.value
        for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets if isinstance(target, ast.Name)
    }
    results: List[dict] = []
    owned = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not methods:
            continue
        owned.update(id(method) for method in methods)
        # Only class views: ones with a method that takes or reads the request.
        if not any(_uses_request(method, _Reads(classes, False).request_names(method)) for method in methods):
            continue
        # A class view is read as a whole: its handlers and helpers share self.request.
        reads = _Reads(classes, flask_request, functions)
        reads.constants = constants
        reads.methods = {method.name for method in methods}
        reads.plain_class = bool(node.bases) and all(_last(base) in _PLAIN_VIEWS for base in node.bases)
        if not reads.plain_class:
            reads.complete = False
        for method in methods:
            reads.walk(method, reads.request_names(method))
        for method in methods:
            results.append({"function": f"{node.name}.{method.name}", "file": filename,
                            "inputs": dict(reads.inputs), "complete": reads.complete})
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(node) not in owned:
            reads = _Reads(classes, flask_request, functions)
            reads.constants = constants
            reads.active.add(id(node))
            requests = reads.request_names(node)
            # Only what can be a view: a route-decorated function, or one that takes the request.
            routed = _route_decorated(node)
            if not routed and not _uses_request(node, requests):
                continue
            if routed:
                reads.parameters(node)
            reads.walk(node, requests)
            results.append({"function": node.name, "file": filename, "inputs": reads.inputs, "complete": reads.complete})
    return results


def source_inputs(source: str, filename: str = "<source>") -> List[dict]:
    return module_inputs(ast.parse(source, filename=filename), filename)


def merge_into(routes: dict, path: str, entry: dict) -> None:
    """Adds one function's inputs to a route; a route served by several takes them all."""
    data = routes[path]
    inputs = data.setdefault("inputs", {})
    for name, kind in entry["inputs"].items():
        _merge(inputs, name, kind)
    data["inputs_complete"] = bool(data.get("inputs_complete", True)) and bool(entry["complete"])
