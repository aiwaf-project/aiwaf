"""What each view reads from the query and body, with types, for the route manifest.

``init`` saves these as each route's ``inputs`` (name to int, float, bool, uuid or text) and
``inputs_complete``: true only when every read was understood, so the request can't carry an
input the view uses that isn't listed. Requests are then reviewed against them: see
``AIWAF_ROUTE_SCHEMA_MODE`` in route_risks.

Anything not understood makes the list incomplete rather than guessed, and says why in
``inputs_incomplete``: the request passed to a helper that isn't followed, a query or body
container used whole, a key that isn't a constant, a raw body read, a class view that inherits
behaviour, a dependency, or a form, serializer or model whose fields can't be read. Forms,
serializers, models and helpers are followed into the other modules of the scanned project,
through imports.

Reason codes (stable; other tools and the docs refer to them):

- ``helper``: the request is passed to code that isn't followed.
- ``container_whole``: a query or body container is used whole, or read some other way.
- ``dynamic_key``: an input is read with a key that isn't a constant.
- ``raw_body``: the raw query string or body is read.
- ``class_view``: a class view inherits handlers from a base class that isn't read.
- ``dependency``: a FastAPI dependency may read anything.
- ``external_class``: a form, serializer or model that isn't in the scanned project.
- ``model_fields``: a ModelForm or ModelSerializer, or a model with extra fields, whose fields
  come from its model.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Request attributes that hold the query or body. Called ones, as get_json(), are read by calling.
_CONTAINERS = {"GET", "POST", "args", "form", "json", "values", "data", "query_params", "body", "get_json", "get_data",
               "files", "FILES"}
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
# Casts that reject what isn't their type. bool() isn't one: bool("no") is True, so code that
# calls it (or Flask's type=bool) accepts any text, and a bool restriction would block values it
# handles. A declared bool (a Pydantic field, a form's BooleanField) is a restriction.
# Container labels by location: Django's request.GET and Flask's request.args are the query
# string; POST, form, json, get_json(), data and body are the body. request.values is both.
_QUERY_LABEL = re.compile(r"\.(?:GET|args|query_params)\b")
_BODY_LABEL = re.compile(r"\.(?:POST|FILES|files|form|json|get_json|data|body|get_data)\b")
# A class view's handler methods, one per HTTP method.
_HTTP_HANDLERS = {"get", "post", "put", "patch", "delete", "head", "options"}
_CONVERTERS = {"int": "int", "float": "float", "Decimal": "float", "UUID": "uuid", "uuid.UUID": "uuid"}
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
# webargs' use_args and use_kwargs, and flask-smorest's Blueprint.arguments (which uses them).
_ARGUMENT_DECORATORS = {"use_args", "use_kwargs", "arguments"}
_MARSHMALLOW_FIELDS = {
    "Int": "int", "Integer": "int", "Float": "float", "Decimal": "float", "Number": "float",
    "Bool": "bool", "Boolean": "bool", "UUID": "uuid",
}
_UNKNOWN = {"RAISE": "raise", "EXCLUDE": "exclude", "INCLUDE": "include"}
# Class views whose handlers read nothing the class doesn't show.
_PLAIN_VIEWS = {"View", "APIView", "MethodView", "HTTPEndpoint"}
_ROUTE_DECORATORS = {"api_route", "delete", "get", "patch", "post", "put", "route"}
_OPEN_ANNOTATIONS = {"dict", "Dict", "Mapping", "MutableMapping", "Any", "JsonValue", "Json"}
_NON_INPUT_ANNOTATIONS = {"AsyncSession", "BackgroundTasks", "Connection", "Request", "Response", "Session", "WebSocket"}
_NON_INPUT_PARAMETERS = {"self", "cls", "request", "req", "db", "session", "response", "background_tasks"}
# How far imports and helpers are followed.
_MAX_DEPTH = 6

REASON_CODES = (
    "helper", "container_whole", "dynamic_key", "raw_body", "class_view", "dependency", "external_class", "model_fields",
    "unread_view",
)


def _pydantic_config(node: ast.ClassDef) -> Dict[str, Any]:
    """What a Pydantic model's model_config = ConfigDict(...) (or a class Config) says about names."""
    found: Dict[str, Any] = {}
    for item in node.body:
        if isinstance(item, ast.Assign) and any(_dotted(target) == "model_config" for target in item.targets):
            call = item.value
            if isinstance(call, ast.Call):
                for keyword in call.keywords:
                    if keyword.arg in {"populate_by_name", "validate_by_name", "alias_generator"}:
                        found[keyword.arg] = keyword.value.value if isinstance(keyword.value, ast.Constant) else True
            elif isinstance(call, ast.Dict):
                for key, value in zip(call.keys, call.values):
                    if isinstance(key, ast.Constant) and key.value in {"populate_by_name", "validate_by_name", "alias_generator"}:
                        found[key.value] = value.value if isinstance(value, ast.Constant) else True
        elif isinstance(item, ast.ClassDef) and item.name == "Config":
            for entry in item.body:
                if isinstance(entry, ast.Assign):
                    for target in entry.targets:
                        if _dotted(target) in {"populate_by_name", "allow_population_by_field_name", "alias_generator"}:
                            name = "populate_by_name" if _dotted(target) != "alias_generator" else "alias_generator"
                            found[name] = entry.value.value if isinstance(entry.value, ast.Constant) else True
    return found


def _pydantic_keys(attribute: str, value: Optional[ast.AST], config: Mapping[str, Any]) -> Optional[List[str]]:
    """The request keys a Pydantic field takes, from its Field(alias=...), or None when the
    source doesn't spell them out (an alias held in a variable, AliasPath, AliasChoices of paths)."""
    names: List[str] = []
    if isinstance(value, ast.Call) and _last(value.func) == "Field":
        options = {keyword.arg: keyword.value for keyword in value.keywords if keyword.arg}
        validation = options.get("validation_alias")
        alias = options.get("alias")
        if validation is not None:
            if isinstance(validation, ast.Constant) and isinstance(validation.value, str):
                names.append(validation.value)
            elif isinstance(validation, ast.Call) and _last(validation.func) == "AliasChoices" and all(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in validation.args):
                names.extend(arg.value for arg in validation.args)
            else:
                return None
        elif alias is not None:
            if not (isinstance(alias, ast.Constant) and isinstance(alias.value, str)):
                return None
            names.append(alias.value)
    if not names:
        return [attribute]
    if config.get("populate_by_name") or config.get("validate_by_name"):
        names.append(attribute)
    return list(dict.fromkeys(names))


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


class Project:
    """The scanned modules, so a form, model or helper imported from another one can be read."""

    def __init__(self) -> None:
        self._modules: Dict[str, Optional[_Module]] = {}

    def add(self, tree: ast.Module, filename: str, name: str = "", package: bool = False) -> "_Module":
        module = _Module(self, tree, filename, name, package)
        # Known by every dotted suffix too (src/app/forms.py as app.forms or forms), since imports
        # name modules from wherever the project's root is; an ambiguous name isn't followed.
        parts = name.split(".") if name else []
        for start in range(len(parts)):
            key = ".".join(parts[start:])
            self._modules[key] = None if key in self._modules and self._modules[key] is not module else module
        return module

    def get(self, name: str) -> Optional["_Module"]:
        return self._modules.get(name) if name else None


class _Module:
    """One parsed module: its classes, functions, string constants and imports."""

    def __init__(self, project: Project, tree: ast.Module, filename: str, name: str, package: bool) -> None:
        self.project = project
        self.tree = tree
        self.filename = filename
        self.name = name
        self.classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        self.functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.constants = {
            target.id: node.value.value
            for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in node.targets if isinstance(target, ast.Name)
        }
        # Local name to (module, attribute): attribute None for "import a.b as c".
        self.imports: Dict[str, Tuple[str, Optional[str]]] = {}
        package_parts = name.split(".") if name else []
        if not package and package_parts:
            package_parts = package_parts[:-1]
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.imports[alias.asname] = (alias.name, None)
                    else:
                        head = alias.name.split(".")[0]
                        self.imports.setdefault(head, (head, None))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[:len(package_parts) - (node.level - 1)] if node.level > 1 else package_parts
                    absolute = ".".join([*base, *((node.module or "").split(".") if node.module else [])])
                else:
                    absolute = node.module or ""
                for alias in node.names:
                    if alias.name != "*":
                        self.imports[alias.asname or alias.name] = (absolute, alias.name)

    def lookup(self, kind: str, dotted: str, depth: int = 0) -> Optional[Tuple["_Module", ast.AST]]:
        """The class or function a dotted name refers to here, through imports, or None."""
        if not dotted or depth > _MAX_DEPTH:
            return None
        table = self.classes if kind == "class" else self.functions
        head, _, rest = dotted.partition(".")
        if not rest and head in table:
            return self, table[head]
        target = self.imports.get(head)
        if target is None:
            return None
        module_name, attribute = target
        if attribute is None:
            # import a.b as c, then c.X; or import a, then a.b.X.
            parts = rest.split(".") if rest else []
            for cut in range(len(parts) - 1, -1, -1):
                module = self.project.get(".".join([module_name, *parts[:cut]]))
                if module is not None:
                    return module.lookup(kind, ".".join(parts[cut:]), depth + 1)
            return None
        # from a import b: b may be a submodule, or a name defined or re-exported in a.
        submodule = self.project.get(f"{module_name}.{attribute}" if module_name else attribute)
        if submodule is not None and rest:
            return submodule.lookup(kind, rest, depth + 1)
        module = self.project.get(module_name)
        if module is None or module is self and attribute == head and not rest:
            return None
        return module.lookup(kind, attribute + (f".{rest}" if rest else ""), depth + 1)

    def fields(self, name: str, seen: Tuple = ()) -> Tuple[Optional[Dict[str, str]], Optional[Tuple[str, str]]]:
        """A form, serializer or model's fields and types, or None and the (code, detail) why not."""
        found = self.lookup("class", name)
        if found is None:
            return None, ("external_class", name)
        module, node = found
        key = (module.name, module.filename, node.name)
        if key in seen or len(seen) > 8:
            return None, ("external_class", name)
        for item in node.body:
            if isinstance(item, ast.ClassDef) and item.name in {"Meta", "Config"} and any(
                    isinstance(entry, ast.Assign) and any(_dotted(t) in {"model", "fields", "extra"} for t in entry.targets)
                    for entry in item.body):
                return None, ("model_fields", node.name)  # ModelForm and ModelSerializer fields come from the model.
        fields: Dict[str, str] = {}
        for base in node.bases:
            if _last(base) in _DECLARED_BASES:
                continue
            inherited, reason = module.fields(_dotted(base), (*seen, key))
            if inherited is None:
                return None, reason
            fields.update(inherited)
        # A Pydantic model takes each field under its alias, not its attribute name, unless it
        # populates by name; an alias generator renames them in ways the source doesn't show.
        config = _pydantic_config(node)
        if config.get("alias_generator"):
            return None, ("model_fields", node.name)
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                kind = _annotation_type(item.annotation)
                names = _pydantic_keys(item.target.id, item.value, config)
                if names is None:
                    return None, ("model_fields", node.name)
                for accepted in names:
                    fields[accepted] = kind
            elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call) and _last(item.value.func).endswith("Field"):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        fields[target.id] = _FORM_FIELDS.get(_last(item.value.func), "text")
        return fields, None

    def schema_fields(self, name: str, seen: Tuple = ()) -> Tuple[Optional[Dict[str, str]], str, Optional[Tuple[str, str]]]:
        """A marshmallow Schema's loadable fields (by data_key), its Meta unknown setting, and why not."""
        found = self.lookup("class", name)
        if found is None:
            return None, "raise", ("external_class", name)
        module, node = found
        key = (module.name, module.filename, node.name)
        if key in seen or len(seen) > 8:
            return None, "raise", ("external_class", name)
        fields: Dict[str, str] = {}
        unknown = "raise"
        for base in node.bases:
            if _last(base) in _DECLARED_BASES:
                continue
            inherited, unknown, reason = module.schema_fields(_dotted(base), (*seen, key))
            if inherited is None:
                return None, unknown, reason
            fields.update(inherited)
        for item in node.body:
            if isinstance(item, ast.ClassDef) and item.name == "Meta":
                for entry in item.body:
                    if isinstance(entry, ast.Assign) and any(_dotted(t) == "unknown" for t in entry.targets):
                        unknown = _unknown_setting(entry.value) or unknown
            elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        field = _marshmallow_field(target.id, item.value)
                        if field:
                            fields[field[0]] = field[1]
        return fields, unknown, None

    def model(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """The dotted class name an annotation names when it's a class in the scanned project."""
        if isinstance(annotation, ast.Subscript) and _last(annotation.value) in {"Optional", "Annotated"}:
            inner = annotation.slice.elts[0] if isinstance(annotation.slice, ast.Tuple) else annotation.slice
            return self.model(inner)
        name = _dotted(annotation)
        return name if name and self.lookup("class", name) is not None else None


def _unknown_setting(node: Optional[ast.AST]) -> Optional[str]:
    """RAISE, EXCLUDE or INCLUDE (marshmallow's constants, or their string values), lowercased."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _UNKNOWN.values():
        return node.value
    return _UNKNOWN.get(_last(node)) if isinstance(node, (ast.Name, ast.Attribute)) else None


def _keyword(call: ast.Call, name: str) -> Optional[ast.AST]:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _marshmallow_field(name: str, call: ast.Call) -> Optional[Tuple[str, str]]:
    """A marshmallow field's loaded name (its data_key) and type, or None for a dump_only one."""
    dump_only = _keyword(call, "dump_only")
    if isinstance(dump_only, ast.Constant) and dump_only.value is True:
        return None
    data_key = _keyword(call, "data_key")
    if isinstance(data_key, ast.Constant) and isinstance(data_key.value, str):
        name = data_key.value
    kind = _last(call.func)
    if kind == "List" and call.args and isinstance(call.args[0], ast.Call):
        return name, _MARSHMALLOW_FIELDS.get(_last(call.args[0].func), "text")
    return name, _MARSHMALLOW_FIELDS.get(kind, "text")


class _Reads:
    """Walks one function, or one class's methods, for what it reads from the query and body."""

    def __init__(self, module: _Module, flask_request: bool):
        self.module = module
        self.flask_request = flask_request
        self.inputs: Dict[str, str] = {}
        self.reasons: List[dict] = []
        # For a class view read whole: its own methods, and whether its base only routes the request.
        self.methods: set = set()
        self.plain_class = False
        self.active: set = set()
        # Locations whose webargs arguments reject any name they don't list.
        self.closed: Dict[str, bool] = {}
        # Keys read inside an input, as name in data["user"]["name"]; and inputs used any other
        # way, whose keys aren't known.
        self.nested: Dict[str, set] = {}
        self.flat: set = set()
        # Each name's type where it was read ({"query": {...}, "body": {...}}), and names read where
        # the location isn't known (request.values, a form's fields).
        self.locations: Dict[str, Dict[str, str]] = {}
        self.unlocated: set = set()
        # Where each named read is, and the branches on request.method around them, so a
        # function view's if request.method == "POST": reads can be POST's alone. A helper's
        # reads count where it's called (call_site).
        self.sites: List[dict] = []
        self.regions: List[tuple] = []
        self.call_site: Optional[tuple] = None

    def locate(self, label: str, name: str, kind: str, at: Optional[ast.AST] = None) -> None:
        """Records a named read at the location its container's label names."""
        where = "body" if _BODY_LABEL.search(label) else "query" if _QUERY_LABEL.search(label) else None
        if where is None:
            self.unlocated.add(name)
        else:
            _merge(self.locations.setdefault(where, {}), name, kind)
        position = self.call_site or ((at.lineno, at.col_offset) if hasattr(at, "lineno") else None)
        if position is not None:
            self.sites.append({"name": name, "kind": kind, "where": where, "at": position})

    def method_reads(self) -> Optional[List[dict]]:
        """Every read with the methods it's for, when any is inside a branch on request.method."""
        if not self.regions:
            return None
        constrained = False
        reads = []
        for site in self.sites:
            only: Optional[set] = None
            excluded: set = set()
            for start, end, test in self.regions:
                if not start <= site["at"] < end:
                    continue
                if "only" in test:
                    only = set(test["only"]) if only is None else only & set(test["only"])
                else:
                    excluded |= set(test["except"])
            read = {"name": site["name"], "kind": site["kind"], "where": site["where"]}
            if only is not None:
                read["only"] = sorted(only - excluded)
                constrained = True
            elif excluded:
                read["except"] = sorted(excluded)
                constrained = True
            reads.append(read)
        return reads if constrained else None

    def branches(self, node, is_request) -> None:
        """The parts of node that run only for some methods: if request.method == "POST":,
        its else, and what follows a guard that returns (if request.method != "POST": return)."""
        def method_of(expr) -> bool:
            return isinstance(expr, ast.Attribute) and expr.attr == "method" and is_request(expr.value)

        def values(expr) -> Optional[List[str]]:
            items = expr.elts if isinstance(expr, (ast.Tuple, ast.List, ast.Set)) else [expr]
            found = [item.value.upper() for item in items if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            return found if len(found) == len(items) else None

        def test_of(expr) -> Optional[dict]:
            if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
                left, op, right = expr.left, expr.ops[0], expr.comparators[0]
                if not method_of(left) and method_of(right) and isinstance(op, (ast.Eq, ast.NotEq)):
                    left, right = right, left
                named = values(right) if method_of(left) else None
                if named is None:
                    return None
                if isinstance(op, (ast.Eq, ast.In)):
                    return {"only": named}
                if isinstance(op, (ast.NotEq, ast.NotIn)):
                    return {"except": named}
                return None
            if isinstance(expr, ast.BoolOp):
                parts = [test_of(value) for value in expr.values]
                if any(part is None for part in parts):
                    return None
                if isinstance(expr.op, ast.Or) and all("only" in part for part in parts):
                    return {"only": [method for part in parts for method in part["only"]]}
                if isinstance(expr.op, ast.And) and all("except" in part for part in parts):
                    return {"except": [method for part in parts for method in part["except"]]}
                return None
            if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
                return negate(test_of(expr.operand))
            return None

        def negate(test: Optional[dict]) -> Optional[dict]:
            if test is None:
                return None
            return {"except": test["only"]} if "only" in test else {"only": test["except"]}

        def exits(statements) -> bool:
            last = statements[-1] if statements else None
            if isinstance(last, (ast.Return, ast.Raise)):
                return True
            return isinstance(last, ast.If) and exits(last.body) and exits(last.orelse)

        def span(statements) -> tuple:
            return (statements[0].lineno, statements[0].col_offset), (statements[-1].end_lineno, statements[-1].end_col_offset)

        for holder in ast.walk(node):
            for field in ("body", "orelse", "finalbody"):
                statements = getattr(holder, field, None)
                if not isinstance(statements, list):
                    continue
                for index, statement in enumerate(statements):
                    if not isinstance(statement, ast.If):
                        continue
                    test = test_of(statement.test)
                    if test is None:
                        continue
                    self.regions.append((*span(statement.body), test))
                    if statement.orelse:
                        self.regions.append((*span(statement.orelse), negate(test)))
                    # What follows an if (and its elifs) whose branches return is for the other
                    # methods: if request.method != "POST": return, then POST's code.
                    rest = statements[index + 1:]
                    link, chain = statement, []
                    while rest and isinstance(link, ast.If) and exits(link.body):
                        link_test = test_of(link.test)
                        if link_test is None:
                            chain = []
                            break
                        chain.append(negate(link_test))
                        if not link.orelse:
                            break
                        link = link.orelse[0] if len(link.orelse) == 1 else None
                    if chain and (link is None or not isinstance(link, ast.If) or not link.orelse):
                        for each in chain:
                            self.regions.append((*span(rest), each))

    @property
    def complete(self) -> bool:
        return not self.reasons

    def incomplete(self, code: str, detail: str, node: Optional[ast.AST] = None, **extra) -> None:
        reason = {"code": code, "detail": detail}
        line = getattr(node, "lineno", None)
        if line:
            reason["line"] = line
        if self.module.filename:
            reason["file"] = self.module.filename
        reason.update(extra)
        if reason not in self.reasons:
            self.reasons.append(reason)

    def arguments(self, node) -> bool:
        """Names from webargs' @use_args/@use_kwargs and flask-smorest's @blp.arguments on a view.

        Each loads its schema's fields at its location (json by default). The view only receives
        those, so they're inputs; a location that INCLUDEs unknown names passes others through.
        Whether any were found.
        """
        found = False
        for decorator in getattr(node, "decorator_list", ()):
            if not isinstance(decorator, ast.Call) or _last(decorator.func) not in _ARGUMENT_DECORATORS:
                continue
            if _last(decorator.func) == "arguments" and not isinstance(decorator.func, ast.Attribute):
                continue
            found = True
            argmap = decorator.args[0] if decorator.args else (_keyword(decorator, "argmap") or _keyword(decorator, "schema"))
            location_node = _keyword(decorator, "location")
            location = location_node.value if isinstance(location_node, ast.Constant) else "json"
            segment = "query" if location in {"query", "querystring"} else "body" if location in {"json", "form", "json_or_form"} else None
            if segment is None:
                continue
            fields: Optional[Dict[str, str]] = None
            schema_unknown = "raise"
            if isinstance(argmap, ast.Dict):
                fields = {}
                for key, value in zip(argmap.keys, argmap.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Call):
                        field = _marshmallow_field(key.value, value)
                        if field:
                            fields[field[0]] = field[1]
                    elif not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        self.incomplete("dynamic_key", f"a {location} argument's name isn't a constant", decorator)
            else:
                call = argmap if isinstance(argmap, ast.Call) else None
                target = call.func if call is not None else argmap
                fields, schema_unknown, reason = self.module.schema_fields(_dotted(target))
                if call is not None:
                    schema_unknown = _unknown_setting(_keyword(call, "unknown")) or schema_unknown
                if fields is None:
                    self.class_reason(reason, decorator)
                    continue
            for name, kind in fields.items():
                _merge(self.inputs, name, kind)
                _merge(self.locations.setdefault(segment, {}), name, kind)
            unknown_node = _keyword(decorator, "unknown")
            unknown = _unknown_setting(unknown_node) if unknown_node is not None else ("exclude" if segment == "query" else schema_unknown)
            if unknown_node is not None and isinstance(unknown_node, ast.Constant) and unknown_node.value is None:
                unknown = schema_unknown
            if unknown == "include":
                self.incomplete("container_whole", f"the {location} arguments pass names they don't list through to the view", decorator)
            self.closed[segment] = self.closed.get(segment, True) and unknown == "raise"
        return found

    def close(self) -> None:
        """When both the query and the body reject other names, only a raw body read stays unknown."""
        if self.closed.get("query") and self.closed.get("body"):
            self.reasons = [reason for reason in self.reasons if reason["code"] == "raw_body"]

    def class_reason(self, reason: Tuple[str, str], node: Optional[ast.AST]) -> None:
        code, name = reason
        what = "isn't in the scanned project" if code == "external_class" else "takes its fields from a model"
        self.incomplete(code, f"{name} {what}", node, module=self.module.name, name=name)

    def request_names(self, node) -> set:
        names = {"request"} if self.flask_request else set()
        for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if parameter.arg in _REQUEST_NAMES or _last(parameter.annotation) in _REQUEST_ANNOTATIONS:
                names.add(parameter.arg)
        return names

    def walk(self, node, requests: set, containers: Optional[Dict[str, str]] = None) -> None:
        """What node reads from the request (its names in requests), and from query or body
        containers it was given, as a helper's args parameter (containers: name to label)."""
        parents: Dict[int, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        aliases: Dict[str, ast.AST] = {}

        def is_request(expr) -> bool:
            return (isinstance(expr, ast.Name) and expr.id in requests) or _dotted(expr) == "self.request"

        if self.call_site is None:
            self.branches(node, is_request)

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
            # page = request.GET.get("page"), cast later: int(page).
            if isinstance(parent, (ast.Assign, ast.AnnAssign)) and getattr(parent, "value", None) is expr:
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    return cast_later(targets[0].id)
            return "text"

        def checks_only(use) -> bool:
            """A use that only tests the value: if page, not page, page is None, page == ""."""
            parent = parents.get(id(use))
            if isinstance(parent, (ast.If, ast.While, ast.IfExp)) and parent.test is use:
                return True
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                return True
            if isinstance(parent, ast.Compare) and parent.left is use and all(
                    isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)) for op in parent.ops) and all(
                    isinstance(item, ast.Constant) and item.value in (None, "") for item in parent.comparators):
                return True
            return False

        def recast_at(variable: str) -> Optional[Tuple[int, int]]:
            """Where variable = int(variable) replaces the read for the rest of the function: the
            first such assignment that always runs (at the top, or in a try body, not under an if)."""
            for statement in ast.walk(node):
                if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == variable):
                    continue
                value = statement.value
                if not (isinstance(value, ast.Call) and _dotted(value.func) in _CONVERTERS and value.args
                        and isinstance(value.args[0], ast.Name) and value.args[0].id == variable):
                    continue
                current = parents.get(id(statement))
                while current is not None and current is not node and isinstance(current, ast.Try) \
                        and statement in getattr(current, "body", ()):
                    statement, current = current, parents.get(id(current))
                if current is node:
                    end = (getattr(value, "end_lineno", 0) or 0, getattr(value, "end_col_offset", 0) or 0)
                    return end
            return None

        def cast_later(variable: str) -> str:
            """The type a variable holding a read is cast to: every use of it casts it to that one
            type, apart from checks of whether it was sent and uses after an unconditional
            variable = int(variable). Text when any other use does anything else."""
            kinds = set()
            after = recast_at(variable)
            for use in ast.walk(node):
                if not (isinstance(use, ast.Name) and use.id == variable and isinstance(use.ctx, ast.Load)):
                    continue
                parent = parents.get(id(use))
                if isinstance(parent, ast.Call) and parent.args and parent.args[0] is use and _dotted(parent.func) in _CONVERTERS:
                    kinds.add(_CONVERTERS[_dotted(parent.func)])
                elif after is not None and (use.lineno, use.col_offset) > after:
                    continue  # The cast value, not the read.
                elif not checks_only(use):
                    return "text"
            return kinds.pop() if len(kinds) == 1 else "text"

        def label(expr) -> str:
            return _dotted(expr) or "the request body"

        def loop_keys(node) -> Optional[List[str]]:
            """The keys a loop variable takes, as key in for key in ("theme", "lang"), or None."""
            if not isinstance(node, ast.Name):
                return None
            current = parents.get(id(node))
            while current is not None:
                loops = current.generators if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) \
                    else [current] if isinstance(current, ast.For) else []
                for loop in loops:
                    if isinstance(loop.target, ast.Name) and loop.target.id == node.id:
                        values = loop.iter.elts if isinstance(loop.iter, (ast.Tuple, ast.List, ast.Set)) else None
                        keys = [self.key(value) for value in values or ()]
                        return keys if values and all(key is not None for key in keys) else None
                current = parents.get(id(current))
            return None

        def keys(node) -> Optional[List[str]]:
            key = self.key(node)
            return [key] if key is not None else loop_keys(node)

        def key_of_use(use) -> Optional[str]:
            """The constant key a value is read with, as sku in line["sku"] or line.get("sku"), or None."""
            _, parent = up(use)
            if isinstance(parent, ast.Subscript) and parent.value is use:
                return self.key(parent.slice)
            if isinstance(parent, ast.Attribute) and parent.attr == "get":
                call = parents.get(id(parent))
                if isinstance(call, ast.Call) and call.func is parent and call.args:
                    return self.key(call.args[0])
            return None

        def element_keys(read) -> Optional[set]:
            """The keys each element of a list input is read with, when it's looped over
            straightforwardly (for line in lines: line["sku"], [l["qty"] for l in lines]). None when
            the elements are used any other way."""
            _, parent = up(read)
            scopes: list = []
            name = None
            if isinstance(parent, ast.For) and parent.iter is read and isinstance(parent.target, ast.Name):
                name, scopes = parent.target.id, [*parent.body, *parent.orelse]
            elif isinstance(parent, ast.comprehension) and parent.iter is read and isinstance(parent.target, ast.Name):
                owner = parents.get(id(parent))
                name = parent.target.id
                scopes = [*parent.ifs, *(getattr(owner, attribute) for attribute in ("elt", "key", "value") if hasattr(owner, attribute)),
                          *(item for later in getattr(owner, "generators", ()) if later is not parent for item in (later.iter, *later.ifs))]
            if name is None:
                return None
            keys = set()
            for scope in scopes:
                for use in ast.walk(scope):
                    if isinstance(use, ast.Name) and use.id == name and isinstance(use.ctx, ast.Load):
                        key = key_of_use(use)
                        if key is None:
                            return None
                        keys.add(key)
            return keys or None

        def alias_element_keys(variable: str) -> Optional[set]:
            """A list held in a local first, as lines = data["lines"]: every use must loop over it."""
            keys: set = set()
            for use in ast.walk(node):
                if isinstance(use, ast.Name) and use.id == variable and isinstance(use.ctx, ast.Load):
                    found = element_keys(use)
                    if found is None:
                        return None
                    keys |= found
            return keys or None

        def inner(read, key: str) -> None:
            """What's done with an input's value: a constant-key read inside it, or anything else."""
            _, parent = up(read)
            elements = element_keys(read)
            if elements is None and isinstance(parent, ast.Assign) and parent.value is read and len(parent.targets) == 1 \
                    and isinstance(parent.targets[0], ast.Name):
                elements = alias_element_keys(parent.targets[0].id)
            if elements:
                self.nested.setdefault(f"{key}[]", set()).update(elements)
                return
            sub = None
            if isinstance(parent, ast.Subscript) and parent.value is read:
                sub = self.key(parent.slice)
            elif isinstance(parent, ast.Attribute) and parent.attr == "get":
                call = parents.get(id(parent))
                if isinstance(call, ast.Call) and call.func is parent and call.args:
                    sub = self.key(call.args[0])
            if sub is None:
                self.flat.add(key)
            else:
                self.nested.setdefault(key, set()).add(sub)

        def container(expr, name: str) -> None:
            """A query or body container, as request.GET, and what's done with it."""
            expr, parent = up(expr)
            if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or) and parent.values[0] is expr:
                container(parent, name)  # request.get_json(silent=True) or {}
                return
            if isinstance(parent, ast.Subscript) and parent.value is expr:
                found = keys(parent.slice)
                if found is not None:
                    for key in found:
                        kind = converted(parent)
                        _merge(self.inputs, key, kind)
                        self.locate(name, key, kind, parent)
                        inner(parent, key)
                else:
                    self.incomplete("dynamic_key", f"{name} is read with a key that isn't a constant", parent)
                return
            if isinstance(parent, ast.Attribute) and parent.value is expr:
                call = parents.get(id(parent))
                if parent.attr in {"get", "getlist", "get_list", "getall"} and isinstance(call, ast.Call) and call.func is parent:
                    found = keys(call.args[0]) if call.args else None
                    if found is not None:
                        kind = next((_CONVERTERS.get(_dotted(k.value), "text") for k in call.keywords if k.arg == "type"), None)
                        for key in found:
                            read_kind = kind or converted(call)
                            _merge(self.inputs, key, read_kind)
                            self.locate(name, key, read_kind, call)
                            inner(call, key)
                    else:
                        self.incomplete("dynamic_key", f"{name}.{parent.attr}() is called with a key that isn't a constant", call)
                    return
                self.incomplete("container_whole", f"{name}.{parent.attr} reads it whole", parent)
                return
            if isinstance(parent, ast.Compare) and expr is not parent.left:
                return  # "q" in request.GET reads nothing.
            if isinstance(parent, (ast.Assign, ast.AnnAssign)) and getattr(parent, "value", None) is expr:
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    aliases[targets[0].id] = (expr, name)
                    return
            if isinstance(parent, ast.Call) and _last(parent.func) == "loads" and parent.args and parent.args[0] is expr:
                container(parent, name)  # json.loads(request.body)
                return
            if isinstance(parent, (ast.Call, ast.keyword)):
                call = parents.get(id(parent)) if isinstance(parent, ast.keyword) else parent
                if isinstance(call, ast.Call):
                    fields, reason = self.module.fields(_dotted(call.func))
                    if fields is not None:
                        for field, kind in fields.items():
                            _merge(self.inputs, field, kind)
                            self.unlocated.add(field)  # A form or serializer may read either.
                        return
                    if reason is not None and reason[0] == "model_fields":
                        self.class_reason(reason, call)
                        return
                    callee = _dotted(call.func) or "a call"
                    if callee.split(".")[-1][:1].isupper():  # A class, as a form or serializer.
                        self.class_reason(reason or ("external_class", callee), call)
                        return
                    if self.follow(call, expr, parent, container=name):
                        return  # A helper of the project's, read with the container as its parameter.
                    self.incomplete("container_whole", f"{name} is passed whole to {callee}", call)
                    return
            self.incomplete("container_whole", f"{name} is used whole", parent or expr)

        for expr in ast.walk(node):
            if isinstance(expr, ast.Attribute) and is_request(expr.value):
                if expr.attr in _CONTAINERS:
                    _, parent = up(expr)
                    if expr.attr in _CALLED and isinstance(parent, ast.Call) and parent.func is expr:
                        container(parent, f"{label(expr)}()")
                    elif expr.attr not in {"get_json", "get_data"}:
                        container(expr, label(expr))
                    else:
                        self.incomplete("container_whole", f"{label(expr)} is used without being called", expr)
                elif expr.attr in _RAW:
                    self.incomplete("raw_body", f"{label(expr)} reads the raw query or body", expr)
            elif isinstance(expr, ast.Name) and expr.id in requests and isinstance(expr.ctx, ast.Load):
                parent = parents.get(id(expr))
                if isinstance(parent, ast.Attribute) and parent.value is expr:
                    continue
                call = parents.get(id(parent)) if isinstance(parent, ast.keyword) else parent
                if isinstance(call, ast.Call) and (_last(call.func) in _SAFE_REQUEST_CALLS or self.follow(call, expr, parent)):
                    continue
                if isinstance(call, ast.Call):
                    self.incomplete("helper", f"the request is passed to {_dotted(call.func) or 'a call'}, which isn't followed", call)
                else:
                    self.incomplete("helper", "the request is stored or returned where it may be read", expr)
        # A container this function was given, as a helper's args parameter.
        for expr in ast.walk(node):
            if isinstance(expr, ast.Name) and expr.id in (containers or {}) and isinstance(expr.ctx, ast.Load):
                container(expr, containers[expr.id])
        # What's read from an alias, as data = request.json, then data["q"].
        for expr in ast.walk(node):
            if isinstance(expr, ast.Name) and expr.id in aliases and isinstance(expr.ctx, ast.Load):
                container(expr, aliases[expr.id][1])

    def key(self, node: ast.AST) -> Optional[str]:
        """A constant key, written out or held in a module-level string constant, or None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.module.constants:
                return self.module.constants[node.id]
            # A constant imported from another project module.
            target = self.module.imports.get(node.id)
            if target is not None and target[1] is not None:
                source = self.module.project.get(target[0])
                if source is not None:
                    return source.constants.get(target[1])
        return None

    def follow(self, call: ast.Call, request: ast.AST, holder: ast.AST, container: Optional[str] = None) -> bool:
        """Whether a call the request is passed to is one these reads cover, following it if needed.
        With container, what's passed is that query or body container (its label), not the request."""
        callee = _dotted(call.func)
        method = callee.split(".")[-1]
        # A class view is read whole, so its own methods are covered; a plain View's dispatch and
        # setup only route the request to them.
        if container is None and callee.startswith("self.") and callee.count(".") == 1 and method in self.methods:
            return True
        if container is None and callee.startswith("super.") and self.plain_class and method in {"dispatch", "setup", "get", "post", "put", "patch", "delete", "head", "options"}:
            return True
        found = self.module.lookup("function", callee)
        if found is None:
            return False
        module, target = found
        if id(target) in self.active or len(self.active) >= 4:
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
        caller = self.module
        self.module = module
        site = self.call_site
        self.call_site = site or (call.lineno, call.col_offset)
        try:
            if container is None:
                self.walk(target, {name})
            else:
                self.walk(target, set(), {name: container})
        finally:
            self.module = caller
            self.call_site = site
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
                self.incomplete("dependency", f"parameter {parameter.arg} is a dependency, which may read anything", parameter)
                continue
            name = parameter.arg
            # The wire key: an alias on the default (= Body(alias="limit")) or in Annotated's
            # metadata (Annotated[str, Body(embed=True, alias="limit")]), not the argument's name.
            markers = [default] if isinstance(default, ast.Call) else []
            if isinstance(annotation, ast.Subscript) and _last(annotation.value) == "Annotated" and isinstance(annotation.slice, ast.Tuple):
                markers.extend(item for item in annotation.slice.elts[1:] if isinstance(item, ast.Call))
            for marker in markers:
                alias = next((k.value.value for k in marker.keywords if k.arg in {"alias", "validation_alias"}
                              and isinstance(k.value, ast.Constant)), None)
                if isinstance(alias, str):
                    name = alias
                    break
            model = self.module.model(annotation)
            if _last(annotation) in _OPEN_ANNOTATIONS or (isinstance(annotation, ast.Subscript) and _last(annotation.value) in _OPEN_ANNOTATIONS):
                # A dict or Any body takes whatever names the client sends.
                self.incomplete("container_whole", f"parameter {parameter.arg} takes a whole {_last(annotation) or 'body'}", parameter)
            elif model is not None:
                models.append((name, model, parameter))
            elif _annotation_type(annotation) == "text" and _last(annotation) not in {"", "str", "Optional", "List", "list"} \
                    and _last(annotation) not in _ANNOTATIONS:
                self.class_reason(("external_class", _dotted(annotation) or _last(annotation)), parameter)
            else:
                _merge(self.inputs, name, _annotation_type(annotation))
        # One model is the whole body; several are each under their parameter's name.
        for name, model, parameter in models:
            fields, reason = self.module.fields(model)
            if fields is None:
                self.class_reason(reason or ("external_class", model), parameter)
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


def _argument_decorated(node) -> bool:
    return any(isinstance(item, ast.Call) and _last(item.func) in _ARGUMENT_DECORATORS for item in node.decorator_list)


def _flask_request(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"flask", "quart"}
               and any(alias.name == "request" for alias in node.names) for node in ast.walk(tree))


def _entry(function: str, filename: str, reads: _Reads) -> dict:
    entry = {"function": function, "file": filename, "inputs": dict(reads.inputs), "complete": reads.complete}
    # A key path's first name, as lines in lines[], is the input.
    def top(path: str) -> str:
        return path.split(".")[0].removesuffix("[]")

    nested = {key: sorted(keys) for key, keys in reads.nested.items() if top(key) not in reads.flat and top(key) in reads.inputs}
    if nested:
        entry["nested"] = nested
    if reads.reasons:  # Only on incomplete entries, so complete ones keep the shared report shape.
        entry["reasons"] = [dict(reason) for reason in reads.reasons]
    locations = {where: dict(types) for where, types in reads.locations.items() if types}
    if locations:
        entry["locations"] = locations
    if reads.unlocated:
        entry["unlocated"] = sorted(reads.unlocated)
    method_reads = reads.method_reads()
    if method_reads:
        entry["method_reads"] = method_reads
    return entry


def module_inputs(tree: ast.Module, filename: str, module: Optional[_Module] = None) -> List[dict]:
    """Each function's (or class view's) inputs: {function, file, inputs, complete, reasons}.

    With module, from Project.add, forms, models and helpers imported from the project's other
    modules are followed; without, only this module's are.
    """
    if module is None:
        module = Project().add(tree, filename)
    flask_request = _flask_request(tree)
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
        if not any(_uses_request(method, _Reads(module, False).request_names(method)) or _argument_decorated(method)
                   for method in methods):
            continue
        # A class view is read as a whole: its handlers and helpers share self.request. Each HTTP
        # method's handler (get, post, ...) is also read with only the class's other methods, so
        # what it reads is its own: a GET doesn't take the POST's body fields.
        def read_class(chosen):
            reads = _Reads(module, flask_request)
            reads.methods = {method.name for method in methods}
            reads.plain_class = bool(node.bases) and all(_last(base) in _PLAIN_VIEWS for base in node.bases)
            if not reads.plain_class:
                bases = ", ".join(_dotted(base) for base in node.bases) or "no base"
                reads.incomplete("class_view", f"{node.name} inherits handlers from {bases}", node, view=node.name)
            for method in chosen:
                reads.arguments(method)
                reads.walk(method, reads.request_names(method))
            reads.close()
            return reads

        whole = read_class(methods)
        shared = [method for method in methods if method.name not in _HTTP_HANDLERS]
        for method in methods:
            reads = read_class([method, *shared]) if method.name in _HTTP_HANDLERS else whole
            results.append(_entry(f"{node.name}.{method.name}", filename, reads))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(node) not in owned:
            reads = _Reads(module, flask_request)
            reads.active.add(id(node))
            requests = reads.request_names(node)
            # Only what can be a view: a route-decorated function, or one that takes the request
            # (even unused, as def detail(request, pk) that reads nothing).
            routed = _route_decorated(node)
            takes_request = any(parameter.arg in _REQUEST_NAMES or _last(parameter.annotation) in _REQUEST_ANNOTATIONS
                                for parameter in node.args.args[:1])
            if not routed and not takes_request and not _uses_request(node, requests):
                continue
            # webargs arguments say what the view's parameters hold; otherwise each is an input.
            if not reads.arguments(node) and routed:
                reads.parameters(node)
            reads.walk(node, requests)
            reads.close()
            results.append(_entry(node.name, filename, reads))
    return results


def source_inputs(source: str, filename: str = "<source>") -> List[dict]:
    return module_inputs(ast.parse(source, filename=filename), filename)


def project_inputs(sources: Dict[str, str]) -> List[dict]:
    """Inputs for several modules at once, by dotted module name to source, following imports."""
    project = Project()
    modules = []
    for name, source in sources.items():
        package = name.endswith(".__init__")
        dotted = name[: -len(".__init__")] if package else name
        filename = name.replace(".", "/") + ".py"
        tree = ast.parse(source, filename=filename)
        modules.append((tree, filename, project.add(tree, filename, dotted, package)))
    results: List[dict] = []
    for tree, filename, module in modules:
        results.extend(module_inputs(tree, filename, module))
    return results


_MAX_REASONS = 20


def merge_into(routes: dict, path: str, entry: dict) -> None:
    """Adds one function's inputs to a route; a route served by several takes them all."""
    data = routes[path]
    inputs = data.setdefault("inputs", {})
    nested = entry.get("nested") or {}
    for name, kind in entry["inputs"].items():
        _merge(inputs, name, kind)
        # An input's inner keys are only known when every function reading it reads them: its own
        # keys (customer) or its elements' (lines[]).
        paths = [path for path in nested if path.split(".")[0].removesuffix("[]") == name]
        if paths:
            for path in paths:
                data.setdefault("_nested", {}).setdefault(path, set()).update(nested[path])
        else:
            data.setdefault("_flat", set()).add(name)
    data["inputs_complete"] = bool(data.get("inputs_complete", True)) and bool(entry["complete"])
    reasons = data.setdefault("inputs_incomplete", [])
    for reason in entry.get("reasons", ()):
        item = {**reason, "function": entry.get("function", "")}
        if item not in reasons and len(reasons) < _MAX_REASONS:
            reasons.append(item)
    if not reasons:
        data.pop("inputs_incomplete", None)
