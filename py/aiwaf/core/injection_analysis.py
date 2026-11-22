"""Static analysis that follows request input to dangerous operations.

This reports possible injection paths. It reads source files only and never runs
application code or affects request handling. A clean report doesn't prove code is safe:
input that crosses modules, dynamic calls or frameworks it can't see can be missed.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import re
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .input_schema import module_inputs

SCHEMA_VERSION = "1.0"
_SKIPPED_DIRECTORIES = {
    ".git", ".hg", ".tox", ".venv", "__pycache__", "build", "dist", "env", "migrations",
    "node_modules", "site-packages", "venv",
}
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_HELPER_DEPTH = 3

_REQUEST_ATTRIBUTES = {
    "COOKIES", "FILES", "GET", "META", "POST", "args", "body", "content", "cookies", "data",
    "files", "form", "get_data", "get_json", "headers", "json", "path_params", "query_params",
    "query_string", "receive", "receive_bytes", "receive_json", "receive_text", "stream", "values",
}
# Django Channels consumer methods called with what the browser sent.
_CONSUMER_METHODS = {"receive", "receive_json", "websocket_receive"}
# Flask-SocketIO and python-socketio event handlers.
_SOCKET_DECORATORS = {"event", "on", "on_event"}
_REQUEST_NAMES = {"request", "req"}
_NON_INPUT_PARAMETERS = {"self", "cls", "request", "req", "db", "session", "response", "background_tasks"}
_NON_INPUT_ANNOTATIONS = {
    "AsyncSession", "BackgroundTasks", "Connection", "Request", "Response", "Session", "WebSocket",
}
_ROUTE_DECORATORS = {"api_route", "delete", "get", "head", "options", "patch", "post", "put", "route", "websocket"}
# Each sanitizer and the operations its result is safe for; "*" is all of them.
_SANITIZER_KINDS = {
    "UUID": {"*"}, "abs": {"*"}, "bool": {"*"}, "float": {"*"}, "int": {"*"}, "len": {"*"}, "uuid.UUID": {"*"},
    # secure_filename keeps only letters, digits, dots, dashes and underscores.
    "secure_filename": {"*"}, "werkzeug.utils.secure_filename": {"*"},
    "bleach.clean": {"xss"}, "escape": {"xss"}, "html.escape": {"xss"}, "markupsafe.escape": {"xss"},
    "os.path.basename": {"path"}, "shlex.quote": {"command"},
    "quote": {"header", "redirect"}, "urllib.parse.quote": {"header", "redirect"},
    "quote_plus": {"header", "redirect"}, "urllib.parse.quote_plus": {"header", "redirect"},
    "escape_filter_chars": {"ldap"}, "ldap.filter.escape_filter_chars": {"ldap"}, "ldap.filter.filter_format": {"ldap"},
    "ldap3.utils.conv.escape_filter_chars": {"ldap"}, "re.escape": {"redos"},
}
# Project functions named like these are taken to make text safe to show in HTML.
_HTML_ESCAPER_NAME = re.compile(r"escape|sanitiz|html_?encode|encode_?(for_)?html", re.IGNORECASE)
# Responses whose text is shown as HTML.
_HTML_RESPONSES = {"HTMLResponse", "HttpResponse", "Markup", "make_response", "mark_safe"}
_REDIRECTS = {"HttpResponsePermanentRedirect", "HttpResponseRedirect", "RedirectResponse", "redirect"}
_REDIRECT_CHECKS = {"is_safe_url", "url_has_allowed_host_and_scheme"}
_XML_PARSERS = {
    "etree.XML", "etree.fromstring", "etree.parse", "lxml.etree.XML", "lxml.etree.fromstring", "lxml.etree.parse",
    "xml.dom.minidom.parse", "xml.dom.minidom.parseString", "xml.dom.pulldom.parseString", "xml.sax.parse",
    "xml.sax.parseString",
}
_XPATH_FUNCTIONS = {"etree.XPath", "lxml.etree.XPath", "lxml.etree.XPathEvaluator"}
_ELEMENTPATH_FUNCTIONS = {"elementpath.iter_select", "elementpath.select"}
_REGEX_FUNCTIONS = {
    "re.compile", "re.findall", "re.finditer", "re.fullmatch", "re.match", "re.search", "re.split", "re.sub", "re.subn",
    "regex.compile", "regex.match", "regex.search",
}
_SANITIZERS = {name for name, kinds in _SANITIZER_KINDS.items() if kinds == {"*"}}
_SINK_KINDS = {
    "code", "command", "deserialization", "header", "ldap", "nosql", "path", "redirect", "redos", "sql", "ssrf",
    "template", "xpath", "xss", "xxe",
}
_TEXT_PASSTHROUGH = {
    "urllib.parse.urlparse", "urllib.parse.urlsplit", "urlparse", "urlsplit",
    "base64.b64decode", "base64.b64encode", "base64.urlsafe_b64decode", "base64.urlsafe_b64encode",
    "bytes", "codecs.decode", "codecs.encode", "dict", "f", "format", "html.unescape", "list",
    "os.path.abspath", "os.path.join", "os.path.normpath", "repr", "set", "str", "tuple",
    "unquote", "unquote_plus", "urljoin", "urllib.parse.unquote", "urllib.parse.unquote_plus", "urllib.parse.urljoin",
}
_STRING_METHODS = {
    "casefold", "decode", "encode", "format", "get", "getlist", "getvalue", "items", "join", "keys", "lower",
    "lstrip", "pop", "replace", "rstrip", "split", "strip", "title", "upper", "values",
}
_SQL_METHODS = {"execute", "executemany", "executescript", "raw"}
_SQL_FUNCTIONS = {"RawSQL", "sql.text", "sqlalchemy.text", "text"}
_NOSQL_METHODS = {
    "aggregate", "count_documents", "delete_many", "delete_one", "find", "find_one",
    "find_one_and_update", "update_many", "update_one",
}
_NOSQL_RECEIVER_HINTS = ("coll", "db", "mongo", "users", "documents")
_COMMAND_FUNCTIONS = {
    "commands.getoutput", "os.execl", "os.execle", "os.execlp", "os.execlpe", "os.execv", "os.execve", "os.execvp",
    "os.execvpe", "os.popen", "os.posix_spawn", "os.posix_spawnp", "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnv",
    "os.spawnve", "os.spawnvp", "os.system", "pty.spawn",
}
_SUBPROCESS_FUNCTIONS = {
    "subprocess.Popen", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput", "subprocess.run",
}
_PATH_FUNCTIONS = {
    "FileResponse", "Path", "codecs.open", "io.open", "open", "os.listdir", "os.makedirs", "os.mkdir", "os.open",
    "os.path.exists", "os.path.getsize", "os.path.isdir", "os.path.isfile", "os.remove", "os.rename", "os.rmdir",
    "os.stat", "os.unlink", "pathlib.Path",
    "send_file", "shutil.copy", "shutil.move", "shutil.rmtree",
}
# pathlib.Path methods that read, write or test the file a path names.
_PATH_METHODS = {
    "exists", "is_dir", "is_file", "mkdir", "open", "read_bytes", "read_text", "rename", "rmdir", "touch",
    "unlink", "write_bytes", "write_text",
}
_SSRF_FUNCTIONS = {
    "httpx.delete", "httpx.get", "httpx.head", "httpx.post", "httpx.put", "httpx.request",
    "requests.delete", "requests.get", "requests.head", "requests.patch", "requests.post",
    "requests.put", "requests.request", "urllib.request.urlopen", "urlopen",
}
_SCALAR_ANNOTATIONS = {"", "Any", "Optional", "UUID", "bool", "bytes", "float", "int", "str"}
# Parameter types the framework converts before the view runs, so their values can't carry text.
_SAFE_SCALARS = {
    "Decimal", "NegativeFloat", "NegativeInt", "NonNegativeFloat", "NonNegativeInt", "PositiveFloat",
    "PositiveInt", "StrictBool", "StrictFloat", "StrictInt", "UUID", "UUID1", "UUID3", "UUID4", "UUID5", "bool",
    "conint", "confloat", "date", "datetime", "float", "int", "time", "timedelta",
}
# Flask and Werkzeug route converters that only match numbers or UUIDs.
_SAFE_CONVERTER = re.compile(r"<(?:int|float|uuid)(?:\([^)]*\))?:(\w+)>")
# Calls that put a value into a collection: the collection then holds that value.
_COLLECTION_ADDS = {
    "add", "append", "appendleft", "extend", "extendleft", "insert", "setdefault", "update", "write", "writelines",
}
_HTTP_CLIENT_METHODS = {"delete", "get", "head", "patch", "post", "put"}
_CODE_FUNCTIONS = {"__import__", "compile", "eval", "exec"}
# Importing a module named by input runs that module's code.
_IMPORT_FUNCTIONS = {"importlib.__import__", "importlib.import_module", "import_module"}
_TEMPLATE_FUNCTIONS = {"Template", "jinja2.Template", "render_template_string"}
_DESERIALIZE_FUNCTIONS = {
    "cloudpickle.load", "cloudpickle.loads", "dill.load", "dill.loads", "joblib.load", "jsonpickle.decode", "marshal.load",
    "marshal.loads", "pandas.read_pickle", "pickle.load", "pickle.loads", "shelve.open", "torch.load", "yaml.load",
    "yaml.unsafe_load",
}
_SAFE_YAML_LOADERS = {"CSafeLoader", "SafeLoader", "yaml.CSafeLoader", "yaml.SafeLoader"}


_UNKNOWN = object()
# Checks that a value passes before it's used, as in if not name.isalnum(): return.
_VALIDATING_METHODS = {
    "endswith", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier", "isnumeric", "startswith",
}
_VALIDATING_FUNCTIONS = {"re.fullmatch", "re.match", "re.search", "fullmatch", "match"}
_EXIT_CALLS = {"abort", "flask.abort", "sys.exit", "exit"}


def _exits(body: list) -> bool:
    """True when a block always leaves: return, raise, continue, break or abort()."""
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
        return True
    return isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and dotted_name(last.value.func) in _EXIT_CALLS


def _assigned_names(statements: list) -> Set[str]:
    names: Set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _is_http_client_call(receiver: str, method: str) -> bool:
    # A bare "client" or "session" can be a cache or dict, so get() needs a clearer name.
    if method == "request":
        return any(hint in receiver for hint in ("client", "http", "session"))
    return method in _HTTP_CLIENT_METHODS and "http" in receiver


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    function: str
    sink: str
    call: str
    source: str
    confidence: str
    # The file of the view the request entered, when the operation is in another module.
    entry_file: str = ""


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return ""


def _with_key(source: Optional[str], key: ast.AST) -> Optional[str]:
    """Add a literal parameter name to a request source, as in request.GET['q']."""
    if (
        source
        and "[" not in source
        and not source.startswith(("route parameter", "body parameter"))
        and isinstance(key, ast.Constant)
        and isinstance(key.value, str)
    ):
        return f"{source}['{key.value}']"
    return source


def _annotation_name(node: Optional[ast.AST]) -> str:
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    return dotted_name(node).split(".")[-1] if node is not None else ""


def _class_annotation(node: Optional[ast.AST]) -> Optional[ast.AST]:
    """The class an annotation names, through Optional, Annotated and unions with None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(node, ast.Subscript) and _annotation_name(node.value) in {"Optional", "Annotated"}:
        inner = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
        return _class_annotation(inner)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if _is_none(node.right):
            return _class_annotation(node.left)
        return _class_annotation(node.right) if _is_none(node.left) else None
    return node if isinstance(node, (ast.Name, ast.Attribute)) else None


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None or dotted_name(node) == "None"


def _safe_scalar(node: Optional[ast.AST]) -> bool:
    """Whether a parameter's type admits only numbers, booleans, dates or UUIDs (or None)."""
    if node is None:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        sides = (node.left, node.right)
        return all(_safe_scalar(side) or _is_none(side) for side in sides) and not all(_is_none(side) for side in sides)
    if isinstance(node, ast.Subscript):
        base = _annotation_name(node.value)
        inner = node.slice
        elements = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
        if base == "Annotated":
            return _safe_scalar(elements[0])
        if base == "Optional":
            return _safe_scalar(elements[0])
        if base == "Union":
            return all(_safe_scalar(e) or _is_none(e) for e in elements) and not all(_is_none(e) for e in elements)
        if base == "Literal":
            return True
        return False
    if isinstance(node, ast.Call):
        return _annotation_name(node.func) in {"conint", "confloat"}
    return _annotation_name(node) in _SAFE_SCALARS


def _module_base(module_name: str, target: Optional[str], level: int) -> str:
    """The absolute module a "from ... import" names, resolving relative imports."""
    if not level:
        return target or ""
    parts = module_name.split(".")[:-level] if module_name else []
    if target:
        parts.append(target)
    return ".".join(part for part in parts if part)


class _ModuleAnalyzer:
    def __init__(self, tree: ast.Module, filename: str, module_name: str = "", project: Optional[dict] = None,
                 extras: Optional[dict] = None):
        self.filename = filename
        self.module_name = module_name
        # The project's other modules by dotted name, so calls into them can be followed.
        self.project = project if project is not None else {}
        # Input kept on request or flask.g by middleware, by (owner, attribute), shared by the project.
        self.extras: Dict[tuple, str] = extras if extras is not None else {}
        # Helpers already walked with the same input, shared by the project: what they returned and
        # found, so each is walked once however many views call it.
        shared = self.project.get("\0shared") if isinstance(self.project.get("\0shared"), dict) else None
        self.summaries: Dict[tuple, tuple] = shared["summaries"] if shared else {}
        self.recordings: List[list] = shared["recordings"] if shared else []
        self.findings: Dict[tuple, Finding] = {}
        self.functions: Dict[str, ast.AST] = {}
        self.flask_request = False
        # Flask views send returned text as HTML.
        self.uses_flask = False
        self.aliases: Dict[str, str] = {}
        self._returns: Dict[tuple, Optional[str]] = {}
        # Classes by name, with their methods and the classes their attributes hold.
        self.classes: Dict[str, dict] = {}
        self.method_owners: Dict[int, dict] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, node)
            elif isinstance(node, ast.ImportFrom) and (node.module or node.level):
                base = _module_base(module_name, node.module, node.level)
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
                    if alias.name == "request" and (node.module or "").split(".")[0] in {"flask", "quart"}:
                        self.flask_request = True
                    if (node.module or "").split(".")[0] in {"flask", "quart"}:
                        self.uses_flask = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in {"flask", "quart"}:
                        self.uses_flask = True
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
        self.tree = tree
        self.lines: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._collect_class(node)

    def _collect_class(self, node: ast.ClassDef) -> None:
        """Records methods, and attribute classes, as in users: UserService or self.users = UserService()."""
        info: dict = {"methods": {}, "fields": {}, "bases": [dotted_name(base) for base in node.bases]}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info["methods"].setdefault(item.name, item)
                self.method_owners[id(item)] = info
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                annotated = _class_annotation(item.annotation)
                if annotated is not None:
                    info["fields"][item.target.id] = self.canonical(annotated)
        init = info["methods"].get("__init__")
        if init is not None:
            parameters = {
                parameter.arg: _class_annotation(parameter.annotation)
                for parameter in [*init.args.args, *init.args.kwonlyargs]
            }
            for statement in ast.walk(init):
                if isinstance(statement, ast.Assign):
                    targets, value = statement.targets, statement.value
                elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                    targets, value = [statement.target], statement.value
                else:
                    continue
                for target in targets:
                    if not (isinstance(target, ast.Attribute) and dotted_name(target.value) == "self"):
                        continue
                    if isinstance(value, ast.Call):
                        info["fields"][target.attr] = self.canonical(value.func)
                    elif isinstance(value, ast.Name) and parameters.get(value.id) is not None:
                        info["fields"][target.attr] = self.canonical(parameters[value.id])
        self.classes.setdefault(node.name, info)

    def resolve_class(self, reference: str) -> Optional[tuple]:
        """The module and class info a dotted class name refers to, in this module or the project."""
        if not reference:
            return None
        if reference in self.classes:
            return self, self.classes[reference]
        module_name, _, name = reference.rpartition(".")
        target = self.project.get(module_name)
        if target is not None and name in target.classes:
            return target, target.classes[name]
        return None

    def find_method(self, call: ast.Call, analyzer) -> Optional[tuple]:
        """A method call on a value of one of the project's classes, as in service.search(term)."""
        if analyzer is None or not isinstance(call.func, ast.Attribute):
            return None
        value = call.func.value
        reference = ""
        if isinstance(value, ast.Name) and value.id in analyzer.types:
            reference = analyzer.types[value.id]
        elif isinstance(value, ast.Attribute) and dotted_name(value.value) == "self" and analyzer.owner is not None:
            reference = analyzer.owner["fields"].get(value.attr, "")
        elif isinstance(value, ast.Call):
            reference = self.canonical(value.func)
        elif isinstance(value, (ast.Name, ast.Attribute)) and dotted_name(value) not in {"self", "cls"}:
            reference = self.canonical(value)
        found = self.resolve_class(reference)
        if found is None:
            return None
        module, info = found
        method = info["methods"].get(call.func.attr)
        return (module, method) if method is not None else None

    def canonical(self, node: ast.AST) -> str:
        name = dotted_name(node)
        head, _, rest = name.partition(".")
        target = self.aliases.get(head)
        if target:
            return f"{target}.{rest}" if rest else target
        return name

    def run(self) -> List[Finding]:
        owners = {
            item: node.name
            for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)
            for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                label = f"{owners[node]}.{node.name}" if node in owners else node.name
                self._analyze(node, self._initial_taint(node), label, 0, set())
        return list(self.findings.values())

    def _initial_taint(self, node: ast.AST) -> Dict[str, str]:
        tainted: Dict[str, str] = {}
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        names = [parameter.arg for parameter in parameters]
        decorators = [dotted_name(item.func if isinstance(item, ast.Call) else item) for item in node.decorator_list]
        owner = self.method_owners.get(id(node))
        # GraphQL resolvers (Graphene, Strawberry, Ariadne) take their arguments after info.
        graphql = "info" in names and (node.name.startswith("resolve") or any(
            part in decorator for decorator in decorators for part in ("field", "mutation", "query", "strawberry")))
        socket = any(decorator.split(".")[-1] in _SOCKET_DECORATORS and "socket" in decorator.lower() for decorator in decorators)
        consumer = owner is not None and node.name in _CONSUMER_METHODS and any("Consumer" in base for base in owner.get("bases", []))
        if graphql or socket or consumer:
            start = names.index("info") + 1 if graphql else 0
            label = "graphql argument" if graphql else "websocket message"
            for parameter in parameters[start:]:
                if parameter.arg not in {"self", "cls", "info"} and not _safe_scalar(parameter.annotation):
                    tainted[parameter.arg] = f"{label} {parameter.arg}"
            if node.args.kwarg is not None:
                tainted[node.args.kwarg.arg] = f"{label} {node.args.kwarg.arg}"
            return tainted
        routes = [decorator for decorator in node.decorator_list if self._is_route_decorator(decorator)]
        if routes:
            defaults = dict(zip(reversed(node.args.args), reversed(node.args.defaults)))
            # Route values converted to numbers or UUIDs, as <int:id> in Flask.
            converted = {
                name for decorator in routes for argument in decorator.args[:1]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                for name in _SAFE_CONVERTER.findall(argument.value)
            }
            for parameter in parameters:
                default = defaults.get(parameter)
                dependency = isinstance(default, ast.Call) and dotted_name(default).split(".")[-1] in {"Depends", "Security"}
                annotation = _annotation_name(parameter.annotation)
                if parameter.arg in converted or _safe_scalar(parameter.annotation):
                    continue  # Converted by the framework before the view runs.
                if (
                    parameter.arg not in _NON_INPUT_PARAMETERS
                    and annotation not in _NON_INPUT_ANNOTATIONS
                    and annotation != "Annotated"
                    and not dependency
                ):
                    kind = "route" if annotation in _SCALAR_ANNOTATIONS else "body"
                    tainted[parameter.arg] = f"{kind} parameter {parameter.arg}"
                elif annotation == "Annotated" and "Depends" not in ast.dump(parameter.annotation):
                    tainted[parameter.arg] = f"route parameter {parameter.arg}"
        else:
            # Django and Django REST framework views take route values after the request.
            names = [parameter.arg for parameter in node.args.args]
            if "request" in names[:2]:
                for name in names[names.index("request") + 1:]:
                    tainted[name] = f"route parameter {name}"
        return tainted

    @staticmethod
    def _is_route_decorator(decorator: ast.AST) -> bool:
        return (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr in _ROUTE_DECORATORS
        )

    def _analyze(self, node, tainted, label, depth, active, entry_file="", requests=()) -> Optional[str]:
        """Follow input through a function; return the request source its return value carries."""
        key = (node.name, (self.filename,), id(node))
        if (key in active) or depth > _MAX_HELPER_DEPTH:
            return None
        summary_key = (id(node), depth, frozenset(tainted.items()), frozenset(requests)) if depth else None
        if summary_key in self.summaries:
            # Walked before with the same input: record its findings again under this caller.
            old_label, recorded, returned = self.summaries[summary_key]
            for module, item, item_label, sink, call, source, confidence in recorded:
                module.record(item, label + item_label[len(old_label):], sink, call, source, confidence, entry_file or self.filename)
            return returned
        recorded: list = []
        if summary_key:
            self.recordings.append(recorded)
        try:
            returned = self._walk(node, tainted, label, depth, {*active, key}, entry_file, requests)
        finally:
            if summary_key:
                self.recordings.pop()  # Nested walks finish in order, so this one is last.
        if summary_key:
            self.summaries[summary_key] = (label, recorded, returned)
        return returned

    def _walk(self, node, tainted, label, depth, active, entry_file, requests) -> Optional[str]:
        request_names = self._request_names(node) | set(requests)
        analyzer = _FunctionAnalyzer(self, label, dict(tainted), request_names, depth, active, entry_file or self.filename)
        analyzer.owner = self.method_owners.get(id(node))
        analyzer.view = depth == 0 and any(self._is_route_decorator(item) for item in node.decorator_list)
        analyzer.add_parameter_types(node)
        analyzer.block(node.body, nested=False)
        return analyzer.returned

    def find_function(self, call: ast.Call, analyzer=None):
        """The project function a call runs, with the module it's in, or None."""
        method = self.find_method(call, analyzer)
        if method is not None:
            return method
        name = dotted_name(call.func)
        head, _, _ = name.partition(".")
        last = name.split(".")[-1]
        if (name == last or head in {"self", "cls"}) and last in self.functions:
            return self, self.functions[last]
        module, _, function = self.canonical(call.func).rpartition(".")
        target = self.project.get(module)
        if target is not None and function in target.functions:
            return target, target.functions[function]
        if last in self.functions:
            return self, self.functions[last]
        return None

    def _request_names(self, node) -> Set[str]:
        names = {"request"} if self.flask_request else set()
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for parameter in parameters:
            annotation = _annotation_name(parameter.annotation)
            if parameter.arg in _REQUEST_NAMES or parameter.arg == "websocket" or annotation in {"HttpRequest", "Request", "WebSocket"}:
                names.add(parameter.arg)
        return names

    def record(self, node, label, sink, call, source, confidence, entry_file="") -> None:
        for recorded in self.recordings:
            recorded.append((self, node, label, sink, call, source, confidence))
        entry = "" if entry_file in ("", self.filename) else entry_file
        # Per entry view too, so an operation reached from several views links to each one's route.
        key = (node.lineno, sink, call, label.split(" -> ")[0], entry)
        self.findings.setdefault(key, Finding(self.filename, node.lineno, label, sink, call, source, confidence, entry))


class _FunctionAnalyzer:
    def __init__(self, module, label, tainted, request_names, depth, active, entry_file=""):
        self.module = module
        self.label = label
        self.tainted = tainted
        self.request_names = request_names
        self.depth = depth
        self.active = active
        self.entry_file = entry_file
        # The request source a return statement carries, if any.
        self.returned: Optional[str] = None
        # The class of the method being walked, and the classes local names hold.
        self.owner: Optional[dict] = None
        self.types: Dict[str, str] = {}
        # What objects built here keep from their constructor: input by attribute, and the request.
        self.fields: Dict[str, Dict[str, str]] = {}
        self.request_fields: Dict[str, Set[str]] = {}
        # Values made by a project function whose class isn't shown, as factories make them.
        self.made_by: Dict[str, object] = {}
        # Names holding a pathlib.Path, whose methods read and write files.
        self.paths: Set[str] = set()
        self._followed: Dict[int, Optional[str]] = {}
        # Values known before running, so if 7 * 42 - num > 200: can be decided.
        self.constants: Dict[str, object] = {}
        # Dicts filled with literal keys, by key, and lists, by position: the input each holds.
        self.keyed: Dict[str, Dict[object, Optional[str]]] = {}
        # What each followed dict value is safe for, by key.
        self.keyed_kinds: Dict[str, Dict[object, Set[str]]] = {}
        self.lists: Dict[str, list] = {}
        # Names checked before use, and the operations the check makes them safe for ("*" for all).
        self.checked: Dict[str, Set[str]] = {}
        # urlparse(value) results, by the value they parsed: a check on one is a check on the other.
        self.parsed: Dict[str, str] = {}
        # XML parsers set to load external entities.
        self.unsafe_xml: Set[str] = set()
        # True for a view, whose returned text is the response.
        self.view = False
        # Names holding text, as RESPONSE = "" and RESPONSE += ...; not response objects.
        self.texts: Set[str] = set()

    def constant(self, node: Optional[ast.AST]):
        """A value known before the program runs, or _UNKNOWN."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id, _UNKNOWN)
        if isinstance(node, ast.UnaryOp):
            value = self.constant(node.operand)
            if value is _UNKNOWN:
                return _UNKNOWN
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub) and isinstance(value, (int, float)):
                return -value
            return _UNKNOWN
        if isinstance(node, ast.BoolOp):
            values = [self.constant(item) for item in node.values]
            if any(value is _UNKNOWN for value in values):
                return _UNKNOWN
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.IfExp):
            test = self.constant(node.test)
            return _UNKNOWN if test is _UNKNOWN else self.constant(node.body if test else node.orelse)
        if isinstance(node, ast.Subscript):
            value, index = self.constant(node.value), self.constant(node.slice)
            if isinstance(value, (str, tuple)) and isinstance(index, int) and -len(value) <= index < len(value):
                return value[index]
            return _UNKNOWN
        if isinstance(node, ast.BinOp):
            left, right = self.constant(node.left), self.constant(node.right)
            if left is _UNKNOWN or right is _UNKNOWN:
                return _UNKNOWN
            operations = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                          ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}
            operation = operations.get(type(node.op))
            try:
                return operation(left, right) if operation and not isinstance(left, bool) else _UNKNOWN
            except (TypeError, ZeroDivisionError, OverflowError, ValueError):
                return _UNKNOWN
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.constant(node.left), self.constant(node.comparators[0])
            if left is _UNKNOWN or right is _UNKNOWN:
                return _UNKNOWN
            comparisons = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
                           ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
                           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            comparison = comparisons.get(type(node.ops[0]))
            try:
                return comparison(left, right) if comparison else _UNKNOWN
            except TypeError:
                return _UNKNOWN
        return _UNKNOWN

    def matching_case(self, node) -> Optional[list]:
        """The body a match statement on a known value runs, or None when it can't be told."""
        subject = self.constant(node.subject)
        if subject is _UNKNOWN:
            return None

        def matches(pattern) -> Optional[bool]:
            if isinstance(pattern, ast.MatchValue):
                value = self.constant(pattern.value)
                return None if value is _UNKNOWN else value == subject
            if isinstance(pattern, ast.MatchSingleton):
                return pattern.value is subject
            if isinstance(pattern, ast.MatchOr):
                results = [matches(item) for item in pattern.patterns]
                return None if None in results else any(results)
            if isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
                return True
            return None

        for case in node.cases:
            result = matches(case.pattern)
            if result is None or case.guard is not None:
                return None
            if result:
                return case.body
        return []

    def validated(self, test: ast.AST, outcome: bool = True) -> Dict[str, Set[str]]:
        """The tainted names a condition shows are safe when it is outcome, and what for.

        if not name.isalnum(): return leaves only names that passed, so the check is read with
        outcome False after it; if name in ALLOWED: is read with outcome True inside it.
        """
        def names(node) -> Set[str]:
            found = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and item.id in self.tainted}
            return found | {self.parsed[name] for name in found if name in self.parsed}

        def merge(parts) -> Dict[str, Set[str]]:
            found: Dict[str, Set[str]] = {}
            for part in parts:
                for name, kinds in part.items():
                    found.setdefault(name, set()).update(kinds)
            return found

        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self.validated(test.operand, not outcome)
        if isinstance(test, ast.BoolOp):
            # All parts of an and hold when it's true; all parts of an or fail when it's false.
            if isinstance(test.op, ast.And) == outcome:
                return merge(self.validated(value, outcome) for value in test.values)
            return {}
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            operation, left, right = test.ops[0], test.left, test.comparators[0]
            if isinstance(operation, (ast.In, ast.NotIn)):
                present = isinstance(operation, ast.In) == outcome
                if isinstance(left, ast.Constant) and isinstance(left.value, str) and not present:
                    # A character or substring that isn't there: '..' and slashes only make a path safe.
                    kinds = {"path"} if any(part in left.value for part in ("..", "/", "\\")) else {"*"}
                    return {name: set(kinds) for name in names(right)}
                if not names(right) and present:
                    # One of a fixed set of allowed values.
                    return {name: {"*"} for name in names(left)}
                return {}
            if isinstance(operation, (ast.Eq, ast.NotEq)) and (isinstance(operation, ast.Eq) == outcome):
                for subject, other in ((left, right), (right, left)):
                    if isinstance(other, ast.Constant):
                        return {name: {"*"} for name in names(subject)}
            return {}
        if isinstance(test, ast.Call) and outcome:
            function = self.module.canonical(test.func)
            if function.split(".")[-1] in _REDIRECT_CHECKS and test.args:
                return {name: {"redirect", "ssrf"} for name in names(test.args[0])}
            if isinstance(test.func, ast.Attribute) and test.func.attr in _VALIDATING_METHODS - {"startswith", "endswith"}:
                return {name: {"*"} for name in names(test.func.value)}
            if function in _VALIDATING_FUNCTIONS and len(test.args) >= 2:
                return {name: {"*"} for name in names(test.args[1])}
        return {}

    def check(self, name: str, kinds: Set[str]) -> None:
        self.checked.setdefault(name, set()).update(kinds)

    def unsafe_names(self, node: ast.AST, sink: str) -> Set[str]:
        """Tainted names an expression uses that no check or sanitizer made safe for this sink."""
        if isinstance(node, ast.Call):
            kinds = _SANITIZER_KINDS.get(self.module.canonical(node.func)) or _SANITIZER_KINDS.get(dotted_name(node.func))
            if kinds and ({sink, "*"} & kinds):
                return set()
            helpers = self.helpers(node) if sink != "<none>" else []
            called = dotted_name(node.func).split(".")[-1]
            if sink == "xss" and called and _HTML_ESCAPER_NAME.search(called) and (
                    not helpers or all(_HTML_ESCAPER_NAME.search(helper.name) for _, helper in helpers)):
                return set()
            if sink in {"sql", "xpath"} and isinstance(node.func, ast.Attribute) and node.func.attr == "replace" \
                    and len(node.args) >= 2:
                # Quotes replaced or removed: the value can't leave a quoted SQL or XPath literal.
                old, new = self.constant(node.args[0]), self.constant(node.args[1])
                if isinstance(old, str) and old in {"'", '"'} and isinstance(new, str) and old not in new.replace(old * 2, ""):
                    return set()
            if helpers and self.follow_helper(node):
                return {"<helper>"}  # A project function's result, which no check here covers.
        if isinstance(node, ast.Attribute) and dotted_name(node.value) in self.request_names:
            return {"<request>"}  # Read straight from the request.
        if isinstance(node, ast.Name) and node.id in self.keyed and node.id not in self.tainted and sink != "<none>":
            kinds = self.keyed_kinds.get(node.id, {})
            unsafe = [key for key, value in self.keyed[node.id].items() if value and not ({sink, "*"} & kinds.get(key, set()))]
            return {node.id} if unsafe else set()
        if isinstance(node, ast.Name):
            safe = {sink, "*"} & self.checked.get(node.id, set()) if sink != "<none>" else set()
            tainted = node.id in self.tainted or node.id in self.keyed or node.id in self.lists
            return {node.id} if tainted and not safe else set()
        found: Set[str] = set()
        for child in ast.iter_child_nodes(node):
            found |= self.unsafe_names(child, sink)
        return found

    def safe_kinds(self, node: ast.AST) -> Set[str]:
        kinds = {kind for kind in _SINK_KINDS if not self.unsafe_names(node, kind)}
        return {"*"} if kinds == _SINK_KINDS else kinds

    def is_checked(self, node: ast.AST, sink: str) -> bool:
        """True when every input an operation uses was checked or sanitized for it."""
        # Input that came back from a call, with no checked name behind it, can't be shown safe.
        traced = self.unsafe_names(node, "<none>")
        return bool(traced) and not self.unsafe_names(node, sink)

    def add_parameter_types(self, node) -> None:
        defaults = dict(zip(reversed(node.args.args), reversed(node.args.defaults)))
        defaults.update({parameter: default for parameter, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default})
        for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            annotated = _class_annotation(parameter.annotation)
            default = defaults.get(parameter)
            # FastAPI dependencies, as in service: UserService = Depends(UserService).
            if isinstance(default, ast.Call) and dotted_name(default.func).split(".")[-1] == "Depends" and default.args:
                annotated = default.args[0]
            if annotated is not None:
                self.types[parameter.arg] = self.module.canonical(annotated)

    def remember(self, target: ast.AST, value: Optional[ast.AST]) -> None:
        """Records what a new object keeps, as self.request = request in its __init__."""
        if not isinstance(target, ast.Name):
            return
        name = target.id
        self.fields.pop(name, None)
        self.request_fields.pop(name, None)
        self.made_by.pop(name, None)
        self.paths.discard(name)
        if value is None:
            return
        if any(isinstance(item, ast.Call) and dotted_name(item.func).split(".")[-1] in {"Path", "PurePath"}
               for item in ast.walk(value)):
            self.paths.add(name)
        if not isinstance(value, ast.Call):
            return
        found = self.module.resolve_class(self.module.canonical(value.func))
        if found is None:
            helper = self.module.find_function(value, self)
            if helper is not None:
                self.made_by[name] = helper[0]
            return
        init = found[1]["methods"].get("__init__")
        if init is None:
            return
        parameters = [parameter.arg for parameter in init.args.args[1:]]
        inputs: Dict[str, str] = {}
        requests: Set[str] = set()
        arguments = list(zip(parameters, value.args)) + [(keyword.arg, keyword.value) for keyword in value.keywords]
        for parameter, argument in arguments:
            source = self.taint(argument)
            if source:
                inputs[parameter] = source
            elif dotted_name(argument) in self.request_names:
                requests.add(parameter)
        fields: Dict[str, str] = {}
        request_fields: Set[str] = set()
        for statement in init.body:
            if not (isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(statement.value, ast.Name)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for item in targets:
                if isinstance(item, ast.Attribute) and dotted_name(item.value) == "self":
                    if statement.value.id in inputs:
                        fields[item.attr] = inputs[statement.value.id]
                    elif statement.value.id in requests:
                        request_fields.add(item.attr)
        if fields:
            self.fields[name] = fields
        if request_fields:
            self.request_fields[name] = request_fields

    def object_state(self, node: ast.Call) -> tuple:
        """The input and request a method call's object kept, as self.<name> inside the method."""
        value = node.func.value if isinstance(node.func, ast.Attribute) else None
        if not isinstance(value, ast.Name):
            return {}, set()
        tainted = {f"self.{field}": source for field, source in self.fields.get(value.id, {}).items()}
        return tainted, {f"self.{field}" for field in self.request_fields.get(value.id, set())}

    def factory_methods(self, node: ast.Call) -> list:
        """A method call on what a factory made: any class in the factory's module may be it."""
        if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
            return []
        module = self.made_by.get(node.func.value.id)
        if module is None:
            return []
        methods = [info["methods"][node.func.attr] for info in module.classes.values() if node.func.attr in info["methods"]]
        return [(module, method) for method in methods] if len(methods) <= 5 else []

    def note_type(self, target: ast.AST, value: Optional[ast.AST], annotation: Optional[ast.AST] = None) -> None:
        if not isinstance(target, ast.Name):
            return
        annotated = _class_annotation(annotation) if annotation is not None else None
        if annotated is not None:
            self.types[target.id] = self.module.canonical(annotated)
        elif isinstance(value, ast.Call) and self.module.resolve_class(self.module.canonical(value.func)) is not None:
            self.types[target.id] = self.module.canonical(value.func)

    def block(self, statements: Iterable[ast.stmt], nested: bool) -> None:
        for statement in statements:
            self.statement(statement, nested)

    def statement(self, node: ast.stmt, nested: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.scan_calls(child)
        for item in getattr(node, "items", []) if isinstance(node, (ast.With, ast.AsyncWith)) else []:
            # with open(name) as handle: the opened call is an operation too.
            self.scan_calls(item.context_expr)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # A loop may change what it assigns, so none of it stays known inside or after it.
            for name in _assigned_names(node.body + node.orelse):
                self.constants.pop(name, None)
        if isinstance(node, ast.If):
            known = self.constant(node.test)
            if known is not _UNKNOWN:
                # Only one branch can run, and it always does.
                self.block(node.body if known else node.orelse, nested)
                return
            checks = self.validated(node.test, False)
            if checks and not node.orelse and _exits(node.body):
                # A check that leaves when it fails: what follows only sees values that passed.
                self.block(node.body, True)
                for name, kinds in checks.items():
                    self.check(name, kinds)
                return
            checks = self.validated(node.test, True)
            if checks and not _exits(node.body):
                # Inside if value in ALLOWED: the value passed the check.
                before = {name: set(kinds) for name, kinds in self.checked.items()}
                for name, kinds in checks.items():
                    self.check(name, kinds)
                self.block(node.body, True)
                self.checked = before
                self.block(node.orelse, True)
                return
        if isinstance(node, ast.Match):
            body = self.matching_case(node)
            if body is not None:
                self.block(body, nested)
                return
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.lists:
                    index = self.constant(target.slice)
                    items = self.lists[target.value.id]
                    if isinstance(index, int) and -len(items) <= index < len(items):
                        del items[index]
                    else:
                        self.degrade(target.value.id)
        if isinstance(node, ast.Assign) and self.assign_item(node):
            return
        if isinstance(node, ast.Assign):
            source = self.taint(node.value)
            for target in node.targets:
                self.note_type(target, node.value)
                self.remember(target, node.value)
                self.track(target, node.value, nested)
                self.assign(target, source, nested)
            if isinstance(node.value, ast.Call):
                self.collect(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self.note_type(node.target, node.value, node.annotation)
            self.remember(node.target, node.value)
            self.track(node.target, node.value, nested)
            self.assign(node.target, self.taint(node.value), nested)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                self.constants.pop(node.target.id, None)
            source = self.taint(node.value)
            if source and isinstance(node.target, ast.Name):
                # text += value is safe for what both the text so far and the value are safe for.
                name = node.target.id
                added = self.safe_kinds(node.value)
                before = self.checked.get(name) if name in self.tainted else {"*"}
                if before is None:
                    kinds: Set[str] = set()
                elif "*" in before:
                    kinds = added
                elif "*" in added:
                    kinds = before
                else:
                    kinds = before & added
                if kinds:
                    self.checked[name] = set(kinds)
                else:
                    self.checked.pop(name, None)
            if source:
                self.assign(node.target, source, nested)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self.assign(node.target, self.taint(node.iter), True)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, self.taint(item.context_expr), True)
        elif isinstance(node, ast.Return) and node.value is not None:
            if not self.returned:
                self.returned = self.taint(node.value)
            if self.view and self.module.uses_flask and self.returns_text(node.value):
                source = self.taint(node.value)
                if source:
                    self.report(node.value, "xss", "return", source, "medium")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self.collect(node.value)
        for field in ("body", "orelse", "finalbody"):
            self.block(getattr(node, field, []) or [], True)
        for handler in getattr(node, "handlers", []) or []:
            self.block(handler.body, True)
        for case in getattr(node, "cases", []) or []:
            self.block(case.body, True)

    def track(self, target: ast.AST, value: ast.AST, nested: bool) -> None:
        """Records known values, and dicts and lists to follow by key and position."""
        if not isinstance(target, ast.Name):
            return
        name = target.id
        known = self.constant(value)
        if known is not _UNKNOWN and not nested and isinstance(known, (str, int, float, bool, tuple)):
            self.constants[name] = known
        else:
            self.constants.pop(name, None)
        self.keyed.pop(name, None)
        self.keyed_kinds.pop(name, None)
        self.lists.pop(name, None)
        if isinstance(value, ast.Dict) and all(key is not None and self.constant(key) is not _UNKNOWN for key in value.keys):
            self.keyed[name] = {self.constant(key): self.taint(item) for key, item in zip(value.keys, value.values)}
        elif isinstance(value, ast.List) and not any(isinstance(item, ast.Starred) for item in value.elts):
            self.lists[name] = [self.taint(item) for item in value.elts]
        # A value made only from checked or sanitized input keeps what it's safe for.
        kinds = self.safe_kinds(value) if self.taint(value) else set()
        if kinds:
            self.checked[name] = kinds
        else:
            self.checked.pop(name, None)
        if any(isinstance(item, ast.Name) and item.id in self.paths for item in ast.walk(value)):
            self.paths.add(name)
        if isinstance(value, ast.Call) and dotted_name(value.func).split(".")[-1] in {"ConfigParser", "RawConfigParser"}:
            self.keyed[name] = {}
        self.parsed.pop(name, None)
        if isinstance(value, ast.Call) and dotted_name(value.func).split(".")[-1] in {"urlparse", "urlsplit"} \
                and value.args and isinstance(value.args[0], ast.Name):
            self.parsed[name] = value.args[0].id
        if self.returns_text(value):
            self.texts.add(name)
        else:
            self.texts.discard(name)
        self.unsafe_xml.discard(name)
        if isinstance(value, ast.Call) and dotted_name(value.func).split(".")[-1] == "XMLParser":
            options = {keyword.arg: self.constant(keyword.value) for keyword in value.keywords}
            if options.get("resolve_entities") is True or options.get("no_network") is False:
                self.unsafe_xml.add(name)

    def degrade(self, name: str) -> None:
        """A change that can't be followed by key or position: the collection is one value from now on."""
        values = list(self.keyed.pop(name, {}).values()) + list(self.lists.pop(name, []))
        source = next((value for value in values if value), None)
        if source:
            self.tainted[name] = source

    def assign_item(self, node: ast.Assign) -> bool:
        """values['key'] = x or items[0] = x on a followed dict or list."""
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
            return False
        target = node.targets[0]
        if not isinstance(target.value, ast.Name):
            return False
        name, key = target.value.id, self.constant(target.slice)
        if name in self.keyed and key is not _UNKNOWN:
            self.keyed[name][key] = self.taint(node.value)
            self.keyed_kinds.setdefault(name, {})[key] = self.safe_kinds(node.value) if self.keyed[name][key] else {"*"}
            return True
        if name in self.lists and isinstance(key, int) and -len(self.lists[name]) <= key < len(self.lists[name]):
            self.lists[name][key] = self.taint(node.value)
            return True
        if name in self.keyed or name in self.lists:
            self.degrade(name)
        return False

    def read_item(self, name: str, key) -> object:
        """A followed dict's value by key, or list's item by position: a source, None, or _UNKNOWN."""
        if name in self.keyed and key is not _UNKNOWN:
            return self.keyed[name].get(key)
        items = self.lists.get(name)
        if items is not None and isinstance(key, int) and -len(items) <= key < len(items):
            return items[key]
        return _UNKNOWN

    def collect(self, call: ast.Call) -> None:
        """parts.append(value) and the like: the collection now holds the value."""
        if isinstance(call.func, ast.Attribute) and call.func.attr == "setFeature" and isinstance(call.func.value, ast.Name) \
                and len(call.args) == 2 and "external" in dotted_name(call.args[0]).lower() + str(self.constant(call.args[0])).lower() \
                and self.constant(call.args[1]) is True:
            self.unsafe_xml.add(call.func.value.id)
            return
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            name, method = call.func.value.id, call.func.attr
            if name in self.lists:
                items = self.lists[name]
                index = self.constant(call.args[0]) if call.args else _UNKNOWN
                if method == "append" and len(call.args) == 1:
                    items.append(self.taint(call.args[0]))
                elif method == "insert" and len(call.args) == 2 and isinstance(index, int):
                    items.insert(index, self.taint(call.args[1]))
                elif method == "pop" and (not call.args or (isinstance(index, int) and -len(items) <= index < len(items))):
                    if items:
                        items.pop(index if call.args else -1)
                elif method == "clear":
                    items.clear()
                elif method not in {"copy", "count", "index"}:
                    self.degrade(name)
                    if method in _COLLECTION_ADDS:
                        source = self.first(call.args)
                        if source:
                            self.tainted[name] = self.tainted.get(name) or source
                return
            if name in self.keyed:
                key = self.constant(call.args[0]) if call.args else _UNKNOWN
                if method == "set" and len(call.args) == 3 and all(self.constant(item) is not _UNKNOWN for item in call.args[:2]):
                    self.keyed[name][(self.constant(call.args[0]), self.constant(call.args[1]))] = self.taint(call.args[2])
                elif method in {"add_section", "has_option", "has_section", "options", "sections"}:
                    pass
                elif method in {"setdefault"} and key is not _UNKNOWN and len(call.args) == 2:
                    self.keyed[name].setdefault(key, self.taint(call.args[1]))
                elif method == "pop" and key is not _UNKNOWN:
                    self.keyed[name].pop(key, None)
                elif method not in {"get", "keys", "copy", "items", "values"}:
                    self.degrade(name)
                    source = self.first([*call.args, *[keyword.value for keyword in call.keywords]])
                    if source:
                        self.tainted[name] = self.tainted.get(name) or source
                return
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in _COLLECTION_ADDS:
            return
        source = self.first([*call.args, *[keyword.value for keyword in call.keywords]])
        if source and isinstance(call.func.value, (ast.Name, ast.Attribute)):
            self.assign(call.func.value, source, True)

    def assign(self, target: ast.AST, source: Optional[str], nested: bool) -> None:
        if isinstance(target, ast.Name):
            if source:
                self.tainted[target.id] = source
            elif not nested:
                # A value overwritten inside a branch may still be tainted on another path.
                self.tainted.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.assign(element, source, nested)
        elif isinstance(target, ast.Starred):
            self.assign(target.value, source, nested)
        elif isinstance(target, ast.Attribute) and source and (
                dotted_name(target.value) in self.request_names or self.module.canonical(target.value) in {"flask.g", "g"}):
            # Middleware or a decorator keeping input on request or flask.g for later views.
            owner = "g" if self.module.canonical(target.value) in {"flask.g", "g"} else "request"
            self.module.extras.setdefault((owner, target.attr), f"{dotted_name(target)} ({source})")
        elif isinstance(target, (ast.Attribute, ast.Subscript)) and source:
            name = dotted_name(target)
            if isinstance(target, ast.Subscript):
                name = dotted_name(target.value)
            if name:
                self.tainted[name.split(".")[0] if isinstance(target, ast.Subscript) else name] = source

    def taint(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None or isinstance(node, ast.Constant):
            return None
        if isinstance(node, ast.Name):
            source = self.tainted.get(node.id)
            if not source and (node.id in self.keyed or node.id in self.lists):
                values = list(self.keyed.get(node.id, {}).values()) + list(self.lists.get(node.id, []))
                source = next((value for value in values if value), None)
            return source
        if isinstance(node, ast.Await):
            return self.taint(node.value)
        if isinstance(node, ast.Attribute):
            name = dotted_name(node)
            if name in self.tainted:
                return self.tainted[name]
            if dotted_name(node.value) in self.request_names and node.attr in _REQUEST_ATTRIBUTES:
                return f"{dotted_name(node.value)}.{node.attr}"
            if dotted_name(node.value) in self.request_names and ("request", node.attr) in self.module.extras:
                return self.module.extras[("request", node.attr)]
            if self.module.canonical(node.value) in {"flask.g", "g"} and ("g", node.attr) in self.module.extras:
                return self.module.extras[("g", node.attr)]
            return self.taint(node.value)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                item = self.read_item(node.value.id, self.constant(node.slice))
                if item is not _UNKNOWN:
                    return item
            return _with_key(self.taint(node.value), node.slice)
        if isinstance(node, ast.BinOp):
            return self.taint(node.left) or self.taint(node.right)
        if isinstance(node, ast.JoinedStr):
            return self.first(node.values)
        if isinstance(node, ast.FormattedValue):
            return self.taint(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return self.first(node.elts)
        if isinstance(node, ast.Dict):
            return self.first(node.values)
        if isinstance(node, ast.IfExp):
            known = self.constant(node.test)
            if known is not _UNKNOWN:
                return self.taint(node.body if known else node.orelse)
            return self.taint(node.body) or self.taint(node.orelse)
        if isinstance(node, ast.BoolOp):
            return self.first(node.values)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self.taint(node.elt) or self.first(generator.iter for generator in node.generators)
        if isinstance(node, ast.DictComp):
            return self.taint(node.value) or self.first(generator.iter for generator in node.generators)
        if isinstance(node, ast.Starred):
            return self.taint(node.value)
        if isinstance(node, ast.Call):
            return self.call_taint(node)
        return None

    def first(self, nodes: Iterable[ast.AST]) -> Optional[str]:
        for node in nodes:
            source = self.taint(node)
            if source:
                return source
        return None

    def call_taint(self, node: ast.Call) -> Optional[str]:
        name = self.module.canonical(node.func)
        if name in _SANITIZERS or dotted_name(node.func) in _SANITIZERS:
            return None
        if name in _SANITIZER_KINDS or dotted_name(node.func) in _SANITIZER_KINDS:
            # Safe only for some operations, which is checked where the value is used.
            return self.first(node.args)
        # A project function's result carries the input it was given, when it returns it.
        returned = self.follow_helper(node)
        if returned:
            return returned
        if name in _TEXT_PASSTHROUGH:
            return self.first(node.args)
        if isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and node.func.attr in {"get", "pop"} and node.args:
                key = self.constant(node.args[0])
                if node.func.attr == "get" and len(node.args) >= 2 and receiver.id in self.keyed \
                        and isinstance(self.keyed[receiver.id], dict) and any(isinstance(item, tuple) for item in self.keyed[receiver.id]):
                    second = self.constant(node.args[1])
                    key = _UNKNOWN if key is _UNKNOWN or second is _UNKNOWN else (key, second)
                item = self.read_item(receiver.id, key)
                if item is not _UNKNOWN:
                    return item
            if dotted_name(receiver) in self.request_names and node.func.attr in _REQUEST_ATTRIBUTES:
                return f"{dotted_name(receiver)}.{node.func.attr}()"
            if node.func.attr in _STRING_METHODS:
                source = self.taint(receiver)
                if source:
                    if node.func.attr in {"get", "getlist", "pop"} and node.args:
                        return _with_key(source, node.args[0])
                    return source
                if node.func.attr in {"format", "join"}:
                    return self.first([*node.args, *[keyword.value for keyword in node.keywords]])
                return None
            return self.taint(receiver) if node.func.attr in _REQUEST_ATTRIBUTES else None
        return None

    def scan_calls(self, expression: ast.AST) -> None:
        for node in ast.walk(expression):
            if isinstance(node, ast.Call):
                # A method of one of the project's classes is followed into, not guessed at by its name.
                if self.module.find_method(node, self) is None:
                    self.check_sink(node)
                self.follow_helper(node)

    def passed(self, helper, node: ast.Call) -> Dict[str, str]:
        """The helper's parameters that receive request input in this call."""
        parameters = [parameter.arg for parameter in helper.args.args if parameter.arg not in {"self", "cls"}]
        passed: Dict[str, str] = {}
        for parameter, argument in zip(parameters, node.args):
            source = self.taint(argument)
            if source:
                passed[parameter] = source
        for keyword in node.keywords:
            source = self.taint(keyword.value)
            if source and keyword.arg in parameters:
                passed[keyword.arg] = source
        return passed

    def returns_text(self, value: ast.AST) -> bool:
        """True for a returned string, which Flask sends as HTML; not for dicts, lists or JSON."""
        if isinstance(value, (ast.JoinedStr, ast.BinOp)):
            return True
        if isinstance(value, ast.Constant):
            return isinstance(value.value, str)
        if isinstance(value, ast.Name):
            return value.id in self.texts
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            return value.func.attr in {"format", "join", "replace", "strip", "decode"}
        return False

    def helpers(self, node: ast.Call) -> list:
        found = self.module.find_function(node, self)
        return [found] if found is not None else self.factory_methods(node)

    def follow_helper(self, node: ast.Call) -> Optional[str]:
        """Walks the project functions a call runs, once per call; returns the input their result carries."""
        key = id(node)
        if key in self._followed:
            return self._followed[key]
        self._followed[key] = None
        if self.depth >= _MAX_HELPER_DEPTH:
            return None
        result = None
        for module, helper in self.helpers(node):
            passed = self.passed(helper, node)
            state, requests = self.object_state(node)
            for field, source in state.items():
                passed.setdefault(field, source)
            if passed or requests:
                returned = module._analyze(helper, passed, f"{self.label} -> {helper.name}", self.depth + 1,
                                           self.active, self.entry_file, requests)
                result = result or returned
        self._followed[key] = result
        return result

    def check_sink(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            receiver = dotted_name(node.func.value).lower()
            method = node.func.attr
        else:
            receiver, method = "", ""
        name = self.module.canonical(node.func)
        short = dotted_name(node.func)
        first = node.args[0] if node.args else None
        source = self.taint(first)
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}

        if method in _SQL_METHODS or name in _SQL_FUNCTIONS or short in _SQL_FUNCTIONS:
            if source and not (isinstance(first, ast.Name) and method == "execute" and first.id == "text"):
                self.report(node, "sql", short, source, "high")
            return
        if method == "extra":
            source = self.first(keywords[key] for key in ("where", "select", "tables") if key in keywords)
            if source:
                self.report(node, "sql", short, source, "high")
            return
        if method in _NOSQL_METHODS and any(hint in receiver for hint in _NOSQL_RECEIVER_HINTS):
            if source:
                self.report(node, "nosql", short, source, "medium")
            return
        if name in _COMMAND_FUNCTIONS and source:
            self.report(node, "command", short, source, "high")
            return
        if name in _SUBPROCESS_FUNCTIONS:
            shell = keywords.get("shell")
            uses_shell = isinstance(shell, ast.Constant) and shell.value is True
            if source and (uses_shell or not isinstance(first, (ast.List, ast.Tuple))):
                self.report(node, "command", short, source, "high" if uses_shell else "medium")
            return
        if name in _PATH_FUNCTIONS or short in _PATH_FUNCTIONS:
            if source:
                self.report(node, "path", short, source, "medium")
            return
        if method in _PATH_METHODS and isinstance(node.func.value, ast.Name) and node.func.value.id in self.paths:
            path_source = self.taint(node.func.value)
            if path_source:
                self.report(node, "path", short, path_source, "medium")
            return
        if name in _SSRF_FUNCTIONS or short in _SSRF_FUNCTIONS or _is_http_client_call(receiver, method):
            # A URL can also be passed by keyword, or second as in request("GET", url).
            target = self.taint(keywords.get("url"))
            if not target and method == "request" and len(node.args) > 1:
                target = self.taint(node.args[1])
            elif not target and method != "request":
                target = source
            if target:
                self.report(node, "ssrf", short, target, "medium")
            return
        if (short in _CODE_FUNCTIONS or name in _IMPORT_FUNCTIONS) and source:
            self.report(node, "code", short, source, "high")
            return
        if method == "from_string" and source and any(hint in receiver for hint in ("env", "jinja", "template", "loader")):
            # jinja2.Environment().from_string(text) compiles text as a template.
            self.report(node, "template", short, source, "high")
            return
        if (name in _TEMPLATE_FUNCTIONS or short in _TEMPLATE_FUNCTIONS) and source:
            self.report(node, "template", short, source, "high")
            return
        last = name.split(".")[-1]
        if last in _HTML_RESPONSES and (last != "make_response" or self.module.uses_flask):
            if isinstance(first, ast.Tuple) and first.elts:
                # make_response((body, status, headers)): only the body is shown.
                first = first.elts[0]
                source = self.taint(first)
            if source and not self.is_checked(first, "xss"):
                self.report(node, "xss", short, source, "high" if last in {"Markup", "mark_safe"} else "medium")
            return
        if last in _REDIRECTS and (last != "redirect" or name in {"redirect", "flask.redirect", "django.shortcuts.redirect",
                                                                  "quart.redirect", "shortcuts.redirect"}):
            target = source or self.taint(keywords.get("location")) or self.taint(keywords.get("url"))
            if target:
                self.report(node, "redirect", short, target, "medium")
            return
        parser_call = (method in {"feed", "parse", "parseString"} and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name) and node.func.value.id in self.unsafe_xml)
        if name in _XML_PARSERS or short in _XML_PARSERS or parser_call:
            parsers = [*node.args[1:], *keywords.values()]
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                parsers.append(node.func.value)
            if source and any(isinstance(item, ast.Name) and item.id in self.unsafe_xml for item in parsers):
                self.report(node, "xxe", short, source, "high")
            return
        if name in _XPATH_FUNCTIONS or short in _XPATH_FUNCTIONS or method == "xpath":
            if source:
                self.report(node, "xpath", short, source, "high")
            return
        if name in _ELEMENTPATH_FUNCTIONS:
            query = self.taint(node.args[1]) if len(node.args) > 1 else self.taint(keywords.get("path"))
            if query:
                self.report(node, "xpath", short, query, "high")
            return
        if method in {"search", "search_s", "search_st", "search_ext", "search_ext_s"} and (
                method != "search" or any(hint in receiver for hint in ("conn", "ldap"))):
            position = 1 if method == "search" else 2
            query = self.taint(node.args[position]) if len(node.args) > position else self.taint(
                keywords.get("search_filter") or keywords.get("filterstr"))
            if query:
                self.report(node, "ldap", short, query, "high")
            return
        if name in _REGEX_FUNCTIONS and source:
            self.report(node, "redos", short, source, "medium")
            return
        if name in _DESERIALIZE_FUNCTIONS and source:
            loader = keywords.get("Loader")
            if name == "yaml.load" and loader is not None and dotted_name(loader) in _SAFE_YAML_LOADERS:
                return
            self.report(node, "deserialization", short, source, "high")

    def report(self, node, sink, call, source, confidence) -> None:
        if self.is_checked(node, sink):
            return
        self.module.record(node, self.label, sink, call, source, confidence, self.entry_file)


def _as_dict(finding: Finding) -> dict:
    """A finding in the shared report shape; entry_file only when the view is in another file."""
    data = asdict(finding)
    if not data["entry_file"]:
        del data["entry_file"]
    return data


def analyze_source(source: str, filename: str = "<source>") -> List[Finding]:
    """Return injection findings for one Python file. Raises SyntaxError on invalid source."""
    tree = ast.parse(source, filename=filename)
    analyzer = _ModuleAnalyzer(tree, filename)
    for _ in range(3):
        before = len(analyzer.extras)
        findings = analyzer.run()
        if len(analyzer.extras) == before:
            break
        analyzer.summaries.clear()
    return sorted(findings, key=lambda item: (item.line, item.sink))


def _python_files(root: Path, include_tests: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in _SKIPPED_DIRECTORIES and (include_tests or name not in {"test", "tests"})
        )
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            if not include_tests and (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"):
                continue
            yield Path(directory) / name


def analyze_path(path: str = ".", *, include_tests: bool = False) -> dict:
    """Scan a file or folder and return a report in the shared AIWAF audit format."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    findings: List[Finding] = []
    errors: List[dict] = []
    # Every module is parsed first, so calls from one into another can be followed.
    project: Dict[str, Optional[_ModuleAnalyzer]] = {}
    analyzers: List[_ModuleAnalyzer] = []
    extras: Dict[tuple, str] = {}
    project["\0shared"] = {"summaries": {}, "recordings": []}  # type: ignore[assignment]
    base = root if root.is_dir() else root.parent
    for file in _python_files(root, include_tests):
        display = file.as_posix()
        try:
            if file.stat().st_size > _MAX_FILE_BYTES:
                errors.append({"file": display, "error": "File is too large to analyze"})
                continue
            parts = list(file.relative_to(base).with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            module_name = ".".join(parts)
            text = file.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=display)
            analyzer = _ModuleAnalyzer(tree, display, module_name, project, extras)
            analyzer.lines = text.splitlines()
            analyzers.append(analyzer)
            # Known by every dotted suffix too (src/app/queries.py as app.queries or queries),
            # since imports name modules from wherever the project's root is; an ambiguous name
            # isn't followed.
            for start in range(len(parts)):
                name = ".".join(parts[start:])
                project[name] = None if name in project and project[name] is not analyzer else analyzer
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError, RecursionError) as error:
            errors.append({"file": display, "error": str(error)})
    for name in [name for name, analyzer in project.items() if analyzer is None]:
        del project[name]
    summaries = project["\0shared"]["summaries"]  # type: ignore[index]
    scanned = 0
    failed: Set[str] = set()
    # Again when middleware was found keeping input on the request, for the views that ran first.
    for _ in range(3):
        before = len(extras)
        for analyzer in analyzers:
            if analyzer.filename in failed:
                continue
            try:
                analyzer.run()
            except RecursionError as error:
                failed.add(analyzer.filename)
                errors.append({"file": analyzer.filename, "error": str(error)})
        if len(extras) == before:
            break
        summaries.clear()
    scanned = len(analyzers) - len(failed)
    inputs: List[dict] = []
    for analyzer in analyzers:
        findings.extend(analyzer.findings.values())
        if analyzer.filename not in failed:
            try:
                inputs.extend(module_inputs(analyzer.tree, analyzer.filename))
            except RecursionError:
                pass  # The file's routes get no saved inputs, so their checks stay as they are.
    kept, suppressed = _without_ignored(findings, {analyzer.filename: analyzer.lines for analyzer in analyzers})
    report = {
        "schema_version": SCHEMA_VERSION,
        "language": "python",
        "files_scanned": scanned,
        "findings": [_as_dict(finding) for finding in sorted(kept, key=lambda item: (item.file, item.line, item.sink, item.function))],
        "errors": errors,
        "inputs": inputs,
    }
    if suppressed:
        report["suppressed"] = suppressed
    return report


# aiwaf: ignore, or aiwaf: ignore[sql, xss], on a finding's line or the line before it.
_IGNORE = re.compile(r"aiwaf[:-]\s*ignore(?:\[([a-z_,\s]+)\])?", re.IGNORECASE)


def _ignored(lines: List[str], line: int, sink: str) -> bool:
    for number in (line, line - 1):
        if 1 <= number <= len(lines):
            match = _IGNORE.search(lines[number - 1])
            if match and (not match.group(1) or sink in {item.strip().lower() for item in match.group(1).split(",")}):
                return True
    return False


def _without_ignored(findings: List[Finding], lines: Dict[str, List[str]]) -> tuple:
    kept = [finding for finding in findings if not _ignored(lines.get(finding.file, []), finding.line, finding.sink)]
    return kept, len(findings) - len(kept)


def fingerprint(finding: dict) -> str:
    """Identifies a finding across runs, without its line, which moves as code changes."""
    parts = [str(finding.get(key, "")) for key in ("file", "sink", "call", "function", "source")]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]


def apply_baseline(report: dict, baseline: dict) -> dict:
    """Leaves out findings already in an earlier report, so CI fails only on new ones."""
    known = {fingerprint(finding) for finding in baseline.get("findings", [])}
    kept = [finding for finding in report["findings"] if fingerprint(finding) not in known]
    result = dict(report, findings=kept)
    if len(kept) != len(report["findings"]):
        result["baselined"] = len(report["findings"]) - len(kept)
    return result


SINK_TITLES = {
    "code": "Code injection", "command": "Command injection", "deserialization": "Unsafe deserialization",
    "ldap": "LDAP injection", "nosql": "NoSQL injection", "path": "Path traversal", "prototype": "Prototype pollution",
    "redirect": "Open redirect", "redos": "Regular expression denial of service", "sql": "SQL injection",
    "ssrf": "Server-side request forgery", "template": "Template injection", "xpath": "XPath injection",
    "xss": "Cross-site scripting", "xxe": "XML external entity expansion",
}


def to_sarif(report: dict) -> str:
    """The report as SARIF 2.1.0, for GitHub code scanning and other tools that read it."""
    sinks = sorted({finding["sink"] for finding in report["findings"]})
    rules = [{
        "id": f"aiwaf.{sink}",
        "name": SINK_TITLES.get(sink, sink).title().replace(" ", "").replace("-", ""),
        "shortDescription": {"text": SINK_TITLES.get(sink, sink)},
        "fullDescription": {"text": f"Request input can reach an operation where it allows {SINK_TITLES.get(sink, sink).lower()}."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["security"]},
    } for sink in sinks]
    results = [{
        "ruleId": f"aiwaf.{finding['sink']}",
        "ruleIndex": sinks.index(finding["sink"]),
        "level": "error" if finding["confidence"] == "high" else "warning",
        "message": {"text": f"{finding['source']} reaches {finding['call']} in {finding['function']}"},
        "locations": [{"physicalLocation": {
            "artifactLocation": {"uri": str(finding["file"]).replace("\\", "/").removeprefix("./")},
            "region": {"startLine": max(int(finding["line"]), 1)},
        }}],
        "partialFingerprints": {"aiwafFinding/v1": fingerprint(finding)},
    } for finding in report["findings"]]
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "AIWAF audit", "informationUri": "https://github.com/aiwaf-project/aiwaf", "rules": rules}},
            "results": results,
        }],
    }
    return json.dumps(sarif, indent=2)


def format_report(report: dict) -> str:
    """Return a readable report."""
    lines = []
    for finding in report["findings"]:
        lines.append(
            f"{finding['file']}:{finding['line']}: {finding['sink']} ({finding['confidence']} confidence) "
            f"{finding['source']} reaches {finding['call']} in {finding['function']}"
        )
    for error in report["errors"]:
        lines.append(f"{error['file']}: not analyzed: {error['error']}")
    lines.append(
        f"Scanned {report['files_scanned']} files and found {len(report['findings'])} possible injection paths."
    )
    if report.get("suppressed"):
        lines.append(f"Left out {report['suppressed']} marked with an aiwaf: ignore comment.")
    if report.get("baselined"):
        lines.append(f"Left out {report['baselined']} already in the baseline.")
    if not report["findings"]:
        lines.append("This doesn't prove the code is safe, because static analysis can miss paths.")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
