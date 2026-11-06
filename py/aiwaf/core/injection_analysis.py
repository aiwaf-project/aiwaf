"""Static analysis that follows request input to dangerous operations.

This reports possible injection paths. It reads source files only and never runs
application code or affects request handling. A clean report doesn't prove code is safe:
input that crosses modules, dynamic calls or frameworks it can't see can be missed.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

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
    "query_string", "stream", "values",
}
_REQUEST_NAMES = {"request", "req"}
_NON_INPUT_PARAMETERS = {"self", "cls", "request", "req", "db", "session", "response", "background_tasks"}
_NON_INPUT_ANNOTATIONS = {
    "AsyncSession", "BackgroundTasks", "Connection", "Request", "Response", "Session", "WebSocket",
}
_ROUTE_DECORATORS = {"api_route", "delete", "get", "head", "options", "patch", "post", "put", "route", "websocket"}
_SANITIZERS = {
    "UUID", "abs", "bleach.clean", "bool", "escape", "float", "int", "len", "markupsafe.escape",
    "quote", "secure_filename", "shlex.quote", "urllib.parse.quote", "uuid.UUID", "werkzeug.utils.secure_filename",
}
_TEXT_PASSTHROUGH = {
    "bytes", "dict", "f", "format", "list", "os.path.join", "repr", "set", "str", "tuple",
    "unquote", "unquote_plus", "urljoin", "urllib.parse.unquote", "urllib.parse.unquote_plus", "urllib.parse.urljoin",
}
_STRING_METHODS = {
    "casefold", "decode", "encode", "format", "get", "getlist", "join", "lower", "lstrip", "pop",
    "replace", "rstrip", "split", "strip", "title", "upper", "values",
}
_SQL_METHODS = {"execute", "executemany", "executescript", "raw"}
_SQL_FUNCTIONS = {"RawSQL", "sql.text", "sqlalchemy.text", "text"}
_NOSQL_METHODS = {
    "aggregate", "count_documents", "delete_many", "delete_one", "find", "find_one",
    "find_one_and_update", "update_many", "update_one",
}
_NOSQL_RECEIVER_HINTS = ("coll", "db", "mongo", "users", "documents")
_COMMAND_FUNCTIONS = {"os.popen", "os.system"}
_SUBPROCESS_FUNCTIONS = {
    "subprocess.Popen", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput", "subprocess.run",
}
_PATH_FUNCTIONS = {
    "FileResponse", "Path", "io.open", "open", "os.remove", "os.rmdir", "os.unlink", "pathlib.Path",
    "send_file", "shutil.copy", "shutil.move", "shutil.rmtree",
}
_SSRF_FUNCTIONS = {
    "httpx.delete", "httpx.get", "httpx.head", "httpx.post", "httpx.put", "httpx.request",
    "requests.delete", "requests.get", "requests.head", "requests.patch", "requests.post",
    "requests.put", "requests.request", "urllib.request.urlopen", "urlopen",
}
_SCALAR_ANNOTATIONS = {"", "Any", "Optional", "UUID", "bool", "bytes", "float", "int", "str"}
_HTTP_CLIENT_METHODS = {"delete", "get", "head", "patch", "post", "put"}
_CODE_FUNCTIONS = {"compile", "eval", "exec"}
_TEMPLATE_FUNCTIONS = {"Template", "jinja2.Template", "render_template_string"}
_DESERIALIZE_FUNCTIONS = {"dill.loads", "marshal.loads", "pickle.load", "pickle.loads", "shelve.open", "yaml.load"}
_SAFE_YAML_LOADERS = {"CSafeLoader", "SafeLoader", "yaml.CSafeLoader", "yaml.SafeLoader"}


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


class _ModuleAnalyzer:
    def __init__(self, tree: ast.Module, filename: str):
        self.filename = filename
        self.findings: Dict[tuple, Finding] = {}
        self.functions: Dict[str, ast.AST] = {}
        self.flask_request = False
        self.aliases: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, node)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    if alias.name == "request" and node.module.split(".")[0] in {"flask", "quart"}:
                        self.flask_request = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
        self.tree = tree

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
        if any(self._is_route_decorator(decorator) for decorator in node.decorator_list):
            defaults = dict(zip(reversed(node.args.args), reversed(node.args.defaults)))
            for parameter in parameters:
                default = defaults.get(parameter)
                dependency = isinstance(default, ast.Call) and dotted_name(default).split(".")[-1] in {"Depends", "Security"}
                annotation = _annotation_name(parameter.annotation)
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

    def _analyze(self, node, tainted, label, depth, active) -> None:
        if node.name in active or depth > _MAX_HELPER_DEPTH:
            return
        active = {*active, node.name}
        request_names = self._request_names(node)
        analyzer = _FunctionAnalyzer(self, label, dict(tainted), request_names, depth, active)
        analyzer.block(node.body, nested=False)

    def _request_names(self, node) -> Set[str]:
        names = {"request"} if self.flask_request else set()
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for parameter in parameters:
            annotation = _annotation_name(parameter.annotation)
            if parameter.arg in _REQUEST_NAMES or annotation in {"HttpRequest", "Request"}:
                names.add(parameter.arg)
        return names

    def record(self, node, label, sink, call, source, confidence) -> None:
        key = (node.lineno, sink, call)
        self.findings.setdefault(key, Finding(self.filename, node.lineno, label, sink, call, source, confidence))


class _FunctionAnalyzer:
    def __init__(self, module, label, tainted, request_names, depth, active):
        self.module = module
        self.label = label
        self.tainted = tainted
        self.request_names = request_names
        self.depth = depth
        self.active = active

    def block(self, statements: Iterable[ast.stmt], nested: bool) -> None:
        for statement in statements:
            self.statement(statement, nested)

    def statement(self, node: ast.stmt, nested: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.scan_calls(child)
        if isinstance(node, ast.Assign):
            source = self.taint(node.value)
            for target in node.targets:
                self.assign(target, source, nested)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self.assign(node.target, self.taint(node.value), nested)
        elif isinstance(node, ast.AugAssign):
            source = self.taint(node.value)
            if source:
                self.assign(node.target, source, nested)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self.assign(node.target, self.taint(node.iter), True)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, self.taint(item.context_expr), True)
        for field in ("body", "orelse", "finalbody"):
            self.block(getattr(node, field, []) or [], True)
        for handler in getattr(node, "handlers", []) or []:
            self.block(handler.body, True)
        for case in getattr(node, "cases", []) or []:
            self.block(case.body, True)

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
            return self.tainted.get(node.id)
        if isinstance(node, ast.Await):
            return self.taint(node.value)
        if isinstance(node, ast.Attribute):
            name = dotted_name(node)
            if name in self.tainted:
                return self.tainted[name]
            if isinstance(node.value, ast.Name) and node.value.id in self.request_names and node.attr in _REQUEST_ATTRIBUTES:
                return f"{node.value.id}.{node.attr}"
            return self.taint(node.value)
        if isinstance(node, ast.Subscript):
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
        if name in _TEXT_PASSTHROUGH:
            return self.first(node.args)
        if isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in self.request_names and node.func.attr in _REQUEST_ATTRIBUTES:
                return f"{receiver.id}.{node.func.attr}()"
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
                self.check_sink(node)
                self.follow_helper(node)

    def follow_helper(self, node: ast.Call) -> None:
        name = dotted_name(node.func).split(".")[-1]
        helper = self.module.functions.get(name)
        if helper is None or self.depth >= _MAX_HELPER_DEPTH:
            return
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
        if passed:
            self.module._analyze(helper, passed, f"{self.label} -> {helper.name}", self.depth + 1, self.active)

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
        if short in _CODE_FUNCTIONS and source:
            self.report(node, "code", short, source, "high")
            return
        if (name in _TEMPLATE_FUNCTIONS or short in _TEMPLATE_FUNCTIONS) and source:
            self.report(node, "template", short, source, "high")
            return
        if name in _DESERIALIZE_FUNCTIONS and source:
            loader = keywords.get("Loader")
            if name == "yaml.load" and loader is not None and dotted_name(loader) in _SAFE_YAML_LOADERS:
                return
            self.report(node, "deserialization", short, source, "high")

    def report(self, node, sink, call, source, confidence) -> None:
        self.module.record(node, self.label, sink, call, source, confidence)


def analyze_source(source: str, filename: str = "<source>") -> List[Finding]:
    """Return injection findings for one Python file. Raises SyntaxError on invalid source."""
    tree = ast.parse(source, filename=filename)
    return sorted(_ModuleAnalyzer(tree, filename).run(), key=lambda item: (item.line, item.sink))


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
    scanned = 0
    for file in _python_files(root, include_tests):
        display = file.as_posix()
        try:
            if file.stat().st_size > _MAX_FILE_BYTES:
                errors.append({"file": display, "error": "File is too large to analyze"})
                continue
            findings.extend(analyze_source(file.read_text(encoding="utf-8"), display))
            scanned += 1
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError, RecursionError) as error:
            errors.append({"file": display, "error": str(error)})
    return {
        "schema_version": SCHEMA_VERSION,
        "language": "python",
        "files_scanned": scanned,
        "findings": [asdict(finding) for finding in sorted(findings, key=lambda item: (item.file, item.line, item.sink))],
        "errors": errors,
    }


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
    if not report["findings"]:
        lines.append("This doesn't prove the code is safe, because static analysis can miss paths.")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
