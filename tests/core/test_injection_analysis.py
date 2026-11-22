"""The analyzer must find input reaching dangerous operations in each Python framework."""
import json
from pathlib import Path

import pytest

from aiwaf.core.injection_analysis import analyze_path, analyze_source, format_report, to_json


def sinks(source):
    return [(item.sink, item.call) for item in analyze_source(source, "app.py")]


DJANGO_VIEW = """
from django.db import connection

def search(request):
    term = request.GET["q"]
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM items WHERE name = '{term}'")
"""

DRF_VIEW = """
class Items(APIView):
    def get(self, request, pk):
        return Item.objects.raw("SELECT * FROM items WHERE id = %s" % pk)
"""

FLASK_VIEW = """
from flask import request
import os

@app.get("/ping")
def ping():
    os.system("ping " + request.args.get("host"))
"""

FLASK_ROUTE_VALUE = """
@app.route("/files/<name>")
def download(name):
    return send_file("/data/" + name)
"""

FASTAPI_ROUTE = """
from fastapi import Depends, Request
import requests

@app.get("/fetch")
async def fetch(url: str, db=Depends(get_db)):
    return requests.get(url)

@app.post("/body")
async def body(request: Request):
    payload = await request.json()
    return eval(payload["expression"])
"""


@pytest.mark.parametrize("source, expected", [
    (DJANGO_VIEW, [("sql", "cursor.execute")]),
    (DRF_VIEW, [("sql", "Item.objects.raw")]),
    (FLASK_VIEW, [("command", "os.system")]),
    (FLASK_ROUTE_VALUE, [("path", "send_file")]),
    (FASTAPI_ROUTE, [("ssrf", "requests.get"), ("code", "eval")]),
])
def test_frameworks_report_input_reaching_sinks(source, expected):
    assert sinks(source) == expected


@pytest.mark.parametrize("statement, sink", [
    ('cursor.execute("SELECT " + value)', "sql"),
    ('cursor.execute("SELECT %s" % value)', "sql"),
    ('cursor.execute("SELECT {}".format(value))', "sql"),
    ('session.execute(text(f"SELECT {value}"))', "sql"),
    ('Item.objects.extra(where=[value])', "sql"),
    ('db.users.find(value)', "nosql"),
    ('subprocess.run(value, shell=True)', "command"),
    ('subprocess.check_output("ls " + value)', "command"),
    ('os.popen(value)', "command"),
    ('open(value)', "path"),
    ('os.remove(value)', "path"),
    ('httpx.get(value)', "ssrf"),
    ('urlopen(value)', "ssrf"),
    ('exec(value)', "code"),
    ('render_template_string(value)', "template"),
    ('pickle.loads(value)', "deserialization"),
    ('yaml.load(value)', "deserialization"),
])
def test_each_sink_type_is_reported(statement, sink):
    source = f"def view(request):\n    value = request.GET['x']\n    {statement}\n"
    assert [item.sink for item in analyze_source(source)] == [sink]


@pytest.mark.parametrize("statement", [
    'cursor.execute("SELECT * FROM items WHERE id = %s", [value])',
    'Item.objects.raw("SELECT * FROM items WHERE id = %s", [value])',
    'subprocess.run(["ping", value])',
    'yaml.load(value, Loader=yaml.SafeLoader)',
    'text.find(value)',
    'print(value)',
])
def test_safe_uses_are_not_reported(statement):
    source = f"def view(request):\n    value = request.GET['x']\n    {statement}\n"
    assert sinks(source) == []


@pytest.mark.parametrize("cleaner", ["int", "float", "uuid.UUID", "shlex.quote", "secure_filename"])
def test_sanitizers_stop_tracking(cleaner):
    source = f"def view(request):\n    value = {cleaner}(request.GET['x'])\n    os.system('ls ' + value)\n"
    assert sinks(source) == []


def test_input_is_tracked_through_assignments_and_string_methods():
    source = """
def view(request):
    first = request.POST.get("q")
    second = first.strip().lower()
    third = "%" + second
    query = f"SELECT {third}"
    cursor.execute(query)
"""
    finding = analyze_source(source)[0]
    assert (finding.sink, finding.source, finding.line) == ("sql", "request.POST['q']", 7)


def test_overwriting_a_value_stops_tracking_it():
    assert sinks("def view(request):\n    v = request.GET['x']\n    v = 'safe'\n    os.system(v)\n") == []


def test_overwriting_in_one_branch_keeps_tracking_it():
    source = """
def view(request):
    v = request.GET["x"]
    if not v:
        v = "safe"
    os.system(v)
"""
    assert sinks(source) == [("command", "os.system")]


def test_input_is_followed_into_helpers():
    source = """
def run(sql_text):
    cursor.execute(sql_text)

def outer(value):
    run("SELECT " + value)

def view(request):
    outer(request.GET["x"])
"""
    finding = analyze_source(source)[0]
    assert finding.sink == "sql"
    assert finding.function == "view -> outer -> run"


def test_recursive_helpers_finish():
    source = "def loop(v):\n    loop(v)\n    os.system(v)\n\ndef view(request):\n    loop(request.GET['x'])\n"
    assert sinks(source) == [("command", "os.system")]


def test_functions_without_request_input_are_ignored():
    assert sinks("def task(value):\n    os.system(value)\n") == []


def test_a_flask_request_is_only_recognized_when_imported():
    source = "def task():\n    os.system(request.args['x'])\n"
    assert sinks(source) == []
    assert sinks("from flask import request\n" + source) == [("command", "os.system")]


def test_fastapi_dependencies_are_not_treated_as_input():
    source = """
@app.get("/x")
def view(db: Session, service=Depends(get_service)):
    os.system(service)
    cursor.execute(db)
"""
    assert sinks(source) == []


def test_import_aliases_are_resolved():
    source = "import subprocess as sp\nfrom os import system as run\n\ndef view(request):\n    run(request.GET['x'])\n    sp.Popen(request.GET['x'], shell=True)\n"
    assert [item.sink for item in analyze_source(source)] == ["command", "command"]


def test_invalid_syntax_is_reported_by_path_scan(tmp_path):
    (tmp_path / "bad.py").write_text("def (", encoding="utf-8")
    (tmp_path / "good.py").write_text(DJANGO_VIEW, encoding="utf-8")
    report = analyze_path(str(tmp_path))
    assert report["files_scanned"] == 1
    assert len(report["errors"]) == 1
    assert report["findings"][0]["sink"] == "sql"


def test_path_scan_skips_tests_and_dependencies_unless_asked(tmp_path):
    for relative in ("tests/test_a.py", "venv/lib.py", "test_b.py", "node_modules/c.py"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(DJANGO_VIEW, encoding="utf-8")
    assert analyze_path(str(tmp_path))["files_scanned"] == 0
    assert analyze_path(str(tmp_path), include_tests=True)["files_scanned"] == 2


def test_single_file_can_be_scanned(tmp_path):
    file = tmp_path / "view.py"
    file.write_text(DJANGO_VIEW, encoding="utf-8")
    assert analyze_path(str(file))["files_scanned"] == 1


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        analyze_path("no/such/path")


def test_reports_have_the_shared_shape_and_readable_output(tmp_path):
    (tmp_path / "view.py").write_text(DJANGO_VIEW, encoding="utf-8")
    report = analyze_path(str(tmp_path))
    assert set(report) == {"schema_version", "language", "files_scanned", "findings", "errors", "inputs"}
    assert report["inputs"] == [{"function": "search", "file": report["findings"][0]["file"], "inputs": {"q": "text"}, "complete": True}]
    # entry_file is added only for an operation in another module than the view.
    assert set(report["findings"][0]) == {"file", "line", "function", "sink", "call", "source", "confidence"}
    assert json.loads(to_json(report))["language"] == "python"
    text = format_report(report)
    assert "request.GET['q'] reaches cursor.execute in search" in text
    assert "found 1 possible injection paths" in text


def test_a_clean_report_says_it_is_not_proof(tmp_path):
    (tmp_path / "view.py").write_text("x = 1\n", encoding="utf-8")
    assert "doesn't prove the code is safe" in format_report(analyze_path(str(tmp_path)))


def test_oversized_files_are_reported(tmp_path, monkeypatch):
    from aiwaf.core import injection_analysis

    monkeypatch.setattr(injection_analysis, "_MAX_FILE_BYTES", 1)
    (tmp_path / "view.py").write_text("x = 1\n", encoding="utf-8")
    assert "too large" in analyze_path(str(tmp_path))["errors"][0]["error"]


def test_source_is_never_executed(tmp_path):
    marker = tmp_path / "ran"
    source = f"open({str(marker)!r}, 'w')\ndef view(request):\n    eval(request.GET['x'])\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    analyze_path(str(tmp_path))
    assert not Path(marker).exists()


@pytest.mark.parametrize(
    "call",
    [
        "requests.request(method='GET', url=value)",
        "requests.request('GET', value)",
        "requests.get(url=value)",
        "client.request('GET', value)",
        "self.upstream_client.request(method='GET', url=value)",
        "http_client.get(value)",
    ],
)
def test_outbound_requests_report_the_url_however_it_is_passed(call):
    source = f"def view(request):\n    value = request.GET['u']\n    {call}\n"
    assert [finding.sink for finding in analyze_source(source)] == ["ssrf"]


@pytest.mark.parametrize(
    "call",
    [
        "requests.request('POST', 'http://fixed.example', data=value)",
        "requests.post('http://fixed.example', json=value)",
        "session.get(value)",
        "redis_client.get(value)",
    ],
)
def test_input_outside_the_url_is_not_an_outbound_request_path(call):
    source = f"def view(request):\n    value = request.GET['u']\n    {call}\n"
    assert analyze_source(source) == []


@pytest.mark.parametrize(
    ("statement", "sink"),
    [
        ("open(os.path.join('/data', value))", "path"),
        ("requests.get(urljoin('http://fixed.example', value))", "ssrf"),
        ("requests.get(urllib.parse.urljoin('http://fixed.example', value))", "ssrf"),
        ("os.system(unquote(value))", "command"),
    ],
)
def test_input_is_tracked_through_path_and_url_helpers(statement, sink):
    source = (
        "import os, urllib.parse\n"
        "from urllib.parse import unquote, urljoin\n"
        f"def view(request):\n    value = request.GET['u']\n    {statement}\n"
    )
    assert [finding.sink for finding in analyze_source(source)] == [sink]


# Gaps closed: a helper's return value, collections built in loops, typed route values and calls
# into other modules of the project.
HELPER_RETURN = '''
def build(name):
    return "SELECT * FROM t WHERE n='" + name + "'"
def clean(value):
    return int(value)
def view(request):
    cursor.execute(build(request.GET.get("n")))
    cursor.execute(f"SELECT * FROM t WHERE id={clean(request.GET['id'])}")
'''
LOOP_APPEND = '''
def view(request):
    parts = []
    for part in request.GET.getlist("p"):
        parts.append(part)
    cursor.execute("SELECT * FROM t WHERE n IN (" + ",".join(parts) + ")")
'''
TYPED = '''
from flask import request
@app.get("/a/{uid}")
async def a(uid: int):
    await db.execute(f"SELECT * FROM u WHERE id={uid}")
@app.get("/b/{uid}")
async def b(uid: Annotated[int, Path()]):
    await db.execute(f"SELECT * FROM u WHERE id={uid}")
@app.get("/c")
async def c(uid: Optional[UUID] = None):
    await db.execute(f"SELECT * FROM u WHERE id='{uid}'")
@app.route("/d/<int:uid>")
def d(uid):
    cursor.execute(f"SELECT * FROM u WHERE id={uid}")
@app.get("/e/{name}")
async def e(name: str):
    await db.execute(f"SELECT * FROM u WHERE n='{name}'")
@app.get("/f/{value}")
async def f(value: int | str):
    await db.execute(f"SELECT * FROM u WHERE n='{value}'")
@app.route("/g/<name>")
def g(name):
    cursor.execute(f"SELECT * FROM u WHERE n='{name}'")
'''


def test_a_helpers_return_value_carries_its_input():
    findings = analyze_source(HELPER_RETURN, "view.py")
    # build() returns the input in SQL text; clean() returns int(), which can't carry text.
    assert [(f.line, f.sink, f.source) for f in findings] == [(7, "sql", "request.GET['n']")]


def test_a_collection_built_in_a_loop_carries_its_input():
    assert [(f.line, f.sink) for f in analyze_source(LOOP_APPEND, "view.py")] == [(6, "sql")]


def test_route_values_typed_or_converted_as_numbers_or_uuids_are_not_input():
    assert sorted(f.function for f in analyze_source(TYPED, "view.py")) == ["e", "f", "g"]


def test_calls_into_other_modules_are_followed_and_linked_to_the_view(tmp_path):
    from aiwaf.core.route_risks import apply_report

    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "queries.py").write_text(
        "def find_user(name):\n"
        "    cursor.execute(\"SELECT * FROM u WHERE n = '%s'\" % name)\n"
        "def build(name):\n"
        "    return \"SELECT * FROM u WHERE n = '\" + name + \"'\"\n"
        "def safe(uid):\n"
        "    cursor.execute(\"SELECT * FROM u WHERE id = %s\", [uid])\n", encoding="utf-8")
    (package / "views.py").write_text(
        "from .queries import find_user, safe\n"
        "from . import queries\n"
        "def search(request):\n"
        "    find_user(request.GET['q'])\n"
        "def lookup(request):\n"
        "    cursor.execute(queries.build(request.GET.get('n')))\n"
        "def harmless(request):\n"
        "    safe(request.GET['id'])\n", encoding="utf-8")
    report = analyze_path(str(tmp_path))
    found = {(f["file"].rsplit("/", 1)[-1], f["function"], f.get("entry_file", "").rsplit("/", 1)[-1]) for f in report["findings"]}
    assert found == {("queries.py", "search -> find_user", "views.py"), ("views.py", "lookup", "")}
    # The finding in queries.py links to the route of the view in views.py.
    manifest = {"routes": {"/search": {"view": "app.views.search"}, "/lookup": {"view": "app.views.lookup"},
                           "/harmless": {"view": "app.views.harmless"}}}
    assert apply_report(report, manifest)["routes"] == 2
    assert manifest["routes"]["/search"]["injection_risks"] == [{"sink": "sql", "input": "q", "confidence": "high"}]
    assert "injection_risks" not in manifest["routes"]["/harmless"]


def test_methods_on_instances_of_project_classes_are_followed(tmp_path):
    from aiwaf.core.injection_analysis import analyze_path

    package = tmp_path / "app" / "services"
    package.mkdir(parents=True)
    (tmp_path / "app" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "users.py").write_text(
        "class UserService:\n"
        "    def __init__(self, db):\n"
        "        self.db = db\n"
        "    def search(self, term):\n"
        "        return self.db.execute(\"SELECT * FROM u WHERE n = '\" + term + \"'\")\n"
        "    def safe(self, term):\n"
        "        return len(term)\n"
        "    @staticmethod\n"
        "    def run(cmd):\n"
        "        os.system(cmd)\n"
    )
    (tmp_path / "app" / "views.py").write_text(
        "from fastapi import Depends\n"
        "from typing import Optional\n"
        "from .services.users import UserService\n"
        "from app.services import users as users_module\n"
        "@app.get('/a')\n"
        "def a(q: str):\n"
        "    s = UserService(db)\n"
        "    return s.search(q)\n"
        "@app.get('/b')\n"
        "def b(q: str, service: UserService = Depends(UserService)):\n"
        "    return service.search(q)\n"
        "@app.get('/c')\n"
        "def c(q: str, service: Optional['UserService'] = None):\n"
        "    return service.search(q)\n"
        "@app.get('/d')\n"
        "def d(q: str):\n"
        "    return users_module.UserService(db).search(q)\n"
        "@app.get('/e')\n"
        "def e(q: str):\n"
        "    UserService.run(q)\n"
        "@app.get('/f')\n"
        "def f(q: str):\n"
        "    s = UserService(db)\n"
        "    return s.safe(q)\n"
        "class View:\n"
        "    def __init__(self, service: UserService):\n"
        "        self.service = service\n"
        "    @app.get('/g')\n"
        "    def g(self, q: str):\n"
        "        return self.service.search(q)\n"
    )
    findings = analyze_path(str(tmp_path))["findings"]
    assert sorted(finding["function"] for finding in findings) == [
        "View.g -> search", "a -> search", "b -> search", "c -> search", "d -> search", "e -> run"]
    assert all(finding["file"].endswith("services/users.py") for finding in findings)
    assert all(finding["entry_file"].endswith("app/views.py") for finding in findings)


def test_a_known_project_method_is_not_mistaken_for_a_database_call(tmp_path):
    from aiwaf.core.injection_analysis import analyze_path

    (tmp_path / "audit.py").write_text("class AuditLog:\n    def execute(self, text):\n        print(text)\n")
    (tmp_path / "views.py").write_text(
        "from audit import AuditLog\n"
        "@app.get('/a')\n"
        "def a(q: str):\n"
        "    log = AuditLog()\n"
        "    log.execute(q)\n"
    )
    assert analyze_path(str(tmp_path))["findings"] == []


# Each case: what it shows, the operations reported, and the code. Safe look-alikes report nothing.
FLOW_CASES = [
    # New kinds of operation.
    ("redirect", ["redirect"], "def view(request):\n    return redirect(request.GET['next'])\n"),
    ("redirect checked by Django", [], (
        "def view(request):\n    nxt = request.GET['next']\n"
        "    if not url_has_allowed_host_and_scheme(nxt, allowed_hosts={'x'}):\n        return HttpResponse('no')\n"
        "    return redirect(nxt)\n")),
    ("redirect checked by netloc", [], (
        "def view(request):\n    bar = request.GET['next']\n    url = urllib.parse.urlparse(bar)\n"
        "    if url.netloc not in ['example.com'] or url.scheme != 'https':\n        return 'no'\n    return redirect(bar)\n")),
    ("reflected HTML in Flask", ["xss"], (
        "from flask import request\n@app.route('/x')\ndef x():\n    out = ''\n    out += f\"<p>{request.args['q']}</p>\"\n    return out\n")),
    ("escaped HTML in Flask", [], (
        "from flask import request\n@app.route('/x')\ndef x():\n    out = ''\n    out += f\"<p>{escape(request.args['q'])}</p>\"\n    return out\n")),
    ("input only in a header", [], (
        "from flask import request, make_response\n@app.route('/x')\ndef x():\n"
        "    out = make_response(('ok', {'X-Value': request.args['q']}))\n    return out\n")),
    ("FastAPI returns JSON", [], "from fastapi import FastAPI\n@app.get('/x')\ndef x(q: str):\n    return {'q': q}\n"),
    ("mark_safe", ["xss"], "def view(request):\n    return HttpResponse(mark_safe(request.GET['h']))\n"),
    ("LDAP filter", ["ldap"], "def view(request):\n    conn.search('ou=x', f\"(uid={request.GET['u']})\")\n"),
    ("LDAP filter escaped", [], "def view(request):\n    conn.search('ou=x', f\"(uid={escape_filter_chars(request.GET['u'])})\")\n"),
    ("XPath query", ["xpath"], "def view(request):\n    root.xpath(f\"//u[@id='{request.GET['id']}']\")\n"),
    ("XPath with quotes removed", [], "def view(request):\n    v = request.GET['id'].replace(\"'\", '')\n    root.xpath(f\"//u[@id='{v}']\")\n"),
    ("XML with external entities", ["xxe"], (
        "def view(request):\n    parser = xml.sax.make_parser()\n"
        "    parser.setFeature(xml.sax.handler.feature_external_ges, True)\n"
        "    xml.dom.minidom.parseString(request.body, parser)\n")),
    ("XML with defaults", [], "def view(request):\n    xml.dom.minidom.parseString(request.body, xml.sax.make_parser())\n"),
    ("regular expression from input", ["redos"], "def view(request):\n    re.compile(request.GET['p'])\n"),
    ("regular expression escaped", [], "def view(request):\n    re.compile(re.escape(request.GET['p']))\n"),
    # Remote code execution sinks.
    ("import of a module named by input", ["code"], "def view(request):\n    importlib.import_module(request.GET['m'])\n"),
    ("__import__ of input", ["code"], "def view(request):\n    __import__(request.GET['m'])\n"),
    ("os.exec of input", ["command"], "def view(request):\n    os.execvp(request.GET['b'], [])\n"),
    ("Jinja from_string", ["template"], "def view(request):\n    env.from_string(request.GET['t']).render()\n"),
    ("jsonpickle", ["deserialization"], "def view(request):\n    jsonpickle.decode(request.body)\n"),
    ("import of a checked number", [], "def view(request):\n    importlib.import_module('app.' + str(int(request.GET['n'])))\n"),
    # Encoders are safe only for what they encode for.
    ("HTML escaping before SQL", ["sql"], "def view(request):\n    cursor.execute(html.escape(request.GET['q']))\n"),
    ("shell quoting", [], "def view(request):\n    os.system('ls ' + shlex.quote(request.GET['q']))\n"),
    # Checks before use.
    ("isalnum check that returns", [], (
        "def view(request):\n    v = request.GET['v']\n    if not v.isalnum():\n        return\n    os.system(v)\n")),
    ("allowlist branch", [], (
        "def view(request):\n    s = request.GET['s']\n    if s in ('name', 'date'):\n        cursor.execute('ORDER BY ' + s)\n")),
    ("denylist that skips known names", ["sql"], (
        "def view(request):\n    for k in request.GET.keys():\n        if k in SKIP:\n            continue\n"
        "        cursor.execute(k)\n")),
    ("dot-dot check makes a path safe, not SQL", ["sql"], (
        "def view(request):\n    f = request.GET['f']\n    if '..' in f:\n        return\n    open(f)\n    cursor.execute(f)\n")),
    # Known values, dict keys and list positions.
    ("constant condition", [], (
        "def view(request):\n    num = 106\n    if 7 * 18 + num > 200:\n        bar = 'safe'\n"
        "    else:\n        bar = request.GET['q']\n    cursor.execute(bar)\n")),
    ("match on a known value", [], (
        "def view(request):\n    guess = 'ABC'[1]\n    match guess:\n        case 'A':\n            bar = request.GET['q']\n"
        "        case 'B':\n            bar = 'bob'\n        case _:\n            bar = request.GET['q']\n    cursor.execute(bar)\n")),
    ("dict key read back", ["sql"], "def view(request):\n    m = {}\n    m['b'] = request.GET['q']\n    cursor.execute(m['b'])\n"),
    ("other dict key", [], "def view(request):\n    m = {}\n    m['a'] = 'safe'\n    m['b'] = request.GET['q']\n    cursor.execute(m['a'])\n"),
    ("list position", [], (
        "def view(request):\n    l = []\n    l.append('safe')\n    l.append(request.GET['q'])\n    l.append('more')\n"
        "    l.pop(0)\n    cursor.execute(l[1])\n")),
    # A change that can't be followed by position makes the whole list count as input.
    ("list extended with input", ["sql"], (
        "def view(request):\n    l = ['safe']\n    l.extend(request.GET.getlist('q'))\n    cursor.execute(l[0])\n")),
    ("dict updated with input", ["sql"], (
        "def view(request):\n    m = {'a': 'safe'}\n    m.update(request.GET)\n    cursor.execute(m['a'])\n")),
    ("configparser key", ["path"], (
        "def view(request):\n    c = configparser.ConfigParser()\n    c.set('s', 'k', request.GET['q'])\n"
        "    open(c.get('s', 'k'))\n")),
    ("with statement", ["path"], "def view(request):\n    with open(request.GET['f']) as handle:\n        pass\n"),
    # Input from more frameworks.
    ("Graphene resolver", ["sql"], "class Q(graphene.ObjectType):\n    def resolve_user(root, info, id):\n        cursor.execute('SELECT ' + id)\n"),
    ("Strawberry field", ["command"], "@strawberry.field\ndef run(self, info, cmd: str) -> str:\n    return os.system(cmd)\n"),
    ("Strawberry int field", [], "@strawberry.field\ndef user(self, info, id: int):\n    cursor.execute('SELECT ' + str(id))\n"),
    ("FastAPI WebSocket", ["sql"], (
        "@app.websocket('/ws')\nasync def ws(websocket: WebSocket):\n    data = await websocket.receive_text()\n"
        "    cursor.execute('SELECT ' + data)\n")),
    ("Channels consumer", ["command"], "class C(WebsocketConsumer):\n    def receive(self, text_data=None, bytes_data=None):\n        os.system(text_data)\n"),
    ("Flask-SocketIO handler", ["sql"], "@socketio.on('search')\ndef handle(data):\n    cursor.execute('SELECT ' + data['q'])\n"),
    # Values kept for later code.
    ("flask.g set by a before_request hook", ["sql"], (
        "from flask import g, request\n@app.before_request\ndef load():\n    g.term = request.args['q']\n"
        "@app.route('/s')\ndef search():\n    cursor.execute('SELECT ' + g.term)\n    return 'ok'\n")),
    ("request attribute set by middleware", ["sql"], (
        "class M:\n    def __call__(self, request):\n        request.term = request.GET['q']\n        return self.get_response(request)\n"
        "def view(request):\n    cursor.execute('SELECT ' + request.term)\n")),
    ("request kept by a wrapper object", ["command"], (
        "class W:\n    def __init__(self, request):\n        self.request = request\n    def get(self, n):\n"
        "        return self.request.args.get(n)\ndef view(request):\n    w = W(request)\n    os.system(w.get('q'))\n")),
]


@pytest.mark.parametrize("name, expected, source", FLOW_CASES, ids=[case[0] for case in FLOW_CASES])
def test_input_flows_and_their_safe_look_alikes(name, expected, source):
    assert sorted({finding.sink for finding in analyze_source(source)}) == sorted(expected)


def test_ignore_comments_baselines_and_sarif(tmp_path):
    from aiwaf.cli import _handle_audit
    from aiwaf.core.injection_analysis import apply_baseline, fingerprint, to_sarif

    (tmp_path / "v.py").write_text(
        "def view(request):\n    os.system(request.GET['a'])\n    os.system(request.GET['b'])  # aiwaf: ignore[command]\n"
        "    # aiwaf: ignore\n    cursor.execute(request.GET['c'])\n    cursor.execute(request.GET['d'])  # aiwaf: ignore[xss]\n")
    report = analyze_path(str(tmp_path))
    assert [finding["line"] for finding in report["findings"]] == [2, 6]
    assert report["suppressed"] == 2
    assert "Left out 2 marked with an aiwaf: ignore comment." in format_report(report)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(to_json(report))
    with (tmp_path / "v.py").open("a") as handle:
        handle.write("def other(request):\n    os.system(request.GET['z'])\n")
    later = apply_baseline(analyze_path(str(tmp_path)), json.loads(baseline.read_text()))
    assert [finding["function"] for finding in later["findings"]] == ["other"]
    assert later["baselined"] == 2
    # The line isn't part of a finding's identity, so moved code still matches the baseline.
    assert fingerprint(dict(later["findings"][0], line=99)) == fingerprint(later["findings"][0])

    sarif = json.loads(to_sarif(report))
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert [rule["id"] for rule in run["tool"]["driver"]["rules"]] == ["aiwaf.command", "aiwaf.sql"]
    assert run["results"][0]["level"] == "error"
    assert run["results"][0]["locations"][0]["physicalLocation"]["region"] == {"startLine": 2}

    output = tmp_path / "out.sarif"
    assert _handle_audit([str(tmp_path), "--baseline", str(baseline), "--format", "sarif", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["runs"][0]["results"]) == 1
    assert _handle_audit([str(tmp_path), "--baseline", str(tmp_path / "missing.json")]) == 2


def test_shared_helpers_are_walked_once_with_the_same_findings(tmp_path, monkeypatch):
    from aiwaf.core import injection_analysis

    (tmp_path / "helpers.py").write_text(
        "def inner(v):\n    cursor.execute('SELECT ' + v)\n    return v\n"
        "def outer(v):\n    inner(v)\n    return inner(v)\n")
    for index in range(3):
        (tmp_path / f"views_{index}.py").write_text(
            f"from flask import request\nfrom helpers import outer\n@app.route('/r{index}')\n"
            f"def view_{index}():\n    os.system(outer(request.args['q']))\n    return 'ok'\n")
    cached = analyze_path(str(tmp_path))

    class NoCache(dict):
        def __setitem__(self, key, value):
            pass

    original = injection_analysis._ModuleAnalyzer.__init__

    def without_cache(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.summaries = NoCache()

    monkeypatch.setattr(injection_analysis._ModuleAnalyzer, "__init__", without_cache)
    assert analyze_path(str(tmp_path)) == cached
    # Each view links to the helper's operation, and the helper's result carries the input.
    assert sorted(finding["function"] for finding in cached["findings"]) == [
        "view_0", "view_0 -> outer -> inner", "view_1", "view_1 -> outer -> inner", "view_2", "view_2 -> outer -> inner"]
