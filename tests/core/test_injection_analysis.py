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
    assert set(report) == {"schema_version", "language", "files_scanned", "findings", "errors"}
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
