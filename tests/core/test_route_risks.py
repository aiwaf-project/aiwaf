import json
import logging
from pathlib import Path

import pytest

from aiwaf.core import route_risks
from aiwaf.core.injection_analysis import analyze_path, analyze_source
from aiwaf.core.path_manifest import compile_manifest_to_path_rules, load_manifest
from aiwaf.core.route_risks import (
    SINKS,
    RouteRisks,
    apply_report,
    format_counts,
    input_from_source,
    observer_for,
    rule_for,
    update_manifest,
)

CASES = json.loads((Path(__file__).parents[1] / "fixtures" / "route_risk_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def fresh_cache():
    route_risks.clear_cache()
    yield
    route_risks.clear_cache()


def manifest_with(risks, path="/search"):
    return {"routes": {path: {"view": "app.views.search", "injection_risks": risks}, "/health": {"view": "app.views.health"}}}


def write(tmp_path, manifest):
    target = tmp_path / "paths.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


@pytest.mark.parametrize("case", CASES["rules"], ids=lambda case: f"{case['sink']}:{case['value']}")
def test_shared_rule_cases(case):
    assert rule_for(case["sink"], case["value"]) == case["rule"]


def test_only_the_named_input_is_checked():
    risks = RouteRisks(manifest_with([{"sink": "path", "input": "file", "confidence": "medium"}]))
    assert risks.inspect("/search", "other=../x") == []
    assert risks.inspect("/search", "file=../x") == [
        {"route": "/search", "sink": "path", "input": "file", "rule": "path_traversal", "confidence": "medium"}
    ]


def test_an_unknown_input_checks_everything():
    risks = RouteRisks(manifest_with([{"sink": "command", "input": "", "confidence": "high"}]))
    assert risks.inspect("/search", b"", b'{"deep": {"cmd": "$(id)"}}')[0]["rule"] == "command_substitution"
    assert risks.inspect("/search", "a=$(id)")[0]["input"] == ""


def test_nested_values_count_as_the_field_that_holds_them():
    risks = RouteRisks(manifest_with([{"sink": "nosql", "input": "user"}]))
    assert risks.inspect("/search", None, b'{"user": {"$ne": 1}}')[0]["rule"] == "nosql_operator"
    assert risks.inspect("/search", "user[$ne]=1")[0]["rule"] == "nosql_operator"


def test_only_the_rules_for_the_routes_operation_apply():
    risks = RouteRisks(manifest_with([{"sink": "path", "input": "q"}]))
    assert risks.inspect("/search", "q=$(id)") == []


def test_route_values_are_inputs():
    risks = RouteRisks(manifest_with([{"sink": "path", "input": "name"}], "/files/<path:name>"))
    assert risks.inspect("/files/../secret")[0]["rule"] == "path_traversal"


def test_routes_without_saved_risks_get_nothing():
    risks = RouteRisks(manifest_with([{"sink": "path", "input": ""}]))
    assert risks.inspect("/health", "x=../y") == []
    assert risks.inspect("/other", "x=../y") == []
    assert RouteRisks(None).empty
    assert RouteRisks({"routes": "bad"}).empty


def test_exact_routes_win_over_parameter_routes():
    risks = RouteRisks({"routes": {
        "/items/<pk>": {"injection_risks": [{"sink": "path", "input": ""}]},
        "/items/new": {"injection_risks": [{"sink": "command", "input": ""}]},
    }})
    assert risks.inspect("/items/new", "x=$(id)")[0]["route"] == "/items/new"


def test_repeated_matches_log_once():
    risks = RouteRisks(manifest_with([{"sink": "path", "input": ""}]))
    assert len(risks.inspect("/search", "a=../x&b=../y")) == 1


def test_log_mode_logs_names_and_never_values(tmp_path, caplog):
    path = write(tmp_path, manifest_with([{"sink": "path", "input": "", "confidence": "medium"}]))
    observe = observer_for("log", path, "/search", "GET")
    with caplog.at_level(logging.WARNING, logger="aiwaf.injection_risk"):
        assert observe("evil_param_name=../topsecret", None) is None
    event = json.loads(caplog.records[0].getMessage())
    assert event == {
        "event": "aiwaf.injection_risk", "mode": "log", "method": "GET", "route": "/search",
        "sink": "path", "input": "", "rule": "path_traversal", "confidence": "medium",
    }
    assert "topsecret" not in caplog.text
    assert "evil_param_name" not in caplog.text


def test_block_mode_is_the_default_and_blocks(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv(route_risks.MODE_SETTING, raising=False)
    path = write(tmp_path, manifest_with([{"sink": "path", "input": ""}]))
    with caplog.at_level(logging.WARNING, logger="aiwaf.injection_risk"):
        finding = observer_for(None, path, "/search", "GET")("x=../y", None)
    assert (finding.rule, finding.status) == ("path_traversal", 403)
    assert json.loads(caplog.records[0].getMessage())["mode"] == "block"
    assert observer_for("block", path, "/search")("x=fine", None) is None


def test_off_turns_it_off_and_unknown_modes_are_rejected(tmp_path):
    path = write(tmp_path, manifest_with([{"sink": "path", "input": ""}]))
    assert observer_for("off", path, "/search") is None
    for mode in ("", "monitor", "enforce"):
        with pytest.raises(ValueError):
            observer_for(mode, path, "/search")


def test_the_environment_can_turn_on_log_mode(tmp_path, monkeypatch):
    monkeypatch.setenv(route_risks.MODE_SETTING, "LOG")
    assert observer_for(None, write(tmp_path, manifest_with([{"sink": "path", "input": ""}])), "/search") is not None


def test_a_missing_or_empty_manifest_turns_it_off(tmp_path):
    assert observer_for("log", tmp_path / "missing.json", "/search") is None
    assert observer_for("log", write(tmp_path, {"routes": {"/x": {}}}), "/x") is None


def test_a_failing_check_never_reaches_the_request(tmp_path, monkeypatch):
    observe = observer_for("log", write(tmp_path, manifest_with([{"sink": "path", "input": ""}])), "/search")
    monkeypatch.setattr(route_risks.RouteRisks, "inspect", lambda *_: 1 / 0)
    assert observe("a=../x", None) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("request.GET['q']", "q"),
        ("route parameter slug", "slug"),
        ("request.GET", ""),
        ("body parameter item", ""),
    ],
)
def test_input_from_source(source, expected):
    assert input_from_source(source) == expected


@pytest.mark.parametrize(
    ("statement", "source"),
    [
        ("request.GET['q']", "request.GET['q']"),
        ("request.GET.get('q')", "request.GET['q']"),
        ("request.POST.getlist('q')", "request.POST['q']"),
        ("request.GET", "request.GET"),
    ],
)
def test_findings_name_the_parameter(statement, source):
    code = f"import os\ndef view(request):\n    value = {statement}\n    os.system(value)\n"
    assert analyze_source(code)[0].source == source


def test_findings_name_the_class_for_class_based_views():
    code = "import os\nclass Items(View):\n    def get(self, request):\n        os.system(request.GET['q'])\n"
    assert analyze_source(code)[0].function == "Items.get"


def test_model_parameters_have_no_parameter_name():
    code = (
        "import os\nfrom fastapi import FastAPI\napp = FastAPI()\n"
        "@app.post('/x')\ndef view(item: Item, name: str):\n    os.system(item.command)\n    os.system(name)\n"
    )
    sources = sorted(finding.source for finding in analyze_source(code))
    assert sources == ["body parameter item", "route parameter name"]


def report_for(tmp_path, name, code):
    target = tmp_path / "src" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")
    return analyze_path(str(tmp_path / "src"))


@pytest.mark.parametrize(
    ("file", "view"),
    [("views.py", "app.views.search"), ("search.py", "app.views.Other"), ("views.py", "search")],
)
def test_findings_link_to_routes_by_view(tmp_path, file, view):
    report = report_for(tmp_path, file, "import os\ndef search(request):\n    os.system(request.GET['q'])\n")
    manifest = {"routes": {"/search": {"view": view}}}
    counts = apply_report(report, manifest)
    if view.endswith("Other"):
        assert counts == {"routes": 0, "linked": 0, "unlinked": 1, "inputs": 0}
        assert "injection_risks" not in manifest["routes"]["/search"]
    else:
        assert counts == {"routes": 1, "linked": 1, "unlinked": 0, "inputs": 1}
        assert manifest["routes"]["/search"]["inputs"] == {"q": "text"}
        assert manifest["routes"]["/search"]["inputs_complete"] is True
        assert manifest["routes"]["/search"]["injection_risks"] == [{"sink": "command", "input": "q", "confidence": "high"}]


def test_same_named_views_in_other_modules_do_not_link(tmp_path):
    report = report_for(tmp_path, "views.py", "import os\ndef index(request):\n    os.system(request.GET['q'])\n")
    manifest = {"routes": {"/a": {"view": "shop.views.index"}, "/b": {"view": "blog.admin.index"}}}
    apply_report(report, manifest)
    assert "injection_risks" in manifest["routes"]["/a"]
    assert "injection_risks" not in manifest["routes"]["/b"]


def test_class_based_views_and_methods_link(tmp_path):
    report = report_for(tmp_path, "views.py", "import os\nclass Items(View):\n    def get(self, request, pk):\n        os.system(pk)\n")
    for view in ("app.views.Items", "app.views.Items.get"):
        manifest = {"routes": {"/items/<pk>": {"view": view}}}
        apply_report(report, manifest)
        assert manifest["routes"]["/items/<pk>"]["injection_risks"][0]["input"] == "pk"


def test_helpers_link_through_the_view_that_called_them(tmp_path):
    report = report_for(tmp_path, "views.py", "import os\ndef run(v):\n    os.system(v)\ndef search(request):\n    run(request.GET['q'])\n")
    manifest = {"routes": {"/search": {"view": "views.search"}}}
    assert apply_report(report, manifest)["linked"] == 1


def test_a_clean_audit_removes_old_risks_and_nothing_else():
    manifest = {"routes": {"/x": {"view": "a.x", "protections": {"rate_limit": {"requests": 1}}, "injection_risks": [{"sink": "sql"}]}}}
    apply_report({"findings": [], "files_scanned": 3}, manifest)
    assert manifest["routes"]["/x"] == {"view": "a.x", "protections": {"rate_limit": {"requests": 1}}}
    assert manifest["injection_audit"]["files_scanned"] == 3


def test_saved_risks_never_change_path_rules(tmp_path):
    protections = {"rate_limit": {"requests": 5, "window_seconds": 60}, "honeypot": {"enabled": False}}
    plain = {"routes": {"/x": {"view": "views.search", "protections": protections}}}
    before = compile_manifest_to_path_rules(plain)
    report = report_for(tmp_path, "views.py", "import os\ndef search(request):\n    os.system(request.GET['q'])\n")
    apply_report(report, plain)
    assert plain["routes"]["/x"]["injection_risks"]
    assert compile_manifest_to_path_rules(plain) == before


def test_update_manifest_writes_the_file(tmp_path):
    report = report_for(tmp_path, "views.py", "import os\ndef search(request):\n    os.system(request.GET['q'])\n")
    target = write(tmp_path, {"schema_version": "1.0", "routes": {"/search": {"view": "views.search"}}})
    counts = update_manifest(report, str(target))
    assert load_manifest(target)["routes"]["/search"]["injection_risks"][0]["sink"] == "command"
    assert "Saved 1 possible injection paths to 1 routes" in format_counts(counts, str(target))
    assert "couldn't be tied" in format_counts({"linked": 0, "routes": 0, "unlinked": 2}, "x")


def test_update_manifest_needs_an_existing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_manifest({"findings": []}, str(tmp_path / "missing.json"))
    with pytest.raises(ValueError):
        apply_report({"findings": []}, {"routes": []})


def test_audit_into_manifest_saves_findings_and_survives_failures(tmp_path, caplog):
    from aiwaf.core.route_risks import audit_into_manifest, describe_audit

    (tmp_path / "views.py").write_text("import os\ndef search(request):\n    os.system(request.GET['q'])\n", encoding="utf-8")
    manifest = {"routes": {"/search": {"view": "views.search"}, "/x": {"view": "views.x"}}}
    assert audit_into_manifest(manifest, None) is None
    assert describe_audit(manifest) == "Injection checks: not run"
    assert audit_into_manifest(manifest, str(tmp_path)) == {"routes": 1, "linked": 1, "unlinked": 0, "inputs": 1}
    assert describe_audit(manifest) == "Injection checks: 1 routes from 1 files; inputs and types for 1 routes, complete for 1"
    manifest["injection_audit"]["incomplete_input_reasons"] = {"helper": 2, "dynamic_key": 1}
    assert describe_audit(manifest).endswith("complete for 1 (incomplete because of helper 2, dynamic_key 1)")
    manifest["injection_audit"]["unlinked_findings"] = 2
    assert "2 findings couldn't be tied" in describe_audit(manifest)
    with caplog.at_level(logging.WARNING, logger="aiwaf.injection_risk"):
        (tmp_path / "empty").mkdir()
        assert audit_into_manifest({"routes": {}}, str(tmp_path / "empty")) is None
    assert "found no Python files" in caplog.text


def test_audit_into_manifest_refuses_a_missing_or_failing_source(tmp_path, monkeypatch):
    from aiwaf.core import injection_analysis
    from aiwaf.core.route_risks import AuditError, audit_into_manifest

    with pytest.raises(AuditError, match="doesn't exist"):
        audit_into_manifest({"routes": {}}, str(tmp_path / "missing"))

    def broken(_path):
        raise RuntimeError("parser crashed")

    monkeypatch.setattr(injection_analysis, "analyze_path", broken)
    with pytest.raises(AuditError, match="parser crashed"):
        audit_into_manifest({"routes": {}}, str(tmp_path))


def test_an_audit_that_reads_no_files_counts_as_no_audit(tmp_path):
    from aiwaf.core.route_risks import audit_into_manifest

    framework = {"inputs": {"quantity": "int"}, "covers": [], "reasons": [], "standalone": True}
    manifest = {"routes": {"/orders": {"view": "v", "declared_inputs": framework}}}
    assert audit_into_manifest(manifest, str(tmp_path)) is None
    route = manifest["routes"]["/orders"]
    assert route["inputs"] == {"quantity": "int"} and route["inputs_complete"] is False
    assert [reason["code"] for reason in route["inputs_incomplete"]] == ["not_audited"]
    assert "injection_audit" not in manifest


ORDINARY = json.loads((Path(__file__).parents[1] / "fixtures" / "ordinary_text.json").read_text(encoding="utf-8"))["values"]


@pytest.mark.parametrize("use_rust", [True, False])
def test_no_rule_blocks_ordinary_text(use_rust):
    from aiwaf.core.native_runtime import native_override

    # The checks run in aiwaf-rust even when AIWAF_USE_RUST turns off the optional Rust paths.
    with native_override(use_rust):
        blocked = [(sink, value) for value in ORDINARY for sink in SINKS if rule_for(sink, value)]
    assert blocked == []


def test_request_formats_are_inputs():
    risks = RouteRisks(manifest_with([{"sink": "path", "input": "a"}]))
    for query, body in (("a=../x", None), (b"a=../x", None), (None, b"a=../x"), (None, '{"a": ["../x"]}'), (None, {"a": "../x"})):
        assert risks.inspect("/search", query, body)[0]["rule"] == "path_traversal", (query, body)
    assert RouteRisks(manifest_with([{"sink": "path", "input": ""}])).inspect("/search", None, b"../x")[0]["input"] == ""


SCHEMA = json.loads((Path(__file__).parents[1] / "fixtures" / "route_schema_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", SCHEMA["cases"], ids=lambda case: f"{case['path']} {case['expect']}")
def test_shared_schema_cases(case):
    from aiwaf.core.sql_injection import SQLInjectionPolicy

    found = SQLInjectionPolicy("block").inspect(case["query"].encode(), case["body"].encode())
    review = RouteRisks(SCHEMA["manifest"]).review(case["path"], case["query"], case["body"].encode(), 65536,
                                                   found.rule if found else None)
    outcome = review[0] if review else ("block" if found else "pass")
    assert outcome == case["expect"]
    if outcome == "mismatch":
        assert review[2] == case["input"]


@pytest.mark.parametrize("case", SCHEMA["strict_cases"]["cases"], ids=lambda case: f"{case['path']} {case['expect']}")
def test_shared_strict_cases(case):
    from aiwaf.core.sql_injection import SQLInjectionPolicy

    found = SQLInjectionPolicy("block").inspect(case["query"].encode(), case["body"].encode())
    risks = RouteRisks(SCHEMA["manifest"], {"allow": SCHEMA["strict_cases"]["allow"], "exempt": SCHEMA["strict_cases"]["exempt"]})
    review = risks.review(case["path"], case["query"], case["body"].encode(), 65536, found.rule if found else None,
                          case.get("method", ""))
    outcome = review[0] if review else ("block" if found else "pass")
    assert outcome == case["expect"]
    if case.get("input"):
        assert review[2] == case["input"]


def test_strict_inputs_settings_and_modes(tmp_path, caplog, monkeypatch):
    from aiwaf.core.route_risks import strict_inputs

    monkeypatch.delenv("AIWAF_STRICT_INPUTS", raising=False)
    monkeypatch.delenv("AIWAF_STRICT_INPUTS_ALLOW", raising=False)
    assert strict_inputs() is None and strict_inputs("no") is None
    assert strict_inputs(True, "a, b*,") == {"allow": ["a", "b*"], "exempt": []}
    assert strict_inputs("on", ["x", " "]) == {"allow": ["x"], "exempt": []}
    monkeypatch.setenv("AIWAF_STRICT_INPUTS", "1")
    monkeypatch.setenv("AIWAF_STRICT_INPUTS_ALLOW", "trace_*")
    assert strict_inputs() == {"allow": ["trace_*"], "exempt": []}
    path = write(tmp_path, SCHEMA["manifest"])
    with caplog.at_level(logging.WARNING, logger="aiwaf.route_schema"):
        monitor = observer_for("off", path, "/search", "GET")
        assert monitor.review(b"q=a&debug=1", b"", None, 65536) is None
        enforce = observer_for("off", path, "/search", "GET", "enforce")
        assert enforce.review(b"q=a&debug=1", b"", None, 65536).rule == "schema_unexpected_input"
        assert enforce.review(b"q=a&trace_id=1", b"", None, 65536) is None
        off = observer_for("off", path, "/search", "GET", "enforce", strict="off")
        assert off.review(b"q=a&debug=1", b"", None, 65536) is None
    events = [json.loads(record.getMessage()) for record in caplog.records]
    assert events[0] == {"event": "aiwaf.route_schema", "mode": "monitor", "method": "GET", "route": "/search",
                         "action": "unexpected", "input": "debug"}


def test_strict_inputs_warn_when_the_engine_is_too_old(tmp_path, caplog, monkeypatch):
    from aiwaf.core import route_risks

    real = route_risks.rust()

    class Old:
        def __init__(self, manifest):
            self._checker = real.RouteRiskChecker(manifest)

        def __getattr__(self, name):
            if name == "strict":
                raise AttributeError(name)
            return getattr(self._checker, name)

    monkeypatch.setattr(route_risks, "rust", lambda: type("R", (), {"RouteRiskChecker": Old})())
    with caplog.at_level(logging.WARNING, logger="aiwaf.route_schema"):
        RouteRisks(SCHEMA["manifest"], {"allow": []})
    assert "Strict inputs need aiwaf-rust 0.2.16" in caplog.text


def test_schema_review_modes_log_names_and_never_values(tmp_path, caplog):
    from aiwaf.core.sql_injection import Finding

    path = write(tmp_path, SCHEMA["manifest"])
    prose, finding = b'{"body": "use sleep(5) in the query"}', Finding("sql_function")
    with caplog.at_level(logging.WARNING, logger="aiwaf.route_schema"):
        # monitor, the default, logs what enforce would do and keeps the finding.
        assert observer_for("off", path, "/comments", "POST").review(b"", prose, finding, 65536) == finding
        assert observer_for("off", path, "/comments", "POST", "enforce").review(b"", prose, finding, 65536) is None
        mismatch = observer_for("off", path, "/comments", "POST", "enforce").review(b"post_id=1%20OR%201%3D1", b"", None, 65536)
    assert mismatch.rule == "schema_type_mismatch"
    events = [json.loads(record.getMessage()) for record in caplog.records]
    assert events[0] == {"event": "aiwaf.route_schema", "mode": "monitor", "method": "POST", "route": "/comments",
                         "action": "relax", "rule": "sql_function"}
    assert events[2]["input"] == "post_id" and events[2]["type"] == "int"
    assert "sleep" not in caplog.text and "OR" not in caplog.text
    # Off, or a manifest with no saved inputs, gives no review.
    assert not hasattr(observer_for("block", path, "/comments", schema_mode="off"), "review")
    (tmp_path / "plain").mkdir()
    assert observer_for("off", write(tmp_path / "plain", manifest_with([{"sink": "path", "input": ""}])), "/search") is None
    with pytest.raises(ValueError):
        observer_for("block", path, "/comments", schema_mode="strict")


def test_a_failing_review_keeps_the_finding(tmp_path, monkeypatch):
    from aiwaf.core.sql_injection import Finding

    review = observer_for("off", write(tmp_path, SCHEMA["manifest"]), "/comments", schema_mode="enforce").review
    monkeypatch.setattr(RouteRisks, "review", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert review(b"", b"{}", Finding("sql_function"), 65536) == Finding("sql_function")


def test_inputs_from_several_views_merge_and_an_unknown_one_makes_the_list_incomplete(tmp_path):
    code = ("from flask import request\n"
            "def search():\n    return int(request.args['page']), request.args.get('q')\n"
            "def other(request):\n    helper(request)\n")
    report = report_for(tmp_path, "views.py", code)
    manifest = {"routes": {"/search": {"view": "views.search"}, "/other": {"view": "views.other"}, "/none": {"view": "x.y"}}}
    assert apply_report(report, manifest)["inputs"] == 2
    assert manifest["routes"]["/search"] == {"view": "views.search", "inputs": {"page": "int", "q": "text"}, "inputs_complete": True}
    assert manifest["routes"]["/other"]["inputs_complete"] is False
    assert "inputs" not in manifest["routes"]["/none"]
    assert "inputs and types for 2 routes" in route_risks.describe_audit(manifest)


@pytest.mark.parametrize("case", SCHEMA["path_cases"], ids=lambda case: f"{case['path']} {case['expect']}")
def test_shared_path_cases(case):
    from aiwaf.core.sql_injection import SQLInjectionPolicy

    found = SQLInjectionPolicy("block").inspect_path(case["path"])
    review = RouteRisks(SCHEMA["manifest"]).review_path(case["path"], 65536, found.rule) if found else None
    outcome = review[0] if review else ("block" if found else "pass")
    assert outcome == case["expect"]
    if outcome == "mismatch":
        assert review[2] == case["input"]


def test_path_findings_are_reviewed_in_the_request_check(tmp_path):
    from types import SimpleNamespace
    from aiwaf.core.sql_injection import SQLInjectionPolicy, inspect_wsgi_request

    path = write(tmp_path, SCHEMA["manifest"])
    request = SimpleNamespace(query_string=b"", content_type="", path="/posts/use-sleep(5)-in-queries", headers={})
    observer = observer_for("off", path, request.path, "GET", "enforce")
    assert inspect_wsgi_request(request, SQLInjectionPolicy(), observer=observer) is None
    assert inspect_wsgi_request(request, SQLInjectionPolicy(), observer=observer_for("off", path, request.path, "GET")).rule == "sql_function"
    assert observer.review_path("/x", route_risks.Finding("payload_inspection_limit", 413), 1).status == 413


def test_every_view_on_a_shared_path_is_linked():
    from aiwaf.core.request_declarations import merge_entries

    merged = merge_entries({"view": "app.list_posts", "methods": ["GET"]}, {"view": "app.create_post", "methods": ["POST"]})
    assert merged["views"] == ["app.create_post", "app.list_posts"]
    assert "views" not in merge_entries({"view": "app.a", "methods": ["GET"]}, {"view": "app.a", "methods": ["POST"]})
    manifest = {"routes": {"/posts": merged, "/only": {**merged}}}
    report = {"findings": [], "inputs": [
        {"function": "list_posts", "file": "app.py", "inputs": {"q": "text"}, "complete": True},
        {"function": "create_post", "file": "app.py", "inputs": {"title": "text"}, "complete": True},
    ]}
    apply_report(report, manifest)
    assert manifest["routes"]["/posts"]["inputs"] == {"q": "text", "title": "text"}
    assert manifest["routes"]["/posts"]["inputs_complete"] is True
    # One view's code not found: the list is incomplete, and says which.
    manifest = {"routes": {"/posts": {**merged}}}
    apply_report({"findings": [], "inputs": report["inputs"][:1]}, manifest)
    route = manifest["routes"]["/posts"]
    assert route["inputs_complete"] is False
    assert [reason["code"] for reason in route["inputs_incomplete"]] == ["unread_view"]
    assert "app.create_post" in route["inputs_incomplete"][0]["detail"]


def test_path_parameters_are_dropped_from_inputs():
    from aiwaf.core.input_sources import finalize, path_parameters

    assert path_parameters("/a/<int:id>/<slug>/{item_id}/{x:int}/:name/:opt?/[page]/[...rest]") == [
        "id", "slug", "item_id", "x", "name", "opt", "page", "rest"]
    routes = {"/posts/<int:post_id>": {"inputs": {"post_id": "text", "q": "text"}, "inputs_complete": True}}
    finalize(routes)
    assert routes["/posts/<int:post_id>"]["inputs"] == {"q": "text"}


def test_each_method_gets_its_own_list_when_its_views_differ(tmp_path, caplog):
    from aiwaf.core.input_sources import finalize
    from aiwaf.core.request_declarations import merge_entries

    route = merge_entries({"view": "app.list_posts", "methods": ["GET", "HEAD"]},
                          {"view": "app.create_post", "methods": ["POST"],
                           "declared_inputs": {"inputs": {"draft": "bool"}, "covers": [], "reasons": [], "standalone": False}})
    assert route["view_methods"] == {"app.create_post": ["POST"], "app.list_posts": ["GET", "HEAD"]}
    assert route["declared_methods"] == {"POST": ["draft"]}
    manifest = {"routes": {"/posts": route}}
    apply_report({"findings": [], "inputs": [
        {"function": "list_posts", "file": "app.py", "inputs": {"q": "text"}, "complete": True},
        {"function": "create_post", "file": "app.py", "inputs": {"title": "text"}, "complete": True},
    ]}, manifest, {"declarations": {"/posts": {"inputs": {"trace": "text"}}}, "observations": {"/posts": {"q": "text"}}})
    saved = manifest["routes"]["/posts"]
    # A recorded name could be read by any method, so every method's list has it.
    assert saved["inputs_by_method"] == {"GET": ["q", "trace"], "HEAD": ["q", "trace"],
                                         "POST": ["draft", "q", "title", "trace"]}
    assert "_input_methods" not in saved and "declared_methods" not in saved
    # The engine checks each method's own list.
    path = write(tmp_path, {**manifest})
    with caplog.at_level(logging.WARNING, logger="aiwaf.route_schema"):
        get = observer_for("off", path, "/posts", "GET", "enforce", strict=True)
        assert get.review(b"title=x", b"", None, 65536).rule == "schema_unexpected_input"
        assert get.review(b"q=x", b"", None, 65536) is None
        post = observer_for("off", path, "/posts", "POST", "enforce", strict=True)
        assert post.review(b"", b'{"title": "t"}', None, 65536) is None
    # An older engine without review_method checks the whole list.
    risks = RouteRisks(manifest, {"allow": []})
    risks._checker = type("Old", (), {"review": lambda self, *args: ("old",) + args[:1]})()
    assert risks.review("/posts", "title=x", b"", 65536, None, "GET") == ("old", "/posts")
    # One method's code not found, or the list incomplete: no per-method lists.
    finalize(routes := {"/x": {"methods": ["GET", "POST"], "inputs": {"a": "text"}, "inputs_complete": False,
                               "inputs_incomplete": [{"code": "helper"}], "_input_methods": {"GET": ["a"]}}})
    assert "inputs_by_method" not in routes["/x"]
    finalize(routes := {"/y": {"methods": ["GET"], "inputs": {"a": "text"}, "inputs_complete": True, "_input_methods": {"GET": ["a"]}}})
    assert "inputs_by_method" not in routes["/y"]
    finalize(routes := {"/z": {"methods": ["GET", "POST"], "inputs": {"a": "text"}, "inputs_complete": True,
                               "_input_methods": {"GET": ["a"], "POST": ["a"]}}})
    assert "inputs_by_method" not in routes["/z"]


def test_openapi_operations_say_which_method_reads_what():
    from aiwaf.core.input_sources import finalize, openapi_inputs

    spec = {"paths": {"/notes": {
        "get": {"parameters": [{"name": "q", "in": "query", "schema": {"type": "string"}}]},
        "post": {"requestBody": {"content": {"application/json": {"schema": {
            "type": "object", "additionalProperties": False, "properties": {"text": {"type": "string"}}}}}}},
    }}}
    routes = {"/notes": {"methods": ["GET", "POST"]}}
    finalize(routes, openapi=openapi_inputs(spec))
    assert routes["/notes"]["inputs_by_method"] == {"GET": ["q"], "POST": ["text"]}


def test_nested_keys_are_saved_and_checked(tmp_path):
    from aiwaf.core.input_schema import source_inputs
    from aiwaf.core.input_sources import finalize

    entries = {entry["function"]: entry for entry in source_inputs(
        "from flask import request\n"
        "def profile():\n"
        "    data = request.get_json()\n"
        "    return data['user']['name'], data['user'].get('email'), data['title'], data['meta'][key]\n"
        "def leaky():\n"
        "    data = request.get_json()\n"
        "    return data['user']['name'], save(data['user'])\n"
    )}
    assert entries["profile"]["nested"] == {"user": ["email", "name"]}
    assert "nested" not in entries["leaky"]
    # Every view reading user must read only its keys; a declaration's names have none known.
    manifest = {"routes": {"/p": {"view": "app.profile", "methods": ["POST"]}, "/both": merge_both()}}
    apply_report({"findings": [], "inputs": [
        {**entries["profile"], "function": "profile", "file": "app.py", "complete": True, "reasons": []},
        {"function": "a", "file": "app.py", "inputs": {"user": "text"}, "complete": True, "nested": {"user": ["name"]}},
        {"function": "b", "file": "app.py", "inputs": {"user": "text"}, "complete": True},
    ]}, manifest, {"declarations": {"/p": {"inputs": {"title": "text"}}}})
    assert manifest["routes"]["/p"]["inputs_nested"] == {"user": ["email", "name"]}
    assert "inputs_nested" not in manifest["routes"]["/both"]
    routes = {"/d": {"inputs": {"user": "text"}, "inputs_complete": True, "_nested": {"user": {"name"}}}}
    finalize(routes, declarations={"/d": {"inputs": {"user": "text"}}})
    assert "inputs_nested" not in routes["/d"]
    # The engine checks the keys inside, skipping exempt paths.
    path = write(tmp_path, {"routes": {"/p": manifest["routes"]["/p"], "/legacy/p": manifest["routes"]["/p"]}})
    enforce = observer_for("off", path, "/p", "POST", "enforce", strict=True, strict_exempt="/legacy/")
    assert enforce.review(b"", b'{"user": {"name": "a", "role": "admin"}, "title": "t"}', None, 65536).rule == "schema_unexpected_input"
    assert enforce.review(b"", b'{"user": {"name": "a"}, "title": "t"}', None, 65536) is None
    exempt = observer_for("off", path, "/legacy/p", "POST", "enforce", strict=True, strict_exempt="/legacy/")
    assert exempt.review(b"anything=1", b"", None, 65536) is None


def merge_both():
    from aiwaf.core.request_declarations import merge_entries

    return merge_entries({"view": "app.a", "methods": ["GET"]}, {"view": "app.b", "methods": ["POST"]})


def test_strict_exempt_setting(monkeypatch):
    from aiwaf.core.route_risks import strict_inputs

    monkeypatch.setenv("AIWAF_STRICT_INPUTS_EXEMPT", "/webhooks/,/legacy/*")
    assert strict_inputs(True)["exempt"] == ["/webhooks/", "/legacy/*"]
    assert strict_inputs(True, exempt=["/x/"])["exempt"] == ["/x/"]


def test_an_engine_without_exempt_paths_checks_no_strict_inputs(tmp_path, caplog, monkeypatch):
    from aiwaf.core import route_risks

    real = route_risks.rust()

    class Old:
        def __init__(self, manifest):
            self._checker = real.RouteRiskChecker(manifest)

        def __getattr__(self, name):
            if name == "strict_exempt":
                raise AttributeError(name)
            return getattr(self._checker, name)

    monkeypatch.setattr(route_risks, "rust", lambda: type("R", (), {"RouteRiskChecker": Old})())
    manifest = {"routes": {"/s": {"view": "s", "inputs": {"q": "text"}, "inputs_complete": True}}}
    with caplog.at_level(logging.WARNING, logger="aiwaf.route_schema"):
        risks = RouteRisks(manifest, {"allow": [], "exempt": ["/legacy/"]})
    assert "AIWAF_STRICT_INPUTS_EXEMPT needs aiwaf-rust 0.2.17" in caplog.text
    assert risks.review("/s", "q=1&extra=2", b"", 65536, None) is None
    assert RouteRisks(manifest, {"allow": [], "exempt": []}).review("/s", "q=1&extra=2", b"", 65536, None)[0] == "unexpected"


def test_audit_into_manifest_without_source_keeps_framework_inputs():
    from aiwaf.core.route_risks import audit_into_manifest

    framework = {"inputs": {"quantity": "int"}, "covers": [], "reasons": [], "standalone": False}
    manifest = {"routes": {"/orders": {"view": "v", "declared_inputs": framework, "input_declaration": None}}}
    assert audit_into_manifest(manifest, None) is None
    route = manifest["routes"]["/orders"]
    assert route["inputs"] == {"quantity": "int"} and route["inputs_complete"] is False
    assert "declared_inputs" not in route and "input_declaration" not in route
    assert "injection_audit" not in manifest


def test_a_framework_list_for_an_unread_view_is_kept_but_not_complete():
    from aiwaf.core.route_risks import apply_report

    report = {"findings": [], "files_scanned": 1, "inputs": [{"function": "views.read", "file": "views.py", "inputs": {"q": "text"}, "complete": True}]}
    manifest = {"routes": {
        "/read": {"view": "views.read", "methods": ["GET"]},
        "/open": {"view": "views.open", "methods": ["POST"],
                  "declared_inputs": {"inputs": {"n": "int"}, "covers": [], "reasons": [], "standalone": False}},
        "/whole": {"view": "views.whole", "methods": ["POST"],
                   "declared_inputs": {"inputs": {"m": "int"}, "covers": [], "reasons": [], "standalone": True}},
    }}
    apply_report(report, manifest)
    routes = manifest["routes"]
    assert routes["/read"]["inputs_complete"] is True
    for path, name in (("/open", "n"), ("/whole", "m")):
        assert routes[path]["inputs"] == {name: "int"} and routes[path]["inputs_complete"] is False
        assert [reason["code"] for reason in routes[path]["inputs_incomplete"]] == ["unread_view"]
    assert manifest["injection_audit"]["routes_with_complete_inputs"] == 1
    assert manifest["injection_audit"]["incomplete_input_reasons"] == {"unread_view": 2}
