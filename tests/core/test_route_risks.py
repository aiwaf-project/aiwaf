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
        assert counts == {"routes": 0, "linked": 0, "unlinked": 1}
        assert "injection_risks" not in manifest["routes"]["/search"]
    else:
        assert counts == {"routes": 1, "linked": 1, "unlinked": 0}
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
    assert audit_into_manifest(manifest, str(tmp_path)) == {"routes": 1, "linked": 1, "unlinked": 0}
    assert describe_audit(manifest) == "Injection checks: 1 routes from 1 files"
    manifest["injection_audit"]["unlinked_findings"] = 2
    assert "2 findings couldn't be tied" in describe_audit(manifest)
    with caplog.at_level(logging.WARNING, logger="aiwaf.injection_risk"):
        assert audit_into_manifest({"routes": {}}, str(tmp_path / "missing")) is None
    assert "Injection audit skipped" in caplog.text


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
