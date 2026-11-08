import json
import sys

import pytest

from aiwaf.cli import _handle_audit, _handle_init, main


def _drop_app_module():
    sys.modules.pop("app", None)


def test_unified_init_imports_flask_app_from_current_project_directory(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "@app.get('/health/')\n"
        "def health():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    _handle_init(["--framework", "flask", "--app", "app:app"])

    manifest = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))
    assert manifest["framework"] == "flask"
    assert "/health/" in manifest["routes"]


def test_unified_init_imports_fastapi_app_from_current_project_directory(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/health/')\n"
        "def health():\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )

    _handle_init(["--framework", "fastapi", "--app", "app:app"])

    manifest = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))
    assert manifest["framework"] == "fastapi"
    assert "/health/" in manifest["routes"]
@pytest.mark.parametrize("framework", ["flask", "fastapi", "django"])
def test_installed_frameworks_detect_and_main_dispatch(monkeypatch, framework):
    import types
    from aiwaf import cli

    calls = []
    monkeypatch.setattr(cli, "_installed_frameworks", lambda: [framework])
    if framework == "django":
        module = types.ModuleType("django.core.management")
        module.execute_from_command_line = lambda argv: calls.append(argv)
        monkeypatch.setitem(sys.modules, "django.core.management", module)
    else:
        module_name = "aiwaf.flask.cli" if framework == "flask" else "aiwaf.fast.cli"
        module = types.ModuleType(module_name)
        module.main = lambda: calls.append(list(sys.argv))
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, "argv", ["aiwaf", "status"])
    cli.main()
    assert calls == [["aiwaf " + framework, "status"]]


@pytest.mark.parametrize("installed", [[], ["flask", "fastapi"]])
def test_bare_command_requires_explicit_choice_when_no_single_framework(monkeypatch, installed):
    from aiwaf import cli

    monkeypatch.setattr(cli, "_installed_frameworks", lambda: installed)
    monkeypatch.setattr(sys, "argv", ["aiwaf", "status"])
    with pytest.raises(SystemExit, match="aiwaf flask"):
        cli.main()


@pytest.mark.parametrize("alias", ["flask", "fast", "fastapi"])
def test_explicit_framework_bypasses_installed_detection(monkeypatch, alias):
    import types
    from aiwaf import cli

    monkeypatch.setattr(cli, "_installed_frameworks", lambda: pytest.fail("explicit framework should bypass detection"))
    calls = []
    module_name = "aiwaf.flask.cli" if alias == "flask" else "aiwaf.fast.cli"
    module = types.ModuleType(module_name)
    module.main = lambda: calls.append(list(sys.argv))
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, "argv", ["aiwaf", alias, "status"])
    cli.main()
    assert calls == [["aiwaf " + alias, "status"]]


def test_detect_calls_django_trainer(monkeypatch):
    from aiwaf import cli

    calls = []
    monkeypatch.setattr("aiwaf.django.trainer.train", lambda: calls.append(True))
    cli.aiwaf_detect()
    assert calls == [True]


VULNERABLE_VIEW = (
    "import os\n"
    "def view(request):\n"
    "    os.system(request.GET['cmd'])\n"
)


def test_audit_exit_codes_and_reports(tmp_path, capsys):
    (tmp_path / "views.py").write_text(VULNERABLE_VIEW, encoding="utf-8")

    assert _handle_audit([str(tmp_path)]) == 0
    assert "found 1 possible" in capsys.readouterr().out
    assert _handle_audit([str(tmp_path), "--fail-on-findings"]) == 1
    capsys.readouterr()

    report = tmp_path / "report.json"
    assert _handle_audit([str(tmp_path), "--format", "json", "--output", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["language"] == "python"
    assert capsys.readouterr().out == ""


def test_audit_reports_a_missing_path(tmp_path, capsys):
    assert _handle_audit([str(tmp_path / "missing")]) == 2
    assert "missing" in capsys.readouterr().err


def test_main_dispatches_audit(monkeypatch, tmp_path, capsys):
    (tmp_path / "views.py").write_text(VULNERABLE_VIEW, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aiwaf", "audit", str(tmp_path), "--fail-on-findings"])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 1
    assert "os.system" in capsys.readouterr().out


FLASK_APP_WITH_SQL = (
    "from flask import Flask, request\n"
    "app = Flask(__name__)\n"
    "@app.get('/search')\n"
    "def search():\n"
    "    return db.execute('SELECT * FROM t WHERE q = ' + request.args['q'])\n"
    "@app.get('/health')\n"
    "def health():\n"
    "    return 'ok'\n"
)


@pytest.mark.parametrize("no_audit", [False, True])
def test_init_saves_injection_checks_unless_asked_not_to(monkeypatch, tmp_path, capsys, no_audit):
    pytest.importorskip("flask")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(FLASK_APP_WITH_SQL, encoding="utf-8")

    _handle_init(["--framework", "flask", "--app", "app:app"] + (["--no-audit"] if no_audit else []))

    routes = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))["routes"]
    output = capsys.readouterr().out
    if no_audit:
        assert "injection_risks" not in routes["/search"]
        assert "Injection checks: not run" in output
    else:
        assert routes["/search"]["injection_risks"] == [{"confidence": "high", "input": "q", "sink": "sql"}]
        assert "injection_risks" not in routes["/health"]
        assert "Injection checks: 1 routes from 1 files" in output
