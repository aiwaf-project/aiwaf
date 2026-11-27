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


@pytest.mark.parametrize("framework, snippet", [("flask", 'NativeAIWAF(app, "from_settings")'),
                                                ("fastapi", 'config="from_settings"')])
def test_init_prints_the_native_setup_unless_asked_not_to(monkeypatch, tmp_path, capsys, framework, snippet):
    pytest.importorskip(framework)
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    source = ("from flask import Flask\napp = Flask(__name__)\napp.config['AIWAF_RATE_MAX'] = 7\n"
              "app.add_url_rule('/health', 'health', lambda: 'ok')\n" if framework == "flask"
              else "from fastapi import FastAPI\napp = FastAPI()\napp.get('/health')(lambda: {'ok': True})\n")
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    _handle_init(["--framework", framework, "--app", "app:app"])
    output = capsys.readouterr().out
    assert "Native pipeline (recommended for new installs)" in output and snippet in output
    assert "Native pipeline settings:" in output
    if framework == "flask":
        assert "AIWAF_RATE_MAX:" not in output  # Carried over, so not listed as a difference.

    _drop_app_module()
    _handle_init(["--framework", framework, "--app", "app:app", "--no-native"])
    assert "Native pipeline" not in capsys.readouterr().out


def test_native_setup_tells_an_existing_django_install_how_to_switch(monkeypatch):
    from aiwaf.core import native_settings
    from aiwaf.core.native_settings import describe_native_setup, native_config

    regular = ["aiwaf.django.middleware.IPAndKeywordBlockMiddleware"]
    text = describe_native_setup(native_config({}, "django", middleware=regular, environ={}), "django", middleware=regular)
    assert "uses the regular middleware, which keeps working" in text or "runs as the native pipeline automatically" in text
    assert "aiwaf.django.csv_pipeline.NativeAIWAFMiddleware" in text and 'AIWAF_CSV_PIPELINE = "from_settings"' in text
    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: False)
    new = describe_native_setup(native_config({}, "django", middleware=[], environ={}), "fast", middleware=[])
    assert "regular middleware" not in new and "pip install -U aiwaf-rust" in new
    assert 'config="from_settings"' in new


def test_native_setup_says_when_the_switch_is_automatic(monkeypatch):
    from aiwaf.core import native_settings
    from aiwaf.core.native_settings import NativeConversion, describe_native_setup

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: True)
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    text = describe_native_setup(NativeConversion(config={}), "flask")
    assert "runs as the native pipeline automatically" in text and "AIWAF_NATIVE_PIPELINE = False" in text


def test_init_writes_nothing_for_an_app_with_no_routes(monkeypatch, tmp_path, capsys):
    pytest.importorskip("flask")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exited:
        _handle_init(["--framework", "flask", "--app", "app:app", "--no-native"])
    assert exited.value.code == 1
    assert "No routes found in app:app (flask)" in capsys.readouterr().err
    assert not (tmp_path / ".aiwaf" / "paths.json").exists()


def test_init_lists_the_keys_of_a_nested_pydantic_body(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n"
        "class Customer(BaseModel):\n    email: str\n"
        "class Order(BaseModel):\n    sku: str\n    customer: Customer\n"
        "app = FastAPI()\n"
        "@app.post('/orders')\n"
        "def create(order: Order):\n    return {'email': order.customer.email}\n",
        encoding="utf-8",
    )
    _handle_init(["--framework", "fastapi", "--app", "app:app", "--no-native"])
    route = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))["routes"]["/orders"]
    assert route["inputs_nested"] == {"customer": ["email"]}
    assert route["inputs_complete"] is True


def test_init_keeps_the_old_manifest_when_the_audit_source_is_missing(monkeypatch, tmp_path, capsys):
    pytest.importorskip("flask")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\napp.add_url_rule('/a', 'a', lambda: 'ok')\n",
                                     encoding="utf-8")
    output = tmp_path / "paths.json"
    output.write_text('{"routes": {"/kept": {}}}', encoding="utf-8")
    with pytest.raises(SystemExit) as exited:
        _handle_init(["--framework", "flask", "--app", "app:app", "--source", "nowhere", "--output", str(output), "--no-native"])
    assert exited.value.code == 1
    assert "The audit source nowhere doesn't exist" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == '{"routes": {"/kept": {}}}'


def test_init_marks_a_route_the_audit_never_read_incomplete(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "from fastapi import FastAPI\nfrom pydantic import BaseModel\n"
        "class Order(BaseModel):\n    sku: str\n    quantity: int\n"
        "app = FastAPI()\n@app.post('/orders')\ndef create(order: Order):\n    return {'sku': order.sku}\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "other.py").write_text("def other():\n    return 1\n", encoding="utf-8")
    _handle_init(["--framework", "fastapi", "--app", "app:app", "--source", "unrelated", "--no-native"])
    route = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))["routes"]["/orders"]
    assert route["inputs"] == {"quantity": "int", "sku": "text"} and route["inputs_complete"] is False
    assert [reason["code"] for reason in route["inputs_incomplete"]] == ["unread_view"]
    _drop_app_module()
    _handle_init(["--framework", "fastapi", "--app", "app:app", "--source", ".", "--no-native"])
    route = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))["routes"]["/orders"]
    assert route["inputs_complete"] is True


def test_init_keeps_a_post_type_from_a_get_on_the_same_path(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    _drop_app_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text(
        "from fastapi import FastAPI\nfrom pydantic import BaseModel\n"
        "class NewOrder(BaseModel):\n    ref: int\n    note: str\n"
        "app = FastAPI()\n"
        "@app.get('/orders')\ndef list_orders(ref: str = ''):\n    return {'ref': ref}\n"
        "@app.post('/orders')\ndef create_order(order: NewOrder):\n    return {'ref': order.ref}\n",
        encoding="utf-8",
    )
    _handle_init(["--framework", "fastapi", "--app", "app:app", "--source", ".", "--no-native"])
    route = json.loads((tmp_path / ".aiwaf" / "paths.json").read_text(encoding="utf-8"))["routes"]["/orders"]
    assert route["inputs"] == {"note": "text", "ref": "text"}
    assert route["inputs_by_method"] == {"GET": ["ref"], "POST": ["note", "ref"]}
