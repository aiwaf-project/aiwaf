import json
import sys

import pytest

from aiwaf.cli import _handle_init


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
