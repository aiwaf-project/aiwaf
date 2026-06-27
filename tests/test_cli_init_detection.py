import pytest

from aiwaf import cli


class FlaskLikeApp:
    url_map = object()


class FastAPILikeApp:
    routes = []


def test_init_detects_single_installed_framework(monkeypatch):
    monkeypatch.setattr(cli, "_installed_frameworks", lambda: ["django"])

    assert cli._detect_framework(None) == "django"


def test_init_detects_flask_from_app_even_when_multiple_installed(monkeypatch):
    monkeypatch.setattr(cli, "_installed_frameworks", lambda: ["django", "flask", "fastapi"])

    assert cli._detect_framework(None, FlaskLikeApp()) == "flask"


def test_init_detects_fastapi_from_app_even_when_multiple_installed(monkeypatch):
    monkeypatch.setattr(cli, "_installed_frameworks", lambda: ["django", "flask", "fastapi"])

    assert cli._detect_framework(None, FastAPILikeApp()) == "fastapi"


def test_init_requires_framework_when_detection_is_ambiguous(monkeypatch):
    monkeypatch.setattr(cli, "_installed_frameworks", lambda: ["django", "flask"])

    with pytest.raises(SystemExit) as exc_info:
        cli._detect_framework(None)

    assert "Multiple supported frameworks are installed" in str(exc_info.value)


def test_init_normalizes_fast_alias():
    assert cli._detect_framework("fast") == "fastapi"
