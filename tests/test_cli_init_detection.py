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


def test_detect_django_settings_module_from_manage_py(tmp_path):
    manage_py = tmp_path / "manage.py"
    manage_py.write_text(
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')\n",
        encoding="utf-8",
    )

    assert cli._detect_django_settings_module_from_manage_py(tmp_path) == "project.settings"


def test_configure_django_for_init_uses_explicit_settings(monkeypatch):
    calls = {"setup": 0}

    class DjangoStub:
        @staticmethod
        def setup():
            calls["setup"] += 1

    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "django", DjangoStub)

    cli._configure_django_for_init("project.settings")

    assert __import__("os").environ["DJANGO_SETTINGS_MODULE"] == "project.settings"
    assert calls["setup"] == 1


def test_configure_django_for_init_auto_detects_manage_py(monkeypatch, tmp_path):
    calls = {"setup": 0}
    (tmp_path / "manage.py").write_text(
        'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autodetected.settings")\n',
        encoding="utf-8",
    )

    class DjangoStub:
        @staticmethod
        def setup():
            calls["setup"] += 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "django", DjangoStub)

    cli._configure_django_for_init()

    assert __import__("os").environ["DJANGO_SETTINGS_MODULE"] == "autodetected.settings"
    assert calls["setup"] == 1


def test_configure_django_for_init_requires_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli._configure_django_for_init()

    assert "--settings project.settings" in str(exc_info.value)
