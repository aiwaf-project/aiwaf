from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_first() -> None:
    repo_root = Path(__file__).resolve().parent
    repo_root_str = str(repo_root)
    try:
        while repo_root_str in sys.path:
            sys.path.remove(repo_root_str)
    except Exception:
        pass
    sys.path.insert(0, repo_root_str)


def _purge_aiwaf_modules() -> None:
    # If an older/installed `aiwaf` package was imported already, purge it once
    # at session start so imports resolve against the source tree. Do NOT purge
    # during collection, or you'll end up with multiple `aiwaf` module instances
    # (breaking Flask-SQLAlchemy registration, caches, etc.).
    for name in list(sys.modules.keys()):
        if name == "aiwaf" or name.startswith("aiwaf."):
            sys.modules.pop(name, None)


def pytest_configure(config):
    # Keep repo root at the front, but don't purge modules here. Conftest files
    # under `tests/` may import `aiwaf` during configuration, and purging later
    # would create duplicate module instances.
    _ensure_repo_root_first()


def pytest_ignore_collect(collection_path, config):
    marker_expr = (config.getoption("-m") or "").strip()
    if not marker_expr:
        return False

    normalized = Path(str(collection_path)).as_posix()

    # When selecting a suite via `-m flask` or `-m django`, default to collecting
    # only that subtree. Keep the `tests/` root collectable so pytest can descend.
    if "flask" in marker_expr:
        if "tests/django/" in normalized or normalized.endswith("tests/django"):
            return True
        if "tests/flask/" in normalized or normalized.endswith("tests/flask"):
            return False
        if normalized.endswith("tests"):
            return False
        if "tests/" in normalized:
            return True

    if "django" in marker_expr:
        if "tests/flask/" in normalized or normalized.endswith("tests/flask"):
            return True
        if "tests/django/" in normalized or normalized.endswith("tests/django"):
            return False
        if normalized.endswith("tests"):
            return False
        if "tests/" in normalized:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/flask/" in path:
            item.add_marker("flask")
        elif "/tests/django/" in path:
            item.add_marker("django")


# Run once when this root conftest is imported (before sub-conftests under tests/).
_ensure_repo_root_first()
_purge_aiwaf_modules()
