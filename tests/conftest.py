"""Global pytest bootstrap for path and settings stability across environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure local repo imports win over any globally installed `tests` package.
repo_root_str = str(REPO_ROOT)
try:
    while repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
except Exception:
    pass
sys.path.insert(0, repo_root_str)

# If a foreign `tests` module was already imported, drop it so import resolves to repo tests/.
loaded_tests = sys.modules.get("tests")
if loaded_tests is not None:
    module_file = getattr(loaded_tests, "__file__", "") or ""
    if module_file and repo_root_str not in module_file:
        sys.modules.pop("tests", None)

# Default Django settings module for tests that call django.setup() directly.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.django.test_settings")


def pytest_ignore_collect(collection_path, config):
    """Limit collection scope by marker expression to avoid cross-suite imports."""
    markexpr = (getattr(config.option, "markexpr", "") or "").strip()
    if not markexpr:
        return False

    path_str = str(collection_path).replace("\\", "/")

    # Strictly scope framework suites when filtering by framework marker.
    if markexpr == "fast":
        return "/tests/django/" in path_str or "/tests/flask/" in path_str
    if markexpr == "django":
        return "/tests/fast/" in path_str or "/tests/flask/" in path_str
    if markexpr == "flask":
        return "/tests/fast/" in path_str or "/tests/django/" in path_str

    return False
