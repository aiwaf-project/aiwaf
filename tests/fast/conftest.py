"""
Pytest fixtures for AIWAF test suites.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fastapi  # noqa: F401
except Exception:
    pytest.skip("FastAPI is not installed", allow_module_level=True)

from aiwaf.core.runtime_storage import initialize_storage


def pytest_collection_modifyitems(items):
    """Mark framework-parity FastAPI tests as `fast` for `pytest -m fast`."""
    for item in items:
        fspath = str(item.fspath).replace("\\", "/")
        if "/tests/fast/" in fspath:
            item.add_marker(pytest.mark.fast)


@pytest.fixture(autouse=True)
def ensure_memory_storage():
    """
    Ensure each test gets a clean in-memory storage backend.
    """
    # Keep long-running suites opt-in unless explicitly enabled.
    os.environ.setdefault("AIWAF_RUN_PERF", "0")
    os.environ.setdefault("AIWAF_RUN_SOAK", "0")
    initialize_storage(backend="memory")
