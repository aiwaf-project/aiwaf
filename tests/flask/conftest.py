import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure repo root is at the front so `import aiwaf.flask` resolves from source.
try:
    while repo_root in sys.path:
        sys.path.remove(repo_root)
except Exception:
    pass
sys.path.insert(0, repo_root)

# Default CSV storage dir for Flask tests that don't set `AIWAF_DATA_DIR` explicitly.
# This prevents local repo state (e.g. `aiwaf_data/whitelist.csv`) from affecting tests.
if "AIWAF_DATA_DIR" not in os.environ:
    import tempfile

    base = os.path.join(repo_root, "tests", "flask", "test_aiwaf_data")
    os.makedirs(base, exist_ok=True)
    os.environ["AIWAF_DATA_DIR"] = tempfile.mkdtemp(prefix="aiwaf_flask_", dir=base)

# If an older/installed `aiwaf` package was imported before this conftest runs,
# it can get cached in `sys.modules` and break imports like `aiwaf.flask` even
# though the repo root is on `sys.path`. Force a clean import from source tree.
for name in list(sys.modules.keys()):
    if name == "aiwaf" or name.startswith("aiwaf."):
        sys.modules.pop(name, None)

import pytest

try:
    from flask import Flask
    from flask.testing import FlaskClient
except Exception as exc:
    pytest.skip("Flask is not installed", allow_module_level=True)

pytestmark = pytest.mark.flask

from aiwaf.flask.db_models import db

@pytest.fixture
def app():
    """Create and configure a test Flask app."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['AIWAF_RATE_WINDOW'] = 10
    app.config['AIWAF_RATE_MAX'] = 20
    app.config['AIWAF_RATE_FLOOD'] = 40
    app.config['AIWAF_MIN_FORM_TIME'] = 1.0
    
    # Force database mode for tests (disable CSV to test database functionality)
    app.config['AIWAF_USE_CSV'] = False
    
    # Disable path exemptions for tests to ensure middleware blocking works
    app.config['AIWAF_EXEMPT_PATHS'] = set()
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()

@pytest.fixture
def app_context(app):
    """Create an application context."""
    with app.app_context():
        yield app


@pytest.fixture(autouse=True)
def _default_header_injection(monkeypatch):
    """Inject browser-like headers so header validation doesn't block tests."""
    default_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    original_open = FlaskClient.open

    def open_with_defaults(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        merged = {**default_headers, **headers}
        kwargs["headers"] = merged
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(FlaskClient, "open", open_with_defaults)
    yield
