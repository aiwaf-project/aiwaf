from flask import Flask

from aiwaf.flask.storage import add_path_exemption, get_path_exemptions, remove_path_exemption
from aiwaf.flask.utils import is_path_exempt


def test_path_exemptions_csv_storage(tmp_path):
    app = Flask(__name__)
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = str(tmp_path)
    app.config['AIWAF_EXEMPT_PATHS'] = set()

    with app.app_context():
        add_path_exemption("/health", reason="Health check")
        add_path_exemption("/api/status", reason="Status")

        exemptions = get_path_exemptions()
        assert "/health" in exemptions
        assert "/api/status" in exemptions

        assert is_path_exempt("/health") is True
        assert is_path_exempt("/api/status") is True

        remove_path_exemption("/health")
        exemptions = get_path_exemptions()
        assert "/health" not in exemptions
        assert is_path_exempt("/health") is False
