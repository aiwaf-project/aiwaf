from flask import Flask
from unittest.mock import patch

from aiwaf.flask.utils import get_ip


def test_flask_utils_get_ip_delegates_to_core_extractor():
    app = Flask(__name__)
    with app.test_request_context("/x", headers={"X-Forwarded-For": "203.0.113.220"}):
        with patch("aiwaf.flask.utils.extract_ip_from_flask_request", return_value="198.51.100.220") as mock_extract:
            assert get_ip() == "198.51.100.220"
            # called with flask.request proxy object
            assert mock_extract.call_count == 1
