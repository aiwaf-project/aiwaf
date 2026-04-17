# Test each Flask AIWAF middleware individually
import unittest
import pytest

try:
    from flask import Flask, jsonify, request
except Exception as exc:
    raise unittest.SkipTest("Flask is not installed") from exc

pytestmark = pytest.mark.flask
from aiwaf.flask.ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from aiwaf.flask.rate_limit_middleware import RateLimitMiddleware
from aiwaf.flask.honeypot_timing_middleware import HoneypotTimingMiddleware
from aiwaf.flask.header_validation_middleware import HeaderValidationMiddleware
from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware
from aiwaf.flask.uuid_tamper_middleware import UUIDTamperMiddleware


def create_app():
    app = Flask(__name__)
    app.config['AIWAF_RATE_WINDOW'] = 10
    app.config['AIWAF_RATE_MAX'] = 2
    app.config['AIWAF_RATE_FLOOD'] = 3
    app.config['AIWAF_MIN_FORM_TIME'] = 1.0

    # Register each middleware individually for testing
    IPAndKeywordBlockMiddleware(app)
    RateLimitMiddleware(app)
    HoneypotTimingMiddleware(app)
    HeaderValidationMiddleware(app)
    AIAnomalyMiddleware(app)
    UUIDTamperMiddleware(app)

    @app.route('/ipkw', methods=['GET'])
    def test_ipkw():
        # Should trigger keyword block if path contains malicious keyword
        return jsonify({'result': 'ipkw ok'})

    @app.route('/ratelimit', methods=['GET'])
    def test_ratelimit():
        # Should trigger rate limit after 2 requests in window
        return jsonify({'result': 'ratelimit ok'})

    @app.route('/honeypot', methods=['GET', 'POST'])
    def test_honeypot():
        # Should trigger honeypot timing on fast POST after GET
        return jsonify({'result': 'honeypot ok'})

    @app.route('/header', methods=['GET'])
    def test_header():
        # Should trigger header validation if User-Agent is missing/short
        return jsonify({'result': 'header ok'})

    @app.route('/anomaly', methods=['GET'])
    def test_anomaly():
        # Placeholder for anomaly detection
        return jsonify({'result': 'anomaly ok'})

    @app.route('/uuid', methods=['GET'])
    def test_uuid():
        # Should trigger UUID tamper block if uuid param is invalid
        return jsonify({'result': 'uuid ok'})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
