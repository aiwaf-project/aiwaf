from flask import Flask

from aiwaf.flask import AIWAF


def test_flask_middleware_order_matches_django_semantics(tmp_path):
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
            "AIWAF_USE_CSV": True,
            "AIWAF_DATA_DIR": str(tmp_path),
        }
    )

    aiwaf = AIWAF(app)
    names = list(aiwaf.middleware_instances.keys())
    expected = [
        "geo_block",
        "ip_keyword_block",
        "rate_limit",
        "ai_anomaly",
        "honeypot",
        "uuid_tamper",
        "header_validation",
        "logging",
    ]
    assert names == expected

