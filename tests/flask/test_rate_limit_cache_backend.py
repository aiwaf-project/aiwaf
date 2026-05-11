from flask import Flask

from aiwaf.flask.rate_limit_middleware import RateLimitMiddleware, _aiwaf_cache


def test_flask_rate_limit_redis_backend_missing_url_falls_back():
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
            "AIWAF_EXEMPT_PATHS": set(),
            "AIWAF_RATE_WINDOW": 60,
            "AIWAF_RATE_MAX": 1,
            "AIWAF_RATE_FLOOD": 100,
            "AIWAF_RATE_CACHE_BACKEND": "redis",  # no URL, should fall back
        }
    )
    _aiwaf_cache.clear()
    RateLimitMiddleware(app)

    @app.route("/rl")
    def rl():
        return "ok"

    headers = {"User-Agent": "Test Browser 1.0"}
    with app.test_client() as client:
        assert client.get("/rl", headers=headers).status_code == 200
        assert client.get("/rl", headers=headers).status_code == 429

