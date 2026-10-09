from flask import Flask

from aiwaf.flask.rate_limit_middleware import RateLimitMiddleware, _aiwaf_cache


def test_flask_rate_limit_redis_backend_missing_url_fails_configuration():
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
            "AIWAF_EXEMPT_PATHS": set(),
            "AIWAF_RATE_WINDOW": 60,
            "AIWAF_RATE_MAX": 1,
            "AIWAF_RATE_FLOOD": 100,
            "AIWAF_RATE_CACHE_BACKEND": "redis",
        }
    )
    _aiwaf_cache.clear()
    import pytest
    with pytest.raises(ValueError, match='redis_url'):
        RateLimitMiddleware(app)

