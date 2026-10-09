from flask import Flask

from aiwaf.flask.rate_limit_middleware import RateLimitMiddleware


def test_unavailable_shared_cache_returns_generic_503_without_calling_app():
    class BrokenCache:
        is_shared = True

        def consume_rate_limit(self, *_args, **_kwargs):
            raise ConnectionError('redis://secret:password@private-host')

    app = Flask(__name__)
    app._aiwaf_rate_cache_backend = BrokenCache()
    calls = []
    app.add_url_rule('/protected', view_func=lambda: calls.append(True) or 'unexpected')
    RateLimitMiddleware(app)
    response = app.test_client().get('/protected')
    assert response.status_code == 503
    assert response.json == {'error': 'temporarily_unavailable'}
    assert calls == []


def test_invalid_redis_config_does_not_silently_use_memory():
    import pytest
    app = Flask(__name__)
    app.config['AIWAF_RATE_CACHE_BACKEND'] = 'redis'
    with pytest.raises(ValueError, match='redis_url'):
        RateLimitMiddleware(app)
