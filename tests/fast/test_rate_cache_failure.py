from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.rate_limit_middleware import RateLimitMiddleware


def test_unavailable_shared_cache_returns_generic_503_without_calling_app():
    class BrokenCache:
        is_shared = True

        def consume_rate_limit(self, *_args, **_kwargs):
            raise ConnectionError('redis://secret:password@private-host')

    app = FastAPI()
    calls = []

    @app.get('/protected')
    def protected():
        calls.append(True)
        return {'unexpected': True}

    app.add_middleware(RateLimitMiddleware, cache_backend=BrokenCache())
    response = TestClient(app).get('/protected')
    assert response.status_code == 503
    assert response.json() == {'error': 'temporarily_unavailable'}
    assert calls == []


def test_runtime_redis_config_is_bound_instead_of_silent_memory_fallback():
    from aiwaf.fast.config import AIWAFConfig
    app = FastAPI()
    config = AIWAFConfig()
    config.set('rate_limiting.cache_backend', 'redis')
    config.set('rate_limiting.redis_url', None)
    app.state.aiwaf_config = config
    app.add_middleware(RateLimitMiddleware)
    response = TestClient(app).get('/protected')
    assert response.status_code == 503
    assert response.json() == {'error': 'temporarily_unavailable'}
