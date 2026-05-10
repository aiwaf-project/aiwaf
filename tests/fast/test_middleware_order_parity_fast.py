from fastapi import FastAPI

from aiwaf.fast import AIWAF


def test_fast_middleware_order_matches_django_semantics():
    app = FastAPI()
    AIWAF(
        app,
        AIWAF_ACCESS_LOG="",
        geo_block={"enabled": True, "block_countries": ["US"]},
    )

    names = [mw.cls.__name__ for mw in app.user_middleware]
    expected = [
        "GeoBlockMiddleware",
        "IPAndKeywordBlockMiddleware",
        "RateLimitMiddleware",
        "AIAnomalyMiddleware",
        "HoneypotTimingMiddleware",
        "UUIDTamperMiddleware",
        "HeaderValidationMiddleware",
        "AIWAFLoggingMiddleware",
    ]
    # user_middleware order is execution order for incoming requests.
    assert names == expected
