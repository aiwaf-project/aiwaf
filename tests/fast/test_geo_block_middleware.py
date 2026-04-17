from aiwaf.fast.middleware.geo_block_middleware import GeoBlockMiddleware, _normalize_country_list


def test_geo_block_country_list_normalization():
    assert _normalize_country_list(["us", " Ca "]) == {"US", "CA"}
    assert _normalize_country_list("de") == {"DE"}
    assert _normalize_country_list([]) == set()


def test_geo_block_middleware_initializes_block_country_config():
    async def _app(scope, receive, send):  # pragma: no cover - middleware init only
        return None

    middleware = GeoBlockMiddleware(_app, enabled=True, block_countries=["us"])
    assert middleware.enabled is True
    assert middleware.block_countries == {"US"}

