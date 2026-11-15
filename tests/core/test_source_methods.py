from aiwaf.core.source_methods import infer_methods_from_source


def test_source_methods_module_contract():
    def handler(request):
        return request.json()

    assert "POST" in infer_methods_from_source(handler)


def test_shared_manifest_helpers_keep_adapter_behavior():
    from aiwaf.core.source_methods import methods_from_route_hint, normalize_methods
    from aiwaf.django.path_manifest import _methods_from_route_hint as django_hint
    from aiwaf.fast.path_manifest import _methods_from_route_hint as fast_hint
    from aiwaf.flask.path_manifest import _methods_from_route_hint as flask_hint

    assert normalize_methods(["post", "get", "get", "TRACE"]) == ["GET", "POST"]
    assert normalize_methods(None) == []
    assert methods_from_route_hint("/items") == ["GET"]
    for hint in (django_hint, fast_hint, flask_hint):
        assert hint("/account/login") == ["GET", "POST"]
        assert hint("/items", "item-create") == ["GET", "POST"]
    # Only FastAPI treats token routes as POST.
    assert fast_hint("/auth/token") == ["GET", "POST"]
    assert flask_hint("/auth/token") == ["GET"]
    assert django_hint("/auth/token") == ["GET"]
