from aiwaf.core.source_methods import infer_methods_from_source


def test_source_methods_module_contract():
    def handler(request):
        return request.json()

    assert "POST" in infer_methods_from_source(handler)

