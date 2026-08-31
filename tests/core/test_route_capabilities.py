from aiwaf.core.route_capabilities import path_looks_uuid_capable


def test_route_capabilities_module_contract():
    assert path_looks_uuid_capable("/users/<uuid:user_id>") is True
    assert path_looks_uuid_capable("/health") is False

