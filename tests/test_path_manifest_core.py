from aiwaf.core.path_manifest import (
    build_manifest,
    build_route_entry,
    compile_manifest_to_path_rules,
    compute_context_hash,
)


def test_manifest_context_hash_is_stable_for_same_routes():
    routes = {
        "/login/": {
            "methods": ["GET", "POST"],
            "view": "LoginView",
            "protections": {"rate_limit": {"requests": 30, "window_seconds": 60}},
        }
    }

    assert compute_context_hash(routes) == compute_context_hash(dict(routes))


def test_build_route_entry_classifies_api_route():
    path, entry = build_route_entry(
        path="/api/users/",
        methods=["GET", "POST"],
        view="UserViewSet",
        metadata={"response_type": "json", "auth_required": True},
    )

    assert path == "/api/users/"
    assert entry["category"] == "api"
    assert entry["response_type"] == "json"
    assert entry["auth_required"] is True
    assert entry["protections"]["rate_limit"]["requests"] == 120


def test_compile_manifest_to_path_rules_maps_protections_to_existing_rules():
    manifest = build_manifest(
        framework="test",
        routes={
            "/api/users/": {
                "methods": ["GET"],
                "view": "UserViewSet",
                "protections": {
                    "rate_limit": {"requests": 120, "window_seconds": 60, "flood": 200},
                    "header_validation": {"enabled": False},
                    "honeypot": {"enabled": False},
                },
            }
        },
    )

    rules = compile_manifest_to_path_rules(manifest)

    assert rules == [
        {
            "PREFIX": "/api/users/",
            "DISABLE": ["header_validation", "honeypot"],
            "RATE_LIMIT": {"WINDOW": 60, "MAX": 120, "FLOOD": 200},
        }
    ]
