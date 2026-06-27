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


def test_build_route_entry_classifies_portal_route_as_authenticated_app():
    path, entry = build_route_entry(
        path="/portal/classes/",
        methods=["GET"],
        view="portal.views.classes",
    )

    assert path == "/portal/classes/"
    assert entry["methods"] == ["GET"]
    assert entry["category"] == "app"
    assert entry["auth_required"] is True


def test_build_route_entry_keeps_portal_upload_detection_authenticated():
    path, entry = build_route_entry(
        path="/portal/profile/upload/photo/",
        methods=["GET", "POST"],
        view="portal.views.upload_photo",
    )

    assert path == "/portal/profile/upload/photo/"
    assert entry["category"] == "upload"
    assert entry["auth_required"] is True
    assert "payload_validation" in entry["protections"]


def test_build_route_entry_uses_auth_detector_metadata():
    path, entry = build_route_entry(
        path="/token/",
        methods=["POST"],
        view="app.login",
        metadata={
            "auth_action": "token_login",
            "auth_confidence": 0.9,
            "auth_signals": ["OAuth2PasswordRequestForm", "create_access_token"],
        },
    )

    assert path == "/token/"
    assert entry["category"] == "auth"
    assert entry["auth_action"] == "token_login"
    assert entry["auth_confidence"] == 0.9
    assert "create_access_token" in entry["auth_signals"]


def test_build_route_entry_uses_api_detector_metadata():
    path, entry = build_route_entry(
        path="/users/",
        methods=["POST"],
        view="app.users",
        metadata={
            "response_type": "json",
            "api_confidence": 0.94,
            "api_signals": ["JsonResponse", "request.body"],
            "request_body": True,
        },
    )

    assert path == "/users/"
    assert entry["category"] == "api"
    assert entry["response_type"] == "json"
    assert entry["api_confidence"] == 0.94
    assert entry["request_body"] is True
    assert "payload_validation" in entry["protections"]
    assert "content_type_validation" in entry["protections"]


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
