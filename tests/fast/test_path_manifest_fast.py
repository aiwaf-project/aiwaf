from fastapi import FastAPI
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from types import SimpleNamespace

from aiwaf.fast.path_manifest import _methods, extract_fastapi_routes, generate_fastapi_manifest


def test_fastapi_manifest_extracts_routes(tmp_path):
    app = FastAPI()

    @app.post("/api/users/", tags=["users"])
    def create_user():
        return {"ok": True}

    routes = extract_fastapi_routes(app)

    assert "/api/users/" in routes
    assert routes["/api/users/"]["category"] == "api"
    assert routes["/api/users/"]["response_type"] == "json"
    assert routes["/api/users/"]["payload_type"] == "json"
    assert routes["/api/users/"]["methods"] == ["POST"]
    assert routes["/api/users/"]["tags"] == ["users"]

    manifest_path = tmp_path / "paths.json"
    manifest = generate_fastapi_manifest(app, manifest_path)
    assert manifest_path.exists()
    assert manifest["framework"] == "fastapi"
    assert manifest["context_hash"]


def test_fastapi_manifest_filters_head_options_methods():
    route = SimpleNamespace(methods={"GET", "HEAD", "OPTIONS", "DELETE"})

    assert _methods(route) == ["DELETE", "GET"]


def test_fastapi_manifest_uses_endpoint_method_metadata_when_route_missing_methods():
    endpoint = SimpleNamespace(methods=["POST", "PATCH"])
    route = SimpleNamespace(methods=None)

    assert _methods(route, endpoint) == ["PATCH", "POST"]


def test_fastapi_manifest_defaults_unknown_method_route_to_get():
    route = SimpleNamespace(methods=None)

    assert _methods(route) == ["GET"]


def test_fastapi_manifest_reads_source_when_route_metadata_missing():
    def endpoint(request):
        if request.method == "POST":
            return {"posted": True}
        return {"ok": True}

    route = SimpleNamespace(methods=None)

    assert _methods(route, endpoint) == ["GET", "POST"]


def test_fastapi_manifest_reads_request_json_source_when_route_metadata_missing():
    def endpoint(request):
        data = request.json
        return data

    route = SimpleNamespace(methods=None)

    assert _methods(route, endpoint) == ["GET", "POST"]


def create_access_token(data):
    return {"token": data}


def fastapi_token_endpoint(form_data: OAuth2PasswordRequestForm = Depends()):
    return create_access_token({"sub": form_data.username})


class UserPayload(BaseModel):
    name: str


def fastapi_body_endpoint(payload: UserPayload) -> dict:
    return {"name": payload.name}


def test_fastapi_manifest_detects_auth_endpoint_from_token_signals(tmp_path):
    app = SimpleNamespace(routes=[
        SimpleNamespace(path="/token/", endpoint=fastapi_token_endpoint, name="token", methods={"POST"}, tags=[]),
    ])

    routes = extract_fastapi_routes(app)
    route = routes["/token/"]

    assert route["category"] == "auth"
    assert route["auth_action"] == "token_login"
    assert route["auth_confidence"] >= 0.8
    assert "fastapi.security.OAuth2PasswordRequestForm" in route["auth_signals"]
    assert "create_access_token" in route["auth_signals"]


def test_fastapi_manifest_detects_api_endpoint_from_response_model_and_body_model():
    app = SimpleNamespace(routes=[
        SimpleNamespace(
            path="/users/",
            endpoint=fastapi_body_endpoint,
            name="users",
            methods={"POST"},
            tags=[],
            response_model=dict,
        ),
    ])

    routes = extract_fastapi_routes(app)
    route = routes["/users/"]

    assert route["category"] == "api"
    assert route["response_type"] == "json"
    assert route["payload_type"] == "json"
    assert route["api_confidence"] >= 0.5
    assert route["request_body"] is True
    assert "response_model" in route["api_signals"]
