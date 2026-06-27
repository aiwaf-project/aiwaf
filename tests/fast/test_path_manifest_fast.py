from fastapi import FastAPI

from aiwaf.fast.path_manifest import extract_fastapi_routes, generate_fastapi_manifest


def test_fastapi_manifest_extracts_routes(tmp_path):
    app = FastAPI()

    @app.post("/api/users/", tags=["users"])
    def create_user():
        return {"ok": True}

    routes = extract_fastapi_routes(app)

    assert "/api/users/" in routes
    assert routes["/api/users/"]["category"] == "api"
    assert routes["/api/users/"]["response_type"] == "json"
    assert routes["/api/users/"]["methods"] == ["POST"]
    assert routes["/api/users/"]["tags"] == ["users"]

    manifest_path = tmp_path / "paths.json"
    manifest = generate_fastapi_manifest(app, manifest_path)
    assert manifest_path.exists()
    assert manifest["framework"] == "fastapi"
    assert manifest["context_hash"]
