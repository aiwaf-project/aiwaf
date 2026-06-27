from flask import Flask, jsonify

from aiwaf.flask.path_manifest import extract_flask_routes, generate_flask_manifest


def test_flask_manifest_extracts_routes(tmp_path):
    app = Flask(__name__)

    @app.route("/login/", methods=["GET", "POST"])
    def login():
        return "ok"

    @app.route("/api/users/", methods=["GET"])
    def users():
        return jsonify([])

    routes = extract_flask_routes(app)

    assert "/login/" in routes
    assert routes["/login/"]["category"] == "auth"
    assert routes["/login/"]["methods"] == ["GET", "POST"]
    assert "/api/users/" in routes
    assert routes["/api/users/"]["category"] == "api"

    manifest_path = tmp_path / "paths.json"
    manifest = generate_flask_manifest(app, manifest_path)
    assert manifest_path.exists()
    assert manifest["framework"] == "flask"
    assert manifest["context_hash"]
