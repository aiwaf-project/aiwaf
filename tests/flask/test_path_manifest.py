from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request
from werkzeug.security import check_password_hash
from types import SimpleNamespace

from aiwaf.flask.path_manifest import _methods, extract_flask_routes, generate_flask_manifest


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


def test_flask_manifest_excludes_internal_aiwaf_routes():
    app = Flask(__name__)

    @app.route("/aiwaf/status")
    def aiwaf_status():
        return jsonify({"ok": True})

    @app.route("/health")
    def health():
        return "ok"

    routes = extract_flask_routes(app)

    assert "/aiwaf/status/" not in routes
    assert "/aiwaf/status" not in routes
    assert "/health/" in routes or "/health" in routes


def test_flask_manifest_filters_implicit_head_options_methods():
    rule = SimpleNamespace(methods={"GET", "HEAD", "OPTIONS", "POST"})

    assert _methods(rule) == ["GET", "POST"]


def test_flask_manifest_uses_view_method_metadata_when_rule_missing_methods():
    def view():
        return "ok"

    view.methods = ["PUT", "PATCH"]
    rule = SimpleNamespace(methods=None)

    assert _methods(rule, view) == ["PATCH", "PUT"]


def test_flask_manifest_uses_method_view_class_when_rule_missing_methods():
    class UserView:
        methods = ["GET", "POST", "DELETE"]

        def get(self):
            return "ok"

        def post(self):
            return "ok"

    def view():
        return "ok"

    view.view_class = UserView
    rule = SimpleNamespace(methods=None)

    assert _methods(rule, view) == ["GET", "POST"]


def test_flask_manifest_uses_route_hints_when_metadata_missing():
    rule = SimpleNamespace(methods=None, rule="/login/", endpoint="login")

    assert _methods(rule) == ["GET", "POST"]


def test_flask_manifest_defaults_unknown_method_route_to_get():
    rule = SimpleNamespace(methods=None)

    assert _methods(rule) == ["GET"]


def test_flask_manifest_reads_source_when_route_metadata_missing():
    def view():
        from flask import request

        if request.method == "POST":
            return "posted"
        return "ok"

    rule = SimpleNamespace(methods=None)

    assert _methods(rule, view) == ["GET", "POST"]


def test_flask_manifest_reads_request_form_source_when_route_metadata_missing():
    def view():
        from flask import request

        name = request.form.get("name")
        return name or "ok"

    rule = SimpleNamespace(methods=None)

    assert _methods(rule, view) == ["GET", "POST"]


def test_flask_manifest_detects_auth_endpoint_from_login_signals(tmp_path):
    app = Flask(__name__)

    def login_user(user):
        return user

    @app.route("/signin/", methods=["GET", "POST"])
    def signin():
        if check_password_hash("hash", "pw"):
            login_user(object())
        return "ok"

    routes = extract_flask_routes(app)
    route = routes["/signin/"]

    assert route["category"] == "auth"
    assert route["auth_action"] == "login"
    assert route["auth_confidence"] >= 0.5
    assert "check_password_hash" in route["auth_signals"]


def test_flask_manifest_marks_login_required_routes_authenticated(tmp_path):
    app = Flask(__name__)

    def login_required(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @app.route("/dashboard/")
    @login_required
    def dashboard():
        return "ok"

    routes = extract_flask_routes(app)
    route = routes["/dashboard/"]

    assert route["category"] == "app"
    assert route["auth_required"] is True


def test_flask_manifest_detects_api_endpoint_from_jsonify_and_body(tmp_path):
    app = Flask(__name__)

    @app.route("/users-json/", methods=["POST"])
    def users_json():
        payload = request.get_json()
        return jsonify({"payload": payload})

    routes = extract_flask_routes(app)
    route = routes["/users-json/"]

    assert route["category"] == "api"
    assert route["response_type"] == "json"
    assert route["api_confidence"] >= 0.5
    assert route["request_body"] is True
    assert "jsonify" in route["api_signals"]


def test_flask_manifest_detects_form_payload_over_mixed_json_response(tmp_path):
    app = Flask(__name__)

    @app.route("/contact/", methods=["GET", "POST"])
    def contact():
        name = request.form.get("name")
        if not name:
            return jsonify({"error": "name required"})
        if request.method == "POST":
            return redirect("/thanks/")
        return render_template("contact.html")

    routes = extract_flask_routes(app)
    route = routes["/contact/"]

    assert route["category"] == "form"
    assert route["response_type"] == "mixed"
    assert route["payload_type"] == "form"
    assert route["request_body"] is True
    assert route["form_confidence"] >= 0.5
    assert "request.form" in route["form_signals"]


def test_flask_manifest_does_not_classify_get_template_page_as_form(tmp_path):
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        return render_template("home.html")

    routes = extract_flask_routes(app)
    route = routes["/"]

    assert route["category"] == "unknown"
    assert route["response_type"] == "html"
    assert "payload_type" not in route
    assert "form_confidence" not in route


METHOD_APP = '''from flask import Flask, request
app = Flask(__name__)


@app.get("/items")
def list_items():
    return {"q": request.args.get("q"), "page": int(request.args.get("page", "1"))}


@app.post("/items")
def create_item():
    return {"title": request.json["title"], "page": request.json.get("page")}


@app.route("/items", methods=["PATCH"])
def update_item():
    return {"note": request.form["note"]}


@app.post("/search")
def search():
    return {"limit": int(request.args["limit"]), "label": request.form.get("limit"), "flag": request.args.get("flag", type=bool)}
'''


def test_each_method_keeps_its_inputs_and_types_and_each_location_its_own(tmp_path, monkeypatch):
    import importlib
    import sys

    (tmp_path / "method_app.py").write_text(METHOD_APP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "method_app", raising=False)
    app = importlib.import_module("method_app").app
    routes = generate_flask_manifest(app, tmp_path / "paths.json", audit_source=str(tmp_path))["routes"]
    items = routes["/items"]
    assert items["inputs_complete"] is True
    assert items["inputs_by_method"] == {"GET": ["page", "q"], "PATCH": ["note"], "POST": ["page", "title"]}
    # page is the GET's integer and the POST's text.
    assert items["inputs"]["page"] == "text"
    assert items["inputs_types_by_method"] == {"GET": {"page": "int"}}
    search = routes["/search"]
    # An integer query limit beside a text form limit, and type=bool, which accepts any text.
    assert search["inputs"] == {"limit": "text", "flag": "text"}
    assert search["inputs_types_by_location"] == {"POST": {"query": {"limit": "int"}}}
