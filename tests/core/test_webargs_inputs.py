"""webargs and flask-smorest arguments, read from a live Flask app and from source."""

import marshmallow as ma
import pytest
from flask import Flask, request
from flask.views import MethodView
from flask_smorest import Api, Blueprint
from webargs import fields
from webargs.flaskparser import use_args, use_kwargs

from aiwaf.core.input_schema import source_inputs
from aiwaf.core.input_sources import combine, finalize, flask_inputs, webargs_inputs
from aiwaf.core.request_declarations import merge_entries
from aiwaf.flask.path_manifest import extract_flask_routes


class Item(ma.Schema):
    name = fields.Str(data_key="Name")
    qty = fields.Int()
    price = fields.Decimal()
    on = fields.Bool()
    owner = fields.UUID()
    tags = fields.List(fields.Int())
    secret = fields.Str(dump_only=True)


class Loose(Item):
    class Meta:
        unknown = ma.INCLUDE


def build_app():
    app = Flask(__name__)
    app.config.update(API_TITLE="t", API_VERSION="1", OPENAPI_VERSION="3.0.2")

    @app.route("/search")
    @use_args({"q": fields.Str(), "page": fields.Int()}, location="query")
    def search(args):
        return args

    @app.route("/exact", methods=["POST"])
    @use_args(Item, location="json")
    @use_kwargs({"q": fields.Str()}, location="query", unknown=ma.RAISE)
    def exact(args, q):
        return {"q": q}

    @app.route("/headers")
    @use_args({"X-Key": fields.Str()}, location="headers")
    def headers(args):
        return {}

    @app.route("/plain")
    def plain():
        return request.args.get("q", "")

    @app.route("/factory", methods=["POST"])
    @use_args(lambda req: Item(context=req.json), location="json")
    def factory(args):
        return {}

    api = Api(app)
    blp = Blueprint("things", __name__)

    @blp.route("/things")
    class Things(MethodView):
        @blp.arguments(Loose)
        def post(self, data):
            return data

        @blp.arguments(Item, location="query", unknown=None)
        def get(self, args):
            return args

    api.register_blueprint(blp)
    return app


def test_live_flask_views_give_their_webargs_and_smorest_fields():
    routes = extract_flask_routes(build_app())
    search = routes["/search"]["declared_inputs"]
    assert search == {"inputs": {"q": "text", "page": "int"}, "covers": [], "reasons": [], "standalone": False}
    exact = routes["/exact"]["declared_inputs"]
    assert exact["inputs"] == {"Name": "text", "qty": "int", "price": "float", "on": "bool", "owner": "uuid", "tags": "int", "q": "text"}
    assert exact["standalone"] is True and "helper" in exact["covers"] and "raw_body" not in exact["covers"]
    assert routes["/headers"]["declared_inputs"] == {"inputs": {}, "covers": [], "reasons": [], "standalone": False}
    assert "declared_inputs" not in routes["/plain"]
    assert [reason["code"] for reason in routes["/factory"]["declared_inputs"]["reasons"]] == ["external_class"]
    things = routes["/things"]["declared_inputs"]
    assert things["inputs"]["Name"] == "text"
    assert [reason["code"] for reason in things["reasons"]] == ["container_whole"]
    assert things["standalone"] is False


def test_closed_arguments_complete_the_route():
    routes = {"/exact": {"inputs": {"x": "text"}, "inputs_complete": False,
                         "inputs_incomplete": [{"code": "helper"}, {"code": "raw_body"}],
                         "declared_inputs": webargs_inputs(build_app().view_functions["exact"])}}
    finalize(routes)
    assert routes["/exact"]["inputs_incomplete"] == [{"code": "raw_body"}]
    assert "q" in routes["/exact"]["inputs"]


def test_helpers_handle_edges():
    assert webargs_inputs(lambda: None) is None
    assert webargs_inputs(None) is None
    assert flask_inputs(type("V", (), {"view_class": type("C", (), {})})(), ["GET"]) is None
    first = {"inputs": {"a": "text"}, "covers": ["helper", "dependency"], "reasons": [], "standalone": True}
    second = {"inputs": {"b": "int"}, "covers": ["helper"], "reasons": [{"code": "x"}], "standalone": False}
    assert combine(None, first) is first and combine(first, None) is first
    assert combine(first, second) == {"inputs": {"a": "text", "b": "int"}, "covers": ["helper"], "reasons": [{"code": "x"}], "standalone": False}
    merged = merge_entries({"methods": ["GET"], "declared_inputs": first}, {"methods": ["POST"], "declared_inputs": second})
    assert merged["declared_inputs"]["inputs"] == {"a": "text", "b": "int"}

    def outer():
        return inner

    inner = None
    assert webargs_inputs(outer) is None  # An empty closure cell.


SOURCE = '''
from flask import Flask, request
from webargs import fields
from webargs.flaskparser import use_args, use_kwargs, parser
import marshmallow as ma
from flask_smorest import Blueprint
from flask.views import MethodView
app = Flask(__name__)
blp = Blueprint("b", __name__)

class Item(ma.Schema):
    class Meta:
        unknown = "raise"
    name = fields.Str(data_key="Name")
    qty = fields.Integer()
    secret = fields.Str(dump_only=True)
    tags = fields.List(fields.UUID())
    other = make_field()

class Loose(Item):
    class Meta:
        unknown = ma.INCLUDE

class Bad(Missing):
    pass

@app.route("/search")
@use_args({"q": fields.Str(), "page": fields.Int(), KEY: fields.Str(), "x": 1}, location="query")
def search(args):
    return helper(request.args)

@app.route("/exact", methods=["POST"])
@parser.use_args(Item(), location="json")
@use_kwargs({"q": fields.Str()}, location="query", unknown=ma.RAISE)
def exact(args, q):
    return helper(request.args)

@app.route("/open", methods=["POST"])
@use_args(Item(unknown=ma.INCLUDE))
@use_args(argmap=Item, location="query", unknown=None)
@use_args({"h": fields.Str()}, location="headers")
def open_(args, more, h):
    return args

@blp.route("/smorest")
class Things(MethodView):
    @blp.arguments(Loose)
    def post(self, data):
        return data

@app.route("/missing")
@use_args(Unknown(), location="query")
@use_args(Bad, location="json")
@use_args(Item, location=LOCATION)
def missing(args):
    return args

@app.route("/plain/<int:id>")
@arguments(Item)
def plain(id):
    return id

class Recursive(Recursive):
    pass

@app.route("/loop")
@use_args(Recursive)
def loop(args):
    return args
'''


def test_source_reads_webargs_and_smorest_decorators():
    found = {entry["function"]: entry for entry in source_inputs(SOURCE)}
    assert found["search"]["inputs"] == {"q": "text", "page": "int"}
    assert [reason["code"] for reason in found["search"]["reasons"]] == ["dynamic_key", "container_whole"]
    assert found["exact"]["complete"] is True
    assert found["exact"]["inputs"] == {"Name": "text", "qty": "int", "tags": "uuid", "other": "text", "q": "text"}
    assert [reason["code"] for reason in found["open_"]["reasons"]] == ["container_whole"]
    assert [reason["code"] for reason in found["Things.post"]["reasons"]] == ["container_whole"]
    assert sorted(reason["code"] for reason in found["missing"]["reasons"]) == ["external_class", "external_class"]
    assert found["missing"]["inputs"] == {"Name": "text", "qty": "int", "tags": "uuid", "other": "text"}
    assert found["plain"]["inputs"] == {"id": "text"}  # A bare arguments() isn't flask-smorest's.
    assert [reason["code"] for reason in found["loop"]["reasons"]] == ["external_class"]
