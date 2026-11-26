"""Inputs from the framework, declarations, OpenAPI and recordings, combined into each route."""

import dataclasses
import json
import sys
import types
import typing
import uuid
from decimal import Decimal

import pytest

import aiwaf
from aiwaf.core import input_sources
from aiwaf.core.input_sources import (
    canonical, class_fields, declared_on, describe, fastapi_inputs, finalize, inputs, load_declarations, load_observations,
    load_spec, openapi_inputs, resolve_class, sources_from_environment,
)


def test_kinds_from_types_names_and_field_classes():
    kind = input_sources._kind
    assert [kind(int), kind(bool), kind(float), kind(Decimal), kind(uuid.UUID), kind(str)] == ["int", "bool", "float", "float", "uuid", "text"]
    assert [kind("IntegerField"), kind("BooleanField"), kind("DecimalField"), kind("UUIDField"), kind("CharField")] == [
        "int", "bool", "float", "uuid", "text"]
    assert [kind(typing.Optional[int]), kind(typing.List[uuid.UUID]), kind(typing.Annotated[float, "x"])] == ["int", "uuid", "float"]
    assert kind(typing.Union[int, str]) == "text" and kind(int | None) == "int" and kind(list) == "text" and kind(None) == "text"


def test_inputs_declares_a_view_and_its_class():
    @aiwaf.inputs(q=str, page=int, id=uuid.UUID)
    def search(request):
        return None

    assert declared_on(search) == {"inputs": {"q": "text", "page": "int", "id": "uuid"}, "complete": True}

    class View:
        pass

    def as_view(request):
        return None

    as_view.view_class = View
    inputs(complete=False, name="text")(as_view)
    assert declared_on(as_view) == {"inputs": {"name": "text"}, "complete": False}
    assert declared_on(View)["complete"] is False

    def wrapper(request):
        return None

    wrapper.__wrapped__ = search
    assert declared_on(wrapper)["inputs"]["page"] == "int"
    assert declared_on(lambda request: None) is None
    assert inputs(q=str)(len) is len  # A builtin can't hold the declaration; it's returned as it is.
    with pytest.raises(AttributeError):
        aiwaf.not_there  # noqa: B018


def test_class_fields_reads_pydantic_dataclasses_and_typed_dicts():
    from pydantic import BaseModel, ConfigDict, Field

    class Item(BaseModel):
        name: str
        count: int = Field(alias="n")

    class Loose(BaseModel):
        model_config = ConfigDict(extra="allow")
        name: str

    @dataclasses.dataclass
    class Point:
        x: float
        ok: bool

    class Movie(typing.TypedDict):
        title: str
        year: int

    assert class_fields(Item) == {"name": "text", "n": "int"}
    assert class_fields(Loose) is None
    assert class_fields(Point) == {"x": "float", "ok": "bool"}
    assert class_fields(Movie) == {"title": "text", "year": "int"}
    assert class_fields(object) is None and class_fields("not a class") is None


def test_class_fields_reads_marshmallow_and_wtforms_shapes():
    integer = type("Integer", (), {})
    schema = type("Schema", (), {"_declared_fields": {"age": integer(), "name": types.SimpleNamespace(data_key="full_name")}})
    open_schema = type("Open", (), {"_declared_fields": {}, "Meta": type("Meta", (), {"unknown": "include"})})
    assert class_fields(schema) == {"age": "int", "full_name": "text"}
    assert class_fields(open_schema) is None

    unbound = type("UnboundField", (), {"_formfield": True})
    def field(kind):
        made = unbound()
        made.field_class = type(kind, (), {})
        return made

    base = type("FlaskForm", (), {"remember": field("BooleanField")})
    form = type("Login", (base,), {"email": field("StringField")})
    assert class_fields(form) == {"email": "text", "remember": "bool", "csrf_token": "text"}


def test_resolve_class_only_uses_loaded_modules():
    module = types.ModuleType("aiwaf_test_app.signup_forms")
    module.Inner = types.SimpleNamespace(Form=dict)
    module.value = 1
    sys.modules["aiwaf_test_app.signup_forms"] = module
    try:
        assert resolve_class("aiwaf_test_app.signup_forms", "Inner.Form") is dict
        assert resolve_class("signup_forms", "Inner.Form") is dict  # By the end of its name.
        assert resolve_class("signup_forms", "value") is None and resolve_class("signup_forms", "Missing") is None
        assert resolve_class("never.imported", "X") is None and resolve_class("", "X") is None
    finally:
        del sys.modules["aiwaf_test_app.signup_forms"]


def test_fastapi_inputs_come_from_resolved_parameters_and_dependencies():
    from fastapi import Body, Depends, FastAPI, Query, Request
    from pydantic import BaseModel, ConfigDict

    class Item(BaseModel):
        name: str
        price: float

    class Loose(BaseModel):
        model_config = ConfigDict(extra="allow")
        name: str

    def paging(page: int = 1, size: int = Query(10, alias="per_page")):
        return page

    def raw(request: Request):
        return request

    app = FastAPI()

    @app.post("/items/{item_id}")
    def create(item_id: int, item: Item, q: typing.Optional[str] = None, paged=Depends(paging)):
        return None

    @app.post("/two")
    def two(a: Item, b: int = Body(...)):
        return None

    @app.post("/loose")
    def loose(item: Loose, other=Depends(raw)):
        return None

    @app.get("/raw")
    def reads_request(request: Request):
        return None

    routes = {route.path: route for route in app.routes}
    created = fastapi_inputs(routes["/items/{item_id}"])
    assert created["inputs"] == {"q": "text", "page": "int", "per_page": "int", "name": "text", "price": "float"}
    assert created["reasons"] == [] and created["standalone"] is True
    assert "dependency" in created["covers"]
    assert fastapi_inputs(routes["/two"])["inputs"] == {"a": "text", "b": "int"}
    loose_found = fastapi_inputs(routes["/loose"])
    assert [reason["code"] for reason in loose_found["reasons"]] == ["model_fields", "dependency"]
    assert fastapi_inputs(routes["/raw"])["standalone"] is False
    assert fastapi_inputs(types.SimpleNamespace()) is None


def test_canonical_paths_match_across_frameworks():
    assert canonical("posts/<int:id>/") == canonical("/posts/{id}") == canonical("/posts/:id") == "/posts/{}"
    assert canonical("/") == "/" and canonical("/a/b?x=1") == "/a/b" and canonical("/files/*rest") == "/files/{}"


SPEC = {
    "openapi": "3.0.3",
    "paths": {
        "/items/{id}": {
            "parameters": [{"$ref": "#/components/parameters/Verbose"}],
            "get": {"parameters": [
                {"name": "id", "in": "path", "schema": {"type": "integer"}},
                {"name": "page", "in": "query", "schema": {"type": "integer"}},
                {"name": "tags", "in": "query", "schema": {"type": "array", "items": {"type": "string", "format": "uuid"}}},
            ]},
            "put": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Item"}}}}},
            "summary": "not an operation",
        },
        "/open": {"post": {
            "parameters": [{"name": "filter", "in": "query", "schema": {"type": "object"}}, {"$ref": "#/missing"}],
            "requestBody": {"content": {
                "application/json": {"schema": {"type": "object", "additionalProperties": True,
                                                 "properties": {"a": {"type": ["number", "null"]}}}},
                "text/plain": {"schema": {"type": "string"}},
            }},
        }},
        "/union": {"post": {"requestBody": {"content": {"application/x-www-form-urlencoded": {"schema": {
            "oneOf": [{"type": "object", "properties": {"x": {"type": "boolean"}}}, {"$ref": "#/components/schemas/Item"}],
        }}, "application/merge-patch+json": {"schema": {"type": "array"}}}}}},
        "/broken": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "other.json#/X"}}}}}},
    },
    "components": {
        "parameters": {"Verbose": {"name": "verbose", "in": "query", "schema": {"type": "boolean"}}},
        "schemas": {"Item": {"allOf": [{"type": "object", "properties": {"name": {"type": "string"}}},
                                       {"properties": {"qty": {"type": "integer"}}}]}},
    },
}


def test_openapi_operations_list_query_and_body_inputs():
    found = openapi_inputs(SPEC)
    assert found["/items/{}"]["GET"] == {"inputs": {"verbose": "bool", "page": "int", "tags": "uuid"}, "complete": True, "reasons": []}
    assert found["/items/{}"]["PUT"]["inputs"] == {"verbose": "bool", "name": "text", "qty": "int"}
    assert found["/items/{}"]["PUT"]["complete"] is True
    opened = found["/open"]["POST"]
    assert opened["inputs"] == {"a": "float"} and opened["complete"] is False
    assert [reason["code"] for reason in opened["reasons"]] == ["container_whole", "external_class", "container_whole", "raw_body"]
    union = found["/union"]["POST"]
    assert union["inputs"] == {"x": "bool", "name": "text", "qty": "int"}
    assert [reason["code"] for reason in union["reasons"]] == ["container_whole"]
    assert found["/broken"]["POST"]["reasons"][0]["code"] == "external_class"
    assert input_sources._openapi_kind({"type": "number"}) == "float" and input_sources._openapi_kind({"type": ["string", "integer"]}) == "text"


def test_load_spec_reads_json_and_yaml_and_refuses_others(tmp_path):
    spec = tmp_path / "openapi.json"
    spec.write_text(json.dumps(SPEC), encoding="utf-8")
    assert load_spec(str(spec))["paths"]
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no paths"):
        load_spec(str(bad))
    yaml_file = tmp_path / "openapi.yaml"
    yaml_file.write_text("paths: {}\n", encoding="utf-8")
    try:
        import yaml  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match="PyYAML"):
            load_spec(str(yaml_file))
    else:
        assert load_spec(str(yaml_file)) == {"paths": {}}


def test_observations_and_declarations_files(tmp_path):
    seen = tmp_path / "seen.jsonl"
    seen.write_text("\n".join([
        json.dumps({"route": "/search/", "method": "GET", "key": "q"}),
        json.dumps({"route": "/search", "method": "GET", "key": "*"}),
        json.dumps({"route": "/posts/<int:id>", "key": "page", "type": "int"}),
        "not json", json.dumps(["no"]), json.dumps({"route": "/x"}),
    ]), encoding="utf-8")
    assert load_observations(str(seen)) == {"/search": {"q": "text", "*": "*"}, "/posts/{}": {"page": "int"}}
    declared = tmp_path / "inputs.json"
    declared.write_text(json.dumps({"routes": {"/a/{id}": {"inputs": {"n": "int"}}, "/b": {"inputs": {}, "complete": False},
                                               "/c": "skipped"}}), encoding="utf-8")
    assert load_declarations(str(declared)) == {"/a/{}": {"inputs": {"n": "int"}, "complete": True},
                                                "/b": {"inputs": {}, "complete": False}}


def test_finalize_combines_every_source():
    routes = {
        "/read": {"inputs": {"q": "text"}, "inputs_complete": True},
        "/framework": {"declared_inputs": {"inputs": {"title": "text"}, "covers": ["class_view"], "reasons": [], "standalone": True}},
        "/covered": {"inputs": {}, "inputs_complete": False, "inputs_incomplete": [{"code": "class_view", "detail": "x"}],
                     "declared_inputs": {"inputs": {}, "covers": ["class_view"], "reasons": [], "standalone": False}},
        "/decorated": {"inputs": {"q": "text"}, "inputs_complete": False, "inputs_incomplete": [{"code": "helper", "detail": "x"}],
                       "input_declaration": {"inputs": {"q": "int"}, "complete": True}},
        "/filed/<int:id>": {"methods": ["GET"]},
        "/spec": {"methods": ["GET", "POST"], "inputs": {}, "inputs_complete": False,
                  "inputs_incomplete": [{"code": "helper", "detail": "x"}]},
        "/partial": {"methods": ["GET", "POST"]},
        "/seen": {"inputs": {"q": "text"}, "inputs_complete": True},
        "/whole": {"inputs": {}, "inputs_complete": True},
        "/only-seen": {},
        "/untouched": {"view": "x"},
        "/odd": "not a dict",
    }
    counts = finalize(
        routes,
        declarations={"/filed/{}": {"inputs": {"n": "int"}, "complete": True}},
        openapi={"/spec": {"GET": {"inputs": {"a": "int"}, "complete": True, "reasons": []},
                           "POST": {"inputs": {"b": "text"}, "complete": True, "reasons": []}},
                 "/partial": {"GET": {"inputs": {"a": "int"}, "complete": True, "reasons": []}}},
        observations={"/seen": {"q": "text", "extra": "text"}, "/whole": {"*": "*"}, "/only-seen": {"k": "text"}},
    )
    assert routes["/read"]["inputs_complete"] is True
    assert routes["/framework"] == {"inputs": {"title": "text"}, "inputs_complete": True}
    assert routes["/covered"]["inputs_complete"] is True and "inputs_incomplete" not in routes["/covered"]
    assert routes["/decorated"]["inputs"] == {"q": "int"} and routes["/decorated"]["inputs_complete"] is True
    assert routes["/filed/<int:id>"]["inputs"] == {"n": "int"} and routes["/filed/<int:id>"]["inputs_complete"] is True
    assert routes["/spec"]["inputs"] == {"a": "int", "b": "text"} and routes["/spec"]["inputs_complete"] is True
    # A spec that lists only some of a route's methods adds names but doesn't vouch for the rest.
    assert routes["/partial"]["inputs"] == {"a": "int"} and routes["/partial"]["inputs_complete"] is True
    assert routes["/seen"]["inputs"] == {"q": "text", "extra": "text"} and routes["/seen"]["inputs_complete"] is False
    assert routes["/seen"]["inputs_incomplete"][0]["code"] == "observed"
    assert routes["/whole"]["inputs_incomplete"][0]["code"] == "container_whole"
    assert routes["/only-seen"]["inputs"] == {"k": "text"} and routes["/only-seen"]["inputs_complete"] is True
    assert "inputs" not in routes["/untouched"]
    assert counts["declared"] == 2 and counts["openapi"] == 2 and counts["observed"] == 3
    assert counts["reasons"] == {"observed": 1, "container_whole": 1}
    assert describe(counts) == f"complete inputs for {counts['complete']} of {counts['complete'] + 2} routes; incomplete because of container_whole (1), observed (1)"
    assert describe({}) == ""


def test_finalize_reads_classes_the_app_has_loaded():
    from pydantic import BaseModel

    class Address(BaseModel):
        street: str
        number: int

    module = types.ModuleType("aiwaf_test_shop.views")
    module.Address = Address
    sys.modules["aiwaf_test_shop.views"] = module
    try:
        routes = {
            "/a": {"view": "aiwaf_test_shop.views.create", "inputs": {}, "inputs_complete": False, "inputs_incomplete": [
                {"code": "model_fields", "detail": "x", "module": "shop.views", "name": "Address"}]},
            "/b": {"view": "aiwaf_test_shop.views.create", "inputs": {}, "inputs_complete": False, "inputs_incomplete": [
                {"code": "external_class", "detail": "x", "module": "elsewhere", "name": "Missing"}]},
        }
        finalize(routes)
    finally:
        del sys.modules["aiwaf_test_shop.views"]
    assert routes["/a"] == {"view": "aiwaf_test_shop.views.create", "inputs": {"street": "text", "number": "int"}, "inputs_complete": True}
    assert routes["/b"]["inputs_complete"] is False


def test_sources_from_environment(tmp_path, monkeypatch):
    for name in ("AIWAF_INPUTS_FILE", "AIWAF_OPENAPI_SPEC", "AIWAF_OBSERVED_INPUTS"):
        monkeypatch.delenv(name, raising=False)
    assert sources_from_environment(str(tmp_path)) == {}
    (tmp_path / ".aiwaf").mkdir()
    (tmp_path / ".aiwaf" / "inputs.json").write_text(json.dumps({"/a": {"inputs": {"q": "text"}}}), encoding="utf-8")
    (tmp_path / ".aiwaf" / "observed_inputs.jsonl").write_text(json.dumps({"route": "/a", "key": "q"}), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC), encoding="utf-8")
    monkeypatch.setenv("AIWAF_OPENAPI_SPEC", str(spec))
    found = sources_from_environment(str(tmp_path))
    assert set(found) == {"declarations", "openapi", "observations"}
    assert found["declarations"]["/a"]["inputs"] == {"q": "text"}


def test_fastapi_dict_bodies_are_open():
    from fastapi import Body, FastAPI

    app = FastAPI()

    @app.post("/events")
    def events(body: dict = Body(...)):
        return body

    @app.post("/any")
    def anything(body: typing.Any = Body(...)):
        return body

    @app.post("/mapping")
    def mapping(body: typing.Mapping[str, int] = Body(...)):
        return body

    for path in ("/events", "/any", "/mapping"):
        route = next(item for item in app.routes if getattr(item, "path", None) == path)
        found = input_sources.fastapi_inputs(route)
        assert found["inputs"] == {}, path
        assert [reason["code"] for reason in found["reasons"]] == ["container_whole"], path


def test_closure_schemas_of_a_views_own_decorators():
    import functools

    import marshmallow as ma
    from pydantic import BaseModel

    class Body(BaseModel):
        name: str
        qty: int

    class Query(ma.Schema):
        page = ma.fields.Int()

    @dataclasses.dataclass
    class Settings:
        debug: bool = False

    def validate(schema, where="json", settings=Settings()):
        def decorate(view):
            @functools.wraps(view)
            def wrapper(*args, **kwargs):
                return view(*args, **kwargs) if schema and settings else None
            return wrapper
        return decorate

    @validate(Body)
    @validate(Query(), "query")
    def view():
        return None

    assert input_sources.closure_inputs(view)["inputs"] == {"name": "text", "qty": "int", "page": "int"}
    assert input_sources.closure_inputs(validate(Settings)(lambda: None)) is None
    assert input_sources.closure_inputs(lambda: None) is None

    class Items:
        post = view

    assert input_sources.view_arguments(type("V", (), {"view_class": Items})(), ["POST", "GET"])["inputs"]["qty"] == "int"


def test_fastapi_bodies_list_the_keys_inside_nested_models():
    from fastapi import FastAPI
    from pydantic import BaseModel, ConfigDict

    class Customer(BaseModel):
        email: str
        name: typing.Optional[str] = None

    class Loose(BaseModel):
        model_config = ConfigDict(extra="allow")
        anything: str

    class Order(BaseModel):
        sku: str
        customer: Customer
        backup: typing.Optional[Customer] = None
        extra: Loose
        tags: typing.List[str] = []

    app = FastAPI()

    @app.post("/orders")
    def create(order: Order):
        return {"email": order.customer.email}

    route = next(item for item in app.routes if getattr(item, "path", "") == "/orders")
    found = fastapi_inputs(route)
    assert found["nested"] == {"backup": ["email", "name"], "customer": ["email", "name"]}
    assert set(found["inputs"]) == {"sku", "customer", "backup", "extra", "tags"}
    assert input_sources.nested_fields(str) == {} and input_sources.nested_fields(Customer) == {}
    assert input_sources._object_class(typing.Union[int, str]) is None


def test_finalize_keeps_a_frameworks_nested_keys_over_the_codes_whole_reading():
    framework = {"inputs": {"customer": "text", "sku": "text"}, "covers": [], "reasons": [], "standalone": True,
                 "nested": {"customer": ["email"]}}
    routes = {"/orders": {"methods": ["POST"], "inputs": {"customer": "text"}, "inputs_complete": True,
                          "_flat": {"customer"}, "declared_inputs": framework}}
    finalize(routes)
    assert routes["/orders"]["inputs_nested"] == {"customer": ["email"]}


def test_combining_handlers_keeps_nested_keys_only_where_neither_takes_the_field_whole():
    from aiwaf.core.input_sources import combine

    nested = {"inputs": {"customer": "text"}, "covers": ["a"], "reasons": [], "standalone": True, "nested": {"customer": ["email"]}}
    other = {"inputs": {"note": "text"}, "covers": ["a"], "reasons": [], "standalone": True, "nested": {"note": ["body"]}}
    whole = {"inputs": {"customer": "text"}, "covers": ["a"], "reasons": [], "standalone": True}
    assert combine(nested, other)["nested"] == {"customer": ["email"], "note": ["body"]}
    assert "nested" not in combine(nested, whole) and "nested" not in combine(whole, nested)
