"""What init saves as each route's inputs: names, types and whether the list is complete."""

from aiwaf.core.input_schema import merge_into, source_inputs


def inputs(code):
    return {entry["function"]: (entry["inputs"], entry["complete"]) for entry in source_inputs(code)}


def test_flask_reads_types_and_aliases():
    found = inputs(
        "from flask import request\n"
        "import json\n"
        "def search():\n"
        "    return request.args.get('q'), request.args.get('page', 1, type=int), float(request.form['ratio'])\n"
        "def body():\n"
        "    data = request.get_json()\n"
        "    return data['name'], int(data.get('count')), 'x' in request.args\n"
        "def raw():\n"
        "    body = json.loads(request.data)\n"
        "    return body['id']\n")
    assert found["search"] == ({"q": "text", "page": "int", "ratio": "float"}, True)
    assert found["body"] == ({"name": "text", "count": "int"}, True)
    assert found["raw"] == ({"id": "text"}, True)


def test_a_name_read_as_two_types_is_text():
    found = inputs("def v(request):\n    return int(request.GET['n']), request.GET.get('n')\n")
    assert found["v"] == ({"n": "text"}, True)


def test_anything_not_understood_makes_the_list_incomplete():
    found = inputs(
        "def helper_gets_request(request):\n    helper(request)\n"
        "def whole(request):\n    return dict(request.GET)\n"
        "def dynamic(request, key):\n    return request.GET[key], request.POST.get(key)\n"
        "def raw(request):\n    return request.META, request.get_full_path()\n"
        "def method(request):\n    return request.GET.urlencode()\n"
        "def stream(request):\n    return request.body.decode()\n"
        "def safe(request):\n    messages.success(request, 'ok')\n    return render(request, 't.html')\n")
    for name in ("helper_gets_request", "whole", "dynamic", "raw", "method", "stream"):
        assert found[name][1] is False, name
    assert found["safe"] == ({}, True)


def test_django_forms_and_serializers_give_their_fields():
    found = inputs(
        "from django import forms\n"
        "class Comment(forms.Form):\n"
        "    body = forms.CharField()\n    post = forms.IntegerField()\n    token = forms.UUIDField()\n"
        "class Reply(Comment):\n    ok = forms.BooleanField()\n"
        "class Model(forms.ModelForm):\n    class Meta:\n        model = object\n        fields = '__all__'\n"
        "def comment(request):\n    return Reply(request.POST)\n"
        "def serialized(request):\n    return ReplySerializer(data=request.data)\n"
        "def model(request):\n    return Model(request.POST)\n")
    assert found["comment"] == ({"body": "text", "post": "int", "token": "uuid", "ok": "bool"}, True)
    assert found["serialized"][1] is False  # The serializer isn't in this module.
    assert found["model"][1] is False


def test_fastapi_parameters_and_models():
    found = inputs(
        "from fastapi import FastAPI, Query, Depends\n"
        "from pydantic import BaseModel\n"
        "from uuid import UUID\n"
        "from typing import Optional, List\n"
        "app = FastAPI()\n"
        "class Item(BaseModel):\n    name: str\n    price: float\n    tags: List[int] = []\n    owner: Optional[UUID] = None\n"
        "class Other(BaseModel):\n    note: str\n"
        "@app.get('/items/{item_id}')\n"
        "async def read(item_id: int, q: Optional[str] = Query(None, alias='query'), flag: bool = False):\n    return {}\n"
        "@app.post('/items')\nasync def create(item: Item):\n    return item\n"
        "@app.post('/both')\nasync def both(item: Item, other: Other):\n    return item\n"
        "@app.post('/dep')\nasync def dep(user=Depends(lambda: 1), q: str = ''):\n    return {}\n"
        "@app.post('/foreign')\nasync def foreign(body: SomethingElse):\n    return body\n"
        "@app.post('/raw')\nasync def raw(request: Request):\n    data = await request.json()\n    return data['a']\n")
    assert found["read"] == ({"item_id": "int", "query": "text", "flag": "bool"}, True)
    assert found["create"] == ({"name": "text", "price": "float", "tags": "int", "owner": "uuid"}, True)
    assert found["both"] == ({"item": "text", "other": "text"}, True)
    assert found["dep"][1] is False
    assert found["foreign"][1] is False
    assert found["raw"] == ({"a": "text"}, True)


def test_class_views_are_read_whole_and_only_plain_ones_are_complete():
    found = inputs(
        "class Search(View):\n"
        "    def get(self, request):\n        return self.page(), request.GET.get('q')\n"
        "    def page(self):\n        return int(self.request.GET['page'])\n"
        "class Listing(ListView):\n    def get_queryset(self):\n        return self.request.GET.get('q')\n")
    assert found["Search.get"] == ({"q": "text", "page": "int"}, True)
    assert found["Search.page"] == found["Search.get"]
    assert found["Listing.get_queryset"][1] is False  # ListView reads ?page= itself.


def test_merge_into_combines_views_of_one_route():
    routes = {"/x": {}}
    merge_into(routes, "/x", {"inputs": {"a": "int", "b": "text"}, "complete": True})
    merge_into(routes, "/x", {"inputs": {"a": "text"}, "complete": False})
    assert routes["/x"] == {"inputs": {"a": "text", "b": "text"}, "inputs_complete": False, "_flat": {"a", "b"}}


def test_the_request_is_followed_into_module_helpers_and_class_methods():
    found = inputs(
        "def _render(request, page):\n    return int(request.GET['n']), request.GET.get('q')\n"
        "def view(request):\n    return _render(request, 1)\n"
        "def by_keyword(request):\n    return _render(page=1, request=request)\n"
        "def deep(request):\n    return helper_elsewhere(request)\n"
        "def loop(request):\n    return loop(request)\n"
        "class Plain(View):\n"
        "    def dispatch(self, request, *args):\n        self.check(request)\n        return super().dispatch(request, *args)\n"
        "    def check(self, request):\n        return request.GET.get('token')\n"
        "class Generic(TemplateView):\n"
        "    def get(self, request):\n        return super().get(request)\n")
    assert found["view"] == ({"n": "int", "q": "text"}, True)
    assert found["by_keyword"] == ({"n": "int", "q": "text"}, True)
    assert found["deep"][1] is False
    assert found["loop"][1] is False
    assert found["Plain.dispatch"] == ({"token": "text"}, True)
    assert found["Generic.get"][1] is False


def test_keys_held_in_module_constants_are_known():
    found = inputs("PAGE = 'page'\ndef view(request):\n    return int(request.GET.get(PAGE)), request.GET[OTHER]\n"
                   "def known(request):\n    return request.POST[PAGE]\n")
    assert found["view"] == ({"page": "int"}, False)
    assert found["known"] == ({"page": "text"}, True)


def reasons(code):
    return {entry["function"]: [reason["code"] for reason in entry.get("reasons", [])] for entry in source_inputs(code)}


def test_each_gap_says_why():
    found = reasons(
        "from fastapi import Depends\n"
        "from django.views.generic import FormView\n"
        "def helper_gets_request(request):\n    helper(request)\n"
        "def stored(request):\n    keep = [request]\n"
        "def dynamic(request, key):\n    return request.GET[key], request.POST.get(key)\n"
        "def raw(request):\n    return request.get_full_path()\n"
        "def whole(request):\n    return dict(request.GET), request.GET.urlencode(), request.GET\n"
        "def called(request):\n    return request.get_json\n"
        "def model(request):\n    return SignupForm(request.POST)\n"
        "@app.get('/d')\ndef dep(user=Depends(current), item: Item = None):\n    return user\n"
        "class Edit(FormView):\n    def post(self, request):\n        return self.request.POST['x']\n")
    assert found["helper_gets_request"] == ["helper"]
    assert found["stored"] == ["helper"]
    assert found["dynamic"] == ["dynamic_key", "dynamic_key"]
    assert found["raw"] == ["raw_body"]
    assert found["whole"] == ["container_whole", "container_whole", "container_whole"]
    assert found["called"] == ["container_whole"]
    assert found["model"] == ["external_class"]
    assert found["dep"] == ["dependency", "external_class"]
    assert found["Edit.post"] == ["class_view"]
    entry = next(item for item in source_inputs("def v(request):\n    x = 1\n    return helper(request)\n") if item["function"] == "v")
    assert entry["reasons"] == [{"code": "helper", "detail": "the request is passed to helper, which isn't followed",
                                 "line": 3, "file": "<source>"}]


def test_forms_models_helpers_and_constants_are_followed_across_modules():
    from aiwaf.core.input_schema import project_inputs

    found = {entry["function"]: entry for entry in project_inputs({
        "shop.__init__": "from .forms import ReExported\n",
        "shop.forms": (
            "from django import forms\n"
            "from shop.base import Base\n"
            "class ReExported(forms.Form):\n    code = forms.UUIDField()\n"
            "class Address(Base):\n    street = forms.CharField()\n"
            "class Profile(forms.ModelForm):\n    class Meta:\n        model = User\n        fields = ['name']\n"
            "class Broken(Missing):\n    pass\n"
            "PAGE = 'page'\n"),
        "shop.base": "from django import forms\nclass Base(forms.Form):\n    number = forms.IntegerField()\n",
        "shop.helpers": "def paginate(request):\n    return int(request.GET['page'])\n",
        "shop.views": (
            "import shop.helpers\n"
            "import shop.forms as shop_forms\n"
            "from . import forms\n"
            "from .forms import Address, Profile, Broken, PAGE\n"
            "from shop import ReExported\n"
            "from ..elsewhere import Thing\n"
            "def address(request):\n    return Address(request.POST), request.GET.get(PAGE)\n"
            "def dotted(request):\n    return forms.Address(data=request.POST), shop_forms.ReExported(request.POST)\n"
            "def reexported(request):\n    return ReExported(request.POST)\n"
            "def helper(request):\n    return shop.helpers.paginate(request)\n"
            "def profile(request):\n    return Profile(request.POST)\n"
            "def broken(request):\n    return Broken(request.POST), Thing(request.POST)\n"),
    })}
    assert found["address"]["inputs"] == {"number": "int", "street": "text", "page": "text"}
    assert found["address"]["complete"] is True
    assert found["dotted"]["inputs"] == {"number": "int", "street": "text", "code": "uuid"}
    assert found["reexported"]["inputs"] == {"code": "uuid"}
    assert found["helper"]["inputs"] == {"page": "int"} and found["helper"]["complete"] is True
    profile = found["profile"]["reasons"][0]
    assert profile["code"] == "model_fields" and profile["name"] == "Profile" and profile["module"] == "shop.views"
    assert [reason["code"] for reason in found["broken"]["reasons"]] == ["external_class", "external_class"]


def test_ambiguous_and_missing_modules_are_not_guessed():
    from aiwaf.core.input_schema import Project

    project = Project()
    import ast as _ast

    one = project.add(_ast.parse("class A:\n    pass\n"), "a/forms.py", "a.forms")
    project.add(_ast.parse("class A:\n    pass\n"), "b/forms.py", "b.forms")
    assert project.get("forms") is None and project.get("a.forms") is one and project.get("") is None
    views = project.add(_ast.parse("import forms\nfrom forms import A\nimport a.forms\n"), "v.py", "v")
    assert views.lookup("class", "A") is None
    assert views.lookup("class", "a.forms.A")[1].name == "A"
    assert views.lookup("class", "a.missing.A") is None
    assert views.lookup("function", "") is None
    assert views.fields("Nope") == (None, ("external_class", "Nope"))
    looping = project.add(_ast.parse("class L(L):\n    pass\n"), "l.py", "l")
    assert looping.fields("L")[0] is None


def test_merge_into_keeps_each_functions_reasons():
    routes = {"/a": {}}
    merge_into(routes, "/a", {"function": "one", "inputs": {"q": "text"}, "complete": True})
    assert "inputs_incomplete" not in routes["/a"]
    reason = {"code": "helper", "detail": "x"}
    merge_into(routes, "/a", {"function": "two", "inputs": {}, "complete": False, "reasons": [reason, reason]})
    assert routes["/a"]["inputs_complete"] is False
    assert routes["/a"]["inputs_incomplete"] == [{"code": "helper", "detail": "x", "function": "two"}]


def test_helpers_given_a_container_or_defaults_and_constant_key_loops():
    from aiwaf.core.input_schema import project_inputs

    found = {entry["function"]: entry for entry in project_inputs({
        "app": (
            "from flask import request\n"
            "from helpers import parse_filters, dump\n"
            "def search():\n"
            "    return parse_filters(request.args), request.args.get('q')\n"
            "def dumped():\n"
            "    return dump(request.form)\n"
            "def unknown_helper():\n"
            "    return missing(request.args)\n"
            "def settings():\n"
            "    payload = request.get_json(silent=True) or {}\n"
            "    return {key: payload.get(key) for key in ('theme', 'lang')}\n"
            "def loops():\n"
            "    for name in ['a', 'b']:\n"
            "        request.form[name]\n"
            "    return [request.args[k] for k in keys], [request.args.get(n) for n in ('x', other)]\n"
            "def upload():\n"
            "    return request.files['file'], request.files.get('other')\n"
        ),
        "helpers": (
            "def parse_filters(args):\n"
            "    return {'sort': args.get('sort'), 'order': args['order']}\n"
            "def dump(form):\n"
            "    return dict(form)\n"
        ),
    })}
    assert found["search"]["inputs"] == {"sort": "text", "order": "text", "q": "text"} and found["search"]["complete"]
    assert [reason["code"] for reason in found["dumped"]["reasons"]] == ["container_whole"]
    assert found["dumped"]["reasons"][0]["file"] == "helpers.py"
    assert [reason["code"] for reason in found["unknown_helper"]["reasons"]] == ["container_whole"]
    assert found["settings"] == {**found["settings"], "inputs": {"theme": "text", "lang": "text"}, "complete": True}
    assert found["loops"]["inputs"] == {"a": "text", "b": "text"}
    assert sorted({reason["code"] for reason in found["loops"]["reasons"]}) == ["dynamic_key"]
    assert found["upload"]["inputs"] == {"file": "text", "other": "text"} and found["upload"]["complete"]


def test_views_that_read_nothing_and_dict_bodies():
    found = {entry["function"]: entry for entry in source_inputs(
        "from fastapi import Body\n"
        "def detail(request, pk):\n"
        "    return pk\n"
        "@app.post('/events')\n"
        "def events(body: dict = Body(...)):\n"
        "    return body\n"
        "@app.post('/typed')\n"
        "def typed(body: Dict[str, Any]):\n"
        "    return body\n"
        "def helper(value):\n"
        "    return value\n"
    )}
    assert found["detail"]["inputs"] == {} and found["detail"]["complete"]
    assert [reason["code"] for reason in found["events"]["reasons"]] == ["container_whole"]
    assert [reason["code"] for reason in found["typed"]["reasons"]] == ["container_whole"]
    assert "helper" not in found


def test_a_read_assigned_to_a_variable_takes_the_type_of_its_later_casts():
    found = inputs(
        "def later(request):\n    page = request.GET.get('page', '1')\n    if not page:\n        page = '1'\n    return int(page)\n"
        "def recast(request):\n    size = request.GET.get('size')\n    try:\n        size = int(size)\n"
        "    except (TypeError, ValueError):\n        size = 10\n    return size\n"
        "def top(request):\n    n = request.GET['n']\n    n = float(n)\n    return n * 2\n"
        "def none(request):\n    n: str = request.GET.get('n')\n    if n is None or n == '':\n        return 0\n    return float(n)\n"
        "def under_if(request):\n    n = request.GET['n']\n    if flag:\n        n = int(n)\n    return n\n"
        "def before(request):\n    n = request.GET['n']\n    log(n)\n    n = int(n)\n    return n\n"
        "def two(request):\n    n = request.POST['n']\n    return int(n), float(n)\n"
        "def other(request):\n    n = request.GET['n']\n    n = str(n)\n    return n\n"
    )
    assert found["later"][0] == {"page": "int"}
    assert found["recast"][0] == {"size": "int"}
    assert found["top"][0] == {"n": "float"}
    assert found["none"][0] == {"n": "float"}
    for name in ("under_if", "before", "two", "other"):
        assert found[name][0] == {"n": "text"}, name


def test_pydantic_models_read_from_source_use_their_request_keys():
    head = ("from fastapi import FastAPI\nfrom pydantic import BaseModel, Field, ConfigDict, AliasChoices\n"
            "app = FastAPI()\n")
    tail = "@app.post('/x')\ndef v(d: D):\n    return d\n"
    alias = inputs(head + "class D(BaseModel):\n    event_id: str = Field(alias='eventId')\n    ok: bool\n" + tail)
    assert alias["v"] == ({"eventId": "text", "ok": "bool"}, True)
    by_name = inputs(head + "class D(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n"
                     "    event_id: str = Field(alias='eventId')\n" + tail)
    assert by_name["v"][0] == {"eventId": "text", "event_id": "text"}
    choices = inputs(head + "class D(BaseModel):\n    ref: int = Field(validation_alias=AliasChoices('ref', 'reference'))\n" + tail)
    assert choices["v"][0] == {"ref": "int", "reference": "int"}
    for model in ("class D(BaseModel):\n    model_config = ConfigDict(alias_generator=str.upper)\n    event_id: str\n",
                  "NAME = 'x'\nclass D(BaseModel):\n    event_id: str = Field(alias=NAME)\n",
                  "class D(BaseModel):\n    class Config:\n        alias_generator = str.upper\n    event_id: str\n"):
        assert inputs(head + model + tail)["v"] == ({}, False)


def test_fastapi_parameter_aliases_in_defaults_and_annotated_metadata():
    found = inputs(
        "from typing import Annotated\nfrom fastapi import Body, FastAPI, Query\napp = FastAPI()\n"
        "@app.post('/a')\ndef a(body: Annotated[str, Body(embed=True, alias='limit')], n: Annotated[int, Query(alias='count')] = 1):\n"
        "    return {}\n"
        "@app.post('/b')\ndef b(note: str = Body(embed=True, alias='memo'), plain: Annotated[int, 'doc'] = 1):\n    return {}\n")
    assert found["a"][0] == {"limit": "text", "count": "int"}
    assert found["b"][0] == {"memo": "text", "plain": "int"}


def test_truthiness_casts_are_not_boolean_restrictions():
    found = {entry["function"]: entry["inputs"] for entry in source_inputs(
        "from flask import request\n"
        "def a():\n    return bool(request.args.get('flag'))\n"
        "def b():\n    return request.args.get('flag', type=bool)\n"
        "def c():\n    return request.args.get('flag') == 'true'\n")}
    assert found == {"a": {"flag": "text"}, "b": {"flag": "text"}, "c": {"flag": "text"}}


def test_reads_keep_their_location_and_class_handlers_their_own_reads():
    found = {entry["function"]: entry for entry in source_inputs(
        "import json\nfrom django.views import View\n"
        "class Items(View):\n"
        "    def get(self, request):\n        return int(request.GET.get('page', '1'))\n"
        "    def post(self, request):\n        return json.loads(request.body)['page'], self.request.POST.get('title')\n"
        "    def audit(self):\n        return self.request.GET.get('trace')\n")}
    found.update({entry["function"]: entry for entry in source_inputs(
        "from flask import request\n"
        "def search():\n"
        "    return int(request.args['limit']), request.form.get('limit'), request.values.get('q')\n")})
    # Each handler has its own reads, with the class's other methods' (audit) beside them.
    assert found["Items.get"]["inputs"] == {"page": "int", "trace": "text"}
    assert found["Items.post"]["inputs"] == {"page": "text", "title": "text", "trace": "text"}
    assert found["Items.post"]["locations"] == {"body": {"page": "text", "title": "text"}, "query": {"trace": "text"}}
    # A helper method is read with the whole class, as before.
    assert found["Items.audit"]["inputs"] == {"page": "text", "title": "text", "trace": "text"}
    assert found["search"]["locations"] == {"query": {"limit": "int"}, "body": {"limit": "text"}}
    assert found["search"]["unlocated"] == ["q"]


def test_looping_over_a_request_list_gives_its_elements_keys():
    found = {entry["function"]: entry.get("nested") for entry in source_inputs(
        "import json\nfrom flask import request\n"
        "def looped():\n    for line in request.json['lines']:\n        save(line['sku'], int(line.get('qty')))\n"
        "def comprehension():\n    return [l['sku'] for l in request.json['lines'] if l.get('active')]\n"
        "def held():\n    lines = request.json['lines']\n    return [l['sku'] for l in lines]\n"
        "def parsed():\n    data = json.loads(request.data)\n    return {l['sku']: l['qty'] for l in data['lines']}\n"
        "def whole():\n    for line in request.json['lines']:\n        save(line)\n"
        "def also_counted():\n    lines = request.json['lines']\n    total = len(lines)\n    return [l['sku'] for l in lines]\n")}
    assert found["looped"] == {"lines[]": ["qty", "sku"]}
    assert found["comprehension"] == {"lines[]": ["active", "sku"]}
    assert found["held"] == {"lines[]": ["sku"]}
    assert found["parsed"] == {"lines[]": ["qty", "sku"]}
    assert found["whole"] is None and found["also_counted"] is None


def test_marshmallow_nested_fields_give_their_keys():
    import marshmallow

    from aiwaf.core.input_sources import serializer_nested

    class Line(marshmallow.Schema):
        sku = marshmallow.fields.Str()
        qty = marshmallow.fields.Int()
        id = marshmallow.fields.Int(dump_only=True)

    class Order(marshmallow.Schema):
        lines = marshmallow.fields.List(marshmallow.fields.Nested(Line))
        more = marshmallow.fields.Nested(Line, many=True)
        one = marshmallow.fields.Nested(Line)
        tags = marshmallow.fields.List(marshmallow.fields.Str())

    assert serializer_nested(Order) == {"lines[]": ["qty", "sku"], "more[]": ["qty", "sku"], "one": ["qty", "sku"]}


def test_a_function_view_that_branches_on_request_method_marks_each_read():
    found = {entry["function"]: entry.get("method_reads") for entry in source_inputs(
        "import json\n"
        "def items(request):\n"
        "    trace = request.GET.get('trace')\n"
        "    if request.method == 'POST':\n"
        "        return json.loads(request.body)['title']\n"
        "    elif request.method in ('PUT', 'PATCH'):\n"
        "        return request.POST['note']\n"
        "    return int(request.GET.get('page', '1'))\n"
        "def guarded(request):\n"
        "    if request.method != 'POST':\n"
        "        return None\n"
        "    return request.POST['title'] if not request.method == 'GET' else None\n"
        "def either(request):\n"
        "    if request.method == 'GET' or request.method == 'HEAD':\n"
        "        return request.GET['q']\n"
        "    if request.method != 'PUT' and request.method != 'PATCH':\n"
        "        return request.POST['x']\n"
        "def plain(request):\n"
        "    return request.GET.get('q')\n"
        "def unknown(request):\n"
        "    if request.method == METHOD:\n"
        "        return request.GET.get('q')\n")}
    assert found["items"] == [
        {"name": "trace", "kind": "text", "where": "query"},
        {"name": "title", "kind": "text", "where": "body", "only": ["POST"]},
        {"name": "note", "kind": "text", "where": "body", "only": ["PATCH", "PUT"]},
        {"name": "page", "kind": "int", "where": "query", "except": ["PATCH", "POST", "PUT"]},
    ]
    assert found["guarded"] == [{"name": "title", "kind": "text", "where": "body", "only": ["POST"]}]
    assert found["either"] == [
        {"name": "q", "kind": "text", "where": "query", "only": ["GET", "HEAD"]},
        # After the GET/HEAD guard, and inside the PUT/PATCH exclusion: none of the four.
        {"name": "x", "kind": "text", "where": "body", "except": ["GET", "HEAD", "PATCH", "PUT"]},
    ]
    assert found["plain"] is None and found["unknown"] is None
