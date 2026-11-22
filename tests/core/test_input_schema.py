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
    assert routes["/x"] == {"inputs": {"a": "text", "b": "text"}, "inputs_complete": False}


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
