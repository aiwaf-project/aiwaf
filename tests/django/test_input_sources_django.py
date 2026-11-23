"""Inputs Django and Django REST framework declare, and recording what Django views read."""

import json
import os
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView
from django.http import HttpResponse
from django.test import RequestFactory, SimpleTestCase, override_settings
from django.urls import path
from django.views.generic import CreateView, FormView, ListView, RedirectView, TemplateView
from rest_framework import filters, generics, serializers
from rest_framework.pagination import PageNumberPagination

from aiwaf.core import input_recorder
from aiwaf.core.input_sources import class_fields, django_inputs


class ContactForm(forms.Form):
    name = forms.CharField()
    age = forms.IntegerField()
    subscribe = forms.BooleanField(required=False)


class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "email"]


class NoteSerializer(serializers.Serializer):
    title = serializers.CharField()
    rating = serializers.IntegerField()
    created = serializers.DateTimeField(read_only=True)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username"]


class NeedsArguments(serializers.Serializer):
    def __init__(self, required):
        super().__init__()


class Contact(FormView):
    form_class = ContactForm
    template_name = "x.html"


class Signup(CreateView):
    model = User
    fields = ["username", "first_name"]


class Listing(ListView):
    model = User
    paginate_by = 10


class Away(RedirectView):
    url = "/elsewhere/"
    query_string = True


class Pagination(PageNumberPagination):
    page_size_query_param = "size"


class Notes(generics.ListCreateAPIView):
    serializer_class = NoteSerializer
    pagination_class = Pagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]


class Chooses(generics.CreateAPIView):
    def get_serializer_class(self):
        return NoteSerializer


class Mixin:
    pass


class Mixed(Mixin, generics.CreateAPIView):
    serializer_class = NoteSerializer


class Unknown(generics.ListAPIView):
    serializer_class = NoteSerializer
    filter_backends = [type("CustomFilter", (), {})]


class ClassFieldsTests(SimpleTestCase):
    def test_django_forms_and_model_forms(self):
        self.assertEqual(class_fields(ContactForm), {"name": "text", "age": "int", "subscribe": "bool"})
        self.assertEqual(class_fields(UserForm), {"username": "text", "email": "text"})

    def test_drf_serializers_without_read_only_fields(self):
        self.assertEqual(class_fields(NoteSerializer), {"title": "text", "rating": "int"})
        self.assertEqual(class_fields(UserSerializer), {"username": "text"})
        self.assertIsNone(class_fields(NeedsArguments))


class DjangoInputsTests(SimpleTestCase):
    def test_function_views_are_left_to_code_reading(self):
        self.assertIsNone(django_inputs(lambda request: None, ["GET"]))

    def test_generic_views_list_their_form_and_query_inputs(self):
        contact = django_inputs(Contact.as_view(), ["GET", "POST"])
        self.assertEqual(contact["inputs"], {"name": "text", "age": "int", "subscribe": "bool"})
        self.assertTrue(contact["standalone"])
        self.assertIn("class_view", contact["covers"])
        self.assertEqual(django_inputs(Signup.as_view(), ["POST"])["inputs"], {"username": "text", "first_name": "text"})
        self.assertEqual(django_inputs(Listing.as_view(), ["GET"])["inputs"], {"page": "text"})
        self.assertEqual(django_inputs(TemplateView.as_view(), ["GET"])["inputs"], {})
        login = django_inputs(LoginView.as_view(), ["GET", "POST"])
        self.assertEqual(login["inputs"], {"username": "text", "password": "text", "next": "text"})
        self.assertEqual(django_inputs(Away.as_view(), ["GET"])["reasons"][0]["code"], "raw_body")

    def test_drf_views_list_serializer_pagination_and_filters(self):
        notes = django_inputs(Notes.as_view(), ["GET", "POST"])
        self.assertEqual(notes["inputs"], {"format": "text", "title": "text", "rating": "int", "page": "text", "size": "int",
                                           "search": "text", "ordering": "text"})
        self.assertEqual(notes["reasons"], [])
        self.assertEqual(django_inputs(Chooses.as_view(), ["POST"])["reasons"][0]["code"], "class_view")
        mixed = django_inputs(Mixed.as_view(), ["POST"])
        self.assertFalse(mixed["standalone"])
        self.assertIn("Mixin", mixed["reasons"][0]["detail"])
        self.assertIn("CustomFilter", django_inputs(Unknown.as_view(), ["GET"])["reasons"][0]["detail"])


def plain(request):
    return HttpResponse(request.GET.get("q", ""))


from django.views import View  # noqa: E402
from webargs import fields  # noqa: E402
from webargs.djangoparser import use_args  # noqa: E402


@use_args({"q": fields.Str(), "page": fields.Int()}, location="query")
def search(request, args):
    return HttpResponse(args.get("q", ""))


class Items(View):
    @use_args({"name": fields.Str()}, location="json")
    def post(self, request, args):
        return HttpResponse(args["name"])

    def get(self, request):
        return HttpResponse("")


urlpatterns = [
    path("contact/", Contact.as_view()),
    path("notes/", Notes.as_view()),
    path("plain/", plain),
    path("search/", search),
    path("items/", Items.as_view()),
]


class ManifestTests(SimpleTestCase):
    def test_init_combines_the_framework_with_code_reading(self):
        from aiwaf.django.path_manifest import generate_django_manifest

        with TemporaryDirectory() as folder, override_settings(ROOT_URLCONF=__name__):
            source = Path(folder) / "src"
            (source / "shop").mkdir(parents=True)
            (source / "shop" / "__init__.py").write_text("", encoding="utf-8")
            (source / "shop" / "forms.py").write_text(
                "from django import forms\nclass AddressForm(forms.ModelForm):\n    class Meta:\n        model = None\n", encoding="utf-8")
            (source / "shop" / "views.py").write_text(
                "from .forms import AddressForm\ndef plain(request):\n    return request.GET.get('q'), AddressForm(request.POST)\n",
                encoding="utf-8")
            # The view's module is loaded, as in a running app; its ModelForm resolves from there.
            loaded = types.ModuleType("shop.views")
            loaded.AddressForm = UserForm
            sys.modules["shop.views"] = loaded
            routes = generate_django_manifest(str(Path(folder) / "paths.json"), audit_source=str(source))["routes"]
            del sys.modules["shop.views"]
        self.assertEqual(routes["/contact/"]["inputs"], {"name": "text", "age": "int", "subscribe": "bool"})
        self.assertTrue(routes["/contact/"]["inputs_complete"])
        self.assertTrue(routes["/notes/"]["inputs_complete"])
        self.assertNotIn("declared_inputs", routes["/notes/"])
        # plain's view is tests.django...plain, not shop.views.plain, so its code isn't linked.
        self.assertNotIn("inputs", routes["/plain/"])


class WebargsTests(SimpleTestCase):
    def test_webargs_views_give_their_arguments(self):
        from aiwaf.django.path_manifest import extract_django_routes

        with override_settings(ROOT_URLCONF=__name__):
            routes = extract_django_routes()
        self.assertEqual(routes["/search/"]["declared_inputs"]["inputs"], {"q": "text", "page": "int"})
        self.assertEqual(routes["/items/"]["declared_inputs"]["inputs"], {"name": "text"})
        self.assertNotIn("declared_inputs", routes["/plain/"])


class RecorderTests(SimpleTestCase):
    def test_records_what_the_view_reads_after_other_middleware(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "seen.jsonl"
            os.environ[input_recorder.ENVIRONMENT] = str(target)
            try:
                def view(request):
                    request.GET.get("q")
                    request.POST["name"]
                    request.POST.get("csrfmiddlewaretoken")
                    list(request.GET.items())
                    return HttpResponse("ok")

                middleware = input_recorder.DjangoInputRecorder(lambda request: view(request))
                request = RequestFactory().post("/search/?q=1&extra=2", {"name": "a", "csrfmiddlewaretoken": "t"})
                request.GET.get("before")  # Read before the view starts: not the view's.
                request.resolver_match = types.SimpleNamespace(route="search/")
                middleware.process_view(request, view, (), {})
                middleware(request)
                lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            finally:
                del os.environ[input_recorder.ENVIRONMENT]
        self.assertEqual(sorted(line["key"] for line in lines), ["*", "name", "q"])
        self.assertEqual({line["route"] for line in lines}, {"/search/"})

    def test_does_nothing_when_off(self):
        os.environ.pop(input_recorder.ENVIRONMENT, None)
        middleware = input_recorder.DjangoInputRecorder(lambda request: HttpResponse("ok"))
        request = RequestFactory().get("/")
        self.assertIsNone(middleware.process_view(request, plain, (), {}))
        self.assertEqual(middleware(request).status_code, 200)
        self.assertFalse(hasattr(request, "_aiwaf_recording"))

    def test_aiwaf_adds_the_recorder_when_recording(self):
        os.environ.pop(input_recorder.ENVIRONMENT, None)
        with override_settings(MIDDLEWARE=["django.middleware.common.CommonMiddleware"]):
            self.assertFalse(input_recorder.add_django_recorder())
            from django.conf import settings
            self.assertNotIn(input_recorder.DJANGO_RECORDER, settings.MIDDLEWARE)
        with override_settings(MIDDLEWARE=("django.middleware.common.CommonMiddleware",), AIWAF_RECORD_INPUTS="1"):
            from django.apps import apps
            from django.conf import settings
            apps.get_app_config("aiwaf").ready()
            self.assertEqual(settings.MIDDLEWARE[-1], input_recorder.DJANGO_RECORDER)
            self.assertTrue(input_recorder.add_django_recorder())
            self.assertEqual(settings.MIDDLEWARE.count(input_recorder.DJANGO_RECORDER), 1)
            self.assertEqual(input_recorder.DjangoInputRecorder(lambda request: None).path, Path(".aiwaf/observed_inputs.jsonl"))
