from django.test import override_settings
from django.http import HttpResponse, JsonResponse
from django.urls import path
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.contrib.auth import authenticate, login
from tempfile import TemporaryDirectory
from pathlib import Path

from tests.django.base_test import AIWAFTestCase


def _helper_checks_method(request):
    if request.method == "POST":
        return HttpResponse("posted")
    return HttpResponse("ok")


def _helper_reads_payload(req):
    return req.POST.get("name")


def _helper_reads_files(request):
    return request.FILES.get("photo")


def view_delegates_method_check(request):
    return _helper_checks_method(request)


def view_delegates_payload(request):
    if _helper_reads_payload(request):
        return HttpResponse("posted")
    return HttpResponse("ok")


def view_passes_post_to_helper(request):
    return _helper_reads_payload(request.POST)


def view_delegates_files(request):
    return _helper_reads_files(request)


def _process_login(request):
    user = authenticate(request, username="user", password="pw")
    if user is not None:
        login(request, user)
    return HttpResponse("ok")


def view_delegates_django_login(request):
    return _process_login(request)


def _api_payload_helper(request):
    return request.body


def django_json_endpoint(request):
    _api_payload_helper(request)
    return JsonResponse({"ok": True})


class DjangoPathManifestTest(AIWAFTestCase):
    @override_settings(ROOT_URLCONF="tests.django.test_urls")
    def test_django_manifest_extracts_routes(self):
        from aiwaf.django.path_manifest import extract_django_routes, generate_django_manifest

        routes = extract_django_routes()

        assert "/admin/login/" in routes
        assert routes["/admin/login/"]["category"] == "admin"
        assert "/api/users/" in routes
        assert routes["/api/users/"]["category"] == "api"
        assert routes["/test/"]["methods"] == ["GET"]
        assert routes["/test-post/"]["methods"] == ["GET", "POST"]

    @override_settings(ROOT_URLCONF="tests.django.test_urls")
    def test_django_manifest_writes_file(self):
        from aiwaf.django.path_manifest import generate_django_manifest

        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "paths.json"
            manifest = generate_django_manifest(str(manifest_path))

            assert manifest_path.exists()
        assert manifest["framework"] == "django"
        assert manifest["context_hash"]

    def test_django_manifest_defaults_function_views_to_get_method(self):
        from aiwaf.django.path_manifest import _methods

        def view(request):
            return HttpResponse("ok")

        assert _methods(view) == ["GET"]

    def test_django_manifest_infers_post_from_django_method_decorator(self):
        from aiwaf.django.path_manifest import _methods

        @require_POST
        def view(request):
            return HttpResponse("ok")

        assert _methods(view) == ["POST"]

    def test_django_manifest_infers_get_from_django_method_decorator(self):
        from aiwaf.django.path_manifest import _methods

        @require_GET
        def view(request):
            return HttpResponse("ok")

        assert _methods(view) == ["GET"]

    def test_django_manifest_infers_methods_from_django_method_list_decorator(self):
        from aiwaf.django.path_manifest import _methods

        @require_http_methods(["GET", "POST"])
        def view(request):
            return HttpResponse("ok")

        assert _methods(view) == ["GET", "POST"]

    def test_django_manifest_reads_request_post_usage_from_view_source(self):
        from aiwaf.django.path_manifest import _methods

        def view(request):
            if request.POST.get("name"):
                return HttpResponse("posted")
            return HttpResponse("ok")

        assert _methods(view) == ["GET", "POST"]

    def test_django_manifest_reads_request_files_usage_from_view_source(self):
        from aiwaf.django.path_manifest import _methods

        def view(request):
            uploaded = request.FILES.get("photo")
            return HttpResponse("ok" if uploaded is None else "uploaded")

        assert _methods(view) == ["GET", "POST"]

    def test_django_manifest_reads_request_alias_method_checks(self):
        from aiwaf.django.path_manifest import _methods

        def view(req):
            if req.method in ("PUT", "DELETE"):
                return HttpResponse("mutating")
            return HttpResponse("ok")

        assert _methods(view) == ["DELETE", "GET", "PUT"]

    def test_django_manifest_follows_same_module_helper_method_checks(self):
        from aiwaf.django.path_manifest import _methods

        assert _methods(view_delegates_method_check) == ["GET", "POST"]

    def test_django_manifest_follows_same_module_helper_payload_reads(self):
        from aiwaf.django.path_manifest import _methods

        assert _methods(view_delegates_payload) == ["GET", "POST"]

    def test_django_manifest_detects_post_object_passed_to_helper(self):
        from aiwaf.django.path_manifest import _methods

        assert _methods(view_passes_post_to_helper) == ["GET", "POST"]

    def test_django_manifest_follows_same_module_helper_file_reads(self):
        from aiwaf.django.path_manifest import _methods

        assert _methods(view_delegates_files) == ["GET", "POST"]

    def test_django_manifest_unwraps_decorator_closure_for_view_name(self):
        from aiwaf.django.path_manifest import _view_name

        def real_view(request):
            return HttpResponse("ok")

        def decorator(view_func):
            def _wrapped_view(request, *args, **kwargs):
                return view_func(request, *args, **kwargs)

            return _wrapped_view

        wrapped = decorator(real_view)

        assert _view_name(wrapped).endswith(".real_view")

    def test_django_manifest_marks_portal_routes_authenticated_and_unwrapped(self):
        from aiwaf.django.path_manifest import _collect_routes

        def real_portal_view(request):
            return HttpResponse("ok")

        def decorator(view_func):
            def _wrapped_view(request, *args, **kwargs):
                return view_func(request, *args, **kwargs)

            return _wrapped_view

        routes = _collect_routes([
            path("portal/classes/", decorator(real_portal_view), name="portal_classes"),
        ])

        route = routes["/portal/classes/"]
        assert route["methods"] == ["GET"]
        assert route["category"] == "app"
        assert route["auth_required"] is True
        assert route["view"].endswith(".real_portal_view")

    def test_django_manifest_detects_auth_endpoint_from_helper_calls(self):
        from aiwaf.django.path_manifest import _collect_routes

        routes = _collect_routes([
            path("custom-login/", view_delegates_django_login, name="custom_login"),
        ])

        route = routes["/custom-login/"]
        assert route["category"] == "auth"
        assert route["auth_action"] == "login"
        assert route["auth_confidence"] >= 0.8
        assert "django.contrib.auth.authenticate" in route["auth_signals"]
        assert "django.contrib.auth.login" in route["auth_signals"]

    def test_django_manifest_detects_api_endpoint_from_json_response(self):
        from aiwaf.django.path_manifest import _collect_routes

        routes = _collect_routes([
            path("json-users/", django_json_endpoint, name="json_users"),
        ])

        route = routes["/json-users/"]
        assert route["category"] == "api"
        assert route["response_type"] == "json"
        assert route["api_confidence"] >= 0.5
        assert route["request_body"] is True
        assert "JsonResponse" in route["api_signals"]
