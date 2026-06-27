from django.test import override_settings
from tempfile import TemporaryDirectory
from pathlib import Path

from tests.django.base_test import AIWAFTestCase


class DjangoPathManifestTest(AIWAFTestCase):
    @override_settings(ROOT_URLCONF="tests.django.test_urls")
    def test_django_manifest_extracts_routes(self):
        from aiwaf.django.path_manifest import extract_django_routes, generate_django_manifest

        routes = extract_django_routes()

        assert "/admin/login/" in routes
        assert routes["/admin/login/"]["category"] == "admin"
        assert "/api/users/" in routes
        assert routes["/api/users/"]["category"] == "api"

    @override_settings(ROOT_URLCONF="tests.django.test_urls")
    def test_django_manifest_writes_file(self):
        from aiwaf.django.path_manifest import generate_django_manifest

        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "paths.json"
            manifest = generate_django_manifest(str(manifest_path))

            assert manifest_path.exists()
        assert manifest["framework"] == "django"
        assert manifest["context_hash"]
