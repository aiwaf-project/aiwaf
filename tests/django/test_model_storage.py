import sys
import types
from django.conf import settings
from django.core.cache import cache
from django.test import override_settings

from .base_test import AIWAFTestCase


class TestModelStorage(AIWAFTestCase):
    def test_cache_model_storage_roundtrips_json_model_artifact(self):
        from aiwaf.django.model_store import load_model_data, save_model_data

        model_data = {
            "model_type": "aiwaf_rust.IsolationForest",
            "model_state": {"trees": [], "threshold": 0.0},
            "model_backend": "aiwaf_rust",
            "framework": "django",
        }

        with override_settings(
            CACHES={
                "default": {
                    "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                    "LOCATION": "aiwaf-model-store-json",
                }
            },
            AIWAF_MODEL_STORAGE="cache",
            AIWAF_MODEL_CACHE_KEY="aiwaf:test:json-model",
            AIWAF_MODEL_STORAGE_FALLBACK=False,
        ):
            cache.clear()
            assert save_model_data(model_data, metadata={"source": "json-test"}) is True
            loaded = load_model_data()

        assert loaded["model_backend"] == "aiwaf_rust"
        assert loaded["framework"] == "django"
        assert loaded["model_state"] == model_data["model_state"]

    def test_cache_model_storage_rejects_python_object_artifact(self):
        from aiwaf.core.model_artifacts import sklearn_model_artifact
        from aiwaf.django.model_store import save_model_data

        class UnsafeModelObject:
            pass

        model_data = sklearn_model_artifact(
            UnsafeModelObject(),
            "test",
            ["f1"],
            1,
            "django",
        )

        with override_settings(
            CACHES={
                "default": {
                    "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                    "LOCATION": "aiwaf-model-store-reject",
                }
            },
            AIWAF_MODEL_STORAGE="cache",
            AIWAF_MODEL_CACHE_KEY="aiwaf:test:reject-model",
            AIWAF_MODEL_STORAGE_FALLBACK=False,
        ):
            cache.clear()
            assert save_model_data(model_data, metadata={"source": "reject-test"}) is False

    @override_settings(
        CACHES={
            "default": {
                "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                "LOCATION": "aiwaf-model-store",
            }
        },
        AIWAF_MODEL_STORAGE="cache",
        AIWAF_MODEL_CACHE_KEY="aiwaf:test:model",
        AIWAF_MODEL_STORAGE_FALLBACK=False,
    )
    def test_cache_model_storage_roundtrip(self):
        from aiwaf.django.model_store import load_model_data, save_model_data

        cache.clear()
        model_data = {"model": {"stub": True}, "sklearn_version": "1.0"}
        assert save_model_data(model_data, metadata={"source": "cache-test"}) is True

        loaded = load_model_data()
        assert loaded == model_data

    @override_settings(
        AIWAF_MODEL_STORAGE="db",
        AIWAF_MODEL_STORAGE_FALLBACK=False,
    )
    def test_db_model_storage_roundtrip(self):
        from aiwaf.django.model_store import load_model_data, save_model_data
        from aiwaf.django.models import AIModelArtifact

        AIModelArtifact.objects.all().delete()
        model_data = {"model": {"stub": True}, "sklearn_version": "1.0"}
        assert save_model_data(model_data, metadata={"source": "db-test"}) is True

        loaded = load_model_data()
        assert loaded == model_data

    @override_settings(
        AIWAF_MODEL_STORAGE="db",
        AIWAF_MODEL_STORAGE_FALLBACK=False,
    )
    def test_db_missing_model_message_mentions_db(self):
        from aiwaf.django.models import AIModelArtifact
        import aiwaf.django.middleware as middleware

        AIModelArtifact.objects.all().delete()
        if "sklearn" not in sys.modules:
            sys.modules["sklearn"] = types.SimpleNamespace(__version__="0.0")

        with self.assertLogs("aiwaf.django.middleware", level="WARNING") as captured:
            middleware.load_model_safely()

        output = "\n".join(captured.output)
        assert "aiwaf_aimodelartifact" in output
        assert "model.json" not in output
