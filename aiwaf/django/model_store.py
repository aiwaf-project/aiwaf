import logging
from pathlib import Path
from django.conf import settings
from django.core.cache import cache

from aiwaf.core.model_security import is_trusted_model_path
from aiwaf.core.model_serialization import (
    can_serialize_model_artifact,
    dump_model_artifact,
    dumps_model_artifact,
    load_model_artifact,
    loads_model_artifact,
)

logger = logging.getLogger("aiwaf.django.model_store")


def _normalize_storage_mode(value):
    if not value:
        return "file"
    value = str(value).strip().lower()
    if value in {"filesystem", "fs"}:
        return "file"
    return value


def _load_from_file(path, *, default_path=None, allow_custom=False):
    if not path:
        return None
    if not is_trusted_model_path(path, default_path=default_path, allow_custom=allow_custom):
        return None
    try:
        return load_model_artifact(path)
    except Exception:
        return None


def _dump_to_bytes(model_data):
    if not can_serialize_model_artifact(model_data):
        return None
    try:
        return dumps_model_artifact(model_data)
    except Exception:
        return None


def _load_from_bytes(raw):
    if not raw:
        return None
    try:
        return loads_model_artifact(raw)
    except Exception:
        return None


def load_model_data():
    storage_mode = _normalize_storage_mode(getattr(settings, "AIWAF_MODEL_STORAGE", "file"))
    model_path = getattr(settings, "AIWAF_MODEL_PATH", None)
    fallback = getattr(settings, "AIWAF_MODEL_STORAGE_FALLBACK", True)
    allow_custom = getattr(settings, "AIWAF_ALLOW_CUSTOM_MODEL_PATH", False)
    default_model_path = str(Path(__file__).with_name("resources") / "model.json")

    if storage_mode == "db":
        try:
            from .models import AIModelArtifact
            record = AIModelArtifact.objects.filter(name="default").first()
            if record:
                data = _load_from_bytes(record.data)
                if data is not None:
                    return data
        except Exception as exc:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("AIWAF model DB load failed: %s", exc)
        if fallback:
            return _load_from_file(model_path, default_path=default_model_path, allow_custom=allow_custom)
        return None

    if storage_mode == "cache":
        cache_key = getattr(settings, "AIWAF_MODEL_CACHE_KEY", "aiwaf:model")
        entry = cache.get(cache_key)
        if isinstance(entry, dict):
            data = _load_from_bytes(entry.get("data"))
            if data is not None:
                return data
        if fallback:
            return _load_from_file(model_path, default_path=default_model_path, allow_custom=allow_custom)
        return None

    return _load_from_file(model_path, default_path=default_model_path, allow_custom=allow_custom)


def save_model_data(model_data, metadata=None):
    storage_mode = _normalize_storage_mode(getattr(settings, "AIWAF_MODEL_STORAGE", "file"))
    model_path = getattr(settings, "AIWAF_MODEL_PATH", None)

    if storage_mode == "file":
        if not can_serialize_model_artifact(model_data):
            return False
        try:
            dump_model_artifact(model_data, model_path)
            return True
        except Exception:
            return False

    raw = _dump_to_bytes(model_data)
    if raw is None:
        return False

    if storage_mode == "db":
        try:
            from .models import AIModelArtifact
            AIModelArtifact.objects.update_or_create(
                name="default",
                defaults={"data": raw, "metadata": metadata or {}},
            )
            return True
        except Exception as exc:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("AIWAF model DB save failed: %s", exc)
            return False

    if storage_mode == "cache":
        cache_key = getattr(settings, "AIWAF_MODEL_CACHE_KEY", "aiwaf:model")
        cache_timeout = getattr(settings, "AIWAF_MODEL_CACHE_TIMEOUT", None)
        cache.set(cache_key, {"data": raw, "metadata": metadata or {}}, timeout=cache_timeout)
        return True

    return False
