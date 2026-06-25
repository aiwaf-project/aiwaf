"""Safe model artifact serialization helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import skops.io as skops_io  # type: ignore
    SKOPS_AVAILABLE = True
except Exception:
    skops_io = None
    SKOPS_AVAILABLE = False


def safe_model_serialization_available() -> bool:
    return SKOPS_AVAILABLE


def can_serialize_model_artifact(model_data: Any) -> bool:
    return _dump_json(model_data) is not None or SKOPS_AVAILABLE


def default_model_filename() -> str:
    return "model.skops"


def _dump_json(model_data: Any) -> bytes | None:
    try:
        return json.dumps(model_data, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_model_artifact(model_data: Any, path: str | Path) -> None:
    json_data = _dump_json(model_data)
    if json_data is not None:
        Path(path).write_bytes(json_data)
        return
    if not SKOPS_AVAILABLE:
        raise RuntimeError("skops is not available")
    skops_io.dump(model_data, path)


def load_model_artifact(path: str | Path) -> Any:
    path_obj = Path(path)
    try:
        return _load_json(path_obj)
    except Exception:
        pass
    if not SKOPS_AVAILABLE:
        raise RuntimeError("skops is not available")
    return skops_io.load(path_obj)


def dumps_model_artifact(model_data: Any) -> bytes:
    json_data = _dump_json(model_data)
    if json_data is not None:
        return json_data
    if not SKOPS_AVAILABLE:
        raise RuntimeError("skops is not available")
    fd, tmp_name = tempfile.mkstemp(suffix=".skops")
    os.close(fd)
    try:
        skops_io.dump(model_data, tmp_name)
        return Path(tmp_name).read_bytes()
    finally:
        try:
            Path(tmp_name).unlink()
        except OSError:
            pass


def loads_model_artifact(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass
    if not SKOPS_AVAILABLE:
        raise RuntimeError("skops is not available")
    fd, tmp_name = tempfile.mkstemp(suffix=".skops")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(raw)
        return skops_io.load(tmp_name)
    finally:
        try:
            Path(tmp_name).unlink()
        except OSError:
            pass
