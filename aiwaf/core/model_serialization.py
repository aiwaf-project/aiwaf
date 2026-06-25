"""Safe model artifact serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_model_serialization_available() -> bool:
    return True


def can_serialize_model_artifact(model_data: Any) -> bool:
    return _dump_json(model_data) is not None


def default_model_filename() -> str:
    return "model.json"


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
    raise RuntimeError("model artifact is not JSON serializable")


def load_model_artifact(path: str | Path) -> Any:
    path_obj = Path(path)
    try:
        return _load_json(path_obj)
    except Exception:
        raise RuntimeError("model artifact is not valid JSON")


def dumps_model_artifact(model_data: Any) -> bytes:
    json_data = _dump_json(model_data)
    if json_data is not None:
        return json_data
    raise RuntimeError("model artifact is not JSON serializable")


def loads_model_artifact(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError("model artifact is not valid JSON")
