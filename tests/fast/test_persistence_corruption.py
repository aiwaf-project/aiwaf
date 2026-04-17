"""Corruption and recovery tests for file-backed persistence."""

from pathlib import Path

from aiwaf.core.runtime_config import AIWAFConfig
from aiwaf.core.runtime_storage import FileStorage


def test_filestorage_handles_corrupted_json(tmp_path):
    path = tmp_path / "store.json"
    path.write_text("{not-json", encoding="utf-8")

    storage = FileStorage(str(path))
    assert storage.get("anything") is None

    storage.set("ok", {"x": 1})
    assert storage.get("ok") == {"x": 1}


def test_config_loader_handles_corrupted_json_and_keeps_defaults(tmp_path):
    path = tmp_path / "bad_config.json"
    path.write_text("[1,", encoding="utf-8")

    config = AIWAFConfig(config_file=str(path), load_from_env=False)
    assert config.get("storage.backend") == "memory"
    assert config.get("header_validation.enabled") is True


def test_filestorage_can_recover_after_partial_like_write(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{\n  \"blocked:1.1.1.1\": {\"value\":", encoding="utf-8")

    storage = FileStorage(str(path))
    storage.set("key", "value")
    assert storage.get("key") == "value"
