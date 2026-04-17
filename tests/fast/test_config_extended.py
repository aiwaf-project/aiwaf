"""Extended config lifecycle and initialization tests."""

import json
import logging

import pytest

import aiwaf.core.runtime_config as config_module
from aiwaf.core.runtime_config import AIWAFConfig, initialize_config


def test_load_from_file_deep_merges(tmp_path):
    config_file = tmp_path / "aiwaf.json"
    config_file.write_text(
        json.dumps(
            {
                "header_validation": {"quality_threshold": 7},
                "storage": {"backend": "file", "file_path": "state.json"},
            }
        )
    )

    config = AIWAFConfig(config_file=str(config_file), load_from_env=False)
    assert config.get("header_validation.quality_threshold") == 7
    assert config.get("storage.backend") == "file"
    assert config.get("storage.file_path") == "state.json"


def test_save_and_reload_round_trip(tmp_path):
    path = tmp_path / "saved.json"
    config = AIWAFConfig(load_from_env=False)
    config.set("rate_limiting.max_requests", 321)
    config.save_to_file(str(path))

    loaded = AIWAFConfig(config_file=str(path), load_from_env=False)
    assert loaded.get("rate_limiting.max_requests") == 321


def test_parse_environment_list_and_bool(monkeypatch):
    monkeypatch.setenv("AIWAF_HEADER_EXEMPT_PATHS", "/health,/metrics")
    monkeypatch.setenv("AIWAF_RATE_LIMITING_ENABLED", "yes")

    config = AIWAFConfig(load_from_env=True)
    assert config.get("header_validation.exempt_paths") == ["/health", "/metrics"]
    assert config.get("rate_limiting.enabled") is True


def test_setup_logging_uses_level_and_optional_file(monkeypatch, tmp_path):
    calls = {}

    def fake_basic_config(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    config = AIWAFConfig(load_from_env=False)
    log_file = tmp_path / "aiwaf.log"
    config.set("logging.level", "DEBUG")
    config.set("logging.log_file", str(log_file))
    config.setup_logging()

    assert calls["level"] == logging.DEBUG
    assert calls["filename"] == str(log_file)


def test_initialize_config_raises_on_validation_errors(monkeypatch):
    monkeypatch.setenv("AIWAF_RATE_MAX_REQUESTS", "-10")

    with pytest.raises(ValueError):
        initialize_config()


def test_get_config_returns_singleton_instance():
    config_module._config = None
    first = config_module.get_config()
    second = config_module.get_config()

    assert first is second
