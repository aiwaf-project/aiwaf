"""
Tests for AIWAF configuration management.
"""

from aiwaf.core.runtime_config import AIWAFConfig


def test_config_get_set_update():
    config = AIWAFConfig(load_from_env=False)
    assert config.get("header_validation.quality_threshold") == 3

    config.set("header_validation.quality_threshold", 5)
    assert config.get("header_validation.quality_threshold") == 5

    config.update({"rate_limiting": {"window_seconds": 120}})
    assert config.get("rate_limiting.window_seconds") == 120


def test_validate_reports_invalid_values():
    config = AIWAFConfig(load_from_env=False)
    config.set("storage.backend", "invalid_backend")
    config.set("header_validation.quality_threshold", 100)
    config.set("logging.level", "VERBOSE")

    errors = config.validate()
    assert any("Invalid storage backend" in error for error in errors)
    assert any("quality_threshold" in error for error in errors)
    assert any("Invalid log level" in error for error in errors)


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("AIWAF_RATE_MAX_REQUESTS", "250")
    monkeypatch.setenv("AIWAF_HEADER_VALIDATION_ENABLED", "false")

    config = AIWAFConfig(load_from_env=True)
    assert config.get("rate_limiting.max_requests") == 250
    assert config.get("header_validation.enabled") is False
