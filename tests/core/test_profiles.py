"""AIWAF_PROFILE picks defaults; monitor logs what it would block instead of blocking."""
import logging

import pytest

from aiwaf.core import profiles


@pytest.fixture(autouse=True)
def balanced(monkeypatch):
    monkeypatch.delenv("AIWAF_PROFILE", raising=False)
    yield
    profiles.activate("balanced")


def test_profiles_resolve_from_settings_then_the_environment(monkeypatch):
    assert profiles.resolve() == "balanced"
    monkeypatch.setenv("AIWAF_PROFILE", "Monitor")
    assert profiles.resolve() == "monitor"
    assert profiles.resolve("strict") == "strict"
    with pytest.raises(ValueError, match="AIWAF_PROFILE must be one of strict, balanced, monitor"):
        profiles.resolve("loose")
    profiles.activate("balanced")
    assert not profiles.monitoring()  # The profile the app chose wins over the environment.
    profiles.activate("monitor")
    assert profiles.monitoring()


def test_strict_halves_this_frameworks_rate_limits_and_monitor_logs_instead():
    assert profiles.settings_for("balanced", 20, 40) == {}
    strict = profiles.settings_for("strict", 20, 40)
    assert (strict["AIWAF_RATE_MAX"], strict["AIWAF_RATE_FLOOD"]) == (10, 20)
    assert profiles.settings_for("strict", 1, 1)["AIWAF_RATE_MAX"] == 1
    assert profiles.settings_for("monitor", 20, 40) == {"AIWAF_SQL_INJECTION_MODE": "monitor", "AIWAF_INJECTION_RISK_MODE": "log"}


def test_explicit_settings_win_over_the_profile(caplog):
    settings = {"AIWAF_PROFILE": "strict", "AIWAF_RATE_MAX": 7}
    assert profiles.apply(settings, settings.setdefault) == "strict"
    assert (settings["AIWAF_RATE_MAX"], settings["AIWAF_RATE_FLOOD"]) == (7, 20)
    monitor = {"AIWAF_PROFILE": "monitor"}
    with caplog.at_level(logging.WARNING, "aiwaf.monitor"):
        profiles.apply(monitor, monitor.setdefault)
    assert "nothing is blocked" in caplog.text


def test_monitor_skips_the_blocklist_and_says_so(caplog):
    from aiwaf.core.runtime_blocklist import BlocklistManager

    assert not profiles.skip_blocklist("203.0.113.5", "flood")
    profiles.activate("monitor")
    with caplog.at_level(logging.WARNING, "aiwaf.monitor"):
        assert profiles.skip_blocklist("203.0.113.5", "flood")
        assert BlocklistManager.block("203.0.113.6", "probe") is False
        profiles.would_block("GET", "/x", "203.0.113.5", 403, "")
    assert "would add 203.0.113.6 to the blocklist (probe)" in caplog.text
    assert "would block GET /x from 203.0.113.5 with 403 (blocked)" in caplog.text
