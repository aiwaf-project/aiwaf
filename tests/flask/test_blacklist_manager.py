def test_unblock_delegates_when_enabled(monkeypatch):
    from aiwaf.flask import blacklist_manager

    calls = []
    monkeypatch.setattr(blacklist_manager, "is_ip_whitelisted", lambda _ip: False)
    monkeypatch.setattr(blacklist_manager, "remove_ip_blacklist", lambda ip: calls.append(ip))
    blacklist_manager.BlacklistManager.unblock("203.0.113.120")
    assert calls == ["203.0.113.120"]
