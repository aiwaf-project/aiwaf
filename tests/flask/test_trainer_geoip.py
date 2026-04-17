from aiwaf.flask import trainer


def test_geoip_summary_for_blocklist(capsys, monkeypatch):
    monkeypatch.setattr(trainer, "_get_storage_mode", lambda: "csv")
    monkeypatch.setattr(trainer, "_read_csv_blacklist", lambda: {"1.1.1.1": "test", "2.2.2.2": "test"})
    monkeypatch.setattr(trainer, "lookup_country_name", lambda ip, **kwargs: "United States" if ip == "1.1.1.1" else None)
    monkeypatch.setattr(trainer.os.path, "exists", lambda path: True)
    monkeypatch.setattr(trainer, "_get_geoip_db_path", lambda: "fake.mmdb")

    trainer._print_geoip_blocklist_summary()

    out = capsys.readouterr().out
    assert "GeoIP summary for blocked IPs" in out
    assert "United States: 1" in out
    assert "UNKNOWN: 1" in out


def test_geoip_summary_skips_missing_db(capsys, monkeypatch):
    monkeypatch.setattr(trainer, "_get_storage_mode", lambda: "csv")
    monkeypatch.setattr(trainer, "_read_csv_blacklist", lambda: {"1.1.1.1": "test"})
    monkeypatch.setattr(trainer.os.path, "exists", lambda path: False)
    monkeypatch.setattr(trainer, "_get_geoip_db_path", lambda: "missing.mmdb")

    trainer._print_geoip_blocklist_summary()

    out = capsys.readouterr().out
    assert "GeoIP summary skipped" in out
