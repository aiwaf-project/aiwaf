import pytest
from django.test import override_settings

pytestmark = pytest.mark.django


def test_django_csv_mode_uses_runtime_storage_and_persists(tmp_path):
    from aiwaf.django.storage import (
        get_blacklist_store,
        get_exemption_store,
        get_keyword_store,
    )

    with override_settings(AIWAF_STORAGE_MODE="csv", AIWAF_DATA_DIR=str(tmp_path)):
        exemption_store = get_exemption_store()
        blacklist_store = get_blacklist_store()
        keyword_store = get_keyword_store()

        ip = "203.0.113.10"
        exemption_store.add_exemption(ip, "trusted test ip")
        assert exemption_store.is_exempted(ip) is True

        blacklist_store.block_ip("198.51.100.23", "csv django test block")
        assert blacklist_store.is_blocked("198.51.100.23") is True

        keyword_store.add_keyword("runtime-csv-keyword", count=2)
        top = keyword_store.get_top_keywords(10)
        assert "runtime-csv-keyword" in top

        runtime_csv = tmp_path / "runtime_store.csv"
        assert runtime_csv.exists(), "Expected shared runtime CSV backend file to exist"
