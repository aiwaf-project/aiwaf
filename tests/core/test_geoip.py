from aiwaf.core.geoip import _extract_country_name_from_raw


def test_extract_country_name_supports_database_shapes():
    assert _extract_country_name_from_raw(None) is None
    assert _extract_country_name_from_raw({"country": {"name": "Canada"}}) == "Canada"
    assert _extract_country_name_from_raw({"country": "Canada"}) == "Canada"
    assert _extract_country_name_from_raw({"country_name": "Canada"}) == "Canada"
    assert _extract_country_name_from_raw({}) is None


def test_emit_geoip_summary_counts_countries_and_unknowns():
    from aiwaf.core.geoip import emit_geoip_summary

    lines = []
    names = {"1.1.1.1": "Australia", "1.0.0.1": "Australia", "8.8.8.8": "United States"}
    emit_geoip_summary(list(names) + ["192.0.2.1"], "Summary", names.get, lines.append)
    assert lines == ["Summary", "  - Australia: 2", "  - United States: 1", "  - UNKNOWN: 1"]

    lines.clear()
    emit_geoip_summary([], "Summary", names.get, lines.append)
    assert lines == []
