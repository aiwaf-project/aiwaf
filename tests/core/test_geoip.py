from aiwaf.core.geoip import _extract_country_name_from_raw


def test_extract_country_name_supports_database_shapes():
    assert _extract_country_name_from_raw(None) is None
    assert _extract_country_name_from_raw({"country": {"name": "Canada"}}) == "Canada"
    assert _extract_country_name_from_raw({"country": "Canada"}) == "Canada"
    assert _extract_country_name_from_raw({"country_name": "Canada"}) == "Canada"
    assert _extract_country_name_from_raw({}) is None
