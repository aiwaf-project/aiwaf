from aiwaf.fast.geoip import get_country_for_ip


def test_geoip_lookup_callable():
    result = get_country_for_ip("127.0.0.1")
    assert result is None or isinstance(result, str)

