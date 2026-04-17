from aiwaf.fast import geoip


def test_geoip_module_exposes_lookup():
    assert callable(geoip.get_country_for_ip)

