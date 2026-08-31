from aiwaf.core.middleware_plan import plan_enabled_middlewares


def test_plan_auto_disables_logging_when_access_log_present():
    enabled = plan_enabled_middlewares(
        ordered_available=["geo_block", "ip_keyword_block", "logging"],
        requested=["all"],
        disabled=[],
        access_log="/var/log/nginx/access.log",
        geo_enabled_flag=False,
        static_block_countries=[],
        dynamic_block_countries=[],
    )
    assert "logging" not in enabled


def test_plan_auto_enables_logging_when_no_access_log():
    enabled = plan_enabled_middlewares(
        ordered_available=["geo_block", "ip_keyword_block", "logging"],
        requested=["all"],
        disabled=[],
        access_log=None,
        geo_enabled_flag=False,
        static_block_countries=[],
        dynamic_block_countries=[],
    )
    assert "logging" in enabled


def test_plan_auto_enables_geo_when_dynamic_countries_exist():
    enabled = plan_enabled_middlewares(
        ordered_available=["geo_block", "ip_keyword_block", "logging"],
        requested=["all"],
        disabled=[],
        access_log=None,
        geo_enabled_flag=False,
        static_block_countries=[],
        dynamic_block_countries=["US"],
    )
    assert "geo_block" in enabled


def test_plan_auto_disables_uuid_when_no_uuid_routes():
    enabled = plan_enabled_middlewares(
        ordered_available=["uuid_tamper", "ip_keyword_block", "logging"],
        requested=["all"],
        disabled=[],
        access_log=None,
        geo_enabled_flag=False,
        static_block_countries=[],
        dynamic_block_countries=[],
        has_uuid_routes=False,
    )
    assert "uuid_tamper" not in enabled
