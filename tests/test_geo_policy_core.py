from aiwaf.core.geo_policy import evaluate_geo_policy, normalize_country_list


def test_geo_policy_normalize():
    assert normalize_country_list(["us", " Ca "]) == {"US", "CA"}
    assert normalize_country_list("fr") == {"FR"}
    assert normalize_country_list([]) == set()


def test_geo_policy_allowlist_blocks_non_allowed():
    decision = evaluate_geo_policy(
        country="FR",
        allow_countries={"US"},
        block_countries=set(),
        dynamic_blocked=set(),
    )
    assert decision.should_block is True
    assert decision.country == "FR"


def test_geo_policy_blocklist_and_dynamic():
    blocked = evaluate_geo_policy(
        country="US",
        allow_countries=set(),
        block_countries={"US"},
        dynamic_blocked=set(),
    )
    assert blocked.should_block is True

    dynamic = evaluate_geo_policy(
        country="DE",
        allow_countries=set(),
        block_countries=set(),
        dynamic_blocked={"DE"},
    )
    assert dynamic.should_block is True

    allowed = evaluate_geo_policy(
        country="GB",
        allow_countries=set(),
        block_countries={"US"},
        dynamic_blocked={"DE"},
    )
    assert allowed.should_block is False

