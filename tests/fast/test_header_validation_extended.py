"""Extended middleware behavior tests for uncovered branches."""

from fastapi import FastAPI

from aiwaf.fast.middleware.header_validation import HeaderValidationMiddleware


def _app():
    return FastAPI()


def test_trust_legitimate_bots_allows_matching_agent_when_enabled():
    middleware = HeaderValidationMiddleware(_app(), trust_legitimate_bots=True)
    user_agent = "Googlebot/2.1 (+http://www.google.com/bot.html)"

    assert middleware._check_user_agent(user_agent) is None


def test_legitimate_bot_can_still_be_flagged_when_trust_disabled():
    middleware = HeaderValidationMiddleware(_app(), trust_legitimate_bots=False)
    user_agent = "Googlebot/2.1 (+http://www.google.com/bot.html)"

    reason = middleware._check_user_agent(user_agent)
    assert reason is not None
    assert "Pattern" in reason


def test_block_request_warning_mode_returns_200_payload():
    middleware = HeaderValidationMiddleware(_app(), block_suspicious=False)

    response = middleware._block_request("198.51.100.9", "test reason", "/api/data")
    assert response.status_code == 200
    body = response.body.decode("utf-8")
    assert "suspicious_headers" in body


def test_quality_scoring_and_combination_checks_cover_edge_cases():
    middleware = HeaderValidationMiddleware(_app())

    low_headers = {"user-agent": "Mozilla/5.0", "accept": "*/*"}
    combo_reason = middleware._check_header_combinations(
        low_headers,
        {"scheme": "https", "http_version": "1.1"},
    )
    assert combo_reason is not None

    high_headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "accept": "text/html,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "en-US,en;q=0.9",
        "accept-encoding": "gzip, deflate",
        "connection": "keep-alive",
        "cache-control": "max-age=0",
    }
    assert middleware._calculate_header_quality(high_headers) >= 8


def test_pattern_management_methods_mutate_lists():
    middleware = HeaderValidationMiddleware(_app())

    middleware.add_suspicious_pattern("evilbot")
    middleware.add_legitimate_pattern("trustedmonitor")
    assert "evilbot" in middleware.suspicious_patterns
    assert "trustedmonitor" in middleware.legitimate_patterns

    middleware.remove_suspicious_pattern("evilbot")
    middleware.remove_legitimate_pattern("trustedmonitor")
    assert "evilbot" not in middleware.suspicious_patterns
    assert "trustedmonitor" not in middleware.legitimate_patterns


def test_enable_disable_and_threshold_mutators_apply_values():
    middleware = HeaderValidationMiddleware(_app(), enabled=False, block_suspicious=False, quality_threshold=1)

    middleware.enable()
    middleware.enable_blocking()
    middleware.set_quality_threshold(4)
    assert middleware.enabled is True
    assert middleware.block_suspicious is True
    assert middleware.quality_threshold == 4

    middleware.disable()
    middleware.disable_blocking()
    assert middleware.enabled is False
    assert middleware.block_suspicious is False
