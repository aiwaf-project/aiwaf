from aiwaf.core.block_responses import blocked_response, throttle_response


def test_blocked_response_defaults():
    payload, status = blocked_response()
    assert status == 403
    assert payload == {"error": "blocked"}


def test_blocked_response_with_message_and_custom_status():
    payload, status = blocked_response("Denied", status_code=405)
    assert status == 405
    assert payload == {"error": "blocked", "message": "Denied"}


def test_throttle_response_contract():
    payload, status = throttle_response()
    assert status == 429
    assert payload == {"error": "too_many_requests"}

