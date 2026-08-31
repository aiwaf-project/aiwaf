from aiwaf.core.header_validation import resolve_required_headers, validate_headers_python


def test_header_validation_module_contract():
    required = resolve_required_headers(None, method="GET")
    assert required == ["HTTP_USER_AGENT", "HTTP_ACCEPT"]
    result = validate_headers_python({"HTTP_USER_AGENT": "Mozilla/5.0", "HTTP_ACCEPT": "text/html"})
    assert result == "Suspicious headers: Missing all browser-standard headers"
