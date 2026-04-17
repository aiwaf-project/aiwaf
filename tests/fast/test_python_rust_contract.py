"""Contract tests comparing Python-only and Rust-enabled behavior surfaces."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.header_validation import HeaderValidationMiddleware


TEST_CASES = [
    ({"user-agent": "python-requests/2.31.0", "accept": "application/json"}, 403),
    ({"user-agent": "curl/8.0", "accept": "*/*"}, 403),
    (
        {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ),
            "accept": "text/html,application/xml;q=0.9,*/*;q=0.8",
            "accept-language": "en-US,en;q=0.9",
            "accept-encoding": "gzip, deflate",
            "connection": "keep-alive",
        },
        200,
    ),
]


def _client(use_rust: bool, monkeypatch):
    monkeypatch.setattr("aiwaf.fast.middleware.header_validation.rust_available", lambda: use_rust)
    if use_rust:
        monkeypatch.setattr(
            "aiwaf.fast.middleware.header_validation.rust_validate_headers",
            lambda headers, required_headers=None, min_score=None: None,
        )

    app = FastAPI()

    @app.get("/api/contract")
    async def endpoint():
        return {"ok": True}

    app.add_middleware(HeaderValidationMiddleware, block_suspicious=True)
    return TestClient(app)


def test_python_and_rust_paths_accept_same_known_good_headers(monkeypatch):
    py_client = _client(False, monkeypatch)
    rust_client = _client(True, monkeypatch)

    headers = TEST_CASES[-1][0]
    assert py_client.get("/api/contract", headers=headers).status_code == 200
    assert rust_client.get("/api/contract", headers=headers).status_code == 200


def test_python_path_expected_outcomes_for_known_cases(monkeypatch):
    client = _client(False, monkeypatch)
    for headers, expected in TEST_CASES:
        assert client.get("/api/contract", headers=headers).status_code == expected
