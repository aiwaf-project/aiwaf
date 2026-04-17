"""Fuzz-like malformed header robustness tests."""

import random
import string

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.header_validation import HeaderValidationMiddleware


def _app():
    app = FastAPI()

    @app.get("/api/fuzz")
    async def fuzz_endpoint():
        return {"ok": True}

    app.add_middleware(HeaderValidationMiddleware, block_suspicious=True)
    return app


def _rand(n):
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{};':,.<>/?\\|\t\n\r"
    return "".join(random.choice(alphabet) for _ in range(n))


def test_malformed_and_oversized_headers_do_not_crash_middleware():
    client = TestClient(_app())

    for size in [0, 1, 10, 512, 4096, 12000]:
        headers = {
            "user-agent": _rand(size),
            "accept": _rand(min(size, 1000)) or "*/*",
            "x-weird-header": _rand(min(size, 500)),
        }
        response = client.get("/api/fuzz", headers=headers)
        assert response.status_code in {200, 403}
