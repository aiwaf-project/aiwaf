from fastapi import FastAPI

from aiwaf.fast import aiwaf_exempt_from


def test_exempt_decorator_marks_endpoint():
    @aiwaf_exempt_from("header_validation")
    async def webhook():
        return {"ok": True}

    assert hasattr(webhook, "_aiwaf_exempt_middlewares")
    assert "header_validation" in webhook._aiwaf_exempt_middlewares
