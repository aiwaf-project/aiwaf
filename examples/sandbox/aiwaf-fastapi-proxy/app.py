import os
import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from aiwaf.fast import AIWAF


TARGET_BASE_URL = os.environ.get("TARGET_BASE_URL", "http://juice:3000").rstrip("/") + "/"
PORT = int(os.environ.get("PORT", "3011"))
UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.upstream_client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        yield
    finally:
        await app.state.upstream_client.aclose()


app = FastAPI(title="AIWAF FastAPI Proxy", lifespan=lifespan)

# Apply AIWAF middleware stack with sandbox-friendly defaults.
AIWAF(
    app,
    storage={"backend": "memory"},
    logging_middleware={
        "enabled": True,
        "log_dir": os.environ.get("AIWAF_LOG_DIR", "/logs"),
        "log_format": os.environ.get("AIWAF_LOG_FORMAT", "json"),
    },
    header_validation={"enabled": True, "block_suspicious": True, "quality_threshold": 3},
    ip_keyword_block={"enabled": True},
    rate_limiting={"enabled": True, "window_seconds": 10, "max_requests": 20, "flood_threshold": 40},
    geo_block={"enabled": False},
    ai_anomaly={"enabled": False},
    uuid_tamper={"enabled": True},
    exemptions={
        "private_ips_exempted": True,
        "auto_exempt_patterns": ["127.0.0.1", "172.20.0.1"],
    },
    path_rules=[
        {
            "PREFIX": "/socket.io/",
            "DISABLE": ["header_validation", "ip_keyword_block", "rate_limit"],
        },
    ],
)


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _filter_request_headers(headers: dict) -> dict:
    filtered = {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() not in {"host", "accept-encoding"}
    }
    filtered["Accept-Encoding"] = "identity"
    return filtered


def _filter_response_headers(headers: httpx.Headers) -> dict:
    excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    upstream_url = urljoin(TARGET_BASE_URL, path)
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    body = await request.body()
    client: httpx.AsyncClient = request.app.state.upstream_client
    upstream = None
    last_exc = None
    for _ in range(8):
        try:
            upstream = await client.request(
                method=request.method,
                url=upstream_url,
                headers=_filter_request_headers(dict(request.headers)),
                content=body if body else None,
            )
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_exc = exc
            await asyncio.sleep(0.5)
        except httpx.HTTPError as exc:
            last_exc = exc
            break

    if upstream is None:
        detail = str(last_exc) if last_exc else "upstream unavailable"
        return Response(content=f"Upstream unavailable: {detail}", status_code=502)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_filter_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    # Run with app object directly to avoid double-import side effects.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
