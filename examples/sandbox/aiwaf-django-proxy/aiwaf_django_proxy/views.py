import requests
import time
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _copy_request_headers(request):
    headers = {}
    for key, value in request.headers.items():
        if key.lower() in _HOP_BY_HOP:
            continue
        # Never forward the incoming Host header to the upstream target.
        # Some upstreams (including Juice Shop) will reject unexpected hosts.
        if key.lower() == "host":
            continue
        headers[key] = value
    return headers


def _copy_response_headers(upstream_headers):
    headers = {}
    for key, value in upstream_headers.items():
        if key.lower() in _HOP_BY_HOP:
            continue
        # requests may transparently decode content while preserving header keys.
        # Do not forward content-encoding/content-length from upstream in that case.
        if key.lower() in {"content-encoding", "content-length"}:
            continue
        headers[key] = value
    return headers


@csrf_exempt
def proxy(request, path=""):
    base = settings.TARGET_BASE_URL.rstrip("/")
    url = f"{base}/{path}" if path else base
    if request.META.get("QUERY_STRING"):
        url = f"{url}?{request.META['QUERY_STRING']}"

    headers = _copy_request_headers(request)

    upstream = None
    last_exc = None
    for _ in range(8):
        try:
            upstream = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                data=request.body if request.body else None,
                allow_redirects=False,
                stream=True,
                timeout=30,
            )
            break
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            time.sleep(0.5)
    if upstream is None and last_exc is not None:
        raise last_exc

    # Some upstream apps return 400 for forwarded browser/proxy headers.
    # Retry idempotent requests with a minimal header set to maximize compatibility.
    if upstream.status_code == 400 and request.method in {"GET", "HEAD"}:
        minimal_headers = {}
        user_agent = request.headers.get("User-Agent")
        accept = request.headers.get("Accept")
        if user_agent:
            minimal_headers["User-Agent"] = user_agent
        if accept:
            minimal_headers["Accept"] = accept
        upstream = requests.request(
            method=request.method,
            url=url,
            headers=minimal_headers,
            allow_redirects=False,
            stream=True,
            timeout=30,
        )

    resp = HttpResponse(
        upstream.content,
        status=upstream.status_code,
        content_type=upstream.headers.get("content-type"),
    )

    for key, value in _copy_response_headers(upstream.headers).items():
        resp[key] = value

    return resp
