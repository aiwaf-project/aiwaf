import os
from urllib.parse import urljoin

import requests
from flask import Flask, Response, request

from aiwaf.flask.middleware import register_aiwaf_middlewares
from aiwaf.flask.storage import add_ip_whitelist


def create_app():
    app = Flask(__name__)

    target_base_url = os.environ.get("TARGET_BASE_URL", "http://juice:3000").rstrip("/") + "/"
    port = int(os.environ.get("PORT", "3010"))

    app.config.update(
        TARGET_BASE_URL=target_base_url,
        PORT=port,
        AIWAF_LOG_DIR=os.environ.get("AIWAF_LOG_DIR", "/logs"),
        AIWAF_LOG_FORMAT=os.environ.get("AIWAF_LOG_FORMAT", "json"),
        AIWAF_MIN_AI_LOGS=int(os.environ.get("AIWAF_MIN_AI_LOGS", "10000")),
        AIWAF_FORCE_AI=os.environ.get("AIWAF_FORCE_AI", "false").lower() == "true",
        AIWAF_USE_RUST=os.environ.get("AIWAF_USE_RUST", "true").lower() == "true",
        AIWAF_PATH_RULES=[
            {
                "PREFIX": "/socket.io/",
                "DISABLE": [
                    "HeaderValidationMiddleware",
                    "IPAndKeywordBlockMiddleware",
                    "RateLimitMiddleware",
                ],
            },
        ],
    )

    # Sandbox defaults: keep local demo traffic from self-blacklisting.
    whitelist_ips = [
        item.strip()
        for item in os.environ.get("AIWAF_WHITELIST_IPS", "127.0.0.1,172.20.0.1").split(",")
        if item.strip()
    ]
    for ip in whitelist_ips:
        try:
            add_ip_whitelist(ip)
        except Exception:
            pass

    register_aiwaf_middlewares(
        app,
        middlewares=[
            "logging",
            "header_validation",
            "ip_keyword_block",
            "rate_limit",
            "geo_block",
            "ai_anomaly",
            "uuid_tamper",
        ],
    )

    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }

    def filter_headers(headers):
        filtered = {
            k: v
            for k, v in headers.items()
            if k.lower() not in hop_by_hop and k.lower() not in {"host", "accept-encoding"}
        }
        # Force identity to avoid forwarding compressed upstream payloads that may
        # not be transparently decoded by the proxy runtime.
        filtered["Accept-Encoding"] = "identity"
        return filtered

    @app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    @app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def proxy(path):
        upstream_url = urljoin(app.config["TARGET_BASE_URL"], path)
        if request.query_string:
            upstream_url = f"{upstream_url}?{request.query_string.decode('utf-8', errors='ignore')}"

        upstream = requests.request(
            method=request.method,
            url=upstream_url,
            headers=filter_headers(dict(request.headers)),
            data=request.get_data(cache=False) if request.data else None,
            allow_redirects=False,
            stream=True,
            timeout=30,
        )

        excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
        response_headers = [
            (k, v) for k, v in upstream.headers.items() if k.lower() not in excluded
        ]
        return Response(upstream.content, status=upstream.status_code, headers=response_headers)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(application.config["PORT"]), debug=False)
