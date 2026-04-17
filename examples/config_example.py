"""FastAPI + AIWAF configuration example.

Run:
    uvicorn examples.config_example:app --reload
"""

from fastapi import FastAPI

from aiwaf.fast import AIWAF

app = FastAPI(title="AIWAF FastAPI Config Example")


@app.get("/")
async def home():
    return {"status": "ok", "service": "fastapi-aiwaf-example"}


@app.get("/api/protected")
async def protected():
    return {"protected": True}


@app.get("/health")
async def health():
    return {"health": "ok"}


# Configure AIWAF with explicit middleware settings.
aiwaf = AIWAF(
    app,
    storage={
        "backend": "memory",  # memory | file | csv | db
        "file_path": "aiwaf_data.json",
    },
    header_validation={
        "enabled": True,
        "block_suspicious": True,
        "quality_threshold": 3,
        "trust_legitimate_bots": False,
        "exempt_paths": ["/health"],
    },
    rate_limiting={
        "enabled": True,
        "window_seconds": 10,
        "max_requests": 20,
        "flood_threshold": 40,
    },
    honeypot={
        "enabled": True,
        "min_form_time": 1.0,
    },
    ip_keyword_block={
        "enabled": True,
        "malicious_keywords": [".env", ".git", "phpmyadmin", "xmlrpc"],
    },
    geo_block={
        "enabled": False,
        "block_countries": ["CN", "RU"],
    },
    ai_anomaly={
        "enabled": False,
    },
    uuid_tamper={
        "enabled": True,
    },
    logging_middleware={
        "enabled": True,
        "log_dir": "aiwaf_logs",
        "log_format": "combined",
    },
    exemptions={
        "private_ips_exempted": True,
        "auto_exempt_patterns": ["127.0.0.1", "10.*.*.*", "192.168.*.*"],
    },
    # Path-specific overrides.
    path_rules=[
        {
            "PREFIX": "/health",
            "DISABLE": ["header_validation", "rate_limit", "ai_anomaly"],
        },
        {
            "PREFIX": "/api/protected",
            "RATE_LIMIT": {"WINDOW": 10, "MAX": 10},
        },
    ],
)

