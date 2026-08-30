# AIWAF Comprehensive Setup Guide

## 1. What This Guide Is
This guide is an end-to-end setup manual for running AIWAF in real projects.

It includes:
- Environment prerequisites.
- Installation paths for Django, Flask, and FastAPI.
- Minimal and production-oriented configurations.
- Storage and model backend options.
- Training setup and verification.
- Rust acceleration enablement.
- Troubleshooting.

If you want internal architecture details, use `PROJECT_DETAILED_REFERENCE.md`.

## 2. Prerequisites

## 2.1 Runtime
- Python `3.8+`
- `pip`
- A running web app in at least one supported framework:
  - Django
  - Flask
  - FastAPI

## 2.2 Recommended for production
- Reverse proxy/web server logs available (Nginx/Apache/Gunicorn/Uvicorn access logs).
- Persistent filesystem path for AIWAF data/logs.
- Scheduled training job (daily or more frequent for high traffic).

## 3. Installation Options

## 3.1 Base install
```bash
pip install aiwaf
```

## 3.2 Framework extras
```bash
pip install "aiwaf[django]"
pip install "aiwaf[flask]"
pip install "aiwaf[fastapi]"
```

## 3.3 Rust acceleration (optional)
```bash
pip install "aiwaf[rust]"
```

Rust mode is optional. If unavailable, AIWAF falls back to Python logic.

## 4. Shared Configuration Concepts
AIWAF uses `AIWAF_*` flat config keys.

Core knobs you will use most:
- `AIWAF_ACCESS_LOG`
- `AIWAF_DISABLE_AI`
- `AIWAF_MIN_AI_LOGS`
- `AIWAF_MIN_TRAIN_LOGS`
- `AIWAF_FORCE_AI_TRAINING`
- `AIWAF_RATE_WINDOW`, `AIWAF_RATE_MAX`, `AIWAF_RATE_FLOOD`
- `AIWAF_REQUIRED_HEADERS`, `AIWAF_HEADER_QUALITY_MIN_SCORE`
- `AIWAF_EXEMPT_PATHS`, `AIWAF_EXEMPT_KEYWORDS`, `AIWAF_ALLOWED_PATH_KEYWORDS`
- `AIWAF_GEO_BLOCK_ENABLED`, `AIWAF_GEO_BLOCK_COUNTRIES`, `AIWAF_GEO_ALLOW_COUNTRIES`
- `AIWAF_USE_RUST`

Model artifact storage (Django path):
- `AIWAF_MODEL_STORAGE = "file" | "db" | "cache"`
- `AIWAF_MODEL_PATH`
- `AIWAF_MODEL_STORAGE_FALLBACK`

## 5. Django Setup

## 5.1 Install
```bash
pip install "aiwaf[django]"
```

## 5.2 Add app
In Django `settings.py`:
```python
INSTALLED_APPS = [
    # ...
    "aiwaf.django",
]
```

## 5.3 Add middleware
Use this baseline order (adjust around your app needs):
```python
MIDDLEWARE = [
    # ... existing framework middleware ...

    "aiwaf.django.middleware.JsonExceptionMiddleware",  # optional JSON 403 for API clients
    "aiwaf.django.middleware.GeoBlockMiddleware",
    "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
    "aiwaf.django.middleware.RateLimitMiddleware",
    "aiwaf.django.middleware.AIAnomalyMiddleware",
    "aiwaf.django.middleware.HoneypotTimingMiddleware",
    "aiwaf.django.middleware.UUIDTamperMiddleware",
    "aiwaf.django.middleware.HeaderValidationMiddleware",
    "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware",  # optional training fallback logs
]
```

## 5.4 Minimum settings
```python
AIWAF_ACCESS_LOG = "/var/log/nginx/access.log"  # or your real path

AIWAF_DISABLE_AI = False
AIWAF_MIN_AI_LOGS = 10000
AIWAF_MIN_TRAIN_LOGS = 50
AIWAF_FORCE_AI_TRAINING = False

AIWAF_RATE_WINDOW = 10
AIWAF_RATE_MAX = 20
AIWAF_RATE_FLOOD = 40

AIWAF_HEADER_QUALITY_MIN_SCORE = 3
AIWAF_REQUIRED_HEADERS = None

AIWAF_EXEMPT_PATHS = ["/favicon.ico", "/robots.txt", "/static/", "/health/"]
AIWAF_ALLOWED_PATH_KEYWORDS = ["profile", "user", "account", "dashboard"]
AIWAF_EXEMPT_KEYWORDS = ["api", "webhook", "health", "static", "media"]
```

## 5.5 Run migrations
```bash
python manage.py makemigrations aiwaf
python manage.py migrate
```

## 5.6 Verify basic health
```bash
python manage.py check
python manage.py aiwaf_logging --status
python manage.py aiwaf_list --all
```

## 5.7 Start training
```bash
python manage.py detect_and_train
```

Notes:
- If AI training thresholds are not met, keyword-focused training still runs when enough minimal logs are available.
- Use `AIWAF_FORCE_AI_TRAINING = True` to force AI model training with low log volume.

## 5.8 Optional Django storage mode
Django supports model mode and CSV mode.

Use CSV mode if you do not want DB-backed runtime stores:
```python
AIWAF_STORAGE_MODE = "csv"
AIWAF_DATA_DIR = "aiwaf_data"
```

## 5.9 Optional Django model artifact backends
```python
AIWAF_MODEL_STORAGE = "file"  # or "db" or "cache"
AIWAF_MODEL_PATH = "py/aiwaf/django/resources/model.pkl"
AIWAF_MODEL_STORAGE_FALLBACK = True
```

## 6. Flask Setup

## 6.1 Install
```bash
pip install "aiwaf[flask]"
```

Optional DB-backed storage:
```bash
pip install Flask-SQLAlchemy
```

## 6.2 Basic app wiring
```python
from flask import Flask
import aiwaf.flask as aiwaf

app = Flask(__name__)

# Core AIWAF config
app.config["AIWAF_ACCESS_LOG"] = "aiwaf_logs/access.log"
app.config["AIWAF_USE_CSV"] = True
app.config["AIWAF_DATA_DIR"] = "aiwaf_data"

# Optional tuning
app.config["AIWAF_RATE_WINDOW"] = 10
app.config["AIWAF_RATE_MAX"] = 20
app.config["AIWAF_RATE_FLOOD"] = 40
app.config["AIWAF_HEADER_QUALITY_MIN_SCORE"] = 3

# Register all default middleware
aiwaf.register_aiwaf_middlewares(app)
```

## 6.3 Selective middleware registration
```python
aiwaf.register_aiwaf_middlewares(
    app,
    middlewares=["ip_keyword_block", "rate_limit", "header_validation", "logging"],
)
```

## 6.4 Exemption decorators (Flask)
```python
from aiwaf.flask import aiwaf_exempt, aiwaf_exempt_from, aiwaf_only

@app.get("/health")
@aiwaf_exempt
def health():
    return {"ok": True}

@app.post("/webhook")
@aiwaf_exempt_from("rate_limit", "ai_anomaly")
def webhook():
    return {"received": True}
```

## 6.5 Train from logs
Use CLI:
```bash
aiwaf flask train
```

Or programmatically:
```python
from aiwaf.flask.trainer import train_from_logs
train_from_logs(app)
```

## 6.6 Verify Flask setup
- Hit a normal endpoint from browser and confirm no false block.
- Send low-header request via curl and confirm header middleware behavior.
- Check generated logs in `AIWAF_LOG_DIR` (default `aiwaf_logs`).

## 7. FastAPI Setup

## 7.1 Install
```bash
pip install "aiwaf[fastapi]"
```

## 7.2 Basic app wiring
```python
from fastapi import FastAPI
from aiwaf.fast import AIWAF

app = FastAPI()

aiwaf = AIWAF(
    app,
    storage={"backend": "file", "file_path": "aiwaf_data.json"},
    rate_limiting={"enabled": True, "max_requests": 20, "window_seconds": 10, "flood_threshold": 40},
    header_validation={"enabled": True, "quality_threshold": 3},
    geo_block={"enabled": False},
)
```

## 7.3 Path-rule controls (FastAPI)
You can pass path rules in config to disable selected middleware by prefix.

Example:
```python
aiwaf = AIWAF(
    app,
    path_rules=[
        {"PREFIX": "/health", "DISABLE": ["rate_limit", "ai_anomaly", "header_validation"]},
        {"PREFIX": "/api/heavy", "RATE_LIMIT": {"WINDOW": 10, "MAX": 60, "FLOOD": 120}},
    ],
)
```

## 7.4 FastAPI decorators
```python
from aiwaf.fast import aiwaf_exempt, aiwaf_exempt_from, aiwaf_only

@app.get("/health")
@aiwaf_exempt
async def health():
    return {"ok": True}
```

## 7.5 FastAPI CLI
Current fast CLI mirrors Flask command surface:
```bash
aiwaf fast --help
```

## 8. Storage Backend Selection

## 8.1 Runtime storage backends (core)
- `memory`
- `file`
- `csv`
- `db` (SQLite in core runtime)

## 8.2 Which should you pick?
- `memory`: easiest for local tests, non-persistent.
- `file`: simple persistent single-node runtime.
- `csv`: human-readable, file-lock-aware utilities.
- `db`: persistent key-value backend with SQLite.

## 8.3 Multi-instance note
For horizontally scaled deployments, ensure shared/centralized state strategy (or custom storage integration) so block/exemption decisions are consistent across instances.

## 9. Training and Data Sources
AIWAF training reads from:
- configured access logs (including rotated and gzip variants), and/or
- adapter logging outputs, and/or
- framework-backed request logs (Django `RequestLog` fallback path).

Recommended:
- Keep logs retained long enough for feature baseline.
- Schedule training job at off-peak times (daily minimum for internet-facing apps).

## 10. Rust Enablement

## 10.1 Install
```bash
pip install "aiwaf[rust]"
```

## 10.2 Configure
```python
AIWAF_USE_RUST = True
```

## 10.3 Verify behavior
- App still starts if Rust extension is absent (Python fallback).
- Enable debug logging and confirm Rust-specific paths where expected.

## 11. Production Baseline Checklist
- [ ] Framework extra installed (`[django]`, `[flask]`, `[fastapi]`).
- [ ] Access logs configured and writable/readable where expected.
- [ ] Middleware integrated in intended order.
- [ ] Exempt paths configured for health/static/webhooks.
- [ ] Rate limits tuned for your traffic profile.
- [ ] Header policy tuned to avoid false positives.
- [ ] Training job scheduled.
- [ ] Geo-blocking disabled unless intentionally configured.
- [ ] Observability/log rotation in place.
- [ ] Verified at least one block event and one exemption scenario in staging.

## 12. Validation Playbook

## 12.1 Smoke checks
1. Request a normal page from browser.
2. Request static asset path.
3. Request health endpoint.
4. Confirm none are blocked unexpectedly.

## 12.2 Rate-limit check
Run a burst test against non-exempt endpoint and verify:
- initial responses pass,
- then `429` soft limit,
- optionally `403` hard flood block.

## 12.3 Header check
Send minimal headers request with curl and verify suspicious request handling.

## 12.4 Keyword/anomaly check
Hit obviously malicious-style probe path in staging and verify blocklist behavior/log entry.

## 13. Troubleshooting

## 13.1 Django model app_label / model registration errors
- Ensure `"aiwaf.django"` is in `INSTALLED_APPS`.
- Re-run migrations.

## 13.2 No training activity
- Confirm `AIWAF_ACCESS_LOG` path exists and is readable.
- Confirm log format includes parseable request lines.
- Lower thresholds (`AIWAF_MIN_TRAIN_LOGS`, `AIWAF_MIN_AI_LOGS`) for test/staging.

## 13.3 Too many false positives
- Add safe paths to `AIWAF_EXEMPT_PATHS`.
- Tune `AIWAF_HEADER_QUALITY_MIN_SCORE` down.
- Expand `AIWAF_ALLOWED_PATH_KEYWORDS` and `AIWAF_EXEMPT_KEYWORDS`.
- Use path rules to disable selected middleware for known safe prefixes.

## 13.4 Unexpected missing persistence
- Check selected storage backend and writable directories.
- For Django, verify whether you are in `models` or `csv` mode.
- For Flask/Fast, verify configured data path and process permissions.

## 13.5 Rust seems inactive
- Confirm `pip show aiwaf-rust` returns installed package.
- Confirm `AIWAF_USE_RUST = True`.
- Verify logs for fallback warnings.

## 14. Local Development Setup (Contributors)

## 14.1 Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

## 14.2 Editable install
```bash
pip install -e .
```

Optional extras:
```bash
pip install -e ".[django,flask,fastapi,rust]"
```

## 14.3 Run tests
```bash
pytest -q
```

Framework-targeted subsets:
```bash
pytest -q -m django
pytest -q -m flask
pytest -q -m fast
```

## 15. Quick Minimal Config Snippets

## 15.1 Django minimal
```python
INSTALLED_APPS += ["aiwaf.django"]
MIDDLEWARE += [
    "aiwaf.django.middleware.HeaderValidationMiddleware",
    "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
]
AIWAF_ACCESS_LOG = "/var/log/nginx/access.log"
```

## 15.2 Flask minimal
```python
app.config["AIWAF_USE_CSV"] = True
import aiwaf.flask as aiwaf
aiwaf.register_aiwaf_middlewares(app, middlewares=["ip_keyword_block", "rate_limit"])
```

## 15.3 FastAPI minimal
```python
from aiwaf.fast import AIWAF
aiwaf = AIWAF(app)
```

## 16. Related Documentation
- `README.md` (feature and operational overview)
- `INSTALLATION.md` (Django-focused install steps)
- `AIWAF_SETTINGS_GUIDE.py` (settings quick reference)
- `PROJECT_DETAILED_REFERENCE.md` (full technical architecture)
- `py/aiwaf/core/KEYWORD_FALLBACK_DETAILED.md` (keyword fallback deep dive)

