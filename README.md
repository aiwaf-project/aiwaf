# AIWAF

> A self-learning Web Application Firewall for Python web applications.
> Framework-agnostic core with optional Django, Flask, and FastAPI adapters.

AIWAF provides context-aware protection with rate limiting, anomaly detection, honeypots, UUID tamper protection, smart keyword learning, file-extension probing detection, exempt path/IP awareness, and scheduled retraining.

## Latest Enhancements

- Smart keyword filtering to avoid blocking legitimate paths like `/profile/`
- Granular reset controls for blacklist, keywords, and exemptions
- Context-aware learning that prioritizes suspicious traffic over normal routes
- Enhanced keyword controls via `AIWAF_ALLOWED_PATH_KEYWORDS` and `AIWAF_EXEMPT_KEYWORDS`
- Comprehensive HTTP method validation in honeypot logic
- Enhanced honeypot timing with page expiry/reload flow
- Header validation with quality scoring and bot-pattern detection

---

## Quick Installation

```bash
pip install aiwaf
```

Optional framework extras:

```bash
pip install "aiwaf[django]"
pip install "aiwaf[flask]"
pip install "aiwaf[fastapi]"
pip install "aiwaf[rust]"
```

Rust extra installs the optional `aiwaf-rust` acceleration package.

Important:
- Use the adapter package for your framework (`aiwaf.django`, `aiwaf.flask`, or `aiwaf.fast`).
- For Django setup and command details, see `INSTALLATION.md` and `REPO_GUIDE_DJANGO.md`.

---

## System Requirements

- Python 3.8+
- CPU-only operation (no GPU required)
- Small deployments: ~1 vCPU and ~512 MB RAM
- Moderate deployments: 2 to 4 vCPU and 2 to 4 GB RAM recommended
- For production, schedule detect/train jobs and rotate logs

---

## Package Structure

```text
aiwaf/
  core/                         # framework-agnostic helpers, training, storage abstractions
  core/geolock/ipinfo_lite.mmdb # bundled GeoIP database
  django/                       # Django adapter (middleware, models, trainer, commands)
  flask/                        # Flask adapter (integration class, middleware, CLI helpers)
  fast/                         # FastAPI adapter (middleware, decorators, CLI helpers)
```

Framework entry points:

```python
# Django
import aiwaf.django as aiwaf

# Flask
import aiwaf.flask as aiwaf

# FastAPI
import aiwaf.fast as aiwaf
```

---

## Features

- **IP blocklist**
  - blocks known suspicious sources quickly
  - supports runtime updates through adapter storage

- **Rate limiting**
  - sliding-window request control (`AIWAF_RATE_WINDOW`, `AIWAF_RATE_MAX`)
  - flood threshold support (`AIWAF_RATE_FLOOD`) for aggressive abuse

- **AI anomaly detection**
  - IsolationForest-based behavioral detection
  - model training updates as traffic grows

- **Dynamic keyword learning**
  - learns suspicious path terms from attack-like traffic
  - excludes exempt/allowed terms to reduce false positives

- **File-extension probing detection**
  - detects repeated probes for extensions like `.php`, `.asp`, `.jsp`

- **Header validation**
  - missing required-header detection
  - suspicious user-agent and header-combination checks
  - header quality scoring
  - static-asset exemption support

- **Enhanced honeypot timing**
  - GET to POST timing checks via `AIWAF_MIN_FORM_TIME`
  - page-age validation via `AIWAF_MAX_PAGE_TIME`
  - method-misuse checks (for example POST to read-only endpoints)

- **UUID tamper protection**
  - blocks invalid/guessed UUID access attempts

- **GeoIP support**
  - optional country-level allow/block behavior
  - local bundled MMDB support by default

- **Built-in logging path**
  - adapter-level request logging can feed training when primary access logs are unavailable

- **Blocked-request debug logging**
  - captures reason, IP, method, path, and user-agent in debug mode

---

## Header Validation Details

What it detects:
- missing core browser-like headers
- low-diversity header sets typical of simple bots
- suspicious or automation-focused user agents
- unrealistic header combinations

What it allows:
- normal browser traffic with complete headers
- well-identified clients and known legitimate bots
- static file requests when exempt patterns are configured

Useful test pattern:

```bash
# often low-quality header profile
curl http://your-app.example/

# compare against normal browser traffic
```

---

## Exemptions and Safe Routing

AIWAF supports:
- exempt paths (`AIWAF_EXEMPT_PATHS`)
- exempt IPs (adapter-managed allowlists)
- exempt keywords (`AIWAF_EXEMPT_KEYWORDS`)
- allowed route keywords (`AIWAF_ALLOWED_PATH_KEYWORDS`)

Effects of exemption:
- excluded from keyword learning
- bypass of selected blocking paths
- reduced false positives on trusted operational routes (webhooks, health, static assets)

Decorator-based exemptions:
- Django adapter and Flask adapter both expose exemption decorators in their adapter modules.

---

## Training and Retraining

Training pipeline:
1. Read configured access logs or adapter logger output
2. Detect suspicious patterns (including heavy 404 probe behavior)
3. Train/update IsolationForest when AI thresholds are met
4. Refresh dynamic keywords from suspicious traffic
5. Remove exempt/allowed noise from learned keyword set

Thresholds:
- `AIWAF_MIN_AI_LOGS` default 10,000 for full AI training
- `AIWAF_MIN_TRAIN_LOGS` default 50 for keyword-focused fallback
- `AIWAF_FORCE_AI_TRAINING` can override AI threshold gating

Daily retraining is recommended for active internet-facing workloads.

---

## Configuration (`AIWAF_*`)

AIWAF uses flat `AIWAF_*` settings/config keys.
Some knobs are adapter-specific; core controls are shared.

Required in most deployments:

```python
AIWAF_ACCESS_LOG = "/var/log/nginx/access.log"
```

Core defaults (examples):

```python
AIWAF_DISABLE_AI = False
AIWAF_MIN_AI_LOGS = 10000
AIWAF_MIN_TRAIN_LOGS = 50
AIWAF_FORCE_AI_TRAINING = False
AIWAF_AI_CONTAMINATION = 0.05

AIWAF_RATE_WINDOW = 10
AIWAF_RATE_MAX = 20
AIWAF_RATE_FLOOD = 10
AIWAF_WINDOW_SECONDS = 60

AIWAF_MIN_FORM_TIME = 1.0
AIWAF_MAX_PAGE_TIME = 240
AIWAF_FILE_EXTENSIONS = [".php", ".asp", ".jsp"]

AIWAF_ALLOWED_PATH_KEYWORDS = ["profile", "user", "account", "dashboard"]
AIWAF_EXEMPT_KEYWORDS = ["api", "webhook", "health", "static", "media"]
AIWAF_EXEMPT_PATHS = ["/favicon.ico", "/robots.txt", "/static/", "/health/"]
```

Model storage:

```python
AIWAF_MODEL_PATH = "aiwaf/resources/model.pkl"
AIWAF_MODEL_STORAGE = "file"          # file | db | cache
AIWAF_MODEL_CACHE_KEY = "aiwaf:model"
AIWAF_MODEL_CACHE_TIMEOUT = None
AIWAF_MODEL_STORAGE_FALLBACK = True
```

Header controls:

```python
AIWAF_REQUIRED_HEADERS = None         # list or method->list mapping
AIWAF_HEADER_QUALITY_MIN_SCORE = 3
```

GeoIP:

```python
AIWAF_GEO_BLOCK_ENABLED = False
AIWAF_GEOIP_DB_PATH = "aiwaf/core/geolock/ipinfo_lite.mmdb"
AIWAF_GEO_BLOCK_COUNTRIES = ["CN", "RU"]
AIWAF_GEO_ALLOW_COUNTRIES = []
AIWAF_GEO_CACHE_SECONDS = 3600
AIWAF_GEO_CACHE_PREFIX = "aiwaf:geo:"
```

Rust acceleration:

```python
AIWAF_USE_RUST = False
```

When enabled, AIWAF attempts Rust-backed helpers and falls back to Python automatically.

Legacy compatibility:
- if you still use nested `AIWAF_SETTINGS`, AIWAF maps common keys into flat `AIWAF_*` values at startup.

---

## Middleware Setup

Order matters in all adapters. Put protection middleware early and logging middleware near the end.

Django example order:

```python
MIDDLEWARE = [
    "aiwaf.django.middleware.JsonExceptionMiddleware",
    "aiwaf.django.middleware.GeoBlockMiddleware",
    "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
    "aiwaf.django.middleware.RateLimitMiddleware",
    "aiwaf.django.middleware.AIAnomalyMiddleware",
    "aiwaf.django.middleware.HoneypotTimingMiddleware",
    "aiwaf.django.middleware.UUIDTamperMiddleware",
    "aiwaf.django.middleware.HeaderValidationMiddleware",
    "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware",
]
```

If JSON API clients need JSON 403 bodies, keep `JsonExceptionMiddleware` near the top.

FastAPI quick integration:

```python
from fastapi import FastAPI
from aiwaf.fast import AIWAF

app = FastAPI()

aiwaf = AIWAF(
    app,
    storage={"backend": "memory"},
    header_validation={"enabled": True, "quality_threshold": 3},
    rate_limiting={"enabled": True, "window_seconds": 10, "max_requests": 20},
    logging_middleware={"enabled": True, "log_dir": "aiwaf_logs", "log_format": "json"},
)
```

---

## Operations

Django adapter examples:

```bash
python manage.py detect_and_train
python manage.py regenerate_model
python manage.py aiwaf_reset --keywords --confirm
python manage.py add_ipexemption 203.0.113.10 --reason "trusted integration"
python manage.py add_pathexemption /api/webhooks/ --reason "partner callbacks"
python manage.py aiwaf_logging --status
python manage.py geo_block_country list
python manage.py geo_block_country add US
python manage.py geo_block_country remove US
```

Flask adapter:
- use `aiwaf.flask.AIWAF` for middleware registration
- use `aiwaf.flask.cli.AIWAFManager` for CSV-backed operational tasks

FastAPI adapter:
- use `aiwaf.fast.AIWAF` for middleware registration
- use `aiwaf fast ...` or `aiwaf-fast ...` for CLI operations

### Django Command Reference

Common management commands:

```bash
python manage.py detect_and_train
python manage.py regenerate_model
python manage.py aiwaf_reset --confirm
python manage.py aiwaf_reset --blacklist --confirm
python manage.py aiwaf_reset --keywords --confirm
python manage.py aiwaf_reset --exemptions --confirm
python manage.py add_ipexemption <ip> --reason "optional reason"
python manage.py add_pathexemption /path/prefix/ --reason "optional reason"
python manage.py aiwaf_pathshell
python manage.py aiwaf_logging --status
python manage.py geo_block_country list
python manage.py geo_block_country add US
python manage.py geo_block_country remove US
python manage.py aiwaf_diagnose
```

`aiwaf_pathshell` helpers:

```text
ls                     # list path tree at current node
cd <index|name>        # enter child path node
up / cd ..             # move up
pwd                    # current path prefix
exempt <index|name|.>  # add exemption for selected/current path
exit                   # quit shell
```

### Flask Adapter Reference

Programmatic integration:

```python
from flask import Flask
from aiwaf.flask import AIWAF

app = Flask(__name__)
app.config["AIWAF_USE_RUST"] = True
app.config["AIWAF_GEO_BLOCK_ENABLED"] = False
app.config["AIWAF_MIN_AI_LOGS"] = 10000

aiwaf = AIWAF(
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
```

Optional Flask CLI manager:

```bash
python -m aiwaf.flask.cli list all
python -m aiwaf.flask.cli add whitelist 203.0.113.10
python -m aiwaf.flask.cli add blacklist 203.0.113.99 --reason "manual test"
python -m aiwaf.flask.cli add keyword ../etc/passwd
python -m aiwaf.flask.cli status
```

### FastAPI Adapter Reference

Programmatic integration:

```python
from fastapi import FastAPI
from aiwaf.fast import AIWAF

app = FastAPI()
AIWAF(app)
```

CLI usage:

```bash
aiwaf fast --help
aiwaf-fast --help
```

### Path-Specific Rules

Use path rules to selectively disable middleware or override rate limits without globally weakening protection:

```python
AIWAF_SETTINGS = {
    "PATH_RULES": [
        {
            "PREFIX": "/api/webhooks/",
            "DISABLE": ["HeaderValidationMiddleware"],
            "RATE_LIMIT": {"WINDOW": 60, "MAX": 2000},
        },
        {
            "PREFIX": "/api/public/",
            "RATE_LIMIT": {"WINDOW": 60, "MAX": 500},
        },
    ]
}
```

Rules are matched by path prefix, and the most specific matching rule applies.

### Blocking Behavior

- Default behavior: blocked requests raise `PermissionDenied("blocked")` and return `403`.
- For JSON APIs (Django): `JsonExceptionMiddleware` converts blocked JSON requests into JSON `403` payloads.
- Rate limiting can emit `429` for soft throttling paths while still escalating repeated abuse to blacklist flow.

### Logging and Training Data Sources

AIWAF trainer can pull from:

1. `AIWAF_ACCESS_LOG` (primary, supports rotated/gzipped parsing where applicable)
2. middleware-captured logs (CSV/DB depending on adapter settings)

This enables training even when reverse proxy logs are unavailable.

---

## Sandbox and Benchmarking

The sandbox in `examples/sandbox/` provides:

- `direct` (no AIWAF)
- `protected_django`
- `protected_flask`
- `protected_fastapi`

Run full benchmark:

```bash
cd examples/sandbox
python run-and-compare.py -n 5
```

Generated outputs:

- `results_direct_*.json`
- `results_protected_django_*.json`
- `results_protected_flask_*.json`
- `comparison_modes_*.json`
- `comparison_aggregate_*.json`

Interpretation guidance:

- `direct` should show low/zero block rate for attacks (baseline)
- protected targets should keep normal traffic blocking near `0%`
- compare attack blocked% and median latency across iterations, not single-run averages

---

## Publish Checklist

Before publishing a new package version:

1. run test suites for both adapters
2. validate sandbox comparison (`run-and-compare.py -n 3` minimum)
3. bump package version in `setup.py`
4. build artifacts (`python -m build`)
5. smoke-test wheel install in clean virtualenv
6. verify `README.md` and extras (`django`, `flask`) match actual package behavior

---

## Reset and Recovery

Granular reset (Django adapter):

```bash
python manage.py aiwaf_reset --blacklist
python manage.py aiwaf_reset --keywords
python manage.py aiwaf_reset --exemptions
python manage.py aiwaf_reset --blacklist --keywords
python manage.py aiwaf_reset --confirm
```

Common recovery path for false positives:
1. clear learned keywords
2. add legitimate route terms to `AIWAF_ALLOWED_PATH_KEYWORDS`
3. add never-block terms to `AIWAF_EXEMPT_KEYWORDS`
4. retrain

---

## Troubleshooting

### Legitimate pages blocked

Cause:
- learned keywords included legitimate app vocabulary

Fix:

```bash
python manage.py aiwaf_reset --keywords --confirm
python manage.py detect_and_train
```

Then tune:
- `AIWAF_ALLOWED_PATH_KEYWORDS`
- `AIWAF_EXEMPT_KEYWORDS`

### AI model not training

- verify log path and permissions
- check volume vs `AIWAF_MIN_AI_LOGS` / `AIWAF_MIN_TRAIN_LOGS`
- use `AIWAF_FORCE_AI_TRAINING=True` only when appropriate

### Geo-blocking not active

- verify `AIWAF_GEO_BLOCK_ENABLED=True`
- verify `AIWAF_GEOIP_DB_PATH`
- confirm geo middleware is enabled in your adapter chain

### Rust mode appears inactive

- set `AIWAF_USE_RUST=True`
- verify environment can import Rust extension
- fallback to Python is expected on Rust import/runtime failure

---

## How It Works

| Layer | Purpose |
|---|---|
| Geo blocking | Country-level allow/block filtering |
| IP/keyword block | Known-bad source and keyword defense |
| Rate limiting | Burst/flood control in sliding windows |
| AI anomaly | ML-based behavior outlier detection |
| Honeypot timing | Automation/timing/method misuse checks |
| UUID tamper | Invalid UUID access blocking |
| Header validation | Bot-like header profile detection |
| Request logger | Optional telemetry capture for analysis/training |

---

## Request Lifecycle (Detailed)

For a typical protected request:

1. Request enters adapter middleware chain.
2. Path/view/IP exemption checks run first.
3. Header validation evaluates required headers and quality score.
4. IP/keyword checks apply static + learned rules.
5. Rate limit checks apply window/flood logic.
6. Geo checks apply country allow/block rules (if enabled).
7. AI anomaly evaluates extracted behavior features (if enabled and model available).
8. Honeypot timing/method checks evaluate form timing and method misuse.
9. UUID tamper checks validate UUID lookup behavior where applicable.
10. Optional logger records request/response metadata.

If any blocking stage denies request:
- status is typically `403` (`PermissionDenied("blocked")`)
- JSON APIs can receive JSON-formatted `403` via JSON exception middleware
- some throttle paths may return `429`

---

## Middleware Notes

`IPAndKeywordBlockMiddleware`:
- blocks already-blacklisted IPs quickly
- checks static suspicious keywords and learned dynamic keywords
- supports exempt keywords and allowed-path keyword logic

`RateLimitMiddleware`:
- enforces short-window max request budgets
- can blacklist persistent flooders
- supports path rule overrides

`GeoBlockMiddleware`:
- resolves country from source IP via MMDB
- supports block-list mode and optional allow-list mode
- can cache lookups for performance

`AIAnomalyMiddleware`:
- requires model load + thresholds
- gracefully disables itself when model/deps are unavailable
- can run Python path or Rust-assisted path depending on config/runtime

`HoneypotTimingMiddleware`:
- enforces minimum submit timing
- enforces max page age semantics where enabled
- includes method misuse detection logic

`UUIDTamperMiddleware`:
- guards UUID access patterns
- usually no-op where no UUID model rules apply

`HeaderValidationMiddleware`:
- checks required headers by method
- scores request realism and can block low-quality profiles
- commonly tuned for API/webhook/socket endpoints via `PATH_RULES`

---

## Advanced Configuration Matrix

Traffic controls:

```python
AIWAF_RATE_WINDOW = 10
AIWAF_RATE_MAX = 20
AIWAF_RATE_FLOOD = 10
AIWAF_WINDOW_SECONDS = 60
```

Header validation:

```python
AIWAF_REQUIRED_HEADERS = None
AIWAF_HEADER_QUALITY_MIN_SCORE = 3
AIWAF_MAX_ACCEPT_LENGTH = 4096
```

AI/model behavior:

```python
AIWAF_DISABLE_AI = False
AIWAF_MIN_AI_LOGS = 10000
AIWAF_MIN_TRAIN_LOGS = 50
AIWAF_FORCE_AI_TRAINING = False
AIWAF_AI_CONTAMINATION = 0.05
```

Model storage:

```python
AIWAF_MODEL_STORAGE = "file"      # file | db | cache
AIWAF_MODEL_PATH = "aiwaf/resources/model.pkl"
AIWAF_MODEL_CACHE_KEY = "aiwaf:model"
AIWAF_MODEL_CACHE_TIMEOUT = None
AIWAF_MODEL_STORAGE_FALLBACK = True
```

Keyword and false-positive controls:

```python
AIWAF_ALLOWED_PATH_KEYWORDS = ["profile", "user", "dashboard"]
AIWAF_EXEMPT_KEYWORDS = ["api", "health", "static", "webhook"]
AIWAF_DYNAMIC_TOP_N = 10
```

Exemptions:

```python
AIWAF_EXEMPT_PATHS = ["/health/", "/static/", "/favicon.ico"]
AIWAF_EXEMPT_IPS = ["127.0.0.1", "::1"]
```

---

## Tuning Playbooks

Reduce false positives without globally weakening protection:
1. reset learned keywords (`--keywords`)
2. add legitimate domain terms to `AIWAF_ALLOWED_PATH_KEYWORDS`
3. add operational terms to `AIWAF_EXEMPT_KEYWORDS`
4. add route-level `PATH_RULES` for webhook/socket endpoints
5. retrain and benchmark again

Harden for sustained attack traffic:
1. tune `AIWAF_RATE_WINDOW`, `AIWAF_RATE_MAX`, `AIWAF_RATE_FLOOD`
2. keep header validation enabled for public paths
3. keep geo rules explicit and minimal
4. enable middleware logging + regular retraining
5. review block reasons before adding broad keyword rules

Stabilize real-time paths:
1. keep global protections enabled
2. disable only strict checks on `/socket.io/` or equivalent via `PATH_RULES`
3. keep blacklist logic for non-realtime paths
4. whitelist trusted internal integration IPs when needed

---

## Rust Verification

Enable:

```python
AIWAF_USE_RUST = True
```

Runtime behavior:
- Rust extension available: selected paths use Rust acceleration
- Rust extension unavailable: automatic fallback to Python

Verification checklist:
1. start app with `AIWAF_USE_RUST=True`
2. confirm startup/runtime logs show Rust availability or fallback path
3. benchmark with multiple iterations and compare medians (`run-and-compare.py -n 5`)

---

## Troubleshooting Decision Tree

Blank page but `/` is `200`:
- inspect JS/CSS/API requests for `403`/`4xx`
- check whether client IP was blacklisted
- confirm `PATH_RULES` for socket/static/API paths

Many `403` immediately after one blocked request:
- likely blacklist cascade
- clear blacklist and add targeted exemption/path rule
- avoid disabling all middleware globally

AI anomaly not active:
- verify model is loadable
- verify AI deps are installed
- verify `AIWAF_DISABLE_AI=False`
- verify thresholds (`AIWAF_MIN_AI_LOGS`, `AIWAF_MIN_TRAIN_LOGS`)

Geo-blocking appears inactive:
- confirm middleware enabled
- confirm MMDB path valid
- confirm allow/block lists are configured as intended

---

## Deployment Patterns

### Reverse Proxy + App Server

Typical production path:
1. internet -> CDN/WAF edge (optional)
2. reverse proxy (Nginx/Traefik/Caddy)
3. application server (Django/Flask with AIWAF middleware)
4. app database/cache + model/log storage

Recommended:
- preserve client IP forwarding correctly (`X-Forwarded-For`)
- keep clock synchronization (NTP) for reliable log timing features
- rotate logs and enforce retention limits
- run periodic retraining as a scheduled job

### Multi-Instance Deployments

When running multiple app instances:
- prefer shared storage mode for model artifacts (`db` or centralized cache)
- ensure blacklist/exemption updates propagate consistently
- avoid host-local-only model paths if instances autoscale

### Blue/Green or Rolling Updates

For safer rollout:
1. deploy with conservative thresholds
2. verify block metrics and false-positive ratio
3. gradually tighten controls
4. promote only after stable benchmark + production canary behavior

---

## Observability and KPIs

Track these indicators per adapter:

- **Normal traffic block rate**: target near `0%`
- **Attack traffic block rate**: target high and stable under replay suite
- **P95/P99 response latency**: compare before/after tuning
- **Blacklist churn**: sudden spikes may indicate noisy rules
- **Top block reasons**: helps tune headers/keywords/rate limits
- **Retraining success/failure counts**: detect model pipeline regressions

Minimum dashboard slices:
- by endpoint family (`/api`, `/socket.io`, static assets)
- by source ASN/country (if geo enabled)
- by middleware reason code
- by deployment version

---

## Security Boundaries and Caveats

AIWAF improves application-layer protection but is not a complete security boundary.

Important caveats:
- does not replace secure coding, authz, secrets management, patching, or network controls
- ML anomaly detection is probabilistic and can drift with traffic profile changes
- aggressive keyword/rate settings can cause self-inflicted outages if not staged
- websocket/realtime paths often require explicit path-rule tuning
- allowlists/exemptions should be tightly scoped and periodically reviewed

---

## Contributor Test Strategy

Recommended local validation flow for changes:

1. unit and adapter tests
2. sandbox startup validation (direct + protected targets)
3. replay benchmark with multiple iterations
4. review aggregate detection and latency medians
5. inspect a sample of blocked and allowed requests for regressions

Suggested benchmark command:

```bash
cd examples/sandbox
python run-and-compare.py -n 5
```

Regression gates (example policy):
- no increase in normal-traffic blocking
- no meaningful drop in attack blocked%
- no unexplained latency regressions beyond agreed budget

---

## FAQ

**Why do I see `403` on `curl` but browser works?**  
Header validation can classify low-quality client headers as automated traffic.

**Why did everything start returning `403` suddenly?**  
Likely blacklist cascade after an initial block event; clear blacklist and add targeted path/IP tuning.

**Can I disable one middleware for a single route?**  
Yes, use `AIWAF_SETTINGS["PATH_RULES"]` with `DISABLE` for that prefix.

**Does Rust mode change detection outcomes?**  
It should preserve behavior while improving some execution paths; verify with A/B multi-iteration benchmarks.

**Do I need Django to use AIWAF?**  
No. Core supports both Django and Flask adapters, but some operational commands are Django-specific.

---

## CLI Entry Point

```bash
aiwaf-detect
```

Current behavior:
- dispatches to Django trainer (`aiwaf.django.trainer.train`)
- requires Django adapter availability

---

## Acknowledgements

GeoIP support uses the bundled IPinfo MMDB format for country mapping.

[DigitalOcean](https://www.digitalocean.com/) provides the cloud infrastructure that powers AIWAF development.

---

## License

MIT. See `LICENSE`.
