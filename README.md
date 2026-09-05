# AIWAF

> A self-learning Web Application Firewall for Python and JavaScript applications.
> Framework adapters for Django, Flask, FastAPI, Express, Fastify, Hapi, Koa,
> Next.js, NestJS, AdonisJS, and Sails.

AIWAF provides context-aware protection across Python and JavaScript runtimes,
with rate limiting, anomaly detection, honeypots, UUID tamper protection, smart
keyword learning, file-extension probing detection, exempt path/IP awareness,
and scheduled retraining.

This monorepo contains both published packages:

- `py/aiwaf/` and the root Python packaging files publish `aiwaf` to PyPI.
- `js/` publishes `aiwaf` to npm and supports Express, Fastify, Hapi, Koa, Next.js, NestJS, AdonisJS, and Sails.

Install the JavaScript package with `npm install aiwaf`. Its usage, configuration, framework adapters, and operational commands are documented in [JavaScript Package](#javascript-package-aiwaf).

For local JavaScript development, run `npm ci` and `npm test` from `js/`. Releases are independent: push a `python-v*` tag (for example, `python-v1.0.8`) to publish the Python package to PyPI, or push a `js-v*` tag (for example, `js-v1.0.1`) to publish `aiwaf` to npm. Both publish workflows can also be started manually from GitHub Actions.

Run the complete monorepo test suite with `python aiwaf_test.py`. Use `--python-only` or `--js-only` when working on one package.

## Latest Enhancements

- Reputation-based IP blocking with weighted offenses and progressive block durations
- Automatic migration of legacy blacklist CSV formats with CLI tools to inspect, convert, or clear imported entries
- Request payload-field inference for generated route manifests
- Smart keyword filtering to avoid blocking legitimate paths like `/profile/`
- Granular reset controls for blacklist, keywords, and exemptions
- Context-aware learning that prioritizes suspicious traffic over normal routes
- Enhanced keyword controls via `AIWAF_ALLOWED_PATH_KEYWORDS` and `AIWAF_EXEMPT_KEYWORDS`
- Comprehensive HTTP method validation in honeypot logic
- Enhanced honeypot timing with page expiry/reload flow
- Header validation with quality scoring and bot-pattern detection

---

## Quick Installation

Python:

```bash
pip install aiwaf
```

JavaScript:

```bash
npm install aiwaf
```

Optional framework extras:

```bash
pip install "aiwaf[django]"
pip install "aiwaf[flask]"
pip install "aiwaf[fastapi]"
pip install "aiwaf[rust]"
```

Rust extra installs `aiwaf-rust`, which accelerates selected paths and is required
for persisted AI model inference from JSON `IsolationForest` artifacts.

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
py/
  aiwaf/                        # Python package (import name remains `aiwaf`)
    core/                       # framework-agnostic helpers and storage
    django/                     # Django adapter
    flask/                      # Flask adapter
    fast/                       # FastAPI adapter
js/                             # JavaScript package, tests, and framework adapters
tests/                          # Python test suites
examples/                       # Python examples and sandbox tooling
```

Contributor setup, test commands, and release conventions are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).

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
  - tracks reason history, reputation score, offense count, and expiration metadata
  - uses progressive temporary blocks while retaining explicit permanent-block support
  - automatically removes expired entries during blacklist checks

- **Rate limiting**
  - sliding-window request control (`AIWAF_RATE_WINDOW`, `AIWAF_RATE_MAX`)
  - flood threshold support (`AIWAF_RATE_FLOOD`) for aggressive abuse

- **AI anomaly detection**
  - IsolationForest-based behavioral detection
  - model training updates as traffic grows
  - persisted runtime models use JSON-only artifacts; Rust `IsolationForest` is required for saved-model inference

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
  - score-based UUID abuse detection
  - malformed UUIDs add high score and can block immediately
  - valid UUID requests that repeatedly end in `404` increase score
  - score decays on successful UUID requests
  - blocks when per-IP UUID score crosses threshold

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

Model persistence is intentionally JSON-only. AIWAF does not load Python object
model artifacts (`pickle`, `joblib`, or `skops`) because those formats can
execute code during deserialization. Scikit-learn models may be used during a
training run for immediate analysis, but they are not persisted. To persist and
reload runtime AI models, install the Rust package so
`aiwaf_rust.IsolationForest` can save/load JSON state. The old bundled
`model.pkl` artifacts have been
removed; retrain to generate a `model.json` artifact.

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

AIWAF_UUID_SCORE_ENABLED = True
AIWAF_UUID_SCORE_WINDOW_SECONDS = 60
AIWAF_UUID_SCORE_BLOCK_THRESHOLD = 5
AIWAF_UUID_SCORE_MALFORMED_WEIGHT = 5
AIWAF_UUID_SCORE_NOT_FOUND_WEIGHT = 1
AIWAF_UUID_SCORE_SUCCESS_DECAY = 2

AIWAF_ALLOWED_PATH_KEYWORDS = ["profile", "user", "account", "dashboard"]
AIWAF_EXEMPT_KEYWORDS = ["api", "webhook", "health", "static", "media"]
AIWAF_EXEMPT_PATHS = ["/favicon.ico", "/robots.txt", "/static/", "/health/"]
```

Model storage:

```python
AIWAF_MODEL_PATH = "aiwaf/resources/model.json"
AIWAF_MODEL_STORAGE = "file"          # file | db | cache
AIWAF_MODEL_CACHE_KEY = "aiwaf:model"
AIWAF_MODEL_CACHE_TIMEOUT = None
AIWAF_MODEL_STORAGE_FALLBACK = True
```

Only JSON-serializable model artifacts are saved. Python object model artifacts
are rejected by design.

Header controls:

```python
AIWAF_REQUIRED_HEADERS = None         # list or method->list mapping
AIWAF_HEADER_QUALITY_MIN_SCORE = 3
```

GeoIP:

```python
AIWAF_GEO_BLOCK_ENABLED = False
AIWAF_GEOIP_DB_PATH = "py/aiwaf/core/geolock/ipinfo_lite.mmdb"
AIWAF_GEO_BLOCK_COUNTRIES = ["CN", "RU"]
AIWAF_GEO_ALLOW_COUNTRIES = []
AIWAF_GEO_CACHE_SECONDS = 3600
AIWAF_GEO_CACHE_PREFIX = "aiwaf:geo:"
```

Rust acceleration:

```python
AIWAF_RUST_ISOLATION_FOREST = True
```

When `aiwaf_rust` is importable, AIWAF automatically uses its supported
accelerators. No `AIWAF_USE_RUST` setting is required. Python fallback remains
automatic when the package or a particular Rust capability is unavailable.
Persisted AI model loading requires a JSON artifact; the Rust `IsolationForest`
backend provides the supported JSON model state. Pickle-based `model.pkl`
artifacts are no longer shipped or loaded.

Legacy compatibility:
- if you still use nested `AIWAF_SETTINGS`, AIWAF maps common keys into flat `AIWAF_*` values at startup.

---

## Middleware Setup

Order matters in all adapters. Put protection middleware early and logging middleware near the end.

### Unified `all` / `auto` Selection

AIWAF now supports a centralized "enable everything with smart defaults" mode across adapters.

- FastAPI and Flask: pass `middlewares=["all"]` (or `["auto"]`)
- Django: use `"aiwaf.django.middleware.all"` in `MIDDLEWARE`

Auto behavior:
- logging middleware is enabled when `AIWAF_ACCESS_LOG` is missing/empty
- logging middleware is disabled when `AIWAF_ACCESS_LOG` is configured
- geo middleware is enabled when any of these are true:
  - explicit geo enable flag is on
  - static geo block list has countries
  - dynamic geo block store/table has countries

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

Django alias example:

```python
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "aiwaf.django.middleware.all",
]
```

FastAPI quick integration:

```python
from fastapi import FastAPI
from aiwaf.fast import AIWAF

app = FastAPI()

aiwaf = AIWAF(
    app,
    middlewares=["all"],
    storage={"backend": "memory"},
    header_validation={"enabled": True, "quality_threshold": 3},
    rate_limiting={"enabled": True, "window_seconds": 10, "max_requests": 20},
    logging_middleware={"enabled": True, "log_dir": "aiwaf_logs", "log_format": "json"},
)
```

Flask quick integration:

```python
from flask import Flask
from aiwaf.flask import AIWAF

app = Flask(__name__)
aiwaf = AIWAF(app, middlewares=["all"])
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
python -m aiwaf.flask.cli blocked
python -m aiwaf.flask.cli unblock 203.0.113.99
python -m aiwaf.flask.cli clear
```

Blacklist migration commands:

```bash
# Detect the configured backend and upgrade legacy rows
aiwaf flask blacklist migrate --app myapp:app
aiwaf fast blacklist migrate --app myapp:app
aiwaf django blacklist migrate

# Inspect all entries or only legacy imports
python -m aiwaf.flask.cli blacklist list
python -m aiwaf.flask.cli blacklist list --legacy

# Convert legacy permanent entries to temporary blocks (default: 24h)
python -m aiwaf.flask.cli blacklist convert-legacy
python -m aiwaf.flask.cli blacklist convert-legacy --duration 1d

# Remove only legacy-imported entries
python -m aiwaf.flask.cli blacklist clear-legacy
```

Durations accept seconds or an `s`, `m`, `h`, or `d` suffix, such as `900`,
`15m`, `24h`, or `1d`. Old headered and headerless blacklist CSV layouts are
upgraded automatically. Imported legacy rows remain permanent until explicitly
converted or cleared.

#### Upgrading an Existing Blocklist

Back up the database or AIWAF data directory before upgrading. Existing entries
should be preserved as permanent blocks unless you deliberately convert them to
temporary blocks.

| Framework/storage | Schema update | Existing entries |
|---|---|---|
| Django ORM | Run Django migrations, then `aiwaf django blacklist migrate` | Command backfills the new reputation fields |
| Flask SQLAlchemy | Run the application's Alembic/Flask-Migrate migration, then `aiwaf flask blacklist migrate --app ...` | Command backfills the new fields |
| Flask `blacklist.csv` | `aiwaf flask blacklist migrate --app ...` | Imported as permanent; use `convert-legacy` to make them temporary |
| Django/FastAPI shared `csv`, `file`, or `db` runtime backend | `aiwaf django blacklist migrate` or `aiwaf fast blacklist migrate --app ...` | Command upgrades legacy key/value records in place |

Django ORM:

```bash
# Generate and apply the model-column migration in the Django project.
python manage.py makemigrations aiwaf
python manage.py migrate aiwaf

aiwaf django blacklist migrate
# Equivalent:
python manage.py aiwaf_migrate_blacklist
```

The command detects `AIWAF_STORAGE_MODE`. In ORM mode it verifies that the new
columns exist and backfills legacy rows. In CSV mode it upgrades the configured
runtime store. It stops with the exact schema-migration instruction if ORM
columns are still missing.

Flask SQLAlchemy ORM:

```bash
# db.create_all() does not alter an existing table.
flask db migrate -m "add AIWAF blacklist reputation fields"
flask db upgrade
aiwaf flask blacklist migrate --app myapp:app
```

The Flask command loads the application so it can inspect `AIWAF_USE_CSV` and
the initialized SQLAlchemy extension. It migrates CSV directly, or verifies and
backfills the ORM table. If ORM columns are missing, it stops and asks for the
application-owned Alembic/Flask-Migrate migration; AIWAF does not silently alter
an application's relational schema.

Migration is optional for runtime compatibility. Until it is run, Django and
Flask ORM adapters detect the deployed table columns and continue using the
legacy `ip`/`reason` behavior. FastAPI and the shared runtime backends continue
to recognize old reason-only `blocked:*` values as permanent blocks. New
reputation metadata takes effect after migration or when a legacy runtime value
is updated. This fallback prevents a package upgrade from silently unblocking
existing IPs.

For Flask CSV storage, set `AIWAF_DATA_DIR` (or Flask
`AIWAF_DATA_DIR`) to the directory containing the current `blacklist.csv`
before running the migration commands. Reading the file rewrites old layouts
to the current columns. The migration recognizes old headered layouts such as
`ip,reason,added_date,extended_request_info` and `ip,timestamp,reason`, plus
headerless `ip[,reason]` rows.

For Django CSV and FastAPI, the shared runtime backends use key/value storage
(`runtime_store.csv`, JSON/file storage, or the `kv_store` SQLite table), so
there are no ORM columns to add. Keep the configured data path unchanged and
run the matching migration command. For FastAPI without an importable app, pass
the backend explicitly, for example:

```bash
aiwaf fast blacklist migrate --backend csv --storage-path aiwaf_data/runtime_store.csv
aiwaf fast blacklist migrate --backend db --storage-path aiwaf_data/aiwaf_data.db
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

AIWAF can generate a route manifest at `.aiwaf/paths.json` and compile it into
runtime path rules. This is the preferred 1.0 workflow because framework-specific
route extraction happens once during init, not on every request.

Generate a manifest:

```bash
# Django
python manage.py aiwaf init

# Flask
aiwaf flask init --app myapp:app

# FastAPI
aiwaf fast init --app myapp:app

# Unified entrypoint
aiwaf init
aiwaf init --app myapp:app
aiwaf init --framework flask --app myapp:app
aiwaf init --framework django --settings myproject.settings
```

`aiwaf init` auto-detects the framework when exactly one supported framework is
installed. If multiple supported frameworks are installed, pass `--app` for
Flask/FastAPI projects or `--framework` to choose the adapter explicitly.
For Django, run from the project root containing `manage.py`, set
`DJANGO_SETTINGS_MODULE`, or pass `--settings`.

During init, AIWAF uses framework metadata first and then bounded static source
analysis when metadata is missing. It can infer:

- route methods from Flask/FastAPI routers, DRF actions, Django CBVs, decorators, and view source
- auth endpoints from signals such as `authenticate`, `login`, `login_user`, `OAuth2PasswordRequestForm`, and helper calls
- API endpoints from combined signals such as `/api/` paths, DRF `APIView`/`ViewSet`, Flask JSON endpoints, FastAPI route metadata, Pydantic/body models, `request.body`, `request.data`, `request.json`, and JSON content-type expectations
- form endpoints from `request.POST`, `request.form`, Django `Form`/`ModelForm`, `render`, `render_template`, `redirect`, and mixed HTML/JSON response flows
- literal payload field names from form/JSON access such as `payload.get("email")`, `request.form["password"]`, and equivalent aliases
- upload/static/app routes from path and source signals

Detector output is explainable: generated routes can include confidence scores
and the exact signals used for auth/API/form classification. Response type alone
does not force API classification; payload type is preferred. For example, a
contact form that redirects on success and returns `JsonResponse` on validation
failure is classified as `category: "form"`, `response_type: "mixed"`, and
`payload_type: "form"`.

Manifest shape:

```json
{
  "schema_version": "1.0",
  "framework": "flask",
  "context_hash": "sha256...",
  "routes": {
    "/api/users/": {
      "methods": ["GET", "POST"],
      "view": "myapp.users",
      "category": "api",
      "response_type": "json",
      "payload_type": "json",
      "payload_fields": ["email", "message"],
      "auth_required": false,
      "api_confidence": 0.94,
      "api_signals": ["path:/api", "JsonResponse", "request.body"],
      "request_body": true,
      "protections": {
        "rate_limit": {"requests": 120, "window_seconds": 60},
        "api_rate_limit": {"requests": 120, "window_seconds": 60},
        "payload_validation": {"max_body_bytes": 1048576, "max_json_depth": 8},
        "content_type_validation": {"require_valid_content_type": true},
        "honeypot": {"enabled": false}
      }
    },
    "/contact/": {
      "methods": ["GET", "POST"],
      "view": "myapp.contact",
      "category": "form",
      "response_type": "mixed",
      "payload_type": "form",
      "auth_required": false,
      "form_confidence": 0.75,
      "form_signals": ["request.POST", "render", "redirect"],
      "request_body": true,
      "protections": {
        "rate_limit": {"requests": 30, "window_seconds": 60},
        "payload_validation": {"max_body_bytes": 1048576},
        "honeypot": {"enabled": true}
      }
    },
    "/login/": {
      "methods": ["GET", "POST"],
      "view": "accounts.views.login_view",
      "category": "auth",
      "response_type": "html",
      "auth_required": false,
      "auth_action": "login",
      "auth_confidence": 0.9,
      "auth_signals": ["django.contrib.auth.authenticate", "django.contrib.auth.login", "POST"],
      "protections": {
        "rate_limit": {"requests": 30, "window_seconds": 60},
        "honeypot": {"enabled": true}
      }
    }
  }
}
```

Method and category detection is deterministic but still best-effort for highly
dynamic code. For maximum accuracy, use framework method decorators such as
Django `@require_http_methods`, explicit Flask route methods, and FastAPI
`@app.get` / `@app.post` decorators.

You can still define manual path rules to selectively disable middleware or
override rate limits without globally weakening protection:

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
Manual rules are applied before generated manifest rules. Path rules are compiled
and cached. If a running application mutates its rules in place, increment
`AIWAF_ROUTE_PLAN_VERSION` (Flask/Django) or `route_plan_version` (FastAPI) so
cached plans are rebuilt. Replacing the rules list with a new object recompiles
it automatically.

### Blocking Behavior

- Default behavior: blocked requests raise `PermissionDenied("blocked")` and return `403`.
- For JSON APIs (Django): `JsonExceptionMiddleware` converts blocked JSON requests into JSON `403` payloads.
- Rate limiting can emit `429` for soft throttling paths while still escalating repeated abuse to blacklist flow.
- Reputation scores accumulate by reason (for example SQL injection, XSS, scanner, brute-force, rate-limit, honeypot, UUID, header, geo, and keyword events).
- The default reputation threshold is 60. Qualifying blocks progress from 15 minutes to 1 hour and then 24 hours for repeated or high-score abuse.
- Storage backends persist score, offenses, reason history, block/expiry timestamps, duration, permanence, and extended request details.
- Passing a positive duration creates a temporary block; the runtime storage API treats a non-positive duration as an explicit permanent block.

### Rate Limiting Cache (Multi-worker)

By default, Flask and FastAPI rate limiting uses an in-process cache (per worker). For multi-worker / multi-instance
deployments, configure the rate limiter to use Redis so all workers share the same counters.

**Flask**

```python
app.config["AIWAF_RATE_CACHE_BACKEND"] = "redis"
app.config["AIWAF_REDIS_URL"] = "redis://localhost:6379/0"
# Optional (defaults to "aiwaf:rate:")
app.config["AIWAF_RATE_CACHE_KEY_PREFIX"] = "aiwaf:rate:"
```

**FastAPI**

```python
from aiwaf.fast import AIWAF

AIWAF(
    app,
    rate_limiting={
        "enabled": True,
        "cache_backend": "redis",
        "redis_url": "redis://localhost:6379/0",
        "cache_key_prefix": "aiwaf:rate:",  # optional
    },
)
```

Environment variables (both adapters):

```bash
set AIWAF_RATE_CACHE_BACKEND=redis
set AIWAF_REDIS_URL=redis://localhost:6379/0
set AIWAF_RATE_CACHE_KEY_PREFIX=aiwaf:rate:
```

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

1. run the Python and JavaScript test suites
2. validate sandbox comparison (`run-and-compare.py -n 3` minimum)
3. bump the Python version in both `pyproject.toml` and `setup.py`, or the JavaScript version in `js/package.json`
4. build artifacts (`python -m build`)
5. smoke-test wheel install in clean virtualenv
6. verify the README and package metadata match actual behavior

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
- install `aiwaf-rust` for persisted runtime ML inference and JSON model artifacts

### Geo-blocking not active

- verify `AIWAF_GEO_BLOCK_ENABLED=True`
- verify `AIWAF_GEOIP_DB_PATH`
- confirm geo middleware is enabled in your adapter chain

### Rust mode appears inactive

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
| UUID tamper | Score-based malformed UUID + repeated UUID-404 abuse detection |
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
9. UUID tamper checks validate UUID format and apply score-based repeated-miss detection.
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
- uses a persisted JSON model when available, otherwise falls back to heuristic/keyword anomaly behavior
- gracefully disables itself when model/deps are unavailable
- persisted ML inference requires the Rust JSON model backend; scikit-learn is training-time only and `model.pkl` is not supported

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
AIWAF_MODEL_PATH = "aiwaf/resources/model.json"
AIWAF_MODEL_CACHE_KEY = "aiwaf:model"
AIWAF_MODEL_CACHE_TIMEOUT = None
AIWAF_MODEL_STORAGE_FALLBACK = True
```

Do not point `AIWAF_MODEL_PATH` at `pickle`, `joblib`, or `skops` artifacts.
AIWAF loads JSON artifacts only. Fresh installs do not include `model.pkl`;
run training with Rust enabled to create `model.json`.

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

Runtime behavior:
- Rust extension available: selected paths use Rust acceleration
- Rust extension unavailable: automatic fallback to Python

Verification checklist:
1. verify `python -c "import aiwaf_rust"` succeeds
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
- verify model is a JSON artifact
- verify AI deps are installed
- verify `AIWAF_DISABLE_AI=False`
- verify thresholds (`AIWAF_MIN_AI_LOGS`, `AIWAF_MIN_TRAIN_LOGS`)
- install `aiwaf-rust` if you need a persisted runtime ML model

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

**Why is model persistence JSON-only?**  
AIWAF is security middleware, so it avoids Python object deserialization formats
such as `pickle`, `joblib`, and `skops`. Persisted AI models should use the Rust
`IsolationForest` JSON state path. Legacy `model.pkl` files are not loaded and
are no longer bundled.

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

## JavaScript Package (`aiwaf`)

The JavaScript package lives in `js/`. Unless a command explicitly installs the
published package, run commands in this section after `cd js`.

AIWAF-JS is a Node.js/Express Web Application Firewall that combines deterministic protections with anomaly detection and continuous learning. It ships as middleware, a CLI for ops workflows, and an offline trainer for IsolationForest models.
Supported frameworks: Express (native), Fastify, Hapi, Koa, NestJS (Express/Fastify wrappers), Next.js (API route wrapper), AdonisJS, and Sails.js.

### What It Does

- Blocks known bad traffic with static keyword rules and IP blacklisting
- Enforces rate limits with flood detection
- Detects bot-like form abuse using honeypot field checks and timing gates
- Enforces optional method policies (405) and suspicious method usage
- Blocks suspicious UUID probing on route prefixes (with optional existence resolver)
- Learns high-frequency malicious segments as dynamic suspicious keywords
- Runs IsolationForest anomaly checks with recent-behavior analysis
- Supports Redis/custom cache backends with memory fallback
- Optional GeoIP blocking (MMDB) with allow/block lists and dynamic blocklist
- CSV fallback storage when DB is unavailable
- Operational CLI for blacklist, exemptions, geo, request logs, training and diagnostics

### Repository Layout

- `index.js`: package entrypoint
- `lib/wafMiddleware.js`: main middleware orchestration
- `lib/rateLimiter.js`: rate-window and flood logic
- `lib/blacklistManager.js`: blocked IP persistence and operations
- `lib/keywordDetector.js`: static keyword checks
- `lib/dynamicKeyword.js`: in-memory dynamic keyword learning/checking
- `lib/uuidDetector.js`: UUID tamper detection
- `lib/honeypotDetector.js`: honeypot trap detection
- `lib/anomalyDetector.js`: pretrained model loading and anomaly scoring
- `lib/featureUtils.js`: request feature extraction and short-lived caching
- `lib/isolationForest.js`: IsolationForest implementation
- `lib/redisClient.js`: optional Redis client lifecycle
- `lib/headerValidation.js`: header caps, suspicious UA, and header quality scoring
- `lib/geoBlocker.js`: GeoIP allow/block checks + MMDB lookup + cache
- `lib/middlewareLogger.js`: JSONL/CSV/DB request logging
- `lib/*Store.js`: DB/CSV storage adapters (blacklist, exemptions, geo, logs, keywords, models)
- `train.js`: offline model training from access logs
- `resources/model.json`: pretrained anomaly model artifact
- `utils/db.js`: SQLite connection (memory DB in test)
- `test/`: Jest test suite

### Request Processing Flow

1. Initialize module options for rate limiter, keyword detectors, honeypot, UUID checks, and anomaly detector.
2. Resolve client IP (`x-forwarded-for` first, then `req.ip`) and normalized path.
3. Enforce optional method policy (405) if enabled.
4. Block immediately if IP is already in blacklist.
5. Header validation (required headers, suspicious UA, header caps, quality score).
6. Geo checks (allow/block lists + DB-backed blocklist).
7. Honeypot field + timing checks.
8. Rate-limit + flood handling.
9. Static keyword blocking.
10. Dynamic keyword blocking.
11. UUID tamper checks (optional existence resolver).
12. Anomaly detection for unknown routes with recent-behavior analysis.
13. Request logging (JSONL/CSV/DB) and optional dynamic keyword learning on 404s.
14. Allow request through `next()` when no rule triggers.

### Installation

```bash
npm install aiwaf
```

#### Optional WASM Acceleration

AIWAF can use the `aiwaf-wasm` optional dependency for faster IsolationForest scoring and deterministic feature validation.
If the WASM module fails to load, it automatically falls back to the JS implementation.

```bash
npm install aiwaf-wasm
```

### Quick Start

```js
const express = require('express');
const aiwaf = require('aiwaf');

const app = express();
app.use(express.json());

app.use(aiwaf({
  staticKeywords: ['.php', '.env', '.git'],
  dynamicTopN: 10,
  WINDOW_SEC: 10,
  MAX_REQ: 20,
  FLOOD_REQ: 40,
  HONEYPOT_FIELD: 'hp_field',
  uuidRoutePrefix: '/user',
  AIWAF_HEADER_VALIDATION: true,
  AIWAF_METHOD_POLICY_ENABLED: true,
  AIWAF_ALLOWED_METHODS: ['GET', 'POST', 'HEAD', 'OPTIONS']
}));

app.get('/', (req, res) => res.send('Protected'));
app.listen(3000);
```

### Path Rules and Auto Middleware

Use path rules to disable selected AIWAF checks or override rate limits for a route prefix:

```js
app.use(aiwaf({
  AIWAF_PATH_RULES: [
    { PREFIX: '/health/', DISABLE: ['header_validation', 'rate_limit'] },
    { PREFIX: '/api/public/', RATE_LIMIT: { WINDOW: 60, MAX: 300 } }
  ]
}));
```

`DISABLE` accepts canonical names such as `ip_keyword_block`, `rate_limit`, `honeypot`, `header_validation`, `geo_block`, `ai_anomaly`, `uuid_tamper`, and `logging`. Class-style names like `HeaderValidationMiddleware` are also accepted.

AIWAF can also read a generated route manifest from `.aiwaf/paths.json` and compile route protections into path rules. For Express apps, generate one after routes are registered. Route extraction uses a Babel-backed JavaScript AST parser, with a conservative string heuristic fallback, to infer API/form/auth signals from handlers:

```js
const manifest = aiwaf.generateExpressManifest(app, '.aiwaf/paths.json');
```

Other frameworks can use `generateFrameworkManifest(framework, app, output, { routes })`. Express/Sails can be introspected from the app, Hapi can be introspected from `server.table()`, and Fastify/Koa/Next/Nest can also accept explicit route lists:

```js
aiwaf.generateFrameworkManifest('fastify', null, '.aiwaf/paths.json', {
  routes: [
    { method: 'GET', path: '/api/users', handler: usersHandler },
    { method: 'POST', path: '/contact', handler: contactHandler }
  ]
});
```

The CLI can generate a manifest from a JSON route list:

```bash
npm run aiwaf -- manifest --framework express --routes routes.json --output .aiwaf/paths.json
```

The Rust/WASM helper surface is available under `aiwaf.wasm` for advanced integrations:

```js
const aiwaf = require('aiwaf');

const model = await aiwaf.wasm.createIsolationForest({ nTrees: 100 });
const features = await aiwaf.wasm.extractWasmFeatures(records, ['.php', '.env']);
const recent = await aiwaf.wasm.analyzeRecentBehavior(entries, ['.php']);
```

Python-parity helper modules are exposed for advanced integrations:

```js
aiwaf.runtimeUtils.isStaticFile('/assets/app.css');
aiwaf.trainingLogic.isScanningPath('/wp-admin/install.php');
aiwaf.geoPolicy.evaluateGeoPolicy({ country: 'US', allowCountries: ['US'] });
aiwaf.blockResponses.blockedPayload('blocked by AIWAF');
```

Use `auto`/`all` middleware selection to enable the canonical protection set while skipping middleware that is not useful for the current app signals:

```js
app.use(aiwaf.auto({
  AIWAF_DISABLE_MIDDLEWARES: ['geo_block']
}));
// Equivalent:
app.use(aiwaf({ AIWAF_MIDDLEWARES: ['auto'] }));
```

### Fastify Usage

```js
const fastify = require('fastify')({ logger: true });
const aiwaf = require('aiwaf');

fastify.register(aiwaf.fastify, {
  staticKeywords: ['.php', '.env', '.git'],
  dynamicTopN: 10,
  WINDOW_SEC: 10,
  MAX_REQ: 20,
  FLOOD_REQ: 40,
  HONEYPOT_FIELD: 'hp_field'
});

fastify.get('/', async () => 'Protected');
fastify.listen({ port: 3000 });
```

### Hapi Usage

```js
const Hapi = require('@hapi/hapi');
const aiwaf = require('aiwaf');

const server = Hapi.server({ port: 3000 });
await server.register({
  plugin: aiwaf.hapi,
  options: {
    staticKeywords: ['.php', '.env', '.git'],
    dynamicTopN: 10,
    WINDOW_SEC: 10,
    MAX_REQ: 20,
    FLOOD_REQ: 40,
    HONEYPOT_FIELD: 'hp_field'
  }
});

server.route({ method: 'GET', path: '/', handler: () => 'Protected' });
await server.start();
```

### Koa Usage

```js
const Koa = require('koa');
const bodyParser = require('koa-bodyparser');
const aiwaf = require('aiwaf');

const app = new Koa();
app.use(bodyParser());

app.use(aiwaf.koa({
  staticKeywords: ['.php', '.env', '.git'],
  dynamicTopN: 10,
  WINDOW_SEC: 10,
  MAX_REQ: 20,
  FLOOD_REQ: 40,
  HONEYPOT_FIELD: 'hp_field'
}));

app.use(ctx => {
  ctx.body = 'Protected';
});

app.listen(3000);
```

### NestJS (Express) Usage

```ts
import { Module, MiddlewareConsumer, NestModule } from '@nestjs/common';
import aiwaf from 'aiwaf';

@Module({})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(aiwaf.nest({
        staticKeywords: ['.php', '.env', '.git'],
        dynamicTopN: 10,
        WINDOW_SEC: 10,
        MAX_REQ: 20,
        FLOOD_REQ: 40,
        HONEYPOT_FIELD: 'hp_field'
      }))
      .forRoutes('*');
  }
}
```

If you need to guarantee ordering before other middleware/proxies, you can also attach the Express middleware directly in `main.ts`:

```ts
import { NestFactory } from '@nestjs/core';
import aiwaf from 'aiwaf';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.use(aiwaf({
    staticKeywords: ['.php', '.env', '.git'],
    dynamicTopN: 10,
    WINDOW_SEC: 10,
    MAX_REQ: 20,
    FLOOD_REQ: 40,
    HONEYPOT_FIELD: 'hp_field'
  }));
  await app.listen(3000);
}
bootstrap();
```

### NestJS (Fastify) Usage

Use the Fastify plugin when running Nest with `FastifyAdapter`:

```ts
import { NestFactory } from '@nestjs/core';
import { FastifyAdapter } from '@nestjs/platform-fastify';
import aiwaf from 'aiwaf';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, new FastifyAdapter());
  await app.register(aiwaf.fastify, {
    staticKeywords: ['.php', '.env', '.git'],
    dynamicTopN: 10,
    WINDOW_SEC: 10,
    MAX_REQ: 20,
    FLOOD_REQ: 40,
    HONEYPOT_FIELD: 'hp_field'
  });
  await app.listen(3000, '0.0.0.0');
}
bootstrap();
```

### Next.js (API Routes) Usage

Use the `aiwaf.next` helper to wrap a Next.js API route handler.

```ts
import aiwaf from 'aiwaf';

function handler(req, res) {
  res.status(200).json({ ok: true });
}

export default aiwaf.next(handler, {
  staticKeywords: ['.php', '.env', '.git'],
  dynamicTopN: 10,
  WINDOW_SEC: 10,
  MAX_REQ: 20,
  FLOOD_REQ: 40,
  HONEYPOT_FIELD: 'hp_field'
});
```

### AdonisJS Usage

Register the middleware in your Adonis middleware stack:

```ts
import aiwaf from 'aiwaf';

export const middleware = [
  () => aiwaf.adonis({
    staticKeywords: ['.php', '.env', '.git'],
    dynamicTopN: 10,
    WINDOW_SEC: 10,
    MAX_REQ: 20,
    FLOOD_REQ: 40,
    HONEYPOT_FIELD: 'hp_field'
  })
];
```

### Sails.js Usage

Use the Express-compatible middleware in your Sails `config/http.js`:

```js
const aiwaf = require('aiwaf');

module.exports.http = {
  middleware: {
    aiwaf: aiwaf.sails({
      staticKeywords: ['.php', '.env', '.git'],
      dynamicTopN: 10,
      WINDOW_SEC: 10,
      MAX_REQ: 20,
      FLOOD_REQ: 40,
      HONEYPOT_FIELD: 'hp_field'
    })
  }
};
```

### Configuration

#### Core Controls

| Option | Default | Description |
|---|---|---|
| `staticKeywords` | `[]` | Substrings that trigger immediate block + blacklist |
| `dynamicTopN` or `DYNAMIC_TOP_N` | `10` | Frequency threshold for dynamic segment blocking |
| `WINDOW_SEC` | `60` | Time window for rate limiting |
| `MAX_REQ` | `100` | Max allowed requests in window before rate block |
| `FLOOD_REQ` | `200` | Hard threshold that blacklists IP |
| `HONEYPOT_FIELD` | `undefined` | Body field name used as bot trap |
| `uuidRoutePrefix` | `"/user"` | Path prefix monitored for UUID tamper attempts |
| `uuidResolver` | `undefined` | Optional async resolver `(uuid, req) => boolean` for existence checks |
| `cache` | fallback memory cache | Custom cache backend used by limiter/features |
| `nTrees` | `100` | IsolationForest trees when model is initialized in-process |
| `sampleSize` | `256` | IsolationForest sample size |
| `AIWAF_WASM_VALIDATION` | `true` | Enable WASM validation when available (headers, URL, content, recent) |
| `AIWAF_WASM_VALIDATE_RECENT` | `false` | Run WASM recent-behavior validation on recent request logs |

#### Header Validation

| Option | Default | Description |
|---|---|---|
| `AIWAF_HEADER_VALIDATION` | `false` | Enable header validation pipeline |
| `AIWAF_REQUIRED_HEADERS` | `[]` | Required headers array, or `{ DEFAULT, GET, POST }` mapping |
| `AIWAF_HEADER_QUALITY_MIN_SCORE` | `3` | Minimum header quality score |
| `AIWAF_MAX_HEADER_BYTES` | `32768` | Max header bytes before blocking |
| `AIWAF_MAX_HEADER_COUNT` | `100` | Max header count before blocking |
| `AIWAF_MAX_USER_AGENT_LENGTH` | `500` | Max User-Agent length |
| `AIWAF_MAX_ACCEPT_LENGTH` | `4096` | Max Accept header length |
| `AIWAF_BLOCKED_USER_AGENTS` | list | Substring deny list |
| `AIWAF_SUSPICIOUS_USER_AGENTS` | regex list | Regex list for suspicious UA detection |
| `AIWAF_LEGITIMATE_BOTS` | regex list | Regex allow list for legitimate crawlers |

#### Method Policy

| Option | Default | Description |
|---|---|---|
| `AIWAF_METHOD_POLICY_ENABLED` | `false` | Enforce method allowlist (returns 405) |
| `AIWAF_ALLOWED_METHODS` | `['GET','POST','HEAD','OPTIONS']` | Allowed methods when policy enabled |
| `AIWAF_POST_ONLY_SUFFIXES` | `['/create/','/submit/','/upload/','/delete/','/process/']` | GET to these triggers 405 when policy enabled |
| `AIWAF_LOGIN_PATH_PREFIXES` | common login paths | Shorten min form time for login |

#### Middleware Selection / Path Rules

| Option | Default | Description |
|---|---|---|
| `AIWAF_MIDDLEWARES` | all compatible checks | Explicit middleware list, or `['auto']` / `['all']` |
| `AIWAF_DISABLE_MIDDLEWARES` | `[]` | Middleware names to disable after selection |
| `AIWAF_PATH_RULES` | `[]` | Prefix rules with `PREFIX`, `DISABLE`, and optional section overrides |
| `AIWAF_PATH_MANIFEST` | `.aiwaf/paths.json` | Route manifest compiled into additional path rules |
| `AIWAF_ROUTE_PLAN_VERSION` | `0` | User-controlled version marker for route policy changes |

#### Keyword Learning

| Option | Default | Description |
|---|---|---|
| `AIWAF_ENABLE_KEYWORD_LEARNING` | `true` | Enable dynamic keyword learning |
| `AIWAF_DYNAMIC_TOP_N` | `10` | Dynamic keyword learning threshold |
| `AIWAF_EXEMPT_KEYWORDS` | `[]` | Skip these keywords |
| `AIWAF_ALLOWED_PATH_KEYWORDS` | `[]` | Allowlist of path fragments |

#### Model / Training

| Option | Default | Description |
|---|---|---|
| `AIWAF_MIN_TRAIN_LOGS` | `50` | Minimum logs to run training |
| `AIWAF_MIN_AI_LOGS` | `10000` | Minimum logs to train AI model |
| `AIWAF_CLEAR_STATE_ON_START` | `false` | Clear blacklist, request logs, and dynamic keywords on startup |
| `AIWAF_FORCE_AI_TRAINING` | `false` | Force AI training below minimum logs |
| `AIWAF_MODEL_STORAGE` | `file` | `file`, `db`, or `cache` |
| `AIWAF_MODEL_PATH` | `resources/model.json` | Model file path (file backend) |
| `AIWAF_MODEL_STORAGE_FALLBACK` | `file` | Fallback model backend |
| `AIWAF_MODEL_CACHE_KEY` | `aiwaf:model` | Cache key when using cache backend |
| `AIWAF_MODEL_CACHE_TTL` | `0` | Cache TTL in seconds |

#### Geo Blocking

| Option | Default | Description |
|---|---|---|
| `AIWAF_GEO_BLOCK_ENABLED` | `false` | Enable geo blocking |
| `AIWAF_GEO_BLOCK_COUNTRIES` | `[]` | Block list (country codes) |
| `AIWAF_GEO_ALLOW_COUNTRIES` | `[]` | Allow list (country codes) |
| `AIWAF_GEO_MMDB_PATH` | `geolock/ipinfo_lite.mmdb` | MMDB path |
| `AIWAF_GEO_CACHE_SECONDS` | `3600` | Geo cache TTL |
| `AIWAF_GEO_CACHE_PREFIX` | `aiwaf:geo:` | Geo cache key prefix |

#### Logging / Storage

| Option | Default | Description |
|---|---|---|
| `AIWAF_MIDDLEWARE_LOGGING` | `false` | Enable JSONL logging |
| `AIWAF_MIDDLEWARE_LOG_PATH` | `logs/aiwaf-requests.jsonl` | JSONL log path |
| `AIWAF_MIDDLEWARE_LOG_DB` | `false` | Store logs in DB |
| `AIWAF_MIDDLEWARE_LOG_CSV` | `false` | Store logs in CSV |
| `AIWAF_MIDDLEWARE_LOG_CSV_PATH` | `logs/aiwaf-requests.csv` | CSV log path |
| `AIWAF_BLOCKED_IPS_CSV_PATH` | `logs/storage/blocked_ips.csv` | CSV fallback for blocked IPs |
| `AIWAF_IP_EXEMPTIONS_CSV_PATH` | `logs/storage/ip_exemptions.csv` | CSV fallback for IP exemptions |
| `AIWAF_PATH_EXEMPTIONS_CSV_PATH` | `logs/storage/path_exemptions.csv` | CSV fallback for path exemptions |
| `AIWAF_GEO_BLOCKED_COUNTRIES_CSV_PATH` | `logs/storage/geo_blocked_countries.csv` | CSV fallback for geo blocklist |
| `AIWAF_REQUEST_LOGS_CSV_PATH` | `logs/storage/request_logs.csv` | CSV fallback for request logs |
| `AIWAF_DYNAMIC_KEYWORDS_CSV_PATH` | `logs/storage/dynamic_keywords.csv` | CSV fallback for dynamic keywords |

#### Redis / Cache
### Redis and Cache Behavior

- Set `REDIS_URL` (or `AIWAF_REDIS_URL`) to enable Redis connectivity (`lib/redisClient.js`).
- If Redis is unavailable, runtime falls back to in-memory behavior.
- You can inject a custom cache object.

Rate limiter custom cache must implement:

- `lPush(key, value)`
- `expire(key, ttl)`
- `lLen(key)`
- `lRange(key, start, end)`

Feature cache custom backend supports:

- `get(key)`
- `set(key, value, ttl)`

### Geo Blocking (MMDB)

- Put your DB at `geolock/ipinfo_lite.mmdb` (default) or set `AIWAF_GEO_MMDB_PATH`.
- Enable with `AIWAF_GEO_BLOCK_ENABLED: true`.
- Configure `AIWAF_GEO_BLOCK_COUNTRIES` and/or `AIWAF_GEO_ALLOW_COUNTRIES`.
- Install MMDB reader dependency in your app:
  - `npm install maxmind`
- If MMDB is unavailable, the middleware falls back to `x-country-code` header.

### Offline Training

Train a model using access logs:

```bash
AIWAF_ACCESS_LOG=/path/to/access.log npm run train
```

Optional rotated/gz support:

```bash
NODE_LOG_GLOB='/path/to/access.log.*' npm run train
```

Training pipeline in `train.js`:

- Reads raw and rotated (including `.gz`) access logs
- Parses request fields (IP, URI, status, response time, timestamp)
- Builds feature vectors: `[pathLen, kwHits, statusIdx, responseTime, burst, total404]`
- Enforces `AIWAF_MIN_TRAIN_LOGS` and `AIWAF_MIN_AI_LOGS`
- Trains IsolationForest when log volume is sufficient
- Learns dynamic keywords from suspicious 4xx/5xx traffic
- Removes exempt keywords and unblocks exempt IPs
- Writes model artifact to `resources/model.json` with metadata
- Model storage backends:
  - `AIWAF_MODEL_STORAGE`: `file` (default), `db`, `cache`
  - `AIWAF_MODEL_PATH` (file backend)
  - `AIWAF_MODEL_STORAGE_FALLBACK` (fallback backend)
  - `AIWAF_MODEL_CACHE_KEY`, `AIWAF_MODEL_CACHE_TTL` (cache backend)

### Testing

```bash
npm test
```

Current tests cover:

- Header validation (caps, suspicious UA, quality scoring)
- Method policy enforcement
- Geo blocking and MMDB lookup
- Honeypot timing policies
- UUID tamper detection (with resolver)
- Anomaly detection and recent-behavior analysis
- Dynamic keyword learning and trainer behaviors
- CSV/DB fallback storage
- CLI and settings compatibility

### Data and Persistence

#### Reputation-aware blacklist

Blacklist records use the same reputation lifecycle as the Python package. Each
record stores the original reason, accumulated reasons, score, offense count,
block timestamp, expiry, duration, permanent status, and optional redacted
request metadata. Repeated offenses progress from 15-minute blocks to one-hour
and 24-hour blocks. Expired entries are ignored and can be removed explicitly:

```bash
npm run aiwaf -- blacklist cleanup
npm run aiwaf -- blacklist migrate
npm run aiwaf -- blacklist migrate --duration 24h
```

Set `AIWAF_CAPTURE_EXTENDED_REQUEST_INFO=true` to retain redacted request
headers and fingerprints with blacklist decisions. Authorization and cookie
headers are redacted by default.

- Runtime blacklist storage uses SQLite through `utils/db.js`.
- Production DB file defaults to `./aiwaf.sqlite`.
- Test environment uses in-memory SQLite (`NODE_ENV=test`).
- Primary blocked IP table: `blocked_ips`.
- Middleware logging supports JSONL, optional SQLite, and CSV fallback.
- CSV settings:
  - `AIWAF_MIDDLEWARE_LOG_CSV`
  - `AIWAF_MIDDLEWARE_LOG_CSV_PATH`
- Table storage CSV fallbacks are enabled automatically when DB operations fail:
  - `blocked_ips` -> `logs/storage/blocked_ips.csv` (`AIWAF_BLOCKED_IPS_CSV_PATH`)
  - `ip_exemptions` -> `logs/storage/ip_exemptions.csv` (`AIWAF_IP_EXEMPTIONS_CSV_PATH`)
  - `path_exemptions` -> `logs/storage/path_exemptions.csv` (`AIWAF_PATH_EXEMPTIONS_CSV_PATH`)
  - `geo_blocked_countries` -> `logs/storage/geo_blocked_countries.csv` (`AIWAF_GEO_BLOCKED_COUNTRIES_CSV_PATH`)
  - `request_logs` -> `logs/storage/request_logs.csv` (`AIWAF_REQUEST_LOGS_CSV_PATH`)
  - `dynamic_keywords` -> `logs/storage/dynamic_keywords.csv` (`AIWAF_DYNAMIC_KEYWORDS_CSV_PATH`)

### Operational Notes

- Middleware order matters; place AIWAF after body parsers if honeypot checks depend on parsed JSON/form body.
- If no trained model exists or loading fails, anomaly detector fails open.
- Dynamic keyword learning persists to DB/CSV via `dynamicKeywordStore`.
- Multi-instance deployments should use Redis/custom shared cache for limiter consistency.

### Development

```bash
npm install
npm test
npm run train
npm run aiwaf -- help
```

### Operations CLI

```bash
npm run aiwaf -- list blacklist
npm run aiwaf -- list exemptions
npm run aiwaf -- add blacklist 203.0.113.9 "manual block"
npm run aiwaf -- remove blacklist 203.0.113.9
npm run aiwaf -- add ip-exemption 203.0.113.10 "trusted monitor"
npm run aiwaf -- add path-exemption /health "health probes"
npm run aiwaf -- add dynamic-keyword scanner 5
npm run aiwaf -- remove dynamic-keyword scanner
npm run aiwaf -- geo block CN "manual block"
npm run aiwaf -- geo summary
npm run aiwaf -- whois example.com
npm run aiwaf -- diagnose 203.0.113.10
npm run aiwaf -- reset --all --confirm
npm run aiwaf -- pathshell
npm run aiwaf -- status
npm run aiwaf -- stats 5000
npm run aiwaf -- logs analyze 5000
npm run aiwaf -- export aiwaf-export.json
npm run aiwaf -- import aiwaf-export.json
npm run aiwaf -- model info
npm run aiwaf -- model export aiwaf-model.json
npm run aiwaf -- model import aiwaf-model.json
npm run aiwaf -- model clear
```

### Sandbox (OWASP Juice Shop)

The repository includes a runnable sandbox that proxies OWASP Juice Shop behind AIWAF.

```bash
docker compose -f examples/sandbox/docker-compose.yml up --build
```

Fastify proxy is also available on `http://localhost:3002`.

---

## Acknowledgements

GeoIP support uses the bundled IPinfo MMDB format for country mapping.

[DigitalOcean](https://www.digitalocean.com/) provides the cloud infrastructure that powers AIWAF development.

---

## License

MIT. See `LICENSE`.
