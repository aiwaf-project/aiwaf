# AIWAF Repository Guide

This document is a practical, code-oriented guide to the `aiwaf` repository for maintainers and contributors.

## 1. What This Repository Is

AIWAF is a Django-first web application firewall package with a shared core and a Flask integration.
It combines:

- Rule-based protections (IP blacklist, keyword checks, rate limiting, honeypot timing, UUID tamper checks, header validation, optional geo blocking)
- Data-driven behavior analysis (IsolationForest anomaly detection)
- Continuous learning from access logs and/or middleware-captured request logs
- Django-native storage and operations through models and management commands

The package is published as `aiwaf` and includes optional extras for GeoIP and Rust acceleration.

## 2. Top-Level Repository Layout

Key paths:

- `py/aiwaf/`: main Python package code
- `py/aiwaf/core/`: shared, framework-agnostic helpers (exemptions, utils, geoip, logs, training, training_logic, training_features, whois, storage_csv, storage_schema, storage_ops, storage_csv_impl, keyword_fallback, storage_interfaces, model_artifacts) and Rust bindings
- `py/aiwaf/django/`: Django entrypoint (middleware exports)
- `py/aiwaf/flask/`: Flask entrypoint (integration exports)
- `py/aiwaf/django/management/commands/`: Django management commands
- `py/aiwaf/django/resources/model.json`: bundled baseline model artifact
- `py/aiwaf/core/geolock/ipinfo_lite.mmdb`: bundled GeoIP database
- `tests/`: Django-based test suite
- `README.md`: user-facing feature and setup documentation
- `INSTALLATION.md`: step-by-step installation/troubleshooting
- `AIWAF_SETTINGS_GUIDE.py`, `AIWAF_SETTINGS_EXAMPLE.py`: configuration references
- `run_tests.py`, `tests/run_working_tests.py`: helper test runners
- `scripts/`: benchmark/stress helper scripts

## 3. Core Runtime Architecture

### 3.0 Framework Entry Points

Use the unified entrypoints:

```python
# Django
import aiwaf.django as aiwaf

# Flask
import aiwaf.flask as aiwaf
```

### 3.1 Middleware-Centric Protection

Main runtime checks live in `py/aiwaf/django/middleware.py` and are designed to be chained in Django `MIDDLEWARE`.

Important middleware classes:

- `JsonExceptionMiddleware`: converts `PermissionDenied` to JSON 403 for JSON requests
- `GeoBlockMiddleware`: optional country-level blocking (GeoIP-backed)
- `IPAndKeywordBlockMiddleware`: blacklist and keyword logic
- `HeaderValidationMiddleware`: bot/suspicious header checks
- `RateLimitMiddleware`: short-window request throttling/flood handling
- `AIAnomalyMiddleware`: model-based anomaly blocking
- `HoneypotTimingMiddleware`: GET->POST timing + method misuse checks
- `UUIDTamperMiddleware`: blocks guessed/nonexistent UUID accesses

Blocking generally raises `PermissionDenied("blocked")` via internal helper logic.

### 3.2 Training and Learning Pipeline

`py/aiwaf/django/trainer.py` handles feature extraction and model/keyword updates:

- Reads logs from:
  - configured `AIWAF_ACCESS_LOG` (including rotated/gzipped files), or
  - fallback middleware request logs (`RequestLog`)
- Computes feature vectors
- Trains/updates IsolationForest when data thresholds are met
- Learns dynamic suspicious keywords from error-heavy traffic
- Removes exempt/allowed keywords to reduce false positives
- Saves model through `model_store.py`

### 3.3 Storage Abstraction

`py/aiwaf/django/storage.py` provides store wrappers with model-backed persistence and fallback behavior.

It supports operations for:

- Blacklist entries
- IP exemptions
- Path exemptions
- Dynamic keywords
- Feature samples

`py/aiwaf/django/model_store.py` handles model artifact loading/saving with multiple backends:

- `file` (default): filesystem path
- `db`: `AIModelArtifact` table
- `cache`: Django cache key

Optional fallback-to-file behavior is supported.

### 3.4 Optional Rust Backend

`py/aiwaf/core/rust_backend.py` tries to import `aiwaf_rust`. If unavailable, functions return graceful fallbacks.

Rust-assisted functions:

- Header validation
- Feature extraction
- Recent-behavior analysis

Python paths remain the default and always available.

## 4. Data Model (Django ORM)

Defined in `py/aiwaf/django/models.py`:

- `BlacklistEntry`: blocked IPs + reason + optional request context
- `IPExemption`: allowlisted IPs
- `ExemptPath`: path prefix exemptions
- `DynamicKeyword`: learned suspicious keywords
- `FeatureSample`: training feature rows
- `RequestLog`: middleware-captured request telemetry
- `AIModelArtifact`: model binary + metadata for DB model storage
- `GeoBlockedCountry`: dynamic geo-block list

Operationally, these models back almost all runtime decisions and admin tooling.

## 5. Configuration Model

### 5.1 Primary Style

The package expects flat `AIWAF_*` Django settings (for example, rate, model storage, logging, exemptions, header checks, geo settings).

### 5.2 Legacy Compatibility

`py/aiwaf/django/settings_compat.py` maps older nested `AIWAF_SETTINGS` keys into flat settings once at startup. This preserves backward compatibility without overriding explicitly defined modern settings.

### 5.3 High-Impact Settings to Understand First

- Log/training:
  - `AIWAF_ACCESS_LOG`
  - `AIWAF_MIDDLEWARE_LOGGING`
  - `AIWAF_MIN_AI_LOGS`
  - `AIWAF_MIN_TRAIN_LOGS`
  - `AIWAF_FORCE_AI_TRAINING`
- Model storage:
  - `AIWAF_MODEL_STORAGE`
  - `AIWAF_MODEL_PATH`
  - `AIWAF_MODEL_STORAGE_FALLBACK`
  - cache/db keys when applicable
- Request filtering:
  - `AIWAF_EXEMPT_PATHS`
  - `AIWAF_EXEMPT_IPS`
  - `AIWAF_ALLOWED_PATH_KEYWORDS`
  - `AIWAF_EXEMPT_KEYWORDS`
- Abuse controls:
  - `AIWAF_RATE_WINDOW`
  - `AIWAF_RATE_MAX`
  - `AIWAF_RATE_FLOOD`
  - `AIWAF_MIN_FORM_TIME`
  - `AIWAF_MAX_PAGE_TIME`
- Optional modules:
  - `AIWAF_GEO_BLOCK_ENABLED`
  - `AIWAF_GEO_BLOCK_COUNTRIES` / allow list settings
  - `AIWAF_USE_RUST`

## 6. Request Lifecycle (Simplified)

1. Request enters middleware chain.
2. Exemption checks evaluate view/path/IP bypass rules.
3. Header, blacklist/keyword, rate, anomaly, honeypot, UUID checks run per middleware order.
4. A suspicious request is denied (`PermissionDenied` -> 403).
5. Optional logger middleware records request + response metadata to CSV/DB.
6. Offline/cron training (`detect_and_train`) updates dynamic keywords and anomaly model.

## 7. Management Commands

Located in `py/aiwaf/django/management/commands/`:

- Exemptions/paths:
  - `add_exemption`, `add_ipexemption`, `add_pathexemption`, `aiwaf_pathshell`
- Inspection/debug:
  - `aiwaf_list`, `aiwaf_logging`, `aiwaf_diagnose`, `aiwaf_whois`, `diagnose_blocking`, `debug_csv`, `geo_traffic_summary`
- Maintenance:
  - `aiwaf_reset`, `clear_blacklist`, `clear_cache`, `setup_models`
- Training/model:
  - `detect_and_train`, `regenerate_model`
- Geo controls:
  - `geo_block_country`
- Test helpers:
  - `test_exemption`, `test_exemption_fix`

For expected usage patterns and examples, see `README.md` and `INSTALLATION.md`.

## 8. Local Development and Test Workflow

### 8.1 Environment

Project metadata is in `pyproject.toml`:

- Python `>=3.8`
- Django `>=3.2`
- core learning deps (`numpy`, `pandas`, `scikit-learn`)
- optional extras: `learning`, `geoblock`, `rust`

### 8.2 Running Tests

Primary options:

- `python run_tests.py`: runs Django tests using `tests.test_settings`
- `python manage.py test`: direct Django test execution
- `python tests/run_working_tests.py`: curated "known working" subset runner

Test suite coverage includes middleware behavior, storage backends, training logic, settings compatibility, rust fallback/integration behavior, path/method/keyword edge cases, and command-level behavior.

## 9. Notable Design Patterns in This Repo

- Lazy model imports (`apps.ready` checks) to avoid AppRegistry issues at import time
- Defensive optional dependency handling (`try/except ImportError` with feature fallback)
- Best-effort logging/writes in middleware to avoid taking down request handling
- Backward-compatible setting migration from nested config style
- Explicit exemption-first strategy to reduce false positives

## 10. Common Contributor Tasks

### Add a New Middleware Rule

1. Implement logic in `py/aiwaf/django/middleware.py` (or helper module if reusable).
2. Respect exemption checks (`is_exempt`, IP exemptions, path rules).
3. Emit useful debug context through existing logger conventions.
4. Add tests under `tests/` for allow + block + edge conditions.
5. Document settings and ordering impact in `README.md`.

### Add a New Training Feature

1. Update extraction logic in `py/aiwaf/django/trainer.py`.
2. Keep rust fallback parity if feature touches rust-extractable data.
3. Ensure model serialization compatibility (`model_store.py`).
4. Add migration/docs only if schema changes are needed.
5. Add regression tests for low-log and threshold-gated behavior.

### Add a New Management Command

1. Place command in `py/aiwaf/django/management/commands/`.
2. Keep command idempotent where possible.
3. Reuse storage/model abstractions rather than raw ad-hoc queries.
4. Add focused tests for success and failure paths.

## 11. Operational Notes

- Middleware order materially changes protection behavior and false-positive profile.
- If `AIWAF_ACCESS_LOG` is unavailable, enabling `AIWAFLoggerMiddleware` provides in-app training data.
- AI model loading failures are intentionally non-fatal; anomaly detection gracefully disables until retraining.
- Geo-blocking requires valid MMDB access and correct middleware/config enablement.
- Rust mode is optional; Python should remain fully functional.

## 12. Suggested Documentation Improvements (Future)

- Consolidate duplicated README sections ("Running Detection & Training", "How It Works").
- Add a dedicated architecture diagram (runtime flow + data flow).
- Add an explicit command reference table with arguments/examples per command.
- Add a contributor checklist for schema changes + migration expectations.

---

If you are new to this codebase, start in this order:

1. `README.md`
2. `INSTALLATION.md`
3. `py/aiwaf/django/middleware.py`
4. `py/aiwaf/django/trainer.py`
5. `py/aiwaf/django/storage.py` and `py/aiwaf/django/models.py`
6. relevant tests under `tests/` for the area you plan to modify

## 13. README Draft (Can Be Reused in `README.md`)

Use this section as a complete repository README draft for AIWAF.

### 13.1 What AIWAF Does

AIWAF is a Django-native Web Application Firewall package that combines deterministic rules and machine-learning anomaly detection. It protects incoming requests in middleware, logs suspicious behavior, and supports ongoing retraining so protection can adapt over time.

Core outcomes:

- Blocks known bad traffic quickly (blacklist, keywords, headers, method misuse, honeypot timing, UUID tampering)
- Slows and blocks abusive clients (rate limiting + flood logic)
- Detects anomalies with IsolationForest from request-derived features
- Learns suspicious keywords from attack patterns while preserving legitimate route keywords
- Supports optional GeoIP blocking and traffic summaries
- Provides operational controls through Django management commands

### 13.2 Main Components

Package layout (high-level):

- `py/aiwaf/core/`
  - Framework-agnostic Rust bindings and shared helpers (exemptions, utils, geoip, logs, training, training_logic, training_features, whois, storage_csv, storage_schema, storage_ops, storage_csv_impl, keyword_fallback, storage_interfaces, model_artifacts, constants)
- `py/aiwaf/django/`
  - Django entrypoint that exports middleware classes
- `py/aiwaf/flask/`
  - Flask entrypoint that exports the Flask integration
- `py/aiwaf/django/middleware.py`
  - Runtime protection middleware classes:
    - `JsonExceptionMiddleware`
    - `GeoBlockMiddleware`
    - `IPAndKeywordBlockMiddleware`
    - `HeaderValidationMiddleware`
    - `RateLimitMiddleware`
    - `AIAnomalyMiddleware`
    - `HoneypotTimingMiddleware`
    - `UUIDTamperMiddleware`
- `py/aiwaf/django/trainer.py`
  - Offline detection/training logic, feature extraction, dynamic keyword learning, model refresh
- `py/aiwaf/django/storage.py`
  - Persistence wrappers for blacklist/exemptions/keywords/features
- `py/aiwaf/django/model_store.py`
  - Model artifact backend abstraction (`file`, `db`, `cache`)
- `py/aiwaf/django/models.py`
  - ORM models used by middleware, training, and admin/ops commands
- `py/aiwaf/core/rust_backend.py`
  - Optional bridge to Rust extension (`aiwaf_rust`) with safe Python fallback
- `py/aiwaf/django/geoip.py`
  - GeoIP lookup helpers backed by bundled MMDB file
- `py/aiwaf/django/middleware_logger.py`
  - Middleware request logging support for observability and training fallback data
- `py/aiwaf/django/blacklist_manager.py`, `py/aiwaf/django/utils.py`, `py/aiwaf/django/decorators.py`
  - Utility and helper layers used across runtime logic
- `py/aiwaf/django/management/commands/`
  - Operational CLI commands for setup, debug, exemptions, retraining, cleanup

### 13.3 Data and Artifacts

- `py/aiwaf/django/resources/model.json`
  - Bundled baseline model artifact
- `py/aiwaf/core/geolock/ipinfo_lite.mmdb`
  - Local GeoIP database used for country detection
- Database models (`py/aiwaf/django/models.py`):
  - `BlacklistEntry`, `IPExemption`, `ExemptPath`, `DynamicKeyword`, `FeatureSample`, `RequestLog`, `AIModelArtifact`, `GeoBlockedCountry`

### 13.4 Request Processing Flow

1. Request enters configured AIWAF middleware.
2. Exemption checks run first (path/IP/view contexts).
3. Header, keyword/IP, method/timing, rate, UUID, geo, and anomaly checks run based on middleware order.
4. Block decisions raise `PermissionDenied("blocked")`.
5. `JsonExceptionMiddleware` can convert applicable 403 results into JSON responses.
6. Optional logger/trainer workflows record traffic and update model/keywords over time.

### 13.5 Training and Learning

`detect_and_train` and `trainer.py` support:

- Access log parsing (including rotated/gzipped patterns) when `AIWAF_ACCESS_LOG` is configured
- Automatic fallback to `RequestLog` data when access logs are missing/unavailable
- Threshold-aware model training using IsolationForest
- Dynamic keyword learning from suspicious/non-legitimate request contexts
- Exemption-aware cleanup of keywords to reduce false positives

### 13.6 Management Commands (Operational Surface)

Representative commands:

- Setup and reset:
  - `setup_models`
  - `aiwaf_reset`
  - `clear_blacklist`
  - `clear_cache`
- Exemptions:
  - `add_exemption`
  - `add_ipexemption`
  - `add_pathexemption`
  - `aiwaf_pathshell`
- Monitoring and debugging:
  - `aiwaf_list`
  - `aiwaf_logging`
  - `aiwaf_diagnose`
  - `diagnose_blocking`
  - `aiwaf_whois`
  - `geo_traffic_summary`
  - `debug_csv`
- Training:
  - `detect_and_train`
  - `regenerate_model`
- Geo operations:
  - `geo_block_country`

### 13.7 Configuration Model

Primary configuration is through flat Django settings prefixed with `AIWAF_` (for example rate controls, model backend, logging behavior, exempt paths/IPs, geo controls, rust toggle).

Compatibility layer:

- `py/aiwaf/django/settings_compat.py` maps legacy nested `AIWAF_SETTINGS` values into modern flat keys at startup.

### 13.8 The Rust Package (`aiwaf-rust`) and How It Fits

AIWAF includes optional Rust acceleration via the separate package `aiwaf-rust` (imported in Python as `aiwaf_rust`).

Where it is declared:

- `pyproject.toml` optional dependency: `rust = ["aiwaf-rust>=0.1.6"]`
- `setup.py` extras: `rust: ["aiwaf-rust>=0.1.6"]`

How to install with rust support:

```bash
pip install "aiwaf[rust]"
```

Rust-accelerated operations exposed through `py/aiwaf/core/rust_backend.py`:

- `validate_headers(...)`
- `extract_features(...)`
- `extract_features_batch(...)` + `finalize_feature_state(...)` for chunked/stateful extraction when supported
- `analyze_recent_behavior(...)`

Runtime behavior:

- If `aiwaf_rust` is importable and provides expected symbols, AIWAF can use it for faster execution in selected hot paths.
- If rust extension is absent or raises runtime errors, helper functions return `None` and Python paths remain active.
- This keeps rust support optional and non-breaking.

Quick verification snippet:

```python
from aiwaf.core.rust_backend import rust_available
print(rust_available())  # True when aiwaf_rust is installed and importable
```

### 13.9 Dependencies and Compatibility

- Python `>=3.8`
- Django `>=3.2`
- Core libraries: `numpy`, `pandas`, `scikit-learn`
- Networking/ops helpers: `requests`, `python-whois`
- Geo support: `geoip2`
- Optional extras:
  - `learning`
  - `geoblock`
  - `rust`
  - `light`

### 13.10 Testing Coverage

The `tests/` suite covers:

- Middleware blocking/allow logic and edge cases
- Settings compatibility and configuration behavior
- Training/model store paths and thresholds
- Geo controls and reporting
- Rust optional integration/fallback contracts
- Command-level behavior for operations/debug flows

### 13.11 Suggested README Quick Start Block

```bash
pip install aiwaf
# or
pip install "aiwaf[rust]"
```

Then:

1. Add `aiwaf` to `INSTALLED_APPS`.
2. Add AIWAF middleware in your preferred order.
3. Run migrations/setup commands for AIWAF models.
4. Configure key `AIWAF_*` settings (exemptions, rates, logging, model storage).
5. Run `detect_and_train` on a schedule (cron/Celery/worker).
