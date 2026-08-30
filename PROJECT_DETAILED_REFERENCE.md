# AIWAF Full Project Deep-Dive Reference

## 1) Purpose and Scope
This document is a full-project technical reference for the AIWAF repository.

It covers:
- System purpose and architecture.
- Execution flows for Django, Flask, and FastAPI adapters.
- Core modules and shared abstractions.
- Storage, learning, and model lifecycle.
- Security controls, exemptions, and path rules.
- CLI and management command surface.
- Test topology and operational artifacts.
- Build/release pipeline and packaging internals.

This is intended to be the single exhaustive orientation doc for engineers working in this repo.

## 2) Project Snapshot
- Package: `aiwaf`
- Version: `0.1.9.7.2` (from `pyproject.toml`/`setup.py`)
- Python requirement: `>=3.8`
- Main frameworks supported:
  - Django (`aiwaf.django`)
  - Flask (`aiwaf.flask`)
  - FastAPI (`aiwaf.fast`)
- Optional Rust acceleration package: `aiwaf-rust`
- Console entry points:
  - `aiwaf`
  - `aiwaf-detect`
  - `aiwaf-fast`

Core dependencies:
- `numpy`, `pandas`, `scikit-learn` (AI/training path)
- `geoip2` (GeoIP)
- `requests`, `python-whois`

## 3) Top-Level Repository Structure
Major top-level items:
- `py/aiwaf/`: Python library source.
- `tests/`: framework-specific and shared tests.
- `examples/`: config/examples and sandbox replay/comparison harness.
- `scripts/`: focused benchmark/stress scripts.
- `README.md`, `INSTALLATION.md`, `REPO_GUIDE_DJANGO.md`: user/operator docs.
- `.github/workflows/`: build/publish workflows.

Operational/runtime artifact dirs often present in local/dev runs:
- `aiwaf_data/`, `aiwaf_logs/`, `test_logs/`, `test_data*`.

## 4) High-Level Architecture
AIWAF is structured as:
- Shared, framework-agnostic core (`py/aiwaf/core/*`).
- Framework adapters (Django/Flask/FastAPI) that apply middleware hooks and use shared core logic.
- Storage abstractions supporting memory/file/CSV/DB modes.
- Learning/training pipelines that build behavior features and optionally train anomaly models.
- Optional Rust backend used for performance-sensitive operations (header validation, feature extraction, behavior analysis, optional Rust IsolationForest model).

Core design intent:
- Safe degradation when optional dependencies are missing.
- Incremental hardening with path/IP exemptions and route-aware heuristics to reduce false positives.
- Portability across Python web frameworks.

## 5) Execution Model by Framework

### 5.1 Django
Primary entry:
- `aiwaf.django` lazily exports middleware classes from `py/aiwaf/django/middleware.py`.

Request pipeline:
- Middleware classes evaluate exemptions and path rules first.
- Blocking actions go through `BlacklistManager` and storage layer.
- AI anomaly middleware caches request behavior and can block with contextual heuristics.
- Trainer reads logs/RequestLog rows and retrains model + keywords.

Storage modes in Django:
- `models` (ORM-backed stores)
- `csv` (runtime CSV adapter)

Model storage for AI artifact:
- `file`, `db`, or `cache` via `py/aiwaf/django/model_store.py`.

### 5.2 Flask
Primary entry:
- `aiwaf.flask` exports integration class `AIWAF` and middleware registration utilities.

Execution style:
- Middleware behavior implemented via `before_request` and `after_request` hooks.
- Exemption decorators (`aiwaf_exempt`, `aiwaf_exempt_from`, `aiwaf_only`, `aiwaf_require_protection`) apply route-level gating.
- Storage can run in DB, CSV, or memory fallback modes.

### 5.3 FastAPI
Primary entry:
- `aiwaf.fast.AIWAF` (alias of core runtime orchestrator in `py/aiwaf/core/runtime_fastapi.py`).

Execution style:
- Starlette `BaseHTTPMiddleware` classes for each protection layer.
- Path-rule and decorator-aware middleware gating via shared runtime decorators.
- Shared runtime storage and blacklist managers used directly.

## 6) Core Shared Layer (`py/aiwaf/core`) File-by-File

### 6.1 Exports and constants
- `__init__.py`: central export surface and lazy alias to FastAPI `AIWAF`.
- `constants.py`: shared status index constants.
- `defaults.py`: default exempt path sets for Django/Flask-like contexts.

### 6.2 Storage primitives
- `storage_schema.py`: canonical CSV filenames and header schema.
- `storage_interfaces.py`: protocol contracts for blacklist/exemption/keyword stores.
- `storage_csv.py`: cross-platform file locking + retry helpers.
- `storage_ops.py`: generic CSV read/write helpers with locking.
- `storage_csv_impl.py`: concrete CSV operations for whitelist/blacklist/keywords/geo/path exemptions.
- `runtime_storage.py`: generic backend abstraction and implementations:
  - `MemoryStorage`
  - `FileStorage` (atomic temp-file replace)
  - `CSVStorage`
  - `DBStorage` (SQLite)
  - plus typed store facades (`ExemptionStore`, `BlacklistStore`, `KeywordStore`, `GeoBlockStore`).

### 6.3 Security logic
- `blacklist.py`: pure decision helpers (`should_block_ip`, `should_unblock_ip`).
- `runtime_blacklist.py`: framework-agnostic blacklist manager with exemption-aware behavior.
- `exemptions.py`: path normalization/matching/path-rule selection/middleware-name normalization.
- `header_validation.py`: canonical header validation logic and heuristics.
- `geoip.py`: GeoIP country/country-name lookups with optional caching hooks.
- `runtime_utils.py`: request/IP helpers, static detection, fingerprinting, allowlist checks, and in-memory rate limiter utility.

### 6.4 Training and model helpers
- `training_logic.py`: malicious-context and scanning heuristics; legitimate keyword defaults.
- `training_features.py`: record normalization and batched Python feature extraction.
- `training.py`: batched helpers and parallel Rust-feature extraction helper.
- `model_artifacts.py`: standardized model artifact payload shape for sklearn and Rust models.

### 6.5 Rust integration
- `rust_backend.py`: optional `aiwaf_rust` bridge:
  - header validation
  - feature extraction (single and chunked)
  - recent-behavior analysis
  - Rust IsolationForest model serialization/deserialization support.

### 6.6 Logging and misc
- `logs.py`: log parsing for combined/common formats + CSV log writing helper.
- `utils.py`: shared route tree and IP extraction helper functions.
- `whois.py`: domain/IP WHOIS helper.
- `keyword_fallback.py`: file-backed fallback keyword counter used by Django model-storage fallback paths.
- `KEYWORD_FALLBACK_DETAILED.md`: subsystem-specific deep dive already added.

## 7) Django Adapter Deep-Dive (`py/aiwaf/django`)

### 7.1 Package and app integration
- `__init__.py`: lazy middleware exports.
- `apps.py`: app config + legacy settings compatibility hook in `ready()`.
- `settings_compat.py`: maps legacy `AIWAF_SETTINGS` nested config into flat `AIWAF_*` settings.

### 7.2 ORM models
`models.py` defines:
- `FeatureSample`: behavior features for training.
- `BlacklistEntry`: blocked IP + reason + optional extended request info.
- `DynamicKeyword`: learned keyword counts.
- `IPExemption`: exempted IP list.
- `ExemptPath`: path exemptions.
- `RequestLog`: middleware-generated request logs.
- `AIModelArtifact`: persisted model bytes + metadata.
- `GeoBlockedCountry`: dynamic geo blocklist.

### 7.3 Storage adapter
`storage.py` provides:
- Model-backed stores for features/blacklist/exemptions/path exemptions/keywords.
- Legacy schema compatibility logic for blacklist table columns.
- CSV-mode adapters wired into shared `core.runtime_storage` for unified runtime behavior.
- Keyword fallback behavior via `KeywordFallbackStore` when DB/model operations fail.

### 7.4 Blacklist and utility layer
- `blacklist_manager.py`: exemption-aware block/is_blocked/unblock operations.
- `utils.py`: Django request IP extraction, exemption checks, path-rule disable logic, rate-limit overrides.
- `geoip.py`: Django cache wrappers around core GeoIP lookup.

### 7.5 Model artifact store
- `model_store.py`: load/save model artifacts from `file`, `db`, or `cache` storage modes with fallback behavior.

### 7.6 Middleware stack (`middleware.py`)
Middleware classes:
- `JsonExceptionMiddleware`: converts `PermissionDenied` to JSON 403 for JSON requests.
- `IPAndKeywordBlockMiddleware`:
  - learns suspicious segments from non-existing paths and malicious context.
  - merges static and dynamic keywords.
  - route-aware filtering against legitimate/exempt keywords.
  - conservative blocking on valid paths; stronger criteria required.
- `RateLimitMiddleware`:
  - cache-backed per-IP timestamps with optional path-rule overrides.
  - flood -> block; soft overflow -> 429.
- `GeoBlockMiddleware`:
  - allow/block mode with static and DB dynamic country list.
- `AIAnomalyMiddleware`:
  - loads model safely; disables AI if insufficient logs.
  - computes per-request feature vector.
  - model anomaly result is post-filtered via behavior heuristics before blocking.
  - keyword learning from 404 non-existent paths in malicious context.
- `HoneypotTimingMiddleware`:
  - GET/POST timing checks.
  - method-acceptance checks for routes.
  - page expiry flow with reload guidance.
- `UUIDTamperMiddleware`:
  - validates UUID path params against app model UUID fields.
- `HeaderValidationMiddleware`:
  - required headers, user-agent heuristics, suspicious combinations, quality score.
  - caps for total bytes, header count, UA/Accept lengths.
  - optional Rust validator path.

### 7.7 Trainer (`trainer.py`)
Key behavior:
- Reads logs from files (including rotated/gz) and CSV; falls back to `RequestLog` model.
- Two-pass analysis:
  - pass 1: aggregate counts (404s, timing, etc.)
  - pass 2: feature extraction + keyword candidate collection.
- Supports Rust feature extraction (chunked when supported) and Python fallback.
- Optional AI model training:
  - sklearn IsolationForest or Rust IsolationForest backend.
  - model persisted through `model_store` with fallback behavior.
- Intelligent anomaly-to-block decisioning using context thresholds.
- Smart keyword learning with legitimate-keyword exclusion and malicious context checks.
- Geo summary reporting helpers for blocked/anomalous IPs.

### 7.8 Django management command surface
Under `py/aiwaf/django/management/commands/`:
- `detect_and_train`: run training/retraining.
- `aiwaf_list`: list blocked/exempt/keyword state.
- `aiwaf_reset`: clear blacklist/exemptions/keywords selectively.
- `add_exemption`, `add_ipexemption`, `add_pathexemption`.
- `geo_block_country`, `geo_traffic_summary`.
- `aiwaf_logging`, `aiwaf_diagnose`, `diagnose_blocking`, `debug_csv`.
- `clear_blacklist`, `clear_cache`.
- `regenerate_model`, `setup_models`.
- `test_exemption`, `test_exemption_fix`.
- `aiwaf_whois`.

## 8) Flask Adapter Deep-Dive (`py/aiwaf/flask`)

### 8.1 Integration and exports
- `__init__.py`: lazy imports for integration class, middleware, decorators, logging, and CLI manager.
- `flask_integration.py`: `AIWAF` orchestration class with middleware registry and enable/disable control.
- `middleware.py`: convenience registration and DB init helpers.

### 8.2 Storage subsystem (`storage.py`)
Supports:
- Database mode (SQLAlchemy models in `db_models.py`) when configured.
- CSV mode via shared `core.storage_csv_impl` (default behavior in many flows).
- Memory fallback.

Tracks:
- Whitelist/blacklist.
- Keywords.
- Geo-blocked countries.
- Path exemptions.

### 8.3 Middleware modules
- `ip_and_keyword_block_middleware.py`: static + learned keyword blocking.
- `rate_limit_middleware.py`: per-IP+path request windows and flood behavior.
- `honeypot_timing_middleware.py`: GET/POST timing checks.
- `header_validation_middleware.py`: shared core header validation with optional Rust fast path.
- `geo_block_middleware.py`: static+dynamic country gating.
- `anomaly_middleware.py`: optional model loading, anomaly decisioning, keyword learning from suspicious 404 contexts.
- `uuid_tamper_middleware.py`: query UUID format checks.
- `logging_middleware.py`: combined/json/csv access logging + analysis utilities.
- `middleware_logger.py`: compact logger feed for training pipelines.

### 8.4 Exemption and route gating
`exemption_decorators.py` provides:
- full exemption (`aiwaf_exempt`), partial exemption (`aiwaf_exempt_from`), allow-only subset (`aiwaf_only`), and required protections (`aiwaf_require_protection`).
- `should_apply_middleware` merges:
  - route decorator metadata
  - runtime exemption state
  - path-rule disable/override entries.

### 8.5 Trainer
`trainer.py` mirrors Django training philosophy:
- log ingestion from access/csv/json sources.
- feature extraction using Rust or Python.
- optional AI model training and anomaly-driven block decisions.
- smart keyword learning with route-awareness and malicious-context checks.

### 8.6 Flask CLI
`flask/cli.py` exposes a broad management surface (used also by FastAPI CLI shim):
- inspect/list state.
- add/remove whitelist/blacklist/keyword/path exemptions/geo blocks.
- export/import style operations.
- route shell helpers.
- WHOIS passthrough support via `whois` command path.

## 9) FastAPI Adapter Deep-Dive (`py/aiwaf/fast` + core runtime)

Fast package mostly re-exports shared runtime implementations from `py/aiwaf/core`.

### 9.1 Runtime orchestrator
- `core/runtime_fastapi.py` `AIWAF` class:
  - initializes config.
  - initializes storage backend.
  - auto-seeds exemption patterns.
  - installs middleware in controlled order.
  - attaches lifespan startup/shutdown logic.

### 9.2 Middleware behavior
- `ip_and_keyword_block_middleware.py`: static and learned keyword gating.
- `rate_limit_middleware.py`: path-aware overrideable rate limits.
- `honeypot_timing_middleware.py`.
- `header_validation.py`: robust header handling, Rust helper integration, quality checks.
- `geo_block_middleware.py`.
- `anomaly_middleware.py`: keyword-centric anomaly heuristics.
- `uuid_tamper_middleware.py`.
- `logging_middleware.py`.

### 9.3 Decorator/path-rule gating
- `fast/decorators.py` re-exports `core/runtime_fastapi_decorators.py`.
- `should_apply_middleware` combines endpoint metadata + path rules + required middleware flags.

### 9.4 CLI
- `fast/cli.py` currently delegates command surface to Flask CLI implementation for operational parity.

## 10) Configuration Model
Two principal configuration styles coexist:
- Flat framework settings/env (`AIWAF_*`) used by modern codepaths.
- Legacy nested `AIWAF_SETTINGS` (mapped in Django by `settings_compat.py`).

Important configuration domains:
- Storage backend/mode.
- AI enable/disable and training thresholds.
- Rate limiting thresholds.
- Exempt paths/IPs/keywords/allowed path keywords.
- Geo blocking behavior.
- Header validation requirements and quality thresholds.
- Model storage backend (`file`/`db`/`cache`) and fallback behavior.
- Rust backend toggle.
- Path rules (`PREFIX` scoped disable/override behavior).

## 11) Storage Semantics and Data Flow

### 11.1 Runtime storage backends
`core/runtime_storage.py` provides key-value semantics with optional TTL:
- `MemoryStorage`: in-process map + periodic cleanup.
- `FileStorage`: JSON file with temp-file atomic replace.
- `CSVStorage`: CSV key-value store.
- `DBStorage`: SQLite-backed key-value store.

### 11.2 Typed store facades
- `ExemptionStore`: explicit IPs + wildcard/CIDR patterns.
- `BlacklistStore`: block/unblock/metadata/stats.
- `KeywordStore`: count-based top keyword retrieval.
- `GeoBlockStore`: dynamic blocked-country set.

### 11.3 Adapter-specific storage deltas
- Django has model-backed canonical stores plus CSV adapter mode and keyword fallback JSON.
- Flask has DB/CSV/memory tri-mode and some count-less keyword behavior in CSV mode.
- FastAPI uses shared runtime stores directly.

## 12) Training and AI Lifecycle
Common lifecycle across Django/Flask:
1. Read and parse logs (combined/common/csv/json/model-backed fallbacks).
2. Build feature records: path length, keyword hits, response time, status index, burst, 404 counts.
3. Optionally use Rust accelerated extraction.
4. Train anomaly model if thresholds/dependencies satisfied.
5. Persist model artifact.
6. Learn suspicious keywords from error traffic with malicious-context filtering.
7. Exclude route-legitimate and explicitly exempt keywords to reduce false positives.

AI execution safeguards:
- model loading can be disabled if logs are insufficient.
- anomaly predictions are post-filtered with behavioral heuristics before hard blocks.
- missing AI dependencies degrade to keyword-based protection.

## 13) Header Validation Subsystem
Shared core (`core/header_validation.py`) defines:
- required header resolution (global or per-method overrides).
- suspicious user-agent and suspicious header combination checks.
- cap checks for header bytes/count/value lengths.
- header quality scoring model.

Adapter implementation details:
- Django has full in-file class implementation with optional Rust call path.
- Flask calls shared validator and optionally Rust path under specific conditions.
- FastAPI normalizes ASGI headers and uses Rust-first fallback-to-Python logic.

## 14) GeoIP and Country Blocking
- Geo database: `py/aiwaf/core/geolock/ipinfo_lite.mmdb`.
- Core functions: country code and country name lookup.
- Adapter wrappers add cache integrations:
  - Django uses `django.core.cache`.
  - Flask uses cache extension when available.
  - FastAPI direct helper uses configured DB path.

Blocking modes:
- allowlist mode: block if country not in allow list.
- blocklist mode: block if country in static list or dynamic DB/runtime list.

## 15) Exemption and Path-Rule System
Exemptions are multi-layered:
- Path exemptions (default + configured + persisted).
- IP exemptions (settings and store).
- Route decorators (Django/Flask/FastAPI equivalents).
- Path rules can disable specific middleware and override rate-limits.

This system is central to balancing security with false-positive reduction.

## 16) Logging and Observability
Logging sources:
- Access/error/aiwaf event logs (Flask/FastAPI logging middleware).
- Django `RequestLog` model and optional middleware CSV logs.
- CSV writer utility in `core/logs.py` using file locking.

Analysis helpers:
- Flask `analyze_access_logs` supports csv/json/combined parsing.
- root helper scripts (`log_analyzer.py`, `analyze_storage.py`, `diagnose_*`) support debugging and operations.

## 17) CLI and Operator Surfaces

### 17.1 Package-level CLI
`aiwaf/cli.py`:
- routes to framework-specific command handlers.
- bare `aiwaf` defaults to `fast` command set.
- `aiwaf-detect` bridges into Django training command path.

### 17.2 Django management commands
Broad operational surface under `py/aiwaf/django/management/commands/` for:
- training/reset/listing.
- diagnostics and cache/blacklist maintenance.
- geo controls and path shell tooling.
- whois and exemption tests.

### 17.3 Flask/Fast command surface
- Flask CLI manager provides storage and security list mutation operations.
- FastAPI CLI currently mirrors Flask CLI behavior.

## 18) Tests and Quality Surface
Test topology:
- Total test files: `168`.
- Django tests: `64`.
- Flask tests: `45`.
- FastAPI tests: `58`.

Themes covered by tests:
- Middleware behavior and integration.
- Storage backends and CSV correctness.
- Config compatibility.
- Header validation robustness.
- Geo/block/exemption behavior.
- Training/model paths and Rust parity checks.
- Packaging and route-rule edge cases.

Pytest config includes framework markers and collection filtering logic to avoid cross-framework import bleed.

## 19) Build, Packaging, and Release
Packaging files:
- `pyproject.toml`: canonical project metadata, deps, extras, scripts.
- `setup.py`: setuptools fallback/compat metadata.
- `MANIFEST.in`: package data inclusion (resources/mmdb/models).

GitHub workflows:
- `.github/workflows/workflow.yml`: build distributions and publish to PyPI (release/workflow_dispatch).
- `.github/workflows/python-publish.yml`: manual publish flow.

## 20) Examples and Sandbox
- `examples/config_example.py`: configuration example.
- `examples/sandbox/`: proxy apps (Django/Flask/FastAPI), docker compose, attack suite and comparison tools.
- Sandbox dir also contains many generated result JSON files from prior benchmark/replay runs.

## 21) Root Utility Scripts
Operational/debug scripts in repo root include:
- `run_tests.py`: local test runner helper.
- `check_rate_limiting_logic.py`: logic diagnostics.
- `diagnose_burst_blocking.py`: burst/blocking diagnostics and fix hints.
- `analyze_storage.py`, `log_analyzer.py`, `analyze_wordpress_attack.py`, `debug_aiwaf.py`.
- `example_usage.py`: integration examples.

## 22) Notable Design Strengths
- Multi-framework support with shared core logic to reduce drift.
- Strong fallback story for missing dependencies and storage/model failures.
- Route/path-aware keyword learning to reduce false positives.
- Path-rule + decorator-based selective protection control.
- Optional Rust acceleration without hard dependency.

## 23) Notable Tradeoffs / Gaps
- Some behavior differs across adapters (e.g., keyword count semantics and middleware depth).
- Mixed maturity level between Django and FastAPI/Flask implementations.
- Legacy compatibility layers increase complexity.
- Runtime caches in middleware are in-process and not distributed by default.
- Some fallback stores (like `keyword_fallback.py`) are intentionally simple and need stronger concurrency hardening for heavy multi-worker traffic.

## 24) Suggested “Where to Start” Paths for Contributors
If you are new to this repo:
1. Read `README.md` and `INSTALLATION.md`.
2. Read this document end-to-end once.
3. Focus on `py/aiwaf/core/*` abstractions first.
4. Choose one adapter to work in (`django`, `flask`, or `fast`) and trace request flow through middleware.
5. Run corresponding framework test subset before/after changes.

## 25) Related Deep-Dive Docs
- `py/aiwaf/core/KEYWORD_FALLBACK_DETAILED.md` (keyword fallback subsystem only).
- `REPO_GUIDE_DJANGO.md` (existing repository guide with Django emphasis).

---

## Appendix A: Directory-Level Inventory (Source Modules)

### A.1 `py/aiwaf/core`
- `__init__.py`
- `blacklist.py`
- `constants.py`
- `defaults.py`
- `exemptions.py`
- `geoip.py`
- `header_validation.py`
- `keyword_fallback.py`
- `logs.py`
- `model_artifacts.py`
- `runtime_blacklist.py`
- `runtime_config.py`
- `runtime_fastapi.py`
- `runtime_fastapi_decorators.py`
- `runtime_storage.py`
- `runtime_utils.py`
- `rust_backend.py`
- `storage_csv.py`
- `storage_csv_impl.py`
- `storage_interfaces.py`
- `storage_ops.py`
- `storage_schema.py`
- `training.py`
- `training_features.py`
- `training_logic.py`
- `utils.py`
- `whois.py`

### A.2 `py/aiwaf/django`
- `__init__.py`
- `apps.py`
- `blacklist_manager.py`
- `decorators.py`
- `geoip.py`
- `middleware.py`
- `middleware_logger.py`
- `model_store.py`
- `models.py`
- `settings_compat.py`
- `storage.py`
- `trainer.py`
- `utils.py`
- `management/commands/*`

### A.3 `py/aiwaf/flask`
- `__init__.py`
- `aiwaf_whois.py`
- `anomaly_middleware.py`
- `apps.py`
- `auto_config.py`
- `blacklist_manager.py`
- `cli.py`
- `db_models.py`
- `decorators.py`
- `exemption_decorators.py`
- `flask_integration.py`
- `geo_block_middleware.py`
- `geoip.py`
- `header_validation_middleware.py`
- `honeypot_timing_middleware.py`
- `ip_and_keyword_block_middleware.py`
- `logging_middleware.py`
- `middleware.py`
- `middleware_logger.py`
- `models.py`
- `rate_limit_middleware.py`
- `rust_backend.py`
- `storage.py`
- `trainer.py`
- `utils.py`
- `uuid_tamper_middleware.py`
- `whois_cli.py`

### A.4 `py/aiwaf/fast`
- `__init__.py`
- `blacklist.py`
- `cli.py`
- `config.py`
- `core.py`
- `decorators.py`
- `geoip.py`
- `rust_backend.py`
- `storage.py`
- `utils.py`
- `middleware/anomaly_middleware.py`
- `middleware/geo_block_middleware.py`
- `middleware/header_validation.py`
- `middleware/honeypot_timing_middleware.py`
- `middleware/ip_and_keyword_block_middleware.py`
- `middleware/logging_middleware.py`
- `middleware/rate_limit_middleware.py`
- `middleware/uuid_tamper_middleware.py`
