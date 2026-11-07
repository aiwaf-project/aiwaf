## Unreleased

- Fix log parsers (Python and Rust) to accept PATCH requests, `-` response sizes,
  integer `response-time=` values, and trailing `Nms` times (converted to seconds).
  Training no longer silently drops these valid lines.

# Changelog

## Python 1.1.8 / Rust-Python 0.2.7 / Node.js 1.1.5 / WASM 0.2.5 / Java 1.3.5 — prepared release

- **Security:** Python and Node.js no longer let a client choose its own address with
  `X-Forwarded-For`, which could evade blocks and rate limits or match `AIWAF_EXEMPT_IPS`.
  Forwarded headers now count only from a trusted peer and are walked from the right.
  **Upgrade check:** proxies connecting from public addresses must be added to the new
  `AIWAF_TRUSTED_PROXIES` setting, or all visitors will appear as the proxy.
  `CF-Connecting-IP` and `Forwarded`-style fallbacks are no longer read for the client IP.
- **Security:** Java stops reading `X-Forwarded-For` at a malformed entry instead of
  skipping it, so entries left of it can't be taken as the client address. It then
  falls back to `X-Real-IP` or the peer address.
- Rust and WASM: `validate_headers_with_config` now enforces every required header. It
  previously checked only `User-Agent` and `Accept`, and only `HTTP_`-style names. The
  Django, FastAPI, Flask and Node.js middleware already enforced these headers themselves.
- Add `audit` to the Python, Node.js and Java command lines. It parses source without
  running it and reports where request input can reach dangerous operations. It's
  report-only and doesn't affect request handling.
- Add `AIWAF_INJECTION_RISK_MODE=log` and `audit --manifest`. Saved findings tell AIWAF
  which operation each route's input reaches, and matching requests are logged for that
  operation. It's off by default, never blocks, and logs no request values.
- Java can now read source with the JDK Compiler Tree API. `aiwaf-cli init --source`
  builds a route manifest without starting the application, including for plain Servlet
  applications, and reads method bodies for API, form and login signals. Source analysis
  needs a JDK.
- CI builds `aiwaf-rust` from this repository for Python tests instead of installing it from PyPI.
- Update package versions and matching native dependency requirements for the next release.
- Remove benchmark/profiling utilities and consolidate release documentation into README/CHANGELOG.
- Remove the deprecated Rust floating-point module import.
- Use blocklist/allowlist in Python, JavaScript, Rust/WASM, Java, CLI help and
  website documentation. Retain prior import, method and command aliases.
- Preserve existing CSV filenames, Django database/model identity, stored legacy
  reason tags and native wire compatibility. Accept older configuration exports
  without discarding blocks or exemptions.

## Python 1.1.7 / aiwaf-rust 0.2.6 — native honeypot cache

- Add opt-in Redis and supplied-cache honeypot timing to Flask and FastAPI's
  native pipelines. FastAPI runs cache-backed native checks in its worker thread
  pool. CSV remains the default in both adapters; other policies are unchanged.

- Default Django native honeypot timing to Django's configured cache, with an
  optional `honeypot_cache_alias`. Explicit CSV and native memory remain available.
- Delegate timestamp I/O to Django while Rust owns timing, TTL validation,
  exemptions, and block decisions. Other native policy stores retain CSV behavior.
- Return generic HTTP 503 for cache failures or invalid state. Reject DummyCache
  and configured ignored cache errors; warn when LocMemCache isolates workers.
- Leave expiry cleanup to cache TTL so a stale POST cannot delete a concurrent
  GET refresh. Until the next GET or cache expiry, repeated expired-page POSTs
  continue to return 409. Existing timestamps are not migrated between backends.
