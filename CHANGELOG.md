# Changelog

## Unreleased

- Add `audit` to the Python, Node.js and Java command lines. It parses source without
  running it and reports where request input can reach dangerous operations. It's
  report-only and doesn't affect request handling.
- **Security:** SQL injection checks now use a tokenizer instead of four signature rules.
  It flags a value only when it breaks out of a SQL value and continues as SQL, which
  catches attacks with no quote such as `1 OR 1=1` and `1 AND SLEEP(5)`, and MySQL's
  executed `/*!...*/` comments, which bypassed the signature rules. It no longer blocks
  ordinary text such as `"Item" #3` or `Drop table tennis; update your set list`.
  **Upgrade check:** rule names change from `sql_union_select`, `sql_boolean_tautology`,
  `sql_quote_comment` and `sql_stacked_statement` to `sql_breakout_numeric`,
  `sql_breakout_single_quote` and `sql_breakout_double_quote`. Text such as
  `union select password` with no value to close is no longer blocked.
- **Requires `aiwaf-rust` 0.2.8.** Python's SQL check and route checks now run in Rust, with
  no Python copy: one call per request does the walking, decoding, JSON parsing, size limits
  and tokenizer. `aiwaf-rust` adds `policy_sql_inspect`, `RouteRiskChecker`,
  `policy_route_rule` and `policy_sql_tokenizer`. `policy_sql_signature` and
  `policy_sql_signatures` remain for older releases and return tokenizer results. A JSON body
  is checked by its values instead of its raw text, and a literal `+` is no longer lost to
  form decoding. Shared case files run against every copy in CI.
- **Node.js requires `aiwaf-wasm` 0.2.6 for request checks.** The SQL check and route checks
  run in `aiwaf-wasm`, from the same Rust code as Python, with no JavaScript copy.
  `aiwaf-wasm` adds `sql_inspect`, `RouteRiskChecker`, `route_rule` and `sql_tokenizer`.
  `package.json` keeps `^0.2.5` until 0.2.6 is published; bump it then. Tests and CI build
  `aiwaf-wasm` from this repository with `scripts/build_wasm_from_source.py`.
- **Security:** SQL detection was measured against the OWASP Core Rule Set and libinjection
  test data, and the gaps found were closed. libinjection detection rose from 69.8% to
  94.9%, and its false-positive list from 9 to 1 flagged value. Payloads now caught include
  `WHERE` straight after the quote, `foo) UNION ALL SELECT`, `-6863 union select`,
  `; DROP TABLE`, `-1 + (SELECT ...)`, MSSQL `WAITFOR` and `EXEC` without a semicolon,
  blind payloads such as `' and 526=527`, names found only in SQL such as `pg_sleep(` and
  `information_schema`, `%u` escapes and typographic quotes, and values in XML bodies.
  Text such as `7" #2 CIRCULAR NEEDLE`, `"IS" STAMP` and `BASE; SET SWAROVSKI` passes.
  `scripts/sql_benchmark.py` runs both test sets in CI with minimum results.
- **Security:** detection of OWASP CRS paranoia level 1 and 2 SQL injection tests rose from
  35% to 71% and from 17% to 42%, and libinjection detection to 98.9%, with no more benign
  values flagged. Newly caught: a value that starts with `AND 2771=1709)`, whole statements
  such as `SELECT * FROM users` and `DROP TABLE users;--`, `NOT LIKE` after the quote,
  `x'|email--`, `a'or'1'='1` without spaces, comments glued to keywords such as
  `SELECT/**/password` and `/*!50000`, PostgreSQL casts such as `::int`, calls such as
  `group_concat(`, `char(113)` and `select(user())`, MSSQL names such as `xp_dirtree`,
  `tempdb` and `declare @x`, Oracle packages such as `utl_http`, and MongoDB operators such
  as `[$ne]`. Code such as `await version()` and `delete Math.pow;` passes. Rust and Java
  have the same rules. Also caught: payloads ending in a null byte, as sqlmap sends,
  `ORDER BY 3631` after a breakout, column aliases such as `) AS t FROM`, `admin';`, `'x'!='y`,
  `in (select * from`, `union select col from`, `@x:=(select`, `ALTER TABLE t ADD`, MySQL's
  `1.e(` number trick, `IF((1=2),...)`, `IIF(`, `ELT(`, `CASE WHEN 1=2 THEN`, SQLite's
  `randomblob(`, PostgreSQL's `lo_import(` and `pg_sleep`, and `foo + (SELECT ... WHERE 1=1)`.
  Sizes such as `18';` and phrases such as `5 as a gift from` pass.
  Code such as `await sleep(10)`, `time.sleep(1)` and `sleep(1000)`, Markdown such as
  `` `'limit'` ``, and prose mentioning `sys.tables` now pass, while Oracle's
  `USER_LOCK.SLEEP(5)` is caught.
- **Security:** the SQL check now also reads the URL path, one segment at a time, as in
  `/post/1 OR 1=1`, and the headers: cookie names and values, the path and query string of
  `Referer`, `X-Original-URL` and `X-Rewrite-URL`, and every other header as one value, such as
  `User-Agent`, `X-Forwarded-For`, `Accept-Language` and apps' own `X-` headers. Headers holding
  credentials, framing or browser-set values (`Authorization`, `Accept`, `Content-*`, `Sec-*`,
  `If-*`, `Host`, CSRF tokens, tracing IDs) aren't checked. This applies in Python, Node.js,
  Java and both native pipelines. `aiwaf-rust` adds `policy_sql_inspect_headers` and
  `policy_sql_inspect_path`, and `aiwaf-wasm` adds `sql_inspect_headers` and
  `sql_inspect_path`. Node.js and Python now pass every header and Rust chooses which to
  check. Shared cases are in `tests/fixtures/sql_header_cases.json` and
  `tests/fixtures/sql_path_cases.json`. Python now also
  inspects XML bodies (`application/xml`, `text/xml`, `+xml`). Java's request walker now
  matches the Rust one: it decodes `%uXXXX` escapes, reads XML bodies by their values, and
  checks the same headers.
- Fuzzing for the SQL tokenizer, request walker and route checks, with cargo-fuzz targets
  in `rust/fuzz` run in CI, and random-input tests in `cargo test`. They found that `` in
  the Rust regexes was slow on long non-ASCII input; it's now ASCII-only, up to eight times
  faster on those inputs.
- `init` audits source and saves route-specific checks into the manifest. Requests to those
  routes are checked for patterns that matter to the operation their input reaches, and
  matches are blocked. `AIWAF_INJECTION_RISK_MODE` is `block` by default, and `log` or
  `off` change that. `init --no-audit` and `audit --manifest` are also available.
- Java can now read source with the JDK Compiler Tree API. `aiwaf-cli init --source`
  builds a route manifest without starting the application, including for plain Servlet
  applications, and reads method bodies for API, form and login signals. Source analysis
  needs a JDK.

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
- CI builds `aiwaf-rust` from this repository for Python tests instead of installing it from PyPI.
- Update package versions and matching native dependency requirements for the next release.
- Remove benchmark/profiling utilities and consolidate release documentation into README/CHANGELOG.
- Remove the deprecated Rust floating-point module import.
- Use blocklist/allowlist in Python, JavaScript, Rust/WASM, Java, CLI help and
  website documentation. Retain prior import, method and command aliases.
- Preserve existing CSV filenames, Django database/model identity, stored legacy
  reason tags and native wire compatibility. Accept older configuration exports
  without discarding blocks or exemptions.
- Fix the middleware CSV log path so only a trailing `.log` becomes `.csv`
  (Django and Flask middleware plus the `aiwaf_logging` and `debug_csv`
  commands). A `.log` elsewhere in the path, such as in a folder name, and a
  non-`.log` extension now keep the configured path; anyone who relied on the
  old rewrite will see CSV logs move to the configured folder and may need to
  move existing files.

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

## Node.js 1.1.4 / Java 1.3.4 — blocklist expiry concurrency fixes

- Make Node blocklist request checks read-only. Database cleanup deletes only
  rows still expired when the delete executes; CSV cleanup checks the latest
  rows under the same cross-process lock used by blocklist writes.
- Reload Java CSV and file storage under a shared JVM and OS file lock for each
  operation. Listing filters expiry without rewriting disk; mutations preserve
  updates from other instances and processes and report persistence failures.
- Verify refreshed blocks survive cleanup and concurrent writers retain all
  unrelated blocks. These fixes are separate from offline training replay.

## Python 1.1.6 / aiwaf-rust 0.2.5 — training replay protection

- Make Flask CLI blocklist listing read-only. Expiry filtering no longer writes
  an older snapshot over blocks concurrently added or refreshed by a trainer.
- Display the canonical block date in CLI output instead of reporting Unknown
  when CSV entries use `added_date` rather than the old `timestamp` field.

- Persist Rust-owned enforcement receipts independently of blocklist expiry and
  manual unblocking. Reusing handled requests cannot extend or recreate a block.
- Apply 404 thresholds and anomaly enforcement to fresh evidence in Flask,
  FastAPI, and Django, while retaining full history for model fitting and learning.
- Serialize trainers with a cross-process file lock and atomically persist
  SHA-256 receipts. Preserve receipts across log rotation and application restarts.
- Require the matching native package; missing capabilities and corrupt receipt
  state fail visibly instead of silently allowing replay.

## Node.js dependency security updates

- Node.js 1.1.4 / WASM 0.2.4 share Rust-owned offline enforcement receipts with
  Python. Historical requests cannot extend or recreate handled blocks; fresh
  404 and anomaly evidence must independently satisfy enforcement thresholds.
- Persist Node receipts across restarts and serialize offline trainers. Report
  missing native APIs, invalid state, and storage failures with a CLI failure.
- Keep Java model training free of blocklist side effects, verified by a new
  runtime-state regression test. File storage concurrency is fixed separately.

- Upgrade Jest, Fastify, and the NestJS test dependencies to patched releases.
- Replace the direct deprecated glob 8 dependency with maintained glob 13.
- Remove broad transitive overrides that forced incompatible parser versions;
  apply targeted fixes for qs and the NYC loader's YAML dependency instead.
- Audit development dependencies as well as production dependencies in npm
  release validation.

## Python 1.1.4 

- Detect the single installed framework for bare CLI commands; require an
  explicit choice when detection is ambiguous. Add the `fastapi` CLI alias.
- Reuse existing project and user CSV directories in the Flask management CLI.
- Standardize framework CSV initialization on shared policy files. Import
  legacy runtime CSV data at startup, preserving expiry times and metadata,
  retaining original sources and backups, and preventing migration replay.
- Preserve keyword frequencies and runtime history in auxiliary metadata.
- Use UTF-8 for Python fallback CSV I/O to match Rust reads and writes.
- Retain `aiwaf-rust>=0.2.3`; no new native API is required for these changes.

## Next releases — ELv2 migration

- Python and Node.js 1.1.2, Rust/Python and WASM 0.2.2, and Java 1.3.3 adopt
  standard Elastic License 2.0 for original project code, with retained MIT
  contributions and separately licensed IPinfo Lite data identified in NOTICE.
- Preserve the earlier MIT permissions and notices; do not retroactively
  replace licenses on previously distributed releases.
- Include LICENSE, NOTICE, and licenses/ in release packaging, and align
  package license metadata and WASM generation with the new terms.
- Synchronize the Python runtime version with its package metadata. The
  internal Rust core crate advances to 0.2.1.

See the [README license section](README.md#license) for version boundaries
and the release automation section for validation requirements.

## Java 1.3.0

- Hardened Redis operation with fail-open/fail-closed policies, bounded
  connection and pool waits, latency/pool diagnostics, non-blocking key scans,
  and state-schema compatibility guards for rolling upgrades. Java 1.3
  explicitly supports standalone Redis and rejects Sentinel/Cluster modes.

## Java 1.2.0

- Added scoped per-engine storage contexts and a Redis backend with atomic,
  distributed rate-limit, honeypot, UUID-score, and anomaly-history state.
- Aligned path, IP, geo-only, and route-level logging exemption behavior with
  Python.
- Added Python-style JSON configuration loading, environment overrides, deep
  merge, validation, saving, and broader Spring property coverage.
- Aligned Java defaults for private-IP exemptions and anomaly detection with
  Python.
- Added Spring Boot servlet auto-configuration with live MVC route, annotation,
  and JPA UUID-field discovery.
- Load generated path manifests at runtime while preserving explicit path-rule
  precedence.
- Replaced the external `mmdblookup` process with the native Java MaxMind DB
  reader and bounded country caching.
- Added Python-compatible reputation weights, metadata, and progressive
  15-minute, 1-hour, and 24-hour default blocks; explicit permanent blocking
  remains available.
- Added Spring property and environment configuration for the new runtime
  behavior.

## 1.0.8

- Require `aiwaf-rust>=0.2.1` for the optional Rust extra.
- Normalize HTTP headers before Rust validation so valid FastAPI requests are not blocked.
- Skip the Rust header validator when FastAPI's Rust path is disabled.

## 1.0.7

- Added reputation-based IP blocking with weighted offenses, progressive
  temporary blocks, expiration handling, and richer stored metadata.
- Added automatic legacy blocklist compatibility and backend-aware migration
  commands for Django, Flask, and FastAPI.
- Added legacy CSV schema detection and safe conversion of imported permanent
  blocks.
- Added request payload-field inference to generated path manifests.
- Enabled installed `aiwaf_rust` capabilities automatically, with Python
  fallback when the extension or a specific capability is unavailable.
