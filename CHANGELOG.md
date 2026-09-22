# Changelog

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
- Added automatic legacy blacklist compatibility and backend-aware migration
  commands for Django, Flask, and FastAPI.
- Added legacy CSV schema detection and safe conversion of imported permanent
  blocks.
- Added request payload-field inference to generated path manifests.
- Enabled installed `aiwaf_rust` capabilities automatically, with Python
  fallback when the extension or a specific capability is unavailable.
