# Changelog

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
