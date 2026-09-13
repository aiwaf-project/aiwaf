# AIWAF Rust Repository Guide

This guide is the canonical reference for developing, testing, packaging, and releasing `aiwaf-rust`.

## 1. What This Repo Is

`aiwaf-rust` is a Rust crate compiled as a Python extension module using PyO3 and built/published with maturin.

- Python package name: `aiwaf-rust`
- Python import name: `aiwaf_rust`
- Rust crate/lib name: `aiwaf_rust`
- Current version (at time of writing): `0.1.0`
- Python support: `>=3.8`
- License: MIT

Core value: provide fast WAF-oriented heuristics for request validation and behavior analysis.

## 2. Repository Structure

```text
.
├── Cargo.toml                 # Rust crate metadata + dependencies
├── pyproject.toml             # Python package metadata + maturin config
├── src/
│   └── lib.rs                 # PyO3 module and all exported functions
├── workflows/
│   ├── publish.yml            # Build/publish workflow variant
│   └── rust-publish.yml       # Build/publish workflow variant
├── README.md                  # Public quickstart + overview
└── LICENSE                    # MIT license
```

## 3. Tech Stack and Tooling

- Rust 2021 edition
- PyO3 (`extension-module` feature)
- maturin (`>=1.6,<2.0`)
- once_cell (for lazily initialized regex sets)
- regex (header and user-agent pattern matching)
- GitHub Actions for wheel/sdist build and PyPI publish

## 4. Local Setup

### Prerequisites

- Rust toolchain (`cargo`, `rustc`)
- Python 3.8+ and `pip`
- Virtual environment recommended

### Quick setup

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -U pip maturin
```

### Build extension into active Python env

```bash
maturin develop
```

This compiles Rust and installs `aiwaf_rust` into the current Python environment.

## 5. API Surface (Python)

All public functions are implemented in `src/lib.rs` and exposed in the `#[pymodule]` block.

### `validate_headers(headers: dict) -> Optional[str]`

Basic header validation using default required headers and default scoring.

- Returns `None` when headers look acceptable.
- Returns a reason string when suspicious.

### `validate_headers_with_config(headers: dict, required_headers: Optional[list[str]], min_score: Optional[int]) -> Optional[str]`

Configurable header validator.

Inputs:
- `headers`: request-like mapping with CGI-style keys (`HTTP_USER_AGENT`, `HTTP_ACCEPT`, etc.)
- `required_headers`: defaults to `["HTTP_USER_AGENT", "HTTP_ACCEPT"]`; pass `[]` to disable required-header checks
- `min_score`: defaults to `3`; pass `<= 0` to disable low-score rejection

Checks include:
- Missing required headers
- Suspicious user-agent patterns (with allowlist for legitimate bots)
- Inconsistent browser/protocol patterns
- Header quality scoring

### `extract_features(records: list[dict], static_keywords: list[str]) -> list[dict]`

Builds feature dictionaries for downstream scoring/modeling.

Required per-record keys:
- `ip: str`
- `path_lower: str`
- `path_len: int`
- `timestamp: float`
- `response_time: float`
- `status_idx: int`
- `kw_check: bool`
- `total_404: int`

Output keys per record:
- `ip`
- `path_len`
- `kw_hits`
- `resp_time`
- `status_idx`
- `burst_count`
- `total_404`

Notes:
- Keyword matching is case-insensitive (`static_keywords` lowercased internally).
- Burst count is number of same-IP requests within the last 10 seconds.

### `analyze_recent_behavior(entries: list[dict], static_keywords: list[str]) -> Optional[dict]`

Analyzes request history to decide if traffic should be blocked.

Required entry keys:
- `path_lower: str`
- `timestamp: float`
- `status: int`
- `kw_check: bool`

Output keys:
- `avg_kw_hits: float`
- `max_404s: int`
- `avg_burst: float`
- `total_requests: int`
- `scanning_404s: int`
- `legitimate_404s: int`
- `should_block: bool`

Logic details:
- Detects scanning-like paths (`wp-admin`, `.env`, traversal markers, etc.)
- Uses keyword hit averages + burst intensity + 404 profile
- Returns `None` when `entries` is empty

## 6. Header/Heuristic Internals

Implemented in `src/lib.rs`.

- `LEGITIMATE_BOTS`: regex allowlist for known crawlers/monitors
- `SUSPICIOUS_UA`: regex list of likely automation clients and malformed UAs
- Scoring model:
  - +2 for `HTTP_USER_AGENT`
  - +2 for `HTTP_ACCEPT`
  - +1 each for selected browser-like headers
  - Additional points for common browser combinations
- Behavior model:
  - 404 counts split into scanning/non-scanning
  - ±10s time-window burst evaluation
  - Rule-based `should_block` threshold logic

## 7. Build, Test, and Packaging

### Rust tests

```bash
cargo test
```

### Build wheels

```bash
maturin build --release --out dist
```

### Build source distribution

```bash
maturin sdist --out dist
```

### Build WASM package (npm)

This repo also ships a WASM package with the same API surface as the Python module.

```bash
# from repo root
cd crates/aiwaf_wasm
wasm-pack build --release --target bundler
```

The npm package name is `aiwaf-wasm`. Use `wasm-pack publish` to publish after validating the build.

### Install from local artifacts

```bash
pip install dist/*.whl
```

## 8. CI/CD and Release

There are two workflow files under `workflows/` with equivalent publish goals.

Both workflows:
- Trigger on `release.published` and `workflow_dispatch`
- Build wheels for:
  - OS: Ubuntu, macOS, Windows
  - Python: 3.8, 3.9, 3.10, 3.11, 3.12
- Build one `sdist`
- Upload artifacts
- Publish to PyPI via `pypa/gh-action-pypi-publish@release/v1` using OIDC (`id-token: write`)

Recommended release flow:
1. Bump version in `Cargo.toml` and `pyproject.toml`.
2. Validate locally (`cargo test`, `maturin build`).
3. Tag and create a GitHub Release.
4. Confirm workflow completes and package appears on PyPI.

## 9. Versioning and Compatibility

Current policy target:
- Keep `aiwaf` and `aiwaf-rust` aligned on `major.minor` where possible.
- Allow independent patch versions for bug fixes.

If versions diverge, document compatibility in `README.md` and release notes.

## 10. Common Development Tasks

### Add new Python-exposed function

1. Implement Rust function in `src/lib.rs` with `#[pyfunction]`.
2. Register it in `#[pymodule]` via `m.add_function(...)`.
3. Rebuild with `maturin develop`.
4. Add README/docs examples and tests.

### Change package metadata

- Python metadata: `pyproject.toml`
- Rust crate metadata: `Cargo.toml`

Keep these consistent for:
- `name`
- `version`
- `description`
- repository URL

## 11. Troubleshooting

### Import fails after code changes

Run:

```bash
maturin develop
```

to rebuild and reinstall the extension in the active venv.

### Missing Rust toolchain

Install via rustup, then verify:

```bash
rustc --version
cargo --version
```

### PyO3/Python mismatch

Ensure the intended Python interpreter is active before `maturin develop`:

```bash
which python
python --version
```

## 12. Repository Hygiene Recommendations

These are not fully implemented yet but are recommended:

- Add unit tests for:
  - user-agent edge cases
  - scoring thresholds
  - behavior block/no-block boundaries
- Add benchmark scripts for high-volume record extraction
- Consolidate to a single publish workflow (`workflows/publish.yml` likely enough)
- Replace placeholder repository URL in `Cargo.toml`

## 13. Security Notes

- This package provides heuristics, not definitive abuse attribution.
- Keep allow/deny patterns reviewed to avoid overblocking.
- Treat `should_block` as one signal in a wider defense pipeline when possible.

## 14. Quick Commands Cheat Sheet

```bash
# setup
python -m venv venv && source venv/bin/activate
pip install -U pip maturin

# dev install
maturin develop

# test
cargo test

# package
maturin build --release --out dist
maturin sdist --out dist
```
