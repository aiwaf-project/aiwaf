# aiwaf-rust Comprehensive Setup Guide

## 1. What This Guide Covers
This guide is the end-to-end setup reference for this repository.

It includes:
- Local environment setup for Rust, Python (PyO3), and WASM.
- Function-level API reference for exported Python and WASM bindings.
- Isolation Forest behavior and state model details.
- Integration patterns for Python services and JS/browser runtimes.
- Build/test/release commands used in this repo.

Official package pages:
- npm (`aiwaf-wasm`): https://www.npmjs.com/package/aiwaf-wasm
- PyPI (`aiwaf-rust`): https://pypi.org/project/aiwaf-rust/

## 2. Repository Architecture
- `crates/aiwaf_core`: shared Rust logic (header validation, feature extraction, behavior analysis, Isolation Forest).
- `src/lib.rs`: PyO3 bindings exposed as Python module `aiwaf_rust`.
- `crates/aiwaf_wasm/src/lib.rs`: `wasm-bindgen` bindings exposed as npm package `aiwaf-wasm`.
- `tests/test_python_api.py`: Python API contract tests.
- `crates/aiwaf_wasm/tests/wasm_api.rs`: WASM API contract tests.

Data flow is:
1. Host runtime (Python or JS) calls binding function.
2. Binding validates/converts runtime data into Rust structs.
3. `aiwaf_core` executes logic.
4. Binding converts Rust output back to runtime-native objects.

## 3. Prerequisites

### 3.1 Core tools
- Rust stable (`rustup`, `cargo`).
- Python 3.8+.
- Node.js 18+ (recommended for WASM package workflows).

### 3.2 Python toolchain
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip maturin pytest
```

### 3.3 WASM toolchain
```bash
cargo install wasm-pack
```

## 4. Quick Start

### 4.1 Python local build/install (PyO3 via maturin)
```bash
maturin develop
python -c "import aiwaf_rust; print(aiwaf_rust.__doc__)"
```

### 4.2 Run Python API tests
```bash
pytest tests/test_python_api.py -q
```

### 4.3 Build WASM package
```bash
cd crates/aiwaf_wasm
wasm-pack build --release --target bundler
cd ../..
```

### 4.4 Run WASM tests
```bash
cd crates/aiwaf_wasm
wasm-pack test --node
cd ../..
```

## 5. PyO3 (Python) Binding: Specific Functions
The Python module is defined in `src/lib.rs` with `#[pymodule] fn aiwaf_rust(...)`.

### 5.1 Header validation
- `validate_headers(headers: dict) -> Optional[str]`
- `validate_headers_with_config(headers: dict, required_headers: Optional[list[str]], min_score: Optional[int]) -> Optional[str]`

Input expectation:
- Keys are currently matched against server-style names used by core (`HTTP_USER_AGENT`, `HTTP_ACCEPT`, etc.).
- Return value is `None` when valid, or a reason string when suspicious.

### 5.2 Feature extraction
- `extract_features(records: list[dict], static_keywords: list[str]) -> list[dict]`
- `extract_features_batch_with_state(records: list[dict], static_keywords: list[str], state: Optional[dict]) -> dict`
- `finalize_feature_state(static_keywords: list[str], state: Optional[dict]) -> dict`

Required `records[*]` fields:
- `ip: str`
- `path_lower: str`
- `path_len: int`
- `timestamp: float`
- `response_time: float`
- `status_idx: int`
- `kw_check: bool`
- `total_404: int`

State contract (`extract_features_batch_with_state`):
- Input state shape: `{"timestamps_by_ip": {"<ip>": [<timestamp>, ...]}}`
- Output shape: `{"features": [...], "state": {...}}`

`finalize_feature_state(...)` returns a feature-batch wrapper with an empty feature list. The bound Python signature accepts extra args, but current implementation ignores them and returns reset/empty output.

### 5.3 Recent behavior analysis
- `analyze_recent_behavior(entries: list[dict], static_keywords: list[str]) -> Optional[dict]`

Required `entries[*]` fields:
- `path_lower: str`
- `timestamp: float`
- `status: int`
- `kw_check: bool`

Output fields:
- `avg_kw_hits`, `max_404s`, `avg_burst`, `total_requests`, `scanning_404s`, `legitimate_404s`, `should_block`.

### 5.4 IsolationForest class (PyO3)
Constructor (keyword args):
- `n_estimators=100`
- `max_samples=None` (`"auto" | int | float`)
- `contamination=None` (`"auto" | float in (0, 0.5]`)
- `max_features=None` (`int | float`)
- `bootstrap=False`
- `n_jobs=None` (accepted, ignored in Rust implementation)
- `random_state=None`
- `verbose=0`
- `warm_start=False`

Methods:
- `fit(data: list[list[float]]) -> None`
- `retrain(data: list[list[float]]) -> None`
- `anomaly_score(point: list[float]) -> float`
- `is_anomaly(point: list[float], thresh: float = 0.5) -> bool`
- `score_samples(data: list[list[float]]) -> list[float]`
- `decision_function(data: list[list[float]]) -> list[float]`
- `predict(data: list[list[float]]) -> list[int]`
- `to_json() -> dict`
- `IsolationForest.from_json(state: dict) -> IsolationForest`

## 6. Isolation Forest Internals and Semantics
Core implementation is in `crates/aiwaf_core/src/lib.rs`.

Training and scoring behavior:
- Trees are randomly split per feature subset with variance-gain selection.
- `fit` resets trees unless `warm_start=True` and trees already exist.
- `retrain` temporarily forces warm-start append behavior.
- `anomaly_score` is the raw anomaly score (`higher` means more anomalous).
- `score_samples` matches sklearn-style direction (`higher` means more normal).
- `decision_function = score_samples - offset_`.
- `predict` returns `1` (inlier) and `-1` (outlier).

Contamination behavior:
- `"auto"` keeps default `offset_ = -0.5`.
- Fixed contamination computes percentile-based offset from training scores.

## 7. WASM Binding: Specific Functions
WASM exports are in `crates/aiwaf_wasm/src/lib.rs`.

Top-level functions:
- `validate_headers(headers)`
- `validate_headers_with_config(headers, required_headers, min_score)`
- `extract_features(records, static_keywords)`
- `extract_features_batch_with_state(records, static_keywords, state)`
- `finalize_feature_state()`
- `analyze_recent_behavior(entries, static_keywords)`

`IsolationForest` class exports:
- `new(config?)`
- `fit(data)`
- `retrain(data)`
- `anomaly_score(point)`
- `is_anomaly(point, thresh?)`
- `score_samples(data)`
- `decision_function(data)`
- `predict(data)`
- `to_json()`
- `IsolationForest.from_json(state)`

Header handling differences vs Python:
- WASM accepts plain JS object or `Headers`.
- WASM auto-fills `user-agent` from `navigator.userAgent` if absent.
- Core checks still use `HTTP_*` style names internally, so keep an eye on header key normalization when integrating with server-side JS runtimes.

## 8. Integration Patterns

### 8.1 Python web-service integration
Typical pattern:
1. Build a normalized request dict from framework request metadata.
2. Run `validate_headers` early; block on non-`None` reasons.
3. Build feature records per request/event.
4. Use `extract_features_batch_with_state` with persisted state for incremental windows.
5. Score with `IsolationForest` and combine with rule-based outcomes.

Minimal skeleton:
```python
import aiwaf_rust

state = None
forest = aiwaf_rust.IsolationForest(random_state=42)


def handle(records, headers):
    reason = aiwaf_rust.validate_headers(headers)
    if reason is not None:
        return {"blocked": True, "reason": reason}

    batch = aiwaf_rust.extract_features_batch_with_state(records, ["wp", ".env"], state)
    feats = [[f["kw_hits"], f["burst_count"], f["total_404"]] for f in batch["features"]]

    if feats:
        forest.fit(feats)
        labels = forest.predict(feats)
        if -1 in labels:
            return {"blocked": True, "reason": "anomaly"}

    return {"blocked": False}
```

### 8.2 JS/browser integration with WASM
Typical pattern:
1. `await init()` once at process startup.
2. Use wasm exports in request/event pipeline.
3. Keep forest/state in memory or persist via `to_json()`.

```js
import init, { IsolationForest, validate_headers } from "aiwaf-wasm";

await init();
const forest = new IsolationForest({ random_state: 42 });
const reason = validate_headers({ HTTP_USER_AGENT: "Mozilla/5.0", HTTP_ACCEPT: "text/html" });
```

### 8.3 Python <-> WASM state portability
Isolation forest state shapes are not identical across bindings:
- PyO3 `to_json()` uses camelCase keys like `nEstimators`, `maxSamples`, `estimatorsFeatures`.
- WASM `to_json()` uses serde struct/enums from core (`snake_case` keys and enum objects).

If you need cross-runtime state sharing:
1. Define a canonical transport schema in your app.
2. Add explicit mapping adapters on both sides.
3. Validate with roundtrip tests before rollout.

## 9. Build, Package, and Publish Notes

### 9.1 Python wheels/sdist
- Build backend is `maturin` (`pyproject.toml`).
- Module name is `aiwaf_rust`.
- ABI mode uses `pyo3` `abi3-py38` for broad wheel compatibility.

### 9.2 WASM npm package
- Crate: `crates/aiwaf_wasm`.
- Build output: `crates/aiwaf_wasm/pkg`.
- Packaging helpers: `scripts/patch_wasm_pkg.py` and `scripts/normalize_wheel.py` (repo release workflows).

## 10. Troubleshooting
- `ImportError` after `maturin develop`: confirm active venv matches build interpreter.
- PyO3 build mismatch: ensure `python --version` is 3.8+ and `maturin` is installed in same env.
- WASM runtime errors: confirm `await init()` happens before calling exports.
- Header false positives: verify header key format and required-header config for your runtime.

## 11. Validation Checklist
- Python import works: `import aiwaf_rust`.
- Python tests pass: `pytest tests/test_python_api.py -q`.
- WASM builds cleanly: `wasm-pack build --release --target bundler`.
- WASM tests pass: `wasm-pack test --node`.
- Isolation forest `fit -> to_json -> from_json -> predict` roundtrip is validated in both bindings.
