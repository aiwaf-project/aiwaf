# aiwaf-rust

`aiwaf-rust` is a Rust core with Python and WebAssembly bindings that provides fast request-header, behavior heuristics, and isolation forest anomaly scoring for WAF-style detection.

- PyPI package: `aiwaf-rust`
- Python import: `aiwaf_rust`
- Built with: `PyO3` + `maturin`
- WASM package: `aiwaf-wasm` (npm)
- Built with: `wasm-bindgen` + `wasm-pack`
- Version: `0.2.0`

## Features

- Header validation with configurable required headers and scoring
- Request feature extraction for downstream detection logic
- Recent behavior analysis for suspicious scanning patterns
- Isolation Forest anomaly detection with sklearn-style API
- Cross-platform wheel publishing (Linux, macOS, Windows)
- WASM build for browser and bundler targets

## Release Hardening

Windows wheel builds configure MSVC Security Development Lifecycle checks, Control Flow Guard,
and EH Continuation Guard for any C/C++ compilation invoked during the build:

```text
CL=/sdl /guard:cf /guard:ehcont
CFLAGS=/sdl /guard:cf /guard:ehcont
CXXFLAGS=/sdl /guard:cf /guard:ehcont
LDFLAGS=/guard:cf /guard:ehcont /force:guardehcont
LINK=/guard:cf /guard:ehcont /force:guardehcont
```

The Windows Rust build uses nightly with `rust-src` and `-Z build-std=std,panic_unwind` so Rust
`std`, `core`, `alloc`, and `panic_unwind` are rebuilt with the same linker mitigation flags.
It also preserves debug tables and requests CFG/EH continuation metadata at codegen and link time:

```text
RUSTFLAGS=-C codegen-units=1 -C debuginfo=1 -C strip=none -C control-flow-guard=y -C link-arg=/guard:cf -C link-arg=/guard:ehcont -C link-arg=/force:guardehcont
```

Rust code is not compiled by MSVC `cl.exe`, so `/sdl` applies only to native C/C++ build steps.
`/force:guardehcont` is intentionally included because Cargo also links host build-script
executables during the build; those host artifacts can still encounter prebuilt toolchain objects
that do not carry EH continuation metadata. The force flag lets linking proceed while still asking
MSVC to emit EH continuation metadata where possible.
This nightly `build-std` route is intentionally Windows-only and more fragile than the normal
stable Rust wheel path. Keeping debug information is intentional for scanners that verify
SDL/CFG/EH continuation metadata in Windows binaries.

## Installation

### From PyPI

```bash
pip install aiwaf-rust
```

### Local development install

The root `../LICENSE` is the sole source license file. Stage a copy before local
`maturin` builds; the release workflow does this automatically.

```bash
cp ../LICENSE LICENSE
pip install maturin
maturin develop
```

## Python API

### Function Reference

**Header Validation**

- `validate_headers(headers: dict[str, str]) -> Optional[str]`  
  Expects a dict of header names to values. Accepts either `HTTP_*` style keys
  (e.g., `HTTP_USER_AGENT`) or standard header names (e.g., `user-agent`).  
  Returns `None` when the headers look acceptable, otherwise a short reason string.

- `validate_headers_with_config(headers: dict[str, str], required_headers: list[str], min_score: int) -> Optional[str]`  
  `required_headers` can be empty to disable required-header checks.  
  `min_score` is the header quality threshold (set `0` to disable).  
  Returns `None` or a reason string.

**Feature Extraction**

- `extract_features(records: list[dict], static_keywords: list[str]) -> list[dict]`  
  Each record expects:  
  `ip` (str), `path_lower` (str), `path_len` (int), `timestamp` (float),  
  `response_time` (float), `status_idx` (int), `kw_check` (bool), `total_404` (int).  
  Returns feature dicts with `ip`, `path_len`, `kw_hits`, `resp_time`, `status_idx`, `burst_count`, and `total_404`.

- `extract_features_batch_with_state(records: list[dict], static_keywords: list[str], state: Optional[dict]) -> dict`  
  Returns `{"features": [...], "state": {...}}` to allow incremental batches.

- `finalize_feature_state() -> dict`  
  Returns an empty feature batch with a reset state.

- `build_records(parsed, ip_404, path_exists_fn, path_exempt_fn, status_idx_list) -> list[dict]`  
  Converts parsed request rows into training records, caches path existence/exemption checks per path, and treats callback errors as `False`.

- `rust_payload_from_records(records: list[dict]) -> list[dict]`  
  Converts built training records into the payload accepted by `extract_features`.

- `python_feature_from_record(record, ip_times, static_keywords) -> dict`  
  Computes one feature row from a built training record and an IP timestamp map.

- `python_features_batched(records, ip_times, static_keywords, iter_batches_fn, batch_size, parallel_enabled, parallel_chunk_size, max_workers) -> list[dict]`  
  Computes feature rows in batch-sized chunks. The Rust binding ignores Python threading controls because the work runs natively.

**Behavior Analysis**

- `analyze_recent_behavior(entries: list[dict], static_keywords: list[str]) -> Optional[dict]`  
  Each entry expects: `path_lower` (str), `timestamp` (float), `status` (int), `kw_check` (bool).  
  Returns `None` or a dict like `{"should_block": bool, "reason": str, ...}`.

**Isolation Forest**

- `IsolationForest(...)` constructor parameters:  
  `n_estimators` (int), `max_samples` ("auto"|int|float), `contamination` ("auto"|float),  
  `max_features` (float), `bootstrap` (bool), `random_state` (int|None), `warm_start` (bool).

- Methods:  
  `fit(data: list[list[float]]) -> None`  
  `retrain(data: list[list[float]]) -> None`  
  `anomaly_score(point: list[float]) -> float`  
  `score_samples(data: list[list[float]]) -> list[float]`  
  `decision_function(data: list[list[float]]) -> list[float]`  
  `predict(data: list[list[float]]) -> list[int]` (1 = inlier, -1 = outlier)  
  `to_json() -> dict`  
  `IsolationForest.from_json(state: dict) -> IsolationForest`

```python
import aiwaf_rust

# 1) Basic header validation
reason = aiwaf_rust.validate_headers({
    "HTTP_USER_AGENT": "Mozilla/5.0",
    "HTTP_ACCEPT": "text/html"
})

# 2) Configurable validation
reason = aiwaf_rust.validate_headers_with_config(
    {
        "HTTP_USER_AGENT": "Mozilla/5.0",
        "HTTP_ACCEPT": "text/html"
    },
    ["HTTP_USER_AGENT", "HTTP_ACCEPT"],
    3,
)

# 3) Training record preparation + feature extraction
records = aiwaf_rust.build_records(
    [
        {
            "ip": "1.2.3.4",
            "path": "/wp-admin",
            "response_time": 0.03,
            "status": 404,
            "timestamp": 1700000000.0,
        }
    ],
    {"1.2.3.4": 5},
    lambda path: False,  # path_exists_fn
    lambda path: False,  # path_exempt_fn
    [200, 404, 500],
)
payload = aiwaf_rust.rust_payload_from_records(records)
features = aiwaf_rust.extract_features(payload, ["wp"])

# Direct feature extraction also works if records are already normalized.
direct_features = aiwaf_rust.extract_features(
    [
        {
            "ip": "1.2.3.4",
            "path_lower": "/wp-admin",
            "path_len": 9,
            "timestamp": 1700000000.0,
            "response_time": 0.03,
            "status_idx": 1,
            "kw_check": True,
            "total_404": 5,
        }
    ],
    ["wp"],
)

single_feature = aiwaf_rust.python_feature_from_record(
    records[0],
    {"1.2.3.4": [1699999995.0, 1700000000.0]},
    ["wp"],
)
batched_features = aiwaf_rust.python_features_batched(
    records,
    {"1.2.3.4": [1699999995.0, 1700000000.0]},
    ["wp"],
    lambda rows, size: [rows[i:i + size] for i in range(0, len(rows), size)],
    256,
    False,
    1024,
    1,
)

# 4) Incremental feature extraction with state
batch = aiwaf_rust.extract_features_batch_with_state(
    [
        {
            "ip": "1.2.3.4",
            "path_lower": "/wp-admin",
            "path_len": 9,
            "timestamp": 1700000000.0,
            "response_time": 0.03,
            "status_idx": 1,
            "kw_check": True,
            "total_404": 5,
        }
    ],
    ["wp"],
    None,
)

# 5) Behavior analysis
analysis = aiwaf_rust.analyze_recent_behavior(
    [
        {
            "path_lower": "/wp-admin",
            "timestamp": 1700000000.0,
            "status": 404,
            "kw_check": True,
        }
    ],
    ["wp"],
)

# 6) Isolation Forest
forest = aiwaf_rust.IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    warm_start=False,
)
forest.fit([[0.1, 1.0], [0.2, 1.1], [9.0, 9.0]])
score = forest.anomaly_score([9.0, 9.0])
labels = forest.predict([[0.1, 1.0], [9.0, 9.0]])

# Save and load
state = forest.to_json()
forest2 = aiwaf_rust.IsolationForest.from_json(state)
forest2.retrain([[0.15, 1.05], [0.25, 1.2]])
```

## WASM API (JS)

Install from npm:

```bash
npm install aiwaf-wasm
```

Usage (bundler target):

```js
import init, {
  IsolationForest,
  build_records,
  extract_features,
  extract_features_batch_with_state,
  python_feature_from_record,
  python_features_batched,
  rust_payload_from_records,
  validate_headers,
} from "aiwaf-wasm";

await init();

const reason = validate_headers({
  HTTP_USER_AGENT: "Mozilla/5.0",
  HTTP_ACCEPT: "text/html",
});

const records = build_records(
  [
    {
      ip: "1.2.3.4",
      path: "/wp-admin",
      response_time: 0.03,
      status: 404,
      timestamp: 1700000000.0,
    },
  ],
  { "1.2.3.4": 5 },
  (path) => false,
  (path) => false,
  [200, 404, 500]
);
const payload = rust_payload_from_records(records);
const feats = extract_features(payload, ["wp"]);

const singleFeature = python_feature_from_record(
  records[0],
  { "1.2.3.4": [1699999995.0, 1700000000.0] },
  ["wp"]
);
const batchedFeatures = python_features_batched(
  records,
  { "1.2.3.4": [1699999995.0, 1700000000.0] },
  ["wp"],
  null,
  256,
  false,
  1024,
  1
);

const directFeats = extract_features(
  [
    {
      ip: "1.2.3.4",
      path_lower: "/wp-admin",
      path_len: 9,
      timestamp: 1700000000.0,
      response_time: 0.03,
      status_idx: 1,
      kw_check: true,
      total_404: 5,
    },
  ],
  ["wp"]
);

const batch = extract_features_batch_with_state(payload, ["wp"], null);

const forest = new IsolationForest({
  n_estimators: 100,
  max_samples: "auto",
  contamination: "auto",
  max_features: 1.0,
  bootstrap: false,
  random_state: 42,
  warm_start: false,
});
forest.fit([[0.1, 1.0], [0.2, 1.1], [9.0, 9.0]]);
const score = forest.anomaly_score([9.0, 9.0]);
const state = forest.to_json();
const forest2 = IsolationForest.from_json(state);
forest2.retrain([[0.15, 1.05], [0.25, 1.2]]);
```

### WASM Function Reference

**Header Validation**

- `validate_headers(headers: Record<string, string> | Headers) -> string | null`  
  Accepts a plain object or a `Headers` instance.  
  Returns `null` when OK, otherwise a reason string.  
  In browsers, if `user-agent` is missing, it is filled from `navigator.userAgent`.

- `validate_headers_with_config(headers, requiredHeaders: string[] | null, minScore: number | null) -> string | null`

**Feature Extraction**

- `extract_features(records: Array<Record>, staticKeywords: string[]) -> Array<Record>`
- `extract_features_batch_with_state(records, staticKeywords, state?) -> { features: Array<Record>, state: object }`
- `finalize_feature_state() -> { features: Array<Record>, state: object }`
- `build_records(parsed, ip404, pathExistsFn, pathExemptFn, statusIdxList) -> Array<Record>`
- `rust_payload_from_records(records: Array<Record>) -> Array<Record>`
- `python_feature_from_record(record, ipTimes, staticKeywords) -> Record`
- `python_features_batched(records, ipTimes, staticKeywords, iterBatchesFn, batchSize, parallelEnabled, parallelChunkSize, maxWorkers) -> Array<Record>`

`build_records` accepts parsed rows with `ip`, `path`, `response_time`, `status`, and `timestamp`.
For Python, `timestamp` can be an epoch-second number or any object with `.timestamp()`.
For WASM, `timestamp` can be an epoch-second number or a JavaScript `Date`.

**Behavior Analysis**

- `analyze_recent_behavior(entries: Array<Record>, staticKeywords: string[]) -> object | null`

**Isolation Forest**

- `new IsolationForest(config?: object)`
- `fit(data: number[][]): void`
- `retrain(data: number[][]): void`
- `anomaly_score(point: number[]): number`
- `score_samples(data: number[][]): number[]`
- `decision_function(data: number[][]): number[]`
- `predict(data: number[][]): number[]`
- `to_json(): object`
- `IsolationForest.from_json(state: object): IsolationForest`

## Isolation Forest Details

Isolation Forest isolates points by randomly choosing a feature and a split value between that feature’s min and max. The number of splits required to isolate a point is its path length. Anomalies tend to have shorter path lengths because they are easier to isolate.

Key mechanics:

- A tree is built by selecting a random split value for each candidate feature at a node and choosing the split with the best variance reduction (ExtraTreeRegressor-style random splitter).
- The tree stops when it reaches `max_depth = ceil(log2(max_samples))` or the node has 0 or 1 samples, or all values are identical for the chosen feature.
- Path length for a point is the depth at which it lands, plus the average path length adjustment for the leaf size.

Scoring:

- For each tree, compute the path length `h(x)`.
- Average across trees: `E[h(x)]`.
- Convert to anomaly score: `s(x) = 2 ^ (-E[h(x)] / c(max_samples))`.
- `c(n)` is the average path length of an unsuccessful search in a binary search tree:
  - `c(n) = 0` for `n <= 1`
  - `c(n) = 1` for `n = 2`
  - otherwise `c(n) = 2 * (ln(n-1) + 0.5772156649) - 2*(n-1)/n`

Interpretation:

- Higher `s(x)` means more anomalous.
- `score_samples` returns the opposite of the anomaly score (higher = more normal), like sklearn.
- `decision_function = score_samples - offset_`.
- `predict` returns `1` for inliers and `-1` for outliers.
- With `contamination="auto"`, `offset_ = -0.5`. With numeric contamination, `offset_` is set to the percentile of training scores.

Retraining:

- With `warm_start=True`, `fit` appends new trees up to `n_estimators`.
- `retrain` always appends trees, preserving existing ones.

## Build and Test

```bash
# Run Rust tests
cargo test

# Stage the root license for Python package builds
cp ../LICENSE LICENSE

# Build Python wheel locally
maturin build --release --out dist

# Build source distribution
maturin sdist --out dist

# Build WASM package
cd crates/aiwaf_wasm
wasm-pack build --release --target bundler
```

## Publishing

Publishing the Python package is handled by GitHub Actions using PyPI trusted publishing.

Workflow (at the monorepo root): `.github/workflows/rust-publish.yml`

Trigger conditions:
- A `rust-v*` tag matching the version in `pyproject.toml`, `Cargo.toml`, and `Cargo.lock` (for example, `rust-v0.2.1`)
- `workflow_dispatch`

The workflow builds:
- ABI3 wheels compatible with Python `3.8+` for Linux x86_64/aarch64, musllinux x86_64, macOS x86_64/arm64, and Windows x86_64
- One source distribution (`sdist`)

After tests and builds pass, it publishes only `aiwaf-rust` to PyPI using OIDC (`id-token: write`) and `pypa/gh-action-pypi-publish@release/v1`. It does not publish the separate `aiwaf-wasm` npm package.

Before the first monorepo release, add a trusted publisher to the existing PyPI `aiwaf-rust` project with owner `aayushgauba`, repository `aiwaf`, workflow filename `rust-publish.yml`, and environment `pypi`. These values must exactly match the GitHub workflow. Then commit the `rust/` sources and root workflow, push the tag, and watch the `Publish Rust package to PyPI` action.

## Compatibility Policy

`aiwaf-rust` has its own PyPI version and `rust-v*` release tags. It does not need to match the Python `aiwaf` or JavaScript `aiwaf` version. Test integrations against the intended `aiwaf` versions before release.

## Development Notes

- Module name in Rust and Python is `aiwaf_rust`
- `pyproject.toml` is the source of Python package metadata
- `Cargo.toml` is the source of Rust crate metadata

## License

MIT. See the canonical [`../LICENSE`](../LICENSE). Published packages include a copy.
