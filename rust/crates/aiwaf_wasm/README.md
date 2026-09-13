# aiwaf-wasm

Rust-powered WAF heuristics compiled to WebAssembly. Provides fast, deterministic
feature extraction and anomaly scoring in browsers and Node.

Version: `0.2.1`

## Install

```bash
npm install aiwaf-wasm
```

## Usage (Node / Bundlers)

```js
import init, {
  validate_headers,
  validate_url,
  validate_content,
  validate_recent,
  AiwafIsolationForest,
} from "aiwaf-wasm";

// wasm-pack builds a default init() for the .wasm file
await init();

const headers = {
  "user-agent": "Mozilla/5.0",
  "accept": "text/html,application/xhtml+xml",
  "accept-language": "en-US,en;q=0.9",
  "accept-encoding": "gzip, deflate",
  "connection": "keep-alive",
};

const url = "https://example.com/login";
const content = "username=admin&password=...";

console.log(validate_headers(headers));
console.log(validate_url(url));
console.log(validate_content(content));
console.log(validate_recent([{ path: "/", status: 200 }]));

const model = new AiwafIsolationForest(100, 256, 0.5, 42);
model.fit([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]);
console.log(model.anomaly_score([0.3, 0.2, 0.1]));
```

## API Reference

### Header Validation

- `validate_headers(headers: Record<string, string> | Headers) -> string | null`  
  Accepts a plain object or a `Headers` instance.  
  Returns `null` when OK, otherwise a reason string.  
  In browsers, if `user-agent` is missing, it is filled from `navigator.userAgent`.

- `validate_headers_with_config(headers, requiredHeaders: string[] | null, minScore: number | null) -> string | null`

### Feature Extraction

- `extract_features(records: Array<Record>, staticKeywords: string[]) -> Array<Record>`
- `extract_features_batch_with_state(records, staticKeywords, state?) -> { features: Array<Record>, state: object }`
- `finalize_feature_state() -> { features: Array<Record>, state: object }`
- `build_records(parsed, ip404, pathExistsFn, pathExemptFn, statusIdxList) -> Array<Record>`
- `rust_payload_from_records(records: Array<Record>) -> Array<Record>`
- `python_feature_from_record(record, ipTimes, staticKeywords) -> Record`
- `python_features_batched(records, ipTimes, staticKeywords, iterBatchesFn, batchSize, parallelEnabled, parallelChunkSize, maxWorkers) -> Array<Record>`

`build_records` accepts parsed rows with `ip`, `path`, `response_time`, `status`, and `timestamp`.
`timestamp` can be an epoch-second number or a JavaScript `Date`. Path callback exceptions are treated as `false`.

### Behavior Analysis

- `analyze_recent_behavior(entries: Array<Record>, staticKeywords: string[]) -> object | null`

### Isolation Forest

- `new IsolationForest(config?: object)`
- `fit(data: number[][]): void`
- `retrain(data: number[][]): void`
- `anomaly_score(point: number[]): number`
- `score_samples(data: number[][]): number[]`
- `decision_function(data: number[][]): number[]`
- `predict(data: number[][]): number[]`
- `to_json(): object`
- `IsolationForest.from_json(state: object): IsolationForest`

## Notes

- Browser usage requires bundler support for WASM (Vite, Webpack, etc.).
- The API mirrors the Python bindings for consistency.

## License

MIT
