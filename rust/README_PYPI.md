# aiwaf-rust

Fast WAF heuristics and anomaly scoring implemented in Rust and exposed to Python.
Designed for lightweight request validation and isolation-forest style scoring.

## Install

```bash
pip install aiwaf-rust
```

## Quickstart

```python
from aiwaf_rust import (
    validate_headers,
    validate_url,
    validate_content,
    validate_recent,
    AiwafIsolationForest,
)

headers = {
    "user-agent": "Mozilla/5.0",
    "accept": "text/html,application/xhtml+xml",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate",
    "connection": "keep-alive",
}

print(validate_headers(headers))
print(validate_url("https://example.com/login"))
print(validate_content("username=admin&password=..."))
print(validate_recent([{ "path": "/", "status": 200 }]))

model = AiwafIsolationForest(100, 256, 0.5, 42)
model.fit([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
print(model.anomaly_score([0.3, 0.2, 0.1]))
```

## Notes

- Built with PyO3 and `abi3` so a single wheel works on Python 3.8+ (including 3.12+).
- The WASM package is published as `aiwaf-wasm` on npm.

## License

MIT
