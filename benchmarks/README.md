# AIWAF Benchmarks

Run the auto-middleware request benchmark from the repository root with an
environment that has the Flask dependencies installed:

```bash
python -m benchmarks.benchmark_auto_middleware
```

The default run includes a second scenario with 100 generated path rules. Set
the rule count explicitly when testing cache behavior:

```bash
python -m benchmarks.benchmark_auto_middleware --path-rules 250
```

For a faster smoke run:

```bash
python -m benchmarks.benchmark_auto_middleware --requests 200 --rounds 3 --warmup 50
```

The benchmark reports median microseconds per request and requests per second
for three Flask applications, first without path rules and then with generated
path rules:

- `baseline`: Flask without AIWAF
- `auto`: AIWAF using `middlewares=["auto"]`
- `full`: every middleware explicitly registered

The auto case configures an external access log and has no geo or UUID routes,
so auto selection can omit logging, geo, and UUID middleware. The full case
keeps those middleware active to expose their request-chain cost. Temporary log
files are deleted after the run.

These are in-process Flask test-client measurements. Use the same machine,
Python environment, request count, and background load when comparing results.
They do not include network or production server overhead.
