"""Compare Flask request overhead for AIWAF auto and full middleware stacks."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from flask import Flask

from aiwaf.flask import AIWAF


FULL_MIDDLEWARES = [
    "geo_block",
    "ip_keyword_block",
    "rate_limit",
    "ai_anomaly",
    "honeypot",
    "uuid_tamper",
    "header_validation",
    "logging",
]
REQUEST_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "aiwaf-benchmark/1.0",
}


def build_path_rules(count: int) -> list[dict]:
    if count < 1:
        return []
    rules = [
        {"PREFIX": f"/generated-rule-{index}/", "DISABLE": ["ai_anomaly"]}
        for index in range(count - 1)
    ]
    rules.append(
        {
            "PREFIX": "/health/",
            "RATE_LIMIT": {"WINDOW": 10, "MAX": 10_000_000, "FLOOD": 20_000_000},
        }
    )
    return rules


def build_app(mode: str, log_dir: Path, path_rules: list[dict]) -> tuple[Flask, list[str]]:
    app = Flask(f"aiwaf_benchmark_{mode}_{len(path_rules)}_rules")
    app.config.update(
        TESTING=True,
        AIWAF_ACCESS_LOG="/var/log/external-access.log",
        AIWAF_GEO_BLOCK_ENABLED=False,
        AIWAF_GEO_BLOCK_COUNTRIES=[],
        AIWAF_LOG_DIR=str(log_dir),
        AIWAF_PATH_RULES=path_rules,
        AIWAF_RATE_MAX=10_000_000,
        AIWAF_RATE_FLOOD=20_000_000,
        AIWAF_USE_CSV=True,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if mode == "baseline":
        return app, []

    requested = ["auto"] if mode == "auto" else FULL_MIDDLEWARES
    integration = AIWAF(app, middlewares=requested)
    return app, integration.get_enabled_middlewares()


def run_batch(app: Flask, requests: int) -> float:
    started = time.perf_counter_ns()
    with app.test_client() as client:
        for _ in range(requests):
            response = client.get("/health", headers=REQUEST_HEADERS)
            if response.status_code != 200:
                raise RuntimeError(
                    f"benchmark request returned HTTP {response.status_code}: "
                    f"{response.get_data(as_text=True)}"
                )
    return (time.perf_counter_ns() - started) / 1_000_000_000


def benchmark(app: Flask, requests: int, rounds: int, warmup: int) -> tuple[float, float]:
    run_batch(app, warmup)
    durations = [run_batch(app, requests) for _ in range(rounds)]
    median_seconds = statistics.median(durations)
    return median_seconds * 1_000_000 / requests, requests / median_seconds


def run_scenario(
    label: str,
    path_rules: list[dict],
    temp_dir: Path,
    *,
    requests: int,
    rounds: int,
    warmup: int,
) -> None:
    print(f"\n{label} ({len(path_rules)} path rules)")
    results: dict[str, tuple[float, float]] = {}
    for mode in ("baseline", "auto", "full"):
        app, enabled = build_app(mode, temp_dir / f"{label}-{mode}", path_rules)
        latency_us, requests_per_second = benchmark(
            app,
            requests=requests,
            rounds=rounds,
            warmup=warmup,
        )
        results[mode] = (latency_us, requests_per_second)
        middleware_label = ", ".join(sorted(enabled)) or "none"
        print(
            f"{mode:8} {latency_us:10.2f} us/request "
            f"{requests_per_second:10.0f} requests/s  [{middleware_label}]"
        )

    baseline_latency = results["baseline"][0]
    auto_latency = results["auto"][0]
    full_latency = results["full"][0]
    print(f"auto overhead vs baseline: {auto_latency - baseline_latency:.2f} us/request")
    print(f"full overhead vs baseline: {full_latency - baseline_latency:.2f} us/request")
    print(f"auto latency reduction vs full: {(1 - auto_latency / full_latency) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--path-rules",
        type=int,
        default=100,
        help="number of generated path rules in the rule-heavy scenario",
    )
    args = parser.parse_args()
    if min(args.requests, args.rounds, args.warmup) < 1:
        parser.error("requests, rounds, and warmup must all be positive")
    if args.path_rules < 1:
        parser.error("path-rules must be positive")

    with tempfile.TemporaryDirectory(prefix="aiwaf-benchmark-") as temp_dir:
        root = Path(temp_dir)
        run_scenario(
            "no-rules",
            [],
            root,
            requests=args.requests,
            rounds=args.rounds,
            warmup=args.warmup,
        )
        run_scenario(
            "rule-heavy",
            build_path_rules(args.path_rules),
            root,
            requests=args.requests,
            rounds=args.rounds,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
