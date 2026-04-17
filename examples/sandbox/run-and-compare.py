#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


TARGETS = ("direct", "protected_django", "protected_flask", "protected_fastapi")


def _comparison_files(sandbox_dir: Path) -> set[Path]:
    return {p for p in sandbox_dir.glob("comparison_modes_*.json") if p.is_file()}


def _latest_new_comparison(before: set[Path], sandbox_dir: Path) -> Path:
    after = _comparison_files(sandbox_dir)
    created = list(after - before)
    if not created:
        raise RuntimeError("Could not detect newly generated comparison_modes_*.json")
    created.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return created[0]


def _calc_stats(report: Dict[str, Any]) -> Dict[str, float]:
    attacks = (report or {}).get("attacks") or []
    total_requests = sum(int(a.get("requests_sent") or 0) for a in attacks)
    total_blocked = sum(int(a.get("blocked") or 0) for a in attacks)
    blocked_pct = (total_blocked / total_requests * 100.0) if total_requests else 0.0
    avg_times = [float(a.get("avg_response_time_ms") or 0.0) for a in attacks]
    avg_response_time = (sum(avg_times) / len(avg_times)) if avg_times else 0.0
    return {
        "total_requests": float(total_requests),
        "total_blocked": float(total_blocked),
        "blocked_pct": blocked_pct,
        "avg_response_time_ms": avg_response_time,
    }


def _extract_iteration_metrics(comparison_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    data = json.loads(comparison_path.read_text(encoding="utf-8"))
    normal_reports = data.get("normal") or []
    attack_reports = data.get("attacks") or []
    by_target: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target in TARGETS:
        normal_report = next((r for r in normal_reports if r.get("target") == target), {})
        attack_report = next((r for r in attack_reports if r.get("target") == target), {})
        by_target[target] = {
            "normal": _calc_stats(normal_report),
            "attacks": _calc_stats(attack_report),
        }
    return by_target


def _aggregate(iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"iterations": len(iterations), "targets": {}}
    for target in TARGETS:
        t = {"normal": {}, "attacks": {}}
        for mode in ("normal", "attacks"):
            blocked_pcts = [it["metrics"][target][mode]["blocked_pct"] for it in iterations]
            response_ms = [it["metrics"][target][mode]["avg_response_time_ms"] for it in iterations]
            t[mode] = {
                "blocked_pct_mean": statistics.mean(blocked_pcts),
                "blocked_pct_median": statistics.median(blocked_pcts),
                "response_ms_mean": statistics.mean(response_ms),
                "response_ms_median": statistics.median(response_ms),
            }
        out["targets"][target] = t
    return out


def _print_aggregate(aggregate: Dict[str, Any]) -> None:
    print("\n\nAggregated Results")
    print(f"Iterations: {aggregate['iterations']}")
    for target in TARGETS:
        info = aggregate["targets"][target]
        n = info["normal"]
        a = info["attacks"]
        print(f"\n{target}")
        print(
            "  normal   blocked% mean/median:"
            f" {n['blocked_pct_mean']:.1f}/{n['blocked_pct_median']:.1f}"
            f" | latency ms mean/median: {n['response_ms_mean']:.2f}/{n['response_ms_median']:.2f}"
        )
        print(
            "  attacks  blocked% mean/median:"
            f" {a['blocked_pct_mean']:.1f}/{a['blocked_pct_median']:.1f}"
            f" | latency ms mean/median: {a['response_ms_mean']:.2f}/{a['response_ms_median']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attack-suite + comparison repeatedly and aggregate results.")
    parser.add_argument("-n", "--iterations", type=int, default=1, help="Number of full suite iterations to run.")
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    sandbox_dir = Path(__file__).resolve().parent
    runs: List[Dict[str, Any]] = []

    for i in range(1, args.iterations + 1):
        print(f"\n=== Iteration {i}/{args.iterations} ===")
        print("Running full test suite (Django + Flask + FastAPI, normal + attacks)...")
        before = _comparison_files(sandbox_dir)
        subprocess.check_call([sys.executable, str(sandbox_dir / "attack-suite.py")], cwd=str(sandbox_dir))

        print("\nGenerating comparison report...")
        subprocess.check_call([sys.executable, str(sandbox_dir / "compare-results-modes.py")], cwd=str(sandbox_dir))
        comparison_path = _latest_new_comparison(before, sandbox_dir)
        metrics = _extract_iteration_metrics(comparison_path)
        runs.append({"iteration": i, "comparison_file": comparison_path.name, "metrics": metrics})

    aggregate = _aggregate(runs)
    now = datetime.now(timezone.utc).isoformat().replace(":", "-")
    aggregate_path = sandbox_dir / f"comparison_aggregate_{now}.json"
    aggregate_path.write_text(json.dumps({"runs": runs, "aggregate": aggregate}, indent=2), encoding="utf-8")
    _print_aggregate(aggregate)
    print(f"\nSaved aggregate report: {aggregate_path}")


if __name__ == "__main__":
    main()
