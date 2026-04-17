#!/usr/bin/env python
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def list_comparison_files(directory: Path) -> List[Path]:
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.name.startswith("comparison_modes_") and p.name.endswith(".json")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def pick_latest(directory: Path) -> Optional[Path]:
    files = list_comparison_files(directory)
    return files[0] if files else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pad(value: Any, length: int) -> str:
    s = str("" if value is None else value)
    return s if len(s) >= length else s + (" " * (length - len(s)))


@dataclass
class Stats:
    total_requests: int = 0
    total_blocked: int = 0
    blocked_pct: float = 0.0
    avg_response_time: float = 0.0


def calculate_stats(report: Dict[str, Any]) -> Stats:
    stats = Stats()
    attacks = (report or {}).get("attacks") or []
    for attack in attacks:
        stats.total_requests += int(attack.get("requests_sent") or 0)
        stats.total_blocked += int(attack.get("blocked") or 0)
    stats.blocked_pct = (stats.total_blocked / stats.total_requests * 100.0) if stats.total_requests else 0.0

    total_time = 0.0
    for attack in attacks:
        total_time += float(attack.get("avg_response_time_ms") or 0.0)
    stats.avg_response_time = (total_time / len(attacks)) if attacks else 0.0
    return stats


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    comparison_file = pick_latest(base_dir)
    if not comparison_file:
        raise SystemExit("No comparison_modes_*.json found in examples/sandbox/")

    data = load_json(comparison_file)
    normal_reports = data.get("normal") or []
    attack_reports = data.get("attacks") or []

    generated_at = (data.get("generatedAt") or "").split("T")[0]
    print("\n")
    print(" WAF Comparison: Normal vs Attack Traffic")
    print(f" Generated: {generated_at}")
    print("\n")

    results = []
    for normal_report in normal_reports:
        target = normal_report.get("target")
        attack_report = next((r for r in attack_reports if r.get("target") == target), None)
        if not attack_report:
            continue
        results.append({"target": target, "normal": calculate_stats(normal_report), "attacks": calculate_stats(attack_report)})

    print("Target                   | Normal Traffic       | Attack Traffic       | Status")
    print("                         | Reqs    Blocked %    | Reqs    Blocked %    |")
    print("-" * 90)

    for result in results:
        normal = result["normal"]
        attacks = result["attacks"]
        status = ""
        if normal.blocked_pct > 5:
            status = " HIGH FALSE POS"
        elif attacks.blocked_pct < 50:
            status = " LOW DETECTION"

        print(
            f"{pad(result['target'], 24)}| "
            f"{pad(normal.total_requests, 5)} {pad(f'{normal.blocked_pct:.1f}%', 9)} | "
            f"{pad(attacks.total_requests, 5)} {pad(f'{attacks.blocked_pct:.1f}%', 9)} | {status}"
        )

    print("\n" + ("=" * 90))
    print("\nDetailed Breakdown:\n")

    for result in results:
        normal = result["normal"]
        attacks = result["attacks"]
        print(f"\n{str(result['target']).upper()}")
        print("-" * 50)

        print("\nNormal Traffic:")
        print(f"  Total Requests: {normal.total_requests}")
        print(f"  Blocked: {normal.total_blocked} ({normal.blocked_pct:.1f}%)")
        print(f"  Avg Response Time: {normal.avg_response_time:.2f}ms")

        print("\nAttack Traffic:")
        print(f"  Total Requests: {attacks.total_requests}")
        print(f"  Blocked: {attacks.total_blocked} ({attacks.blocked_pct:.1f}%)")
        print(f"  Avg Response Time: {attacks.avg_response_time:.2f}ms")

        if normal.blocked_pct > 5:
            print(f"\n   WARNING: High false positive rate ({normal.blocked_pct:.1f}% of normal traffic blocked)")
        if attacks.blocked_pct < 50:
            print(f"\n   WARNING: Low attack detection rate ({attacks.blocked_pct:.1f}% of attacks blocked)")

    rel = os.path.relpath(str(comparison_file), str(base_dir))
    print(f"\n\nFull report: {rel}\n")


if __name__ == "__main__":
    main()

