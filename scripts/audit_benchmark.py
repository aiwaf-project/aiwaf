#!/usr/bin/env python3
"""Score AIWAF's injection audit on an OWASP Benchmark checkout.

The OWASP Benchmark projects are small labelled web apps: each BenchmarkTestNNNNN case is
either a real vulnerability or a safe look-alike. This script runs (or reads) an AIWAF audit
report and scores it per category the way OWASP does: the true positive rate, the false
positive rate, and their difference (Youden's index; 0 is no better than guessing).

The benchmarks are GPL-licensed, so they aren't part of this repository. Clone one, then:

    python scripts/audit_benchmark.py --source ../BenchmarkPython --language python
    java -cp <classpath> com.aiwaf.cli.AiwafConsole audit ../BenchmarkJava/src --format json --output java.json
    python scripts/audit_benchmark.py --source ../BenchmarkJava --report java.json

Categories that aren't injection, such as weak hashing or cookie flags, are left out.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

# OWASP Benchmark categories and the audit sinks that report them.
CATEGORIES = {
    "cmdi": {"command"},
    "codeinj": {"code", "template"},
    "deserialization": {"deserialization"},
    "ldapi": {"ldap"},
    "pathtraver": {"path"},
    "redirect": {"redirect"},
    "sqli": {"sql", "nosql"},
    "xpathi": {"xpath"},
    "xss": {"xss"},
    "xxe": {"xxe"},
}
TEST_NAME = re.compile(r"(BenchmarkTest\d+)")


def expected_results(path: Path) -> Dict[str, Tuple[str, bool]]:
    results = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            results[row[0].strip()] = (row[1].strip(), row[2].strip().lower() == "true")
    return results


def reported(findings: Iterable[dict]) -> Dict[str, Set[str]]:
    """The sinks reported in each test case's file, or in code a test case called."""
    sinks: Dict[str, Set[str]] = defaultdict(set)
    for finding in findings:
        for field in ("entry_file", "file"):
            match = TEST_NAME.search(str(finding.get(field) or ""))
            if match:
                sinks[match.group(1)].add(finding["sink"])
                break
    return sinks


def score(expected: Dict[str, Tuple[str, bool]], sinks: Dict[str, Set[str]]) -> Dict[str, dict]:
    table: Dict[str, dict] = {}
    for name, (category, vulnerable) in expected.items():
        if category not in CATEGORIES:
            continue
        row = table.setdefault(category, {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
        found = bool(sinks.get(name, set()) & CATEGORIES[category])
        key = ("tp" if found else "fn") if vulnerable else ("fp" if found else "tn")
        row[key] += 1
    total = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for row in table.values():
        for key in total:
            total[key] += row[key]
    table["all"] = total
    for row in table.values():
        positives = row["tp"] + row["fn"]
        negatives = row["fp"] + row["tn"]
        row["tpr"] = row["tp"] / positives if positives else 0.0
        row["fpr"] = row["fp"] / negatives if negatives else 0.0
        row["score"] = row["tpr"] - row["fpr"]
    return table


def format_table(table: Dict[str, dict]) -> str:
    lines = [f"{'category':16} {'cases':>6} {'found':>7} {'false alarms':>13} {'score':>7}"]
    for category in sorted(table, key=lambda name: (name == "all", name)):
        row = table[category]
        cases = row["tp"] + row["fn"] + row["fp"] + row["tn"]
        lines.append(
            f"{category:16} {cases:6d} {row['tpr']:7.1%} {row['fpr']:13.1%} {row['score'] * 100:+7.1f}"
        )
    return "\n".join(lines)


def run_python_audit(source: Path) -> dict:
    sys.path.insert(0, str(ROOT / "py"))
    from aiwaf.core.injection_analysis import analyze_path

    return analyze_path(str(source), include_tests=True)


def find_expected(source: Path) -> Optional[Path]:
    candidates = sorted(source.glob("expectedresults-*.csv"))
    return candidates[-1] if candidates else None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path, help="An OWASP Benchmark checkout.")
    parser.add_argument("--expected", type=Path, help="Its expectedresults CSV (found in --source by default).")
    parser.add_argument("--report", type=Path, help="A JSON audit report to score, from any AIWAF audit command.")
    parser.add_argument("--language", choices=["python"], help="Run the Python audit instead of reading --report.")
    parser.add_argument("--json", action="store_true", help="Print the scores as JSON.")
    args = parser.parse_args(argv)

    expected_path = args.expected or find_expected(args.source)
    if expected_path is None or not expected_path.exists():
        parser.error("No expectedresults CSV found; pass --expected.")
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    elif args.language == "python":
        report = run_python_audit(args.source)
    else:
        parser.error("Pass --report, or --language python to run the audit.")
    table = score(expected_results(expected_path), reported(report.get("findings", [])))
    print(json.dumps(table, indent=2) if args.json else format_table(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
