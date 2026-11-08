#!/usr/bin/env python3
"""Measure the SQL check against the OWASP Core Rule Set and libinjection test data.

Downloads pinned copies of both test sets into .cache/sql-benchmark (once), reads them as data
only, and runs every payload through aiwaf-rust's whole-request SQL check
(policy_sql_inspect), the same call Python and Node.js make on each request.

Usage:
    python scripts/sql_benchmark.py              # print the report
    python scripts/sql_benchmark.py --require    # also fail if results drop below the floors
    python scripts/sql_benchmark.py --misses 20  # show some misses and false positives

Needs aiwaf-rust built from this repository and PyYAML (pip install pyyaml).

What's measured:
- CRS: the 942 (SQL injection) regression tests, grouped by paranoia level. Each test's whole
  request is checked: the URL path, the query string, the body and the headers.
- libinjection: every payload in data/sqli-*.txt, and the benign data/false_positives.txt.

CRS matches SQL keywords and function names anywhere, especially at higher paranoia levels.
AIWAF flags values that break out of a SQL value, plus names found only in SQL, so it doesn't
aim to match CRS on fragments such as "concat(" alone.
"""

from __future__ import annotations

import argparse
import collections
import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import unquote_plus, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "sql-benchmark"
MAX_BYTES = 65536

# Pinned so results only change when this file does. Update both together with the floors.
SOURCES = {
    "crs": (
        "https://codeload.github.com/coreruleset/coreruleset/tar.gz/b0b345e893ac4a3dcd527c8d90364cf4583e461e",
        ("tests/regression/tests/REQUEST-942-APPLICATION-ATTACK-SQLI/", "rules/REQUEST-942-APPLICATION-ATTACK-SQLI.conf"),
    ),
    "libinjection": (
        "https://codeload.github.com/client9/libinjection/tar.gz/e86ff4019a4343579cc307d96d79272d5efcd1be",
        ("data/sqli-", "data/false_positives.txt"),
    ),
}

# Results must not drop below these. Raise them when the check improves.
FLOORS = {
    "libinjection_detection": 0.985,
    "libinjection_false_positives": 1,
    "crs_pl1_detection": 0.70,
    "crs_pl1_false_positives": 0,
    "crs_pl2_detection": 0.40,
    "crs_pl2_false_positives": 2,
}


def fetch(name: str) -> Path:
    """Download and unpack only the needed files. Nothing downloaded is run."""
    target = CACHE / name
    if target.is_dir() and any(target.iterdir()):
        return target
    url, wanted = SOURCES[name]
    print(f"Downloading {name} test data from {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = member.name.split("/", 1)[-1]
            if not relative.startswith(wanted) or ".." in Path(relative).parts:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tar.extractfile(member).read())
    return target


def crs_levels(conf: str) -> dict[int, int]:
    levels = {}
    for block in re.split(r"\n(?=SecRule)", conf):
        rule = re.search(r"id:(\d+)", block)
        level = re.search(r"paranoia-level/(\d)", block)
        if rule and level:
            levels[int(rule.group(1))] = int(level.group(1))
    return levels


# CRS negative tests only say one rule shouldn't match. These ones still carry SQL injection,
# so flagging them is right: they're listed apart and not counted as false positives.
CRS_NEGATIVES_WITH_INJECTION = {
    (942101, "24+union+all+select+1,2,3+from+aa"),
    (942160, "BENChmARk(2999/**/999,Md5("),
    (942360, "'||(select(pg_sleep(15))where(true))||'"),
}


def measure_crs(check, check_headers, check_path, root: Path) -> tuple[dict, list, list]:
    import yaml

    levels = crs_levels((root / "rules" / "REQUEST-942-APPLICATION-ATTACK-SQLI.conf").read_text(encoding="utf-8"))
    results: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    misses, false_positives = [], []
    for path in sorted((root / "tests" / "regression" / "tests" / "REQUEST-942-APPLICATION-ATTACK-SQLI").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        rule_id = int(document["rule_id"])
        level = levels.get(rule_id, 0)
        for test in document.get("tests", []):
            for stage in test.get("stages", []):
                given = stage.get("input", {})
                log = (stage.get("output") or {}).get("log", {}) or {}
                attack = rule_id in (log.get("expect_ids") or [])
                if not attack and rule_id not in (log.get("no_expect_ids") or []):
                    continue
                address = urlsplit(str(given.get("uri", "/")))
                query = address.query.encode("utf-8", "surrogateescape")
                data = given.get("data", "")
                body = ("".join(map(str, data)) if isinstance(data, list) else str(data or "")).encode("utf-8", "surrogateescape")
                headers = [(str(k), str(v)) for k, v in (given.get("headers") or {}).items()]
                found = check(query, body) or check_headers(headers) or check_path(address.path)
                payload = (query + b" " + body).decode("utf-8", "replace").strip()
                shown = payload or (address.path if address.path not in ("", "/", "/get", "/post") else str(headers))
                if attack:
                    results[level]["caught" if found else "missed"] += 1
                    if not found:
                        misses.append(f"CRS {rule_id} PL{level}: {unquote_plus(shown)[:150]}")
                elif found and any(rule_id == known and part in str(given.get("uri", "")) + str(data)
                                   for known, part in CRS_NEGATIVES_WITH_INJECTION):
                    results[level]["injection"] += 1
                    false_positives.append(f"CRS {rule_id} PL{level} ({found[0]}), negative test with SQL injection: {unquote_plus(shown)[:150]}")
                else:
                    results[level]["flagged" if found else "passed"] += 1
                    if found:
                        false_positives.append(f"CRS {rule_id} PL{level} ({found[0]}): {unquote_plus(shown)[:150]}")
    return results, misses, false_positives


def lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def measure_libinjection(check, root: Path) -> tuple[collections.Counter, list, list]:
    counts: collections.Counter = collections.Counter()
    misses, false_positives = [], []
    for path in sorted((root / "data").glob("sqli-*.txt")):
        for line in lines(path):
            found = check(b"q=" + line.encode("utf-8", "surrogateescape"), b"")
            counts["caught" if found else "missed"] += 1
            if not found:
                misses.append(f"libinjection {path.name}: {unquote_plus(line)[:150]}")
    for line in lines(root / "data" / "false_positives.txt"):
        counts["benign"] += 1
        if check(b"q=" + line.encode("utf-8", "surrogateescape"), b""):
            counts["flagged"] += 1
            false_positives.append(f"libinjection false_positives.txt: {line[:150]}")
    return counts, misses, false_positives


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the SQL check against OWASP CRS and libinjection test data.")
    parser.add_argument("--require", action="store_true", help="Fail if results drop below the floors in this file.")
    parser.add_argument("--misses", type=int, default=0, help="Show this many misses and every false positive.")
    args = parser.parse_args(argv)
    try:
        import aiwaf_rust
        import yaml  # noqa: F401
    except ImportError as error:
        print(f"{error}. Build aiwaf-rust from this repository and run: pip install pyyaml", file=sys.stderr)
        return 2
    check = lambda query, body: aiwaf_rust.policy_sql_inspect(query, body, MAX_BYTES)  # noqa: E731
    check_headers = lambda headers: aiwaf_rust.policy_sql_inspect_headers(headers, MAX_BYTES) if headers else None  # noqa: E731
    check_path = lambda path: aiwaf_rust.policy_sql_inspect_path(path, MAX_BYTES)  # noqa: E731

    crs, crs_misses, crs_false = measure_crs(check, check_headers, check_path, fetch("crs"))
    lib, lib_misses, lib_false = measure_libinjection(check, fetch("libinjection"))

    print("OWASP CRS 942 (SQL injection) regression tests")
    for level in sorted(crs):
        r = crs[level]
        attacks = r["caught"] + r["missed"]
        print(f"  PL{level}: caught {r['caught']}/{attacks} ({100 * r['caught'] / max(attacks, 1):.0f}%);"
              f" benign tests flagged {r['flagged']}/{r['flagged'] + r['passed']}"
              + (f" ({r['injection']} more flagged that carry SQL injection)" if r["injection"] else ""))
    attacks = lib["caught"] + lib["missed"]
    print(f"libinjection: caught {lib['caught']}/{attacks} ({100 * lib['caught'] / attacks:.1f}%);"
          f" false_positives.txt flagged {lib['flagged']}/{lib['benign']}")
    for line in (crs_false + lib_false) if args.misses else []:
        print(f"  false positive: {line}")
    for line in (crs_misses + lib_misses)[: args.misses]:
        print(f"  missed: {line}")

    if not args.require:
        return 0
    pl1 = crs.get(1, collections.Counter())
    pl2 = crs.get(2, collections.Counter())
    measured = {
        "libinjection_detection": lib["caught"] / attacks,
        "libinjection_false_positives": lib["flagged"],
        "crs_pl1_detection": pl1["caught"] / max(pl1["caught"] + pl1["missed"], 1),
        "crs_pl1_false_positives": pl1["flagged"],
        "crs_pl2_detection": pl2["caught"] / max(pl2["caught"] + pl2["missed"], 1),
        "crs_pl2_false_positives": pl2["flagged"],
    }
    failures = []
    for name, floor in FLOORS.items():
        value = measured[name]
        if (name.endswith("false_positives") and value > floor) or (name.endswith("detection") and value < floor):
            failures.append(f"{name} is {value:.3f}, limit {floor}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
