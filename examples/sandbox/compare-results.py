#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _list_result_files(directory: Path) -> List[Path]:
    return [p for p in directory.iterdir() if p.is_file() and p.name.startswith("results_") and p.name.endswith(".json")]


def _pick_latest_for_target(directory: Path, prefix: str) -> Optional[Path]:
    candidates = [p for p in _list_result_files(directory) if p.name.startswith(f"results_{prefix}_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _expand_arg(arg: str, base_dir: Path) -> List[Path]:
    matches = [Path(p) for p in glob.glob(str(base_dir / arg))]
    return [p for p in matches if p.is_file()]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_by_attack(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for a in (report.get("attacks") or []):
        attack_type = a.get("attack_type")
        if attack_type:
            out[str(attack_type)] = a
    return out


def _pct(blocked: int, total: int) -> str:
    return f"{(blocked / total * 100.0):.1f}%" if total else "0.0%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attack-suite result JSONs (Python)")
    parser.add_argument("files", nargs="*", help="Optional result files or globs")
    parser.add_argument("--json", action="store_true", help="Print JSON summary only")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    resolved: List[Path] = []
    for token in args.files:
        if any(ch in token for ch in "*?[]"):
            resolved.extend(_expand_arg(token, base_dir))
        else:
            resolved.append(base_dir / token)
    resolved = [p for p in resolved if p.exists() and p.is_file()]

    direct_file: Optional[Path] = None
    django_file: Optional[Path] = None
    flask_file: Optional[Path] = None
    fastapi_file: Optional[Path] = None

    if resolved:
        for p in resolved:
            name = p.name
            if name.startswith("results_direct_") and direct_file is None:
                direct_file = p
            if name.startswith("results_protected_django_") and django_file is None:
                django_file = p
            if name.startswith("results_protected_flask_") and flask_file is None:
                flask_file = p
            if name.startswith("results_protected_fastapi_") and fastapi_file is None:
                fastapi_file = p
    else:
        direct_file = _pick_latest_for_target(base_dir, "direct")
        django_file = _pick_latest_for_target(base_dir, "protected_django")
        flask_file = _pick_latest_for_target(base_dir, "protected_flask")
        fastapi_file = _pick_latest_for_target(base_dir, "protected_fastapi")

    if not direct_file or not django_file or not flask_file or not fastapi_file:
        raise SystemExit(
            "Need results for direct, protected_django, protected_flask, protected_fastapi.\n"
            "Place results_direct_*.json, results_protected_django_*.json, "
            "results_protected_flask_*.json, results_protected_fastapi_*.json in examples/sandbox/ "
            "or pass them as args."
        )

    direct = _load_json(direct_file)
    django = _load_json(django_file)
    flask = _load_json(flask_file)
    fastapi = _load_json(fastapi_file)

    direct_by = _index_by_attack(direct)
    django_by = _index_by_attack(django)
    flask_by = _index_by_attack(flask)
    fastapi_by = _index_by_attack(fastapi)

    attack_types = sorted(set(direct_by) | set(django_by) | set(flask_by) | set(fastapi_by))
    rows: List[Dict[str, Any]] = []

    totals = {
        "direct": {"blocked": 0, "requests": 0},
        "django": {"blocked": 0, "requests": 0},
        "flask": {"blocked": 0, "requests": 0},
        "fastapi": {"blocked": 0, "requests": 0},
    }
    for attack in attack_types:
        d = direct_by.get(attack, {})
        dj = django_by.get(attack, {})
        fl = flask_by.get(attack, {})
        fa = fastapi_by.get(attack, {})

        row = {
            "attack_type": attack,
            "direct_blocked": int(d.get("blocked") or 0),
            "django_blocked": int(dj.get("blocked") or 0),
            "flask_blocked": int(fl.get("blocked") or 0),
            "fastapi_blocked": int(fa.get("blocked") or 0),
            "direct_requests": int(d.get("requests_sent") or 0),
            "django_requests": int(dj.get("requests_sent") or 0),
            "flask_requests": int(fl.get("requests_sent") or 0),
            "fastapi_requests": int(fa.get("requests_sent") or 0),
        }
        rows.append(row)

        totals["direct"]["blocked"] += row["direct_blocked"]
        totals["direct"]["requests"] += row["direct_requests"]
        totals["django"]["blocked"] += row["django_blocked"]
        totals["django"]["requests"] += row["django_requests"]
        totals["flask"]["blocked"] += row["flask_blocked"]
        totals["flask"]["requests"] += row["flask_requests"]
        totals["fastapi"]["blocked"] += row["fastapi_blocked"]
        totals["fastapi"]["requests"] += row["fastapi_requests"]

    summary = {
        "direct_file": os.path.relpath(str(direct_file), str(base_dir)),
        "django_file": os.path.relpath(str(django_file), str(base_dir)),
        "flask_file": os.path.relpath(str(flask_file), str(base_dir)),
        "fastapi_file": os.path.relpath(str(fastapi_file), str(base_dir)),
        "rows": rows,
        "totals": totals,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("\nAttack Type              | Direct        | Django        | Flask         | FastAPI")
    print("                         | Blocked / Reqs| Blocked / Reqs| Blocked / Reqs| Blocked / Reqs")
    print("-" * 108)
    for row in rows:
        print(
            f"{row['attack_type']:<24} | "
            f"{row['direct_blocked']:>3}/{row['direct_requests']:<4} ({_pct(row['direct_blocked'], row['direct_requests']):>6}) | "
            f"{row['django_blocked']:>3}/{row['django_requests']:<4} ({_pct(row['django_blocked'], row['django_requests']):>6}) | "
            f"{row['flask_blocked']:>3}/{row['flask_requests']:<4} ({_pct(row['flask_blocked'], row['flask_requests']):>6}) | "
            f"{row['fastapi_blocked']:>3}/{row['fastapi_requests']:<4} ({_pct(row['fastapi_blocked'], row['fastapi_requests']):>6})"
        )
    print("-" * 108)
    print(
        f"{'TOTAL':<24} | "
        f"{totals['direct']['blocked']:>3}/{totals['direct']['requests']:<4} ({_pct(totals['direct']['blocked'], totals['direct']['requests']):>6}) | "
        f"{totals['django']['blocked']:>3}/{totals['django']['requests']:<4} ({_pct(totals['django']['blocked'], totals['django']['requests']):>6}) | "
        f"{totals['flask']['blocked']:>3}/{totals['flask']['requests']:<4} ({_pct(totals['flask']['blocked'], totals['flask']['requests']):>6}) | "
        f"{totals['fastapi']['blocked']:>3}/{totals['fastapi']['requests']:<4} ({_pct(totals['fastapi']['blocked'], totals['fastapi']['requests']):>6})"
    )
    print("")


if __name__ == "__main__":
    main()
