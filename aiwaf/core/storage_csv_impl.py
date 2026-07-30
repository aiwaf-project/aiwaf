"""
Shared CSV storage implementations.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

from .storage_schema import (
    CSV_HEADERS,
    WHITELIST_CSV,
    BLACKLIST_CSV,
    KEYWORDS_CSV,
    GEO_BLOCKED_COUNTRIES_CSV,
    PATH_EXEMPTIONS_CSV,
)
from .storage_ops import (
    ensure_csv_files,
    read_csv_set,
    read_csv_dict,
    append_csv_row,
    rewrite_csv_rows,
    now_iso,
)
from .storage_csv import file_lock, safe_csv_operation


def ensure_all(data_dir: Path):
    ensure_csv_files(data_dir, CSV_HEADERS)
    return _migrate_blacklist_csv(data_dir)


def read_whitelist(data_dir: Path) -> set[str]:
    return read_csv_set(data_dir, WHITELIST_CSV, "ip", CSV_HEADERS)


def append_whitelist(data_dir: Path, ip: str):
    return append_csv_row(data_dir, WHITELIST_CSV, [ip, now_iso()], CSV_HEADERS)


def rewrite_whitelist(data_dir: Path, whitelist: set[str]):
    rows = [[ip, now_iso()] for ip in whitelist]
    return rewrite_csv_rows(data_dir, WHITELIST_CSV, rows, CSV_HEADERS)


def _parse_json(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "permanent"}


def _legacy_blacklist_entry(reason=None, migration_time=None, added_date=None) -> dict:
    blocked_at = time.time() if migration_time is None else migration_time
    legacy_detail = str(reason or "").strip()
    reasons = ["legacy_blacklist"]
    if legacy_detail:
        reasons.append(legacy_detail)
    return {
        "reason": "legacy_import",
        "reputation_reason": "legacy_blacklist",
        "added_date": added_date or now_iso(),
        "extended_request_info": None,
        "score": 100,
        "offenses": 1,
        "blocked_at": blocked_at,
        "expires_at": None,
        "duration": None,
        "permanent": True,
        "reasons": reasons,
    }


def _blacklist_entry_from_row(row: dict) -> dict:
    reason = row.get("reason", "") or "Blocked"
    added_date = row.get("added_date") or row.get("timestamp") or ""
    reasons = _parse_json(row.get("reasons"), [])
    if not isinstance(reasons, list):
        reasons = []
    return {
        "reason": reason,
        "reputation_reason": row.get("reputation_reason", ""),
        "added_date": added_date,
        "extended_request_info": _parse_json(row.get("extended_request_info"), None),
        "score": _parse_int(row.get("score"), 0),
        "offenses": _parse_int(row.get("offenses"), 0),
        "blocked_at": _parse_float(row.get("blocked_at")),
        "expires_at": _parse_float(row.get("expires_at")),
        "duration": _parse_int(row.get("duration"), 0) or None,
        "permanent": _parse_bool(row.get("permanent")),
        "reasons": reasons or [reason],
    }


def _migrate_blacklist_csv(data_dir: Path):
    def _migrate():
        csv_file = data_dir / BLACKLIST_CSV
        if not csv_file.exists():
            return
        with file_lock(csv_file, "r") as f:
            raw_rows = list(csv.reader(f))
        current_headers = CSV_HEADERS[BLACKLIST_CSV]
        if not raw_rows:
            rewrite_csv_rows(data_dir, BLACKLIST_CSV, [], CSV_HEADERS)
            return
        fieldnames = raw_rows[0]
        if fieldnames == current_headers:
            return
        migration_time = time.time()
        migrated = {}
        if "ip" in fieldnames:
            for values in raw_rows[1:]:
                row = dict(zip(fieldnames, values))
                ip = (row.get("ip") or "").strip()
                if not ip:
                    continue
                if "permanent" not in fieldnames and not row.get("expires_at") and not row.get("duration"):
                    migrated[ip] = _legacy_blacklist_entry(
                        row.get("reason"),
                        migration_time=migration_time,
                        added_date=row.get("added_date") or row.get("timestamp"),
                    )
                else:
                    migrated[ip] = _blacklist_entry_from_row(row)
        else:
            for values in raw_rows:
                if not values:
                    continue
                ip = str(values[0] or "").strip()
                if not ip:
                    continue
                reason = values[1] if len(values) > 1 else None
                migrated[ip] = _legacy_blacklist_entry(reason, migration_time=migration_time)
        rewrite_csv_rows(data_dir, BLACKLIST_CSV, [_blacklist_row(ip, entry) for ip, entry in migrated.items()], CSV_HEADERS)

    return safe_csv_operation(_migrate)


def _blacklist_row(ip: str, entry) -> list:
    if isinstance(entry, dict):
        reason = str(entry.get("reason") or "Blocked")
        reputation_reason = str(entry.get("reputation_reason") or "")
        added_date = str(entry.get("added_date") or entry.get("timestamp") or now_iso())
        extended_info = entry.get("extended_request_info")
        reasons = entry.get("reasons") or []
        return [
            ip,
            reason,
            reputation_reason,
            added_date,
            json.dumps(extended_info, separators=(",", ":"), ensure_ascii=False) if extended_info else "",
            str(entry.get("score") or ""),
            str(entry.get("offenses") or ""),
            str(entry.get("blocked_at") or ""),
            str(entry.get("expires_at") or ""),
            str(entry.get("duration") or ""),
            str(bool(entry.get("permanent", False))),
            json.dumps(reasons, separators=(",", ":"), ensure_ascii=False) if reasons else "",
        ]
    return [ip, str(entry or "Blocked"), "", now_iso(), "", "", "", "", "", "", "False", ""]


def read_blacklist(data_dir: Path) -> dict[str, dict]:
    def _read():
        ensure_all(data_dir)
        csv_file = data_dir / BLACKLIST_CSV
        items = {}
        with file_lock(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ip = (row.get("ip") or "").strip()
                if not ip:
                    continue
                items[ip] = _blacklist_entry_from_row(row)
        return items
    return safe_csv_operation(_read)


def append_blacklist(data_dir: Path, ip: str, reason: str, info_json: str, metadata: Optional[dict] = None):
    ensure_all(data_dir)
    entry = dict(metadata or {})
    entry.setdefault("reason", reason)
    if info_json and "extended_request_info" not in entry:
        entry["extended_request_info"] = _parse_json(info_json, None)
    return append_csv_row(
        data_dir,
        BLACKLIST_CSV,
        _blacklist_row(ip, entry),
        CSV_HEADERS,
    )


def rewrite_blacklist(data_dir: Path, blacklist: dict):
    rows = [_blacklist_row(ip, entry) for ip, entry in blacklist.items()]
    return rewrite_csv_rows(data_dir, BLACKLIST_CSV, rows, CSV_HEADERS)


def is_legacy_blacklist_entry(entry) -> bool:
    return isinstance(entry, dict) and (
        entry.get("reputation_reason") == "legacy_blacklist"
        or "legacy_blacklist" in [str(reason) for reason in (entry.get("reasons") or [])]
    )


def legacy_blacklist_entries(data_dir: Path) -> dict[str, dict]:
    blacklist = read_blacklist(data_dir)
    return {ip: entry for ip, entry in blacklist.items() if is_legacy_blacklist_entry(entry)}


def convert_legacy_blacklist_entries(data_dir: Path, duration: Optional[int] = None, clear: bool = False) -> int:
    blacklist = read_blacklist(data_dir)
    changed = 0
    now = time.time()
    for ip in list(blacklist.keys()):
        entry = blacklist[ip]
        if not is_legacy_blacklist_entry(entry):
            continue
        changed += 1
        if clear:
            del blacklist[ip]
            continue
        entry["permanent"] = duration is None
        entry["duration"] = duration
        entry["expires_at"] = now + duration if duration else None
        entry["blocked_at"] = entry.get("blocked_at") or now
        entry["reputation_reason"] = "legacy_blacklist_converted" if duration else "legacy_blacklist"
        blacklist[ip] = entry
    if changed:
        rewrite_blacklist(data_dir, blacklist)
    return changed


def read_keywords(data_dir: Path) -> set[str]:
    return read_csv_set(data_dir, KEYWORDS_CSV, "keyword", CSV_HEADERS)


def append_keyword(data_dir: Path, keyword: str):
    return append_csv_row(data_dir, KEYWORDS_CSV, [keyword, now_iso()], CSV_HEADERS)


def rewrite_keywords(data_dir: Path, keywords: set[str]):
    rows = [[keyword, now_iso()] for keyword in keywords]
    return rewrite_csv_rows(data_dir, KEYWORDS_CSV, rows, CSV_HEADERS)


def read_geo_blocked_countries(data_dir: Path) -> set[str]:
    return {c.upper() for c in read_csv_set(data_dir, GEO_BLOCKED_COUNTRIES_CSV, "country", CSV_HEADERS)}


def append_geo_blocked_country(data_dir: Path, country_code: str):
    return append_csv_row(data_dir, GEO_BLOCKED_COUNTRIES_CSV, [country_code, now_iso()], CSV_HEADERS)


def rewrite_geo_blocked_countries(data_dir: Path, countries: set[str]):
    rows = [[country, now_iso()] for country in sorted(countries)]
    return rewrite_csv_rows(data_dir, GEO_BLOCKED_COUNTRIES_CSV, rows, CSV_HEADERS)


def read_path_exemptions(data_dir: Path) -> dict[str, str]:
    return read_csv_dict(data_dir, PATH_EXEMPTIONS_CSV, "path", "reason", CSV_HEADERS)


def append_path_exemption(data_dir: Path, path: str, reason: str):
    return append_csv_row(data_dir, PATH_EXEMPTIONS_CSV, [path, reason, now_iso()], CSV_HEADERS)


def rewrite_path_exemptions(data_dir: Path, exemptions: dict[str, str]):
    rows = [[path, reason, now_iso()] for path, reason in exemptions.items()]
    return rewrite_csv_rows(data_dir, PATH_EXEMPTIONS_CSV, rows, CSV_HEADERS)
