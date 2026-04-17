"""
Shared CSV storage implementations.
"""

from __future__ import annotations

from pathlib import Path

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


def ensure_all(data_dir: Path):
    return ensure_csv_files(data_dir, CSV_HEADERS)


def read_whitelist(data_dir: Path) -> set[str]:
    return read_csv_set(data_dir, WHITELIST_CSV, "ip", CSV_HEADERS)


def append_whitelist(data_dir: Path, ip: str):
    return append_csv_row(data_dir, WHITELIST_CSV, [ip, now_iso()], CSV_HEADERS)


def rewrite_whitelist(data_dir: Path, whitelist: set[str]):
    rows = [[ip, now_iso()] for ip in whitelist]
    return rewrite_csv_rows(data_dir, WHITELIST_CSV, rows, CSV_HEADERS)


def read_blacklist(data_dir: Path) -> dict[str, str]:
    return read_csv_dict(data_dir, BLACKLIST_CSV, "ip", "reason", CSV_HEADERS)


def append_blacklist(data_dir: Path, ip: str, reason: str, info_json: str):
    return append_csv_row(
        data_dir,
        BLACKLIST_CSV,
        [ip, reason, now_iso(), info_json],
        CSV_HEADERS,
    )


def rewrite_blacklist(data_dir: Path, blacklist: dict[str, str]):
    rows = [[ip, reason, now_iso(), ""] for ip, reason in blacklist.items()]
    return rewrite_csv_rows(data_dir, BLACKLIST_CSV, rows, CSV_HEADERS)


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
