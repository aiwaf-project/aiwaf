"""
Shared log parsing helpers.
"""

from __future__ import annotations

import glob
import gzip
import os
import re
import csv
from datetime import datetime
from aiwaf.core.storage_csv import file_lock
from aiwaf.core.native_runtime import csv_function, csv_values


def csv_path(log_file: str) -> str:
    """Return the CSV companion path without changing parent directory names."""
    if log_file.endswith('.csv') or not log_file.endswith('.log'):
        return log_file
    return f'{log_file[:-4]}.csv'


def write_csv_log(csv_file: str, headers: list[str], row: dict, *, durable: bool = False) -> None:
    # Match DictWriter's rejection of extra fields before invoking a native writer.
    if row.keys() - headers:
        return
    native_log = csv_function('csv_log_append')
    if native_log is not None:
        try:
            native_log(str(csv_file), headers, csv_values([row.get(key, '') for key in headers]), durable)
        except Exception:
            pass  # Do not retry a possibly completed append and duplicate a record.
        return
    log_dir = os.path.dirname(csv_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        with file_lock(csv_file, mode="a"):
            needs_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
                if durable:
                    f.flush()
                    os.fsync(f.fileno())
    except Exception:
        pass


def csv_log_path(path: str) -> str:
    """Return the CSV log path for a configured middleware log path.

    Only a ``.log`` at the very end of the path is replaced with ``.csv``.
    A ``.log`` elsewhere in the path (for example inside a folder name) and
    paths already ending in ``.csv`` are returned unchanged.
    """
    if path.endswith(".log"):
        return path[: -len(".log")] + ".csv"
    return path


# Keep in step with PATTERNS in rust/src/training_pipeline.rs. The first pattern that matches
# wins. A response size may be "-", and the response time comes only from `rt` (seconds, from
# response-time=) or `ms` (milliseconds, the trailing "5ms" in AIWAF's own Flask and FastAPI logs).
_LOG_PREFIX = (
    r'(?P<ip>\S+).*\[(?P<ts>.*?)\].*"(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) (?P<path>.*?) HTTP/.*?" '
)
_LOG_RX_PATTERNS = [
    # Combined log format with response-time= (IPv4 + IPv6)
    re.compile(_LOG_PREFIX + r'(?P<status>\d{3}).*?"(.*?)" "(.*?)".*?response-time=(?P<rt>\d+(?:\.\d+)?)'),
    # Combined log format, optionally followed by a response time in milliseconds (IPv4 + IPv6)
    re.compile(_LOG_PREFIX + r'(?P<status>\d{3}) (?:\d+|-) "(.*?)" "(.*?)"(?: (?P<ms>\d+(?:\.\d+)?)ms\b)?'),
    # Common log format (IPv4 + IPv6)
    re.compile(_LOG_PREFIX + r'(?P<status>\d{3}) (?:\d+|-)'),
]


def _response_time(match: re.Match) -> float:
    """Response time in seconds, or 0.0 when the line doesn't record one."""
    fields = match.groupdict()
    if fields.get("rt"):
        return float(fields["rt"])
    if fields.get("ms"):
        return float(fields["ms"]) / 1000
    return 0.0


def read_rotated_logs(base_path: str) -> list[str]:
    lines: list[str] = []
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8", errors="ignore") as f:
            lines.extend(f.readlines())
    for path in sorted(glob.glob(base_path + ".*")):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                lines.extend(f.readlines())
        except OSError:
            continue
    return lines


def parse_log_line(line: str) -> dict | None:
    for pattern in _LOG_RX_PATTERNS:
        m = pattern.search(line)
        if not m:
            continue
        ip, ts_str, path, status = m.group("ip", "ts", "path", "status")
        rt = _response_time(m)

        try:
            ts = datetime.strptime(ts_str.split()[0], "%d/%b/%Y:%H:%M:%S")
        except ValueError:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue

        return {
            "ip": ip,
            "timestamp": ts,
            "path": path,
            "status": status,
            "response_time": rt,
        }

    return None
