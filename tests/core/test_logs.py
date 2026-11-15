import gzip
import csv
import pytest

from aiwaf.core.logs import read_rotated_logs, parse_log_line
from aiwaf.core.logs import write_csv_log
from aiwaf.core.native_runtime import native_override, close_csv_logs


@pytest.mark.parametrize('native_enabled', [False, True])
def test_training_log_immediate_rows_rotation_and_field_parity(tmp_path, native_enabled):
    path = tmp_path / 'nested' / 'requests.csv'
    headers = ['path', 'optional', 'status']
    row = {'path': '/café,"hello"\nnext', 'optional': None, 'status': 404}
    with native_override(native_enabled):
        write_csv_log(str(path), headers, row, durable=True)
        with path.open(newline='', encoding='utf-8') as stream:
            assert list(csv.reader(stream)) == [headers, [row['path'], '', '404']]
        write_csv_log(str(path), headers, {'path': '/ok', 'status': 200})
        rotated = path.with_suffix('.old')
        path.rename(rotated)
        write_csv_log(str(path), headers, row)
        with path.open(newline='', encoding='utf-8') as stream:
            assert list(csv.reader(stream)) == [headers, [row['path'], '', '404']]
        write_csv_log(str(path), headers, {'path': '/bad', 'extra': True})
        with path.open(newline='', encoding='utf-8') as stream:
            assert len(list(csv.reader(stream))) == 2
    close_csv_logs(str(path))


def test_read_rotated_logs_reads_plain_and_gzip_files(tmp_path):
    base = tmp_path / "access.log"
    base.write_text("current\n", encoding="utf-8")
    (tmp_path / "access.log.1").write_text("old\n", encoding="utf-8")
    with gzip.open(tmp_path / "access.log.2.gz", "wt", encoding="utf-8") as stream:
        stream.write("older\n")
    (tmp_path / "access.log.3.gz").write_bytes(b"bad gzip")
    assert read_rotated_logs(str(base)) == ["current\n", "old\n", "older\n"]


def test_parse_log_line_supports_patch_method():
    # 1. Combined with response time
    line_rt = '203.0.113.5 - - [05/Nov/2026:10:00:00 +0000] "PATCH /api/resource HTTP/1.1" 200 12 "-" "ua" response-time=0.045'
    res_rt = parse_log_line(line_rt)
    assert res_rt is not None
    assert res_rt["ip"] == "203.0.113.5"
    assert res_rt["path"] == "/api/resource"
    assert res_rt["status"] == "200"
    assert res_rt["response_time"] == 0.045

    # 2. Standard combined format
    line_comb = '203.0.113.5 - - [05/Nov/2026:10:00:00 +0000] "PATCH /api/resource HTTP/1.1" 200 12 "https://example.com" "ua"'
    res_comb = parse_log_line(line_comb)
    assert res_comb is not None
    assert res_comb["ip"] == "203.0.113.5"
    assert res_comb["path"] == "/api/resource"
    assert res_comb["status"] == "200"

    # 3. Common log format
    line_common = '203.0.113.5 - - [05/Nov/2026:10:00:00 +0000] "PATCH /api/resource HTTP/1.1" 200 12'
    res_common = parse_log_line(line_common)
    assert res_common is not None
    assert res_common["ip"] == "203.0.113.5"
    assert res_common["path"] == "/api/resource"
    assert res_common["status"] == "200"

