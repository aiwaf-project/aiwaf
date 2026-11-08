import gzip
import csv
import pytest

from aiwaf.core.logs import csv_log_path, read_rotated_logs
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


@pytest.mark.parametrize(
    ('configured', 'expected'),
    [
        ('aiwaf_requests.log', 'aiwaf_requests.csv'),
        ('logs/aiwaf.log', 'logs/aiwaf.csv'),
        ('/var/app.logs/access.log', '/var/app.logs/access.csv'),
        ('/srv/my.log.d/aiwaf.log', '/srv/my.log.d/aiwaf.csv'),
        ('/var/app.logs/x.csv', '/var/app.logs/x.csv'),
        ('access.log.old', 'access.log.old'),
        ('requests.txt', 'requests.txt'),
    ],
)
def test_csv_log_path_only_rewrites_trailing_log(configured, expected):
    assert csv_log_path(configured) == expected
