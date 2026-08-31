import gzip

from aiwaf.core.logs import read_rotated_logs


def test_read_rotated_logs_reads_plain_and_gzip_files(tmp_path):
    base = tmp_path / "access.log"
    base.write_text("current\n", encoding="utf-8")
    (tmp_path / "access.log.1").write_text("old\n", encoding="utf-8")
    with gzip.open(tmp_path / "access.log.2.gz", "wt", encoding="utf-8") as stream:
        stream.write("older\n")
    (tmp_path / "access.log.3.gz").write_bytes(b"bad gzip")
    assert read_rotated_logs(str(base)) == ["current\n", "old\n", "older\n"]
