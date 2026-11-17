"""ensure_csv_files does the full check only when the directory may have changed."""
import os

from aiwaf.core import storage_ops

HEADERS = {"a.csv": ["x"], "b.csv": ["y"]}


def test_unchanged_directory_skips_the_check_and_changes_are_caught(tmp_path, monkeypatch):
    calls = []
    real = storage_ops._ensure_csv_files
    monkeypatch.setattr(storage_ops, "_ensure_csv_files", lambda *a: calls.append(1) or real(*a))
    storage_ops._ENSURED.clear()
    storage_ops.ensure_csv_files(tmp_path, HEADERS)
    storage_ops.ensure_csv_files(tmp_path, HEADERS)
    assert len(calls) == 1 and (tmp_path / "a.csv").exists()
    # Deleting a file changes the directory, so it's created again.
    (tmp_path / "a.csv").unlink()
    os.utime(tmp_path, ns=(1, 1))
    storage_ops.ensure_csv_files(tmp_path, HEADERS)
    assert (tmp_path / "a.csv").exists() and len(calls) == 2


def test_a_change_hidden_by_coarse_timestamps_is_caught_within_a_second(tmp_path, monkeypatch):
    storage_ops._ENSURED.clear()
    clock = [100.0]
    monkeypatch.setattr(storage_ops.time, "monotonic", lambda: clock[0])
    storage_ops.ensure_csv_files(tmp_path, HEADERS)
    mtime = os.stat(tmp_path).st_mtime_ns
    (tmp_path / "b.csv").unlink()
    os.utime(tmp_path, ns=(mtime, mtime))  # As if the deletion fell in the same timestamp tick.
    storage_ops.ensure_csv_files(tmp_path, HEADERS)
    assert not (tmp_path / "b.csv").exists()
    clock[0] += 1.5
    storage_ops.ensure_csv_files(tmp_path, HEADERS)
    assert (tmp_path / "b.csv").exists()


def test_a_missing_directory_is_created_and_the_memo_stays_bounded(tmp_path, monkeypatch):
    storage_ops._ENSURED.clear()
    target = tmp_path / "new"
    storage_ops.ensure_csv_files(target, HEADERS)
    assert (target / "a.csv").exists()
    monkeypatch.setattr(storage_ops, "_ENSURED", {("x", (i,)): (0, 0) for i in range(256)})
    storage_ops.ensure_csv_files(tmp_path / "other", HEADERS)
    assert len(storage_ops._ENSURED) == 1
