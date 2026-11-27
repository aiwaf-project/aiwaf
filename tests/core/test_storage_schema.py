"""List files written under their earlier names move to allowlist.csv and blocklist.csv."""
import os

import pytest

from aiwaf.core import storage_schema
from aiwaf.core.storage_schema import CSV_HEADERS, move_legacy_list_files


def _write(path, header, *rows):
    path.write_text('\n'.join([','.join(header), *rows]) + '\n', encoding='utf-8')


def test_earlier_list_files_are_moved_to_their_current_names(tmp_path):
    _write(tmp_path / 'blacklist.csv', CSV_HEADERS['blocklist.csv'])
    _write(tmp_path / 'whitelist.csv', CSV_HEADERS['allowlist.csv'], '9.9.9.9,2026-10-14')
    move_legacy_list_files(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['allowlist.csv', 'blocklist.csv']
    assert '9.9.9.9' in (tmp_path / 'allowlist.csv').read_text(encoding='utf-8')
    move_legacy_list_files(tmp_path)  # Nothing left to move.
    move_legacy_list_files(tmp_path / 'missing')  # A directory that doesn't exist yet.


def test_a_current_list_file_wins_over_an_earlier_one(tmp_path):
    from aiwaf.core.storage_csv_impl import read_allowlist
    _write(tmp_path / 'whitelist.csv', CSV_HEADERS['allowlist.csv'], '1.1.1.1,old')
    _write(tmp_path / 'allowlist.csv', CSV_HEADERS['allowlist.csv'], '2.2.2.2,new')
    assert read_allowlist(tmp_path) == {'2.2.2.2'}
    assert '1.1.1.1' in (tmp_path / 'whitelist.csv').read_text(encoding='utf-8')


def test_a_move_another_process_finished_first_is_not_an_error(tmp_path, monkeypatch):
    (tmp_path / 'whitelist.csv').write_text('ip,added_date\n', encoding='utf-8')

    def moved_elsewhere(source, target):
        os.replace(source, target)
        raise FileNotFoundError(source)

    monkeypatch.setattr(storage_schema.os, 'rename', moved_elsewhere)
    storage_schema.move_legacy_list_files(tmp_path)
    assert (tmp_path / 'allowlist.csv').exists()

    (tmp_path / 'blacklist.csv').write_text('ip\n', encoding='utf-8')

    def refused(source, target):
        raise PermissionError(target)

    monkeypatch.setattr(storage_schema.os, 'rename', refused)
    with pytest.raises(PermissionError):
        storage_schema.move_legacy_list_files(tmp_path)
