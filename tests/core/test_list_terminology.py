"""Naming changes retain existing imports, settings and persisted policy data."""
import importlib


def test_previous_modules_and_symbols_share_implementations():
    for old, new in [('core.runtime_blacklist', 'core.runtime_blocklist'),
                     ('core.blacklist', 'core.blocklist'),
                     ('core.blacklist_migration', 'core.blocklist_migration'),
                     ('flask.blacklist_manager', 'flask.blocklist_manager'),
                     ('fast.blacklist', 'fast.blocklist')]:
        assert importlib.import_module('aiwaf.' + old) is importlib.import_module('aiwaf.' + new)
    from aiwaf.core.runtime_blocklist import BlocklistManager, BlacklistManager
    assert BlacklistManager is BlocklistManager
    assert BlocklistManager.__dict__['add_to_whitelist'] is BlocklistManager.__dict__['add_to_allowlist']


def test_native_exports_support_previous_names():
    import aiwaf_rust
    assert aiwaf_rust.csv_blacklist_check is aiwaf_rust.csv_blocklist_check
    assert aiwaf_rust.csv_blacklist_block is aiwaf_rust.csv_blocklist_block


def test_existing_policy_files_keep_blocks_and_exemptions(tmp_path):
    from aiwaf.core.storage_csv_impl import ensure_all, read_blocklist, read_allowlist
    (tmp_path/'blacklist.csv').write_text('ip,reason\n8.8.8.8,existing block\n', encoding='utf-8')
    (tmp_path/'whitelist.csv').write_text('ip,added_date\n9.9.9.9,2026-10-14\n', encoding='utf-8')
    ensure_all(tmp_path)
    assert '8.8.8.8' in read_blocklist(tmp_path)
    assert '9.9.9.9' in read_allowlist(tmp_path)


def test_old_configuration_names_and_new_name_precedence(monkeypatch):
    from aiwaf.core.runtime_config import AIWAFConfig
    monkeypatch.setenv('AIWAF_BLACKLIST_DEFAULT_DURATION', '1200')
    cfg = AIWAFConfig()
    assert cfg.get('blocklist.default_block_duration') == 1200
    assert cfg.get('blacklist.default_block_duration') == 1200
    monkeypatch.setenv('AIWAF_BLOCKLIST_DEFAULT_DURATION', '1800')
    assert AIWAFConfig().get('blocklist.default_block_duration') == 1800
    cfg.update({'blacklist': {'default_block_duration': 600}, 'blocklist': {'default_block_duration': 900}})
    assert cfg.get('blocklist.default_block_duration') == 900


def test_cli_accepts_previous_names_with_explicit_data_directory(monkeypatch, capsys, tmp_path):
    import sys
    from types import SimpleNamespace
    from aiwaf.flask import cli
    manager = SimpleNamespace(list_allowlist=lambda: [], list_blocklist=lambda: {})
    monkeypatch.setattr(cli, 'AIWAFManager', lambda directory: manager)
    for name in ('allowlist', 'whitelist', 'blocklist', 'blacklist'):
        monkeypatch.setattr(sys, 'argv', ['aiwaf', '--data-dir', str(tmp_path), 'list', name])
        cli.main()
        assert '(0)' in capsys.readouterr().out


def _write(path, header, *rows):
    path.write_text('\n'.join([','.join(header), *rows]) + '\n', encoding='utf-8')


def test_new_legacy_entries_are_marked_with_the_current_reason_and_earlier_ones_still_count():
    from aiwaf.core.blocklist_migration import legacy_metadata
    from aiwaf.core.storage_csv_impl import is_legacy_blocklist_entry
    assert legacy_metadata('imported')['reputation_reason'] == 'legacy_blocklist'
    assert is_legacy_blocklist_entry({'reputation_reason': 'legacy_blacklist'})
    assert is_legacy_blocklist_entry({'reasons': ['legacy_blacklist', 'old']})
    assert not is_legacy_blocklist_entry({'reputation_reason': 'sql_injection', 'reasons': ['sql_injection']})


def test_the_engine_moves_earlier_list_files_itself(tmp_path):
    import json
    import pytest
    from aiwaf.core.storage_schema import CSV_HEADERS
    rust = pytest.importorskip('aiwaf_rust')
    _write(tmp_path / 'blacklist.csv', CSV_HEADERS['blocklist.csv'])
    _write(tmp_path / 'whitelist.csv', CSV_HEADERS['allowlist.csv'], '9.9.9.9,2026-10-14')
    rust.NativeCSVPipeline(json.dumps({'data_dir': str(tmp_path), 'enabled': [], 'exempt_paths': []}))
    assert not (tmp_path / 'blacklist.csv').exists()
    assert not (tmp_path / 'whitelist.csv').exists()
    assert '9.9.9.9' in (tmp_path / 'allowlist.csv').read_text(encoding='utf-8')
    # With both names present the engine keeps the current file and leaves the earlier one.
    _write(tmp_path / 'whitelist.csv', CSV_HEADERS['allowlist.csv'], '1.1.1.1,old')
    rust.NativeCSVPipeline(json.dumps({'data_dir': str(tmp_path), 'enabled': [], 'exempt_paths': []}))
    assert '1.1.1.1' in (tmp_path / 'whitelist.csv').read_text(encoding='utf-8')
    assert '1.1.1.1' not in (tmp_path / 'allowlist.csv').read_text(encoding='utf-8')
