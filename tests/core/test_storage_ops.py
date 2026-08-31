from aiwaf.core.storage_ops import ensure_csv_files, read_csv_dict


def test_storage_ops_module_contract(tmp_path):
    schema = {"items.csv": ["key", "value"]}
    ensure_csv_files(tmp_path, schema)
    assert read_csv_dict(tmp_path, "items.csv", "key", "value", schema) == {}
