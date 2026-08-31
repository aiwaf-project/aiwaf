from aiwaf.core.storage_csv import safe_csv_operation


def test_storage_csv_module_contract(tmp_path):
    assert safe_csv_operation(lambda: "ok") == "ok"
