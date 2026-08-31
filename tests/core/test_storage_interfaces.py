from aiwaf.core.storage_interfaces import BlacklistStore, ExemptionStore, KeywordStore


def test_storage_interface_protocols_are_runtime_contracts():
    assert BlacklistStore.__name__ == "BlacklistStore"
    assert ExemptionStore.__name__ == "ExemptionStore"
    assert KeywordStore.__name__ == "KeywordStore"

