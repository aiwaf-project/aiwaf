from aiwaf.core.model_security import is_trusted_model_path


def test_model_security_module_contract(tmp_path):
    assert is_trusted_model_path(str(tmp_path / "model.json"), allow_custom=True) is True
