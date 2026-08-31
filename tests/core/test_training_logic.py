from aiwaf.core.training_logic import get_default_legitimate_keywords, is_scanning_path


def test_training_logic_module_contract():
    assert is_scanning_path("/.env") is True
    assert "login" in get_default_legitimate_keywords()

