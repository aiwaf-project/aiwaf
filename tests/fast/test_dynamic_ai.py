from aiwaf.core.training_logic import get_default_legitimate_keywords


def test_dynamic_ai_keyword_baseline_has_common_terms():
    keywords = get_default_legitimate_keywords()
    assert "api" in keywords
    assert "login" in keywords

