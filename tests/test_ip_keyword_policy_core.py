from aiwaf.core.ip_keyword import evaluate_keyword_policy


def test_core_policy_allows_normal_existing_route():
    decision = evaluate_keyword_policy(
        path="/api/data",
        query_keys=[],
        path_exists=True,
        keyword_learning_enabled=True,
        static_keywords={".php", "xmlrpc", "wp-"},
        dynamic_keywords=set(),
        legitimate_keywords={"api", "data"},
        exempt_keywords=set(),
        safe_prefixes={"api"},
        malicious_keywords={"xmlrpc"},
        is_malicious_context=lambda _seg: False,
    )
    assert decision.block_reason is None


def test_core_policy_blocks_nonexistent_suspicious_segment():
    decision = evaluate_keyword_policy(
        path="/shellupload",
        query_keys=[],
        path_exists=False,
        keyword_learning_enabled=True,
        static_keywords=set(),
        dynamic_keywords={"shellupload"},
        legitimate_keywords=set(),
        exempt_keywords=set(),
        safe_prefixes=set(),
        malicious_keywords={"shellupload"},
        is_malicious_context=lambda seg: seg == "shellupload",
    )
    assert decision.block_reason is not None
    assert "shellupload" in decision.block_reason


def test_core_policy_learns_only_when_suspicious_context():
    decision = evaluate_keyword_policy(
        path="/unknownpayload",
        query_keys=[],
        path_exists=False,
        keyword_learning_enabled=True,
        static_keywords=set(),
        dynamic_keywords=set(),
        legitimate_keywords=set(),
        exempt_keywords=set(),
        safe_prefixes=set(),
        malicious_keywords=set(),
        is_malicious_context=lambda seg: seg == "unknownpayload",
    )
    assert "unknownpayload" in decision.learned_keywords
