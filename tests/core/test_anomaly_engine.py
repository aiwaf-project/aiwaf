from aiwaf.core.anomaly import (
    analyze_recent_behavior_python,
    evaluate_anomaly,
)


def test_analyze_recent_behavior_python_blocks_scanning_keywords():
    now = 1000.0
    recent = [
        (now - 1, "/wp-admin", 404, 0.01),
        (now - 2, "/xmlrpc.php", 404, 0.01),
        (now - 3, "/.env", 404, 0.01),
        (now - 4, "/phpmyadmin", 404, 0.01),
        (now - 5, "/wp-includes", 404, 0.01),
    ]

    stats = analyze_recent_behavior_python(
        recent,
        static_keywords=["wp-", "xmlrpc", ".env", "phpmyadmin"],
        path_exists=lambda _p: False,
        is_exempt_path=lambda _p: False,
    )
    assert stats.max_404s == 5
    assert stats.should_block is True


def test_evaluate_anomaly_learns_keywords_only_on_404_and_missing_path():
    now = 1000.0
    outcome = evaluate_anomaly(
        ip="1.2.3.4",
        path="/not-a-real/evil-shell",
        status_code=404,
        response_time=0.2,
        now=now,
        history=[],
        window_seconds=60,
        model=None,
        static_keywords=[".php", "xmlrpc"],
        malicious_keywords=[".php", "xmlrpc"],
        keyword_learning_enabled=True,
        path_exists=lambda _p: False,
        is_exempt_path=lambda _p: False,
        is_malicious_context=lambda seg: seg == "shell",
        legitimate_keywords={"health"},
    )
    assert "shell" in outcome.learned_keywords
