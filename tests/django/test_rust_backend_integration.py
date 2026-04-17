#!/usr/bin/env python3
"""
Integration tests for the Rust backend (real extension module).
Skips if aiwaf_rust isn't available.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from django.test import TestCase
import unittest

try:
    import aiwaf_rust
except Exception:
    aiwaf_rust = None


class RustBackendIntegrationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if aiwaf_rust is None:
            raise unittest.SkipTest(
                "aiwaf_rust extension not available (skip Rust integration tests). "
                "Install it with: pip install aiwaf-rust"
            )

    def test_validate_headers_blocks_missing(self):
        result = aiwaf_rust.validate_headers(
            {"HTTP_USER_AGENT": "Mozilla/5.0"}
        )
        self.assertIsNotNone(result)
        self.assertIn("Missing required headers", result)

    def test_validate_headers_allows_legit(self):
        result = aiwaf_rust.validate_headers(
            {
                "HTTP_USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "HTTP_ACCEPT": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "HTTP_ACCEPT_LANGUAGE": "en-US,en;q=0.5",
                "HTTP_ACCEPT_ENCODING": "gzip, deflate",
                "HTTP_CONNECTION": "keep-alive",
            }
        )
        self.assertIsNone(result)

    def test_validate_headers_allows_legit_bot(self):
        result = aiwaf_rust.validate_headers(
            {
                "HTTP_USER_AGENT": "Googlebot/2.1 (+http://www.google.com/bot.html)",
                "HTTP_ACCEPT": "*/*",
                "HTTP_ACCEPT_LANGUAGE": "en-US",
            }
        )
        self.assertIsNone(result)

    def test_validate_headers_blocks_accept_star_missing_lang(self):
        result = aiwaf_rust.validate_headers(
            {
                "HTTP_USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "HTTP_ACCEPT": "*/*",
            }
        )
        self.assertIsNotNone(result)
        self.assertIn("Generic Accept header", result)

    def test_validate_headers_blocks_http10_chrome(self):
        result = aiwaf_rust.validate_headers(
            {
                "HTTP_USER_AGENT": "Mozilla/5.0 Chrome/120.0.0.0",
                "HTTP_ACCEPT": "text/html",
                "HTTP_ACCEPT_LANGUAGE": "en-US",
                "SERVER_PROTOCOL": "HTTP/1.0",
            }
        )
        self.assertIsNotNone(result)
        self.assertIn("HTTP/1.0", result)

    def test_analyze_recent_behavior_basic_metrics(self):
        entries = [
            {"path_lower": "/wp-admin/install.php", "timestamp": 0.0, "status": 404, "kw_check": True},
            {"path_lower": "/home", "timestamp": 5.0, "status": 200, "kw_check": False},
        ]
        result = aiwaf_rust.analyze_recent_behavior(entries, [".php", "wp-"])
        self.assertIsNotNone(result)
        self.assertEqual(result["max_404s"], 1)
        self.assertGreaterEqual(result["avg_kw_hits"], 0)
        self.assertFalse(result["should_block"])

    def test_analyze_recent_behavior_triggers_block(self):
        entries = []
        for i in range(10):
            entries.append({
                "path_lower": f"/wp-admin/{i}.php",
                "timestamp": float(i),
                "status": 404,
                "kw_check": True,
            })
        result = aiwaf_rust.analyze_recent_behavior(entries, ["wp-"])
        self.assertIsNotNone(result)
        self.assertTrue(result["should_block"])

    def test_isolation_forest_round_trip(self):
        forest = aiwaf_rust.IsolationForest(
            n_estimators=50,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            warm_start=False,
        )
        data = [[0.1, 1.0], [0.2, 1.1], [0.3, 0.9], [9.0, 9.0]]
        forest.fit(data)
        preds = forest.predict(data)
        self.assertEqual(len(preds), len(data))
        self.assertEqual(preds[-1], -1)

        state = forest.to_json()
        forest2 = aiwaf_rust.IsolationForest.from_json(state)
        preds2 = forest2.predict(data)
        self.assertEqual(preds2, preds)

        new_data = [[0.15, 1.05], [0.25, 1.2], [8.9, 9.1]]
        forest2.retrain(new_data)
        preds3 = forest2.predict(new_data)
        self.assertEqual(len(preds3), len(new_data))

    def test_rust_model_load_via_middleware(self):
        import aiwaf.django.middleware as mw
        forest = aiwaf_rust.IsolationForest(
            n_estimators=10,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            warm_start=False,
        )
        data = [[0.1, 1.0], [0.2, 1.1], [9.0, 9.0]]
        forest.fit(data)
        state = forest.to_json()

        original_loader = mw.load_model_data
        original_joblib = mw.JOBLIB_AVAILABLE
        try:
            mw.load_model_data = lambda: {
                "model_backend": "aiwaf_rust",
                "model_state": state,
            }
            mw.JOBLIB_AVAILABLE = True
            model = mw.load_model_safely()
        finally:
            mw.load_model_data = original_loader
            mw.JOBLIB_AVAILABLE = original_joblib

        self.assertIsNotNone(model)
        preds = model.predict(data)
        self.assertEqual(len(preds), len(data))

    def test_ai_anomaly_middleware_uses_rust_without_numpy(self):
        import aiwaf.django.middleware as mw
        from types import SimpleNamespace

        forest = aiwaf_rust.IsolationForest(
            n_estimators=10,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            warm_start=False,
        )
        forest.fit([[0.1, 1.0], [0.2, 1.1], [9.0, 9.0]])

        original_numpy = mw.NUMPY_AVAILABLE
        original_is_exempt = mw.is_exempt
        original_is_ip_exempted = mw.is_ip_exempted
        original_is_mw_disabled = mw.is_middleware_disabled
        original_get_ip = mw.get_ip
        original_path_exists = mw.path_exists_in_django
        original_is_exempt_path = mw.is_exempt_path
        try:
            mw.NUMPY_AVAILABLE = False
            mw.is_exempt = lambda _req: False
            mw.is_ip_exempted = lambda _ip: False
            mw.is_middleware_disabled = lambda _req, _cls: False
            mw.get_ip = lambda _req: "203.0.113.10"
            mw.path_exists_in_django = lambda _path: False
            mw.is_exempt_path = lambda _path: False

            middleware = mw.AIAnomalyMiddleware(lambda r: None)
            middleware.model = forest

            request = SimpleNamespace(
                path="/wp-admin",
                META={},
                _start_time=time.time() - 0.01,
            )
            response = SimpleNamespace(status_code=404)
            result = middleware.process_response(request, response)
        finally:
            mw.NUMPY_AVAILABLE = original_numpy
            mw.is_exempt = original_is_exempt
            mw.is_ip_exempted = original_is_ip_exempted
            mw.is_middleware_disabled = original_is_mw_disabled
            mw.get_ip = original_get_ip
            mw.path_exists_in_django = original_path_exists
            mw.is_exempt_path = original_is_exempt_path

        self.assertIsNotNone(result)



if __name__ == "__main__":
    import unittest
    unittest.main()
