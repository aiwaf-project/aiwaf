import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.test_settings")

import django
django.setup()

from django.http import HttpResponse
from django.test import override_settings

from aiwaf.django.middleware import AIAnomalyMiddleware
from aiwaf.django.middleware_logger import AIWAFLoggerMiddleware
from tests.django.base_test import AIWAFTestCase


class RequestLogPersistenceTestCase(AIWAFTestCase):
    @override_settings(
        AIWAF_MIDDLEWARE_LOGGING=False,
        AIWAF_MIDDLEWARE_DB=True,
        AIWAF_ENABLE_KEYWORD_LEARNING=False,
    )
    @patch("aiwaf.django.models.RequestLog.objects.create")
    @patch("aiwaf.core.anomaly.evaluate_anomaly")
    def test_anomaly_middleware_persists_requestlog_when_logger_db_disabled(self, mock_eval, mock_create):
        mock_eval.return_value = SimpleNamespace(
            updated_history=[],
            learned_keywords=[],
            block=False,
            reason=None,
        )

        middleware = AIAnomalyMiddleware(MagicMock())
        request = self.create_request("/api/ping/", headers={"REMOTE_ADDR": "2001:db8::1"})
        response = HttpResponse(status=200)
        response["Content-Length"] = "42"

        middleware.process_request(request)
        middleware.process_response(request, response)

        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["ip_address"], "2001:db8::1")
        self.assertEqual(kwargs["path"], "/api/ping/")
        self.assertEqual(kwargs["status_code"], 200)

    @override_settings(
        AIWAF_MIDDLEWARE_LOGGING=True,
        AIWAF_MIDDLEWARE_DB=True,
        AIWAF_MIDDLEWARE_CSV=False,
        AIWAF_ENABLE_KEYWORD_LEARNING=False,
    )
    @patch("aiwaf.core.anomaly.evaluate_anomaly")
    def test_no_duplicate_requestlog_when_logger_db_enabled(self, mock_eval):
        mock_eval.return_value = SimpleNamespace(
            updated_history=[],
            learned_keywords=[],
            block=False,
            reason=None,
        )
        fake_requestlog = SimpleNamespace(objects=SimpleNamespace(create=MagicMock()))

        with patch("aiwaf.django.middleware_logger._import_models"), \
             patch("aiwaf.django.middleware_logger.RequestLog", fake_requestlog), \
             patch("aiwaf.django.models.RequestLog.objects.create", side_effect=AssertionError("anomaly middleware should skip DB write")):
            anomaly_middleware = AIAnomalyMiddleware(MagicMock())
            logger_middleware = AIWAFLoggerMiddleware(MagicMock())
            request = self.create_request("/api/ping/", headers={"REMOTE_ADDR": "203.0.113.10"})
            response = HttpResponse(status=200)

            anomaly_middleware.process_request(request)
            logger_middleware.process_request(request)
            logger_middleware.process_response(request, response)
            anomaly_middleware.process_response(request, response)

        fake_requestlog.objects.create.assert_called_once()
