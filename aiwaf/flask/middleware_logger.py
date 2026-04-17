"""AIWAF request logger middleware (Django logic adapted to Flask)."""

from __future__ import annotations

import csv
import os
import time

from flask import g, request

from .utils import get_ip
from aiwaf.core.storage_csv import file_lock


class AIWAFLoggerMiddleware:
    """
    Middleware that logs requests for AI-WAF training.
    Acts as a fallback when main access logs are unavailable.
    """

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self.app = app
        app.before_request(self.process_request)
        app.after_request(self.process_response)

    def process_request(self):
        g.aiwaf_start_time = time.time()
        return None

    def process_response(self, response):
        if not self.app.config.get("AIWAF_MIDDLEWARE_LOGGING", False):
            return response

        start_time = getattr(g, "aiwaf_start_time", time.time())
        response_time = time.time() - start_time

        if self.app.config.get("AIWAF_USE_CSV", True):
            self._write_csv_log(response, response_time)

        return response

    def _get_csv_path(self) -> str:
        csv_file = self.app.config.get("AIWAF_MIDDLEWARE_LOG", "aiwaf_requests.log")
        if not csv_file.endswith(".csv"):
            csv_file = csv_file.replace(".log", ".csv")
        return csv_file

    def _write_csv_log(self, response, response_time: float) -> None:
        csv_file = self._get_csv_path()
        headers, row = self._build_csv_row(response, response_time)
        from aiwaf.core.logs import write_csv_log
        write_csv_log(csv_file, headers, row)

    def _build_csv_row(self, response, response_time: float):
        headers = [
            "timestamp",
            "ip",
            "method",
            "path",
            "status_code",
            "content_length",
            "response_time",
            "referer",
            "user_agent",
        ]
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "ip": get_ip(),
            "method": request.method,
            "path": request.path[:500],
            "status_code": response.status_code,
            "content_length": response.headers.get("Content-Length", "-"),
            "response_time": "{:.6f}".format(response_time),
            "referer": request.headers.get("Referer", "")[:500],
            "user_agent": request.headers.get("User-Agent", "")[:2000],
        }
        return headers, row


# file_lock now provided by aiwaf.core.storage_csv
