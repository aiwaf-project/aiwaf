"""FastAPI logging middleware in web-server style formats."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware

from ..decorators import should_apply_middleware
from ..utils import get_ip
from ...core.request_context import extract_logging_context_from_fastapi_request


class AIWAFLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_dir="aiwaf_logs", log_format="combined", path_rules=None):
        super().__init__(app)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_format = log_format
        self.path_rules = path_rules or []
        self.access_log_file = self.log_dir / "access.log"
        self.error_log_file = self.log_dir / "error.log"
        self.aiwaf_log_file = self.log_dir / "aiwaf.log"

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "logging", self.path_rules):
            return await call_next(request)

        start_time = time.time()
        request.state.aiwaf_blocked = getattr(request.state, "aiwaf_blocked", False)
        request.state.aiwaf_block_reason = getattr(request.state, "aiwaf_block_reason", "")
        response = await call_next(request)

        self._log_access(request, response, start_time)
        if response.status_code >= 400:
            self._log_error(request, response)
        if getattr(request.state, "aiwaf_blocked", False):
            self._log_aiwaf_event(request)
        return response

    def _log_access(self, request, response, start_time):
        if self.log_format == "csv":
            self._log_access_csv(request, response, start_time)
        elif self.log_format == "json":
            self._log_access_json(request, response, start_time)
        else:
            self._log_access_combined(request, response, start_time)

    def _log_access_combined(self, request, response, start_time):
        response_time_ms = int((time.time() - start_time) * 1000)
        ip = get_ip(request)
        ctx = extract_logging_context_from_fastapi_request(request, ip=ip)
        timestamp = datetime.now().strftime("[%d/%b/%Y:%H:%M:%S +0000]")
        path = ctx["path_with_query"]
        protocol = ctx["protocol"]
        content_length = response.headers.get("content-length", "-")
        referer = ctx["referer"] or "-"
        user_agent = ctx["user_agent"] or "-"
        blocked = "BLOCKED" if getattr(request.state, "aiwaf_blocked", False) else "-"
        block_reason = getattr(request.state, "aiwaf_block_reason", "-")
        log_line = (
            f'{ip} - - {timestamp} "{request.method} {path} {protocol}" '
            f'{response.status_code} {content_length} "{referer}" "{user_agent}" '
            f'{response_time_ms}ms {blocked} "{block_reason}"'
        )
        with open(self.access_log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def _log_access_csv(self, request, response, start_time):
        headers = [
            "timestamp",
            "ip",
            "method",
            "path",
            "query_string",
            "protocol",
            "status_code",
            "content_length",
            "response_time_ms",
            "referer",
            "user_agent",
            "blocked",
            "block_reason",
        ]
        row = {
            "timestamp": datetime.now().isoformat(),
            **extract_logging_context_from_fastapi_request(request, ip=get_ip(request)),
            "status_code": response.status_code,
            "content_length": response.headers.get("content-length", 0),
            "response_time_ms": int((time.time() - start_time) * 1000),
            "blocked": getattr(request.state, "aiwaf_blocked", False),
            "block_reason": getattr(request.state, "aiwaf_block_reason", ""),
        }
        row.pop("path_with_query", None)
        if not self.access_log_file.exists():
            with open(self.access_log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        with open(self.access_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(key, "") for key in headers])

    def _log_access_json(self, request, response, start_time):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            **extract_logging_context_from_fastapi_request(request, ip=get_ip(request)),
            "status_code": response.status_code,
            "content_length": response.headers.get("content-length", 0),
            "response_time_ms": int((time.time() - start_time) * 1000),
            "blocked": getattr(request.state, "aiwaf_blocked", False),
            "block_reason": getattr(request.state, "aiwaf_block_reason", ""),
        }
        log_data.pop("path_with_query", None)
        with open(self.access_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")

    def _log_error(self, request, response):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        ctx = extract_logging_context_from_fastapi_request(request, ip=get_ip(request))
        line = (
            f"{timestamp} [error] {response.status_code} from {ctx['ip']}: "
            f'{ctx["method"]} {ctx["path"]} "{ctx["user_agent"] or "-"}"'
        )
        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _log_aiwaf_event(self, request):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        reason = getattr(request.state, "aiwaf_block_reason", "unknown")
        ctx = extract_logging_context_from_fastapi_request(request, ip=get_ip(request))
        line = (
            f"{timestamp} [AIWAF] BLOCKED {ctx['ip']} - {reason} - "
            f'{ctx["method"]} {ctx["path"]} "{ctx["user_agent"] or "-"}"'
        )
        with open(self.aiwaf_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
