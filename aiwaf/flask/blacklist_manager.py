from .storage import is_ip_blacklisted, add_ip_blacklist, remove_ip_blacklist, is_ip_whitelisted
from aiwaf.core.blacklist import should_block_ip, should_unblock_ip
from aiwaf.core.request_context import extract_blacklist_extended_info_from_flask_request

from flask import current_app, has_request_context, has_app_context, request


DEFAULT_CAPTURE_HEADERS = [
    "User-Agent",
    "Accept",
    "Accept-Language",
    "X-Forwarded-For",
    "X-Real-IP",
    "Referer",
]
DEFAULT_REDACT_HEADERS = [
    "Authorization",
    "Cookie",
    "Set-Cookie",
    "X-Api-Key",
]

# Flask-adapted BlacklistManager
class BlacklistManager:
    @classmethod
    def is_blocked(cls, ip):
        enabled = True
        if has_app_context():
            enabled = bool(current_app.config.get("AIWAF_ENABLE_IP_BLOCKING", True))
        if not should_block_ip(enabled, is_ip_whitelisted, ip):
            return False
        return is_ip_blacklisted(ip)
    @classmethod
    def block(cls, ip, reason=None, extended_request_info=None):
        enabled = True
        if has_app_context():
            enabled = bool(current_app.config.get("AIWAF_ENABLE_IP_BLOCKING", True))
        if not should_block_ip(enabled, is_ip_whitelisted, ip):
            return
        if extended_request_info is None:
            extended_request_info = _build_request_info()
        add_ip_blacklist(ip, reason, extended_request_info=extended_request_info)
    @classmethod
    def unblock(cls, ip):
        enabled = True
        if has_app_context():
            enabled = bool(current_app.config.get("AIWAF_ENABLE_IP_BLOCKING", True))
        if not should_unblock_ip(enabled, is_ip_whitelisted, ip):
            return
        remove_ip_blacklist(ip)


def _build_request_info():
    if not has_request_context():
        return None

    enabled = False
    if has_app_context():
        enabled = bool(current_app.config.get("AIWAF_CAPTURE_EXTENDED_REQUEST_INFO", False))
    if not enabled:
        return None

    return extract_blacklist_extended_info_from_flask_request(
        request,
        enabled=True,
        max_bytes=int(current_app.config.get("AIWAF_EXTENDED_REQUEST_INFO_MAX_BYTES", 4096)) if has_app_context() else 4096,
        capture_headers=current_app.config.get(
            "AIWAF_EXTENDED_REQUEST_INFO_HEADERS",
            DEFAULT_CAPTURE_HEADERS,
        ) if has_app_context() else DEFAULT_CAPTURE_HEADERS,
        redact_headers=current_app.config.get(
            "AIWAF_EXTENDED_REQUEST_INFO_REDACT_HEADERS",
            DEFAULT_REDACT_HEADERS,
        ) if has_app_context() else DEFAULT_REDACT_HEADERS,
    )
