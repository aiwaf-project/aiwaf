# aiwaf/blacklist_manager.py

from django.conf import settings
from .storage import get_blacklist_store
from .utils import is_ip_exempted
from ..core.blacklist import should_block_ip, should_unblock_ip

class BlacklistManager:
    @staticmethod
    def block(ip, reason, extended_request_info=None):
        """Add IP to blacklist, but only if it's not exempted"""
        if not should_block_ip(
            getattr(settings, "AIWAF_ENABLE_IP_BLOCKING", True),
            is_ip_exempted,
            ip,
        ):
            return
        
        store = get_blacklist_store()
        store.block_ip(ip, reason, extended_request_info=extended_request_info)

    @staticmethod
    def is_blocked(ip):
        """Check if IP is blocked, but respect exemptions"""
        if not should_block_ip(
            getattr(settings, "AIWAF_ENABLE_IP_BLOCKING", True),
            is_ip_exempted,
            ip,
        ):
            return False
        
        # If not exempted, check blacklist
        store = get_blacklist_store()
        return store.is_blocked(ip)

    @staticmethod
    def all_blocked():
        store = get_blacklist_store()
        return store.get_all_blocked_ips()
    
    @staticmethod
    def unblock(ip):
        if not should_unblock_ip(
            getattr(settings, "AIWAF_ENABLE_IP_BLOCKING", True),
            is_ip_exempted,
            ip,
        ):
            return
        store = get_blacklist_store()
        store.unblock_ip(ip)
