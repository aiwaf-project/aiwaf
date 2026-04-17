"""
Main AIWAF class - orchestrates all security middleware components
"""
import logging
from typing import Optional, Dict, Any, Set, List
from fastapi import FastAPI
from contextlib import asynccontextmanager

from .runtime_config import AIWAFConfig, initialize_config
from .runtime_storage import initialize_storage, get_exemption_store
from .runtime_blacklist import BlacklistManager
from aiwaf.fast.middleware import (
    AIAnomalyMiddleware,
    AIWAFLoggingMiddleware,
    GeoBlockMiddleware,
    HeaderValidationMiddleware,
    HoneypotTimingMiddleware,
    IPAndKeywordBlockMiddleware,
    RateLimitMiddleware,
    UUIDTamperMiddleware,
)

logger = logging.getLogger(__name__)


class AIWAF:
    """
    AI Web Application Firewall for FastAPI.
    
    This class provides a comprehensive security middleware suite that includes:
    - Header validation to detect bots and malicious requests
    - IP blacklisting and whitelisting
    - Rate limiting
    - Request monitoring and logging
    
    Usage:
        ```python
        from fastapi import FastAPI
        from aiwaf import AIWAF
        
        app = FastAPI()
        aiwaf = AIWAF(app)
        ```
    """
    
    def __init__(
        self,
        app: Optional[FastAPI] = None,
        config: Optional[AIWAFConfig] = None,
        config_file: Optional[str] = None,
        **config_overrides
    ):
        """
        Initialize AIWAF.
        
        Args:
            app: FastAPI application instance
            config: Configuration object
            config_file: Path to configuration file
            **config_overrides: Configuration overrides
        """
        # Initialize configuration
        if config is not None:
            self.config = config
        else:
            self.config = initialize_config(config_file, **config_overrides)
        
        # Initialize storage
        storage_config = self.config.get_storage_config()
        initialize_storage(**storage_config)
        
        # Auto-setup exemptions
        self._setup_auto_exemptions()
        
        # Initialize middleware
        if app is not None:
            self.init_app(app)
        
        logger.info("AIWAF initialized successfully")
    
    def init_app(self, app: FastAPI):
        """
        Initialize AIWAF with a FastAPI application.
        
        Args:
            app: FastAPI application instance
        """
        self.app = app
        
        # Add middleware in reverse order (FastAPI adds them in reverse)
        self._add_middleware()
        
        # Add startup and shutdown events
        self._add_lifecycle_events(app)
        
        logger.info("AIWAF integrated with FastAPI app")
    
    def _add_middleware(self):
        """Add middleware to the FastAPI app."""
        path_rules = self.config.get("path_rules", []) or self.config.get("PATH_RULES", []) or []

        if self.config.is_enabled("logging_middleware"):
            logging_cfg = self.config.get("logging_middleware", {})
            self.app.add_middleware(
                AIWAFLoggingMiddleware,
                log_dir=logging_cfg.get("log_dir", "aiwaf_logs"),
                log_format=logging_cfg.get("log_format", "combined"),
                path_rules=path_rules,
            )

        if self.config.is_enabled("uuid_tamper"):
            self.app.add_middleware(UUIDTamperMiddleware, path_rules=path_rules)

        if self.config.is_enabled("ai_anomaly"):
            anomaly_cfg = self.config.get("ai_anomaly", {})
            self.app.add_middleware(
                AIAnomalyMiddleware,
                enabled=anomaly_cfg.get("enabled", True),
                path_rules=path_rules,
            )

        if self.config.is_enabled("geo_block"):
            geo_cfg = self.config.get("geo_block", {})
            self.app.add_middleware(
                GeoBlockMiddleware,
                enabled=geo_cfg.get("enabled", False),
                allow_countries=geo_cfg.get("allow_countries", []),
                block_countries=geo_cfg.get("block_countries", []),
                path_rules=path_rules,
            )

        # Header validation middleware
        if self.config.is_enabled('header_validation'):
            header_config = self.config.get_header_validation_config()
            
            # Add the middleware to FastAPI
            self.app.add_middleware(
                HeaderValidationMiddleware,
                enabled=header_config.get('enabled', True),
                block_suspicious=header_config.get('block_suspicious', True),
                quality_threshold=header_config.get('quality_threshold', 3),
                exempt_paths=set(header_config.get('exempt_paths', [])),
                custom_suspicious_patterns=header_config.get('custom_suspicious_patterns', []),
                custom_legitimate_patterns=header_config.get('custom_legitimate_patterns', []),
                trust_legitimate_bots=header_config.get('trust_legitimate_bots', False),
                path_rules=path_rules,
            )
            
            logger.info("Header validation middleware added")
        
        if self.config.is_enabled("honeypot"):
            honeypot_cfg = self.config.get("honeypot", {})
            self.app.add_middleware(
                HoneypotTimingMiddleware,
                min_form_time=honeypot_cfg.get("min_form_time", 1.0),
                path_rules=path_rules,
            )

        if self.config.is_enabled("rate_limiting"):
            rate_cfg = self.config.get_rate_limiting_config()
            self.app.add_middleware(
                RateLimitMiddleware,
                max_requests=rate_cfg.get("max_requests", 20),
                window_seconds=rate_cfg.get("window_seconds", 10),
                flood_threshold=rate_cfg.get("flood_threshold", 40),
                path_rules=path_rules,
            )

        if self.config.is_enabled("ip_keyword_block"):
            ipkw_cfg = self.config.get("ip_keyword_block", {})
            self.app.add_middleware(
                IPAndKeywordBlockMiddleware,
                malicious_keywords=ipkw_cfg.get("malicious_keywords"),
                path_rules=path_rules,
            )
    
    def _add_lifecycle_events(self, app: FastAPI):
        """Add startup and shutdown events."""
        if app.router.lifespan_context is not None:
            logger.debug("Lifespan context already set on app; skipping AIWAF lifecycle registration")
            return

        @asynccontextmanager
        async def aiwaf_lifespan(app: FastAPI):
            """Lifespan context handling AIWAF startup/shutdown tasks."""
            logger.info("AIWAF startup tasks initiated")
            
            # Cleanup expired blocks
            if self.config.get('blacklist.auto_unblock_enabled', True):
                cleaned = BlacklistManager.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired blocks on startup")
            
            try:
                yield
            finally:
                logger.info("AIWAF shutdown tasks initiated")
                storage_backend = self.config.get('storage.backend')
                if storage_backend == 'file':
                    logger.info("Saving AIWAF data to file")
        
        app.router.lifespan_context = aiwaf_lifespan
        logger.info("AIWAF lifespan events registered")
    
    def _setup_auto_exemptions(self):
        """Setup automatic IP exemptions."""
        exemption_config = self.config.get('exemptions', {})
        
        if exemption_config.get('private_ips_exempted', True):
            exemption_store = get_exemption_store()
            
            # Add common private IP patterns
            private_patterns = exemption_config.get('auto_exempt_patterns', [])
            for pattern in private_patterns:
                exemption_store.add_pattern(pattern, "Auto-exempt private IPs")
            
            logger.info(f"Auto-exempted {len(private_patterns)} private IP patterns")
    
    def add_exemption(self, ip: str, reason: str = "Manual exemption"):
        """
        Add an IP to the exemption list.
        
        Args:
            ip: IP address to exempt
            reason: Reason for exemption
        """
        BlacklistManager.add_to_whitelist(ip, reason)
        logger.info(f"IP {ip} added to exemption list: {reason}")
    
    def remove_exemption(self, ip: str) -> bool:
        """
        Remove an IP from the exemption list.
        
        Args:
            ip: IP address to remove
            
        Returns:
            True if removed successfully
        """
        result = BlacklistManager.remove_from_whitelist(ip)
        if result:
            logger.info(f"IP {ip} removed from exemption list")
        return result
    
    def block_ip(self, ip: str, reason: str, duration: Optional[int] = None) -> bool:
        """
        Block an IP address.
        
        Args:
            ip: IP address to block
            reason: Reason for blocking
            duration: Block duration in seconds (None for permanent)
            
        Returns:
            True if blocked successfully
        """
        return BlacklistManager.block(ip, reason, duration)
    
    def unblock_ip(self, ip: str) -> bool:
        """
        Unblock an IP address.
        
        Args:
            ip: IP address to unblock
            
        Returns:
            True if unblocked successfully
        """
        return BlacklistManager.unblock(ip)
    
    def is_blocked(self, ip: str) -> bool:
        """
        Check if an IP is blocked.
        
        Args:
            ip: IP address to check
            
        Returns:
            True if blocked
        """
        return BlacklistManager.is_blocked(ip)
    
    def is_exempted(self, ip: str) -> bool:
        """
        Check if an IP is exempted.
        
        Args:
            ip: IP address to check
            
        Returns:
            True if exempted
        """
        return BlacklistManager.is_whitelisted(ip)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive AIWAF statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'aiwaf': {
                'version': '1.0.0',
                'enabled_features': [
                    feature for feature in ['header_validation', 'rate_limiting', 'blacklist']
                    if self.config.is_enabled(feature)
                ]
            },
            'blacklist': BlacklistManager.get_statistics(),
            'whitelist': BlacklistManager.get_whitelist(),
            'configuration': {
                'storage_backend': self.config.get('storage.backend'),
                'header_validation_enabled': self.config.is_enabled('header_validation'),
                'rate_limiting_enabled': self.config.is_enabled('rate_limiting'),
                'quality_threshold': self.config.get('header_validation.quality_threshold'),
                'rate_limit': {
                    'max_requests': self.config.get('rate_limiting.max_requests'),
                    'window_seconds': self.config.get('rate_limiting.window_seconds')
                }
            }
        }
        
        return stats
    
    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get recent security activity.
        
        Args:
            hours: Hours to look back
            
        Returns:
            Recent activity summary
        """
        recent_blocks = BlacklistManager.get_recent_blocks(hours)
        top_reasons = BlacklistManager.get_top_blocked_reasons()
        
        return {
            'recent_blocks': recent_blocks,
            'top_block_reasons': top_reasons,
            'summary': {
                'blocks_in_period': len(recent_blocks),
                'unique_ips': len(set(block['ip'] for block in recent_blocks)),
                'most_common_reason': top_reasons[0]['reason'] if top_reasons else 'None'
            }
        }
    
    def enable_feature(self, feature: str):
        """
        Enable a security feature.
        
        Args:
            feature: Feature name ('header_validation', 'rate_limiting')
        """
        self.config.enable_feature(feature)
        logger.info(f"Feature enabled: {feature}")
    
    def disable_feature(self, feature: str):
        """
        Disable a security feature.
        
        Args:
            feature: Feature name
        """
        self.config.disable_feature(feature)
        logger.info(f"Feature disabled: {feature}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration at runtime.
        
        Args:
            updates: Configuration updates
        """
        self.config.update(updates)
        
        # Validate updated configuration
        errors = self.config.validate()
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")
        
        logger.info("Configuration updated successfully")
    
    def save_config(self, config_file: str):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration
        """
        self.config.save_to_file(config_file)
    
    def cleanup(self):
        """Perform cleanup tasks."""
        # Cleanup expired blocks
        cleaned_blocks = BlacklistManager.cleanup_expired()
        
        # Cleanup rate limiter
        # (Rate limiter cleans itself automatically)
        
        logger.info(f"Cleanup completed: {cleaned_blocks} expired blocks removed")
        
        return {
            'cleaned_blocks': cleaned_blocks
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of AIWAF components.
        
        Returns:
            Health check results
        """
        health = {
            'status': 'healthy',
            'components': {},
            'errors': []
        }
        
        try:
            # Check storage
            from .runtime_storage import get_storage
            storage = get_storage()
            test_key = '__aiwaf_health_check__'
            storage.set(test_key, 'test', ttl=60)
            if storage.get(test_key) == 'test':
                health['components']['storage'] = 'healthy'
                storage.delete(test_key)
            else:
                health['components']['storage'] = 'unhealthy'
                health['errors'].append('Storage read/write test failed')
        
        except Exception as e:
            health['components']['storage'] = 'error'
            health['errors'].append(f'Storage error: {str(e)}')
        
        try:
            # Check blacklist
            stats = BlacklistManager.get_statistics()
            health['components']['blacklist'] = 'healthy'
        except Exception as e:
            health['components']['blacklist'] = 'error'
            health['errors'].append(f'Blacklist error: {str(e)}')
        
        try:
            # Check exemption store
            exemption_store = get_exemption_store()
            exemption_store.is_exempted('127.0.0.1')  # Test call
            health['components']['exemptions'] = 'healthy'
        except Exception as e:
            health['components']['exemptions'] = 'error'
            health['errors'].append(f'Exemption store error: {str(e)}')
        
        # Overall health
        if health['errors']:
            health['status'] = 'unhealthy' if any('error' in str(e) for e in health['errors']) else 'degraded'
        
        return health
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export AIWAF data for backup or analysis.
        
        Returns:
            Exported data
        """
        return {
            'configuration': self.config.get_all(),
            'statistics': self.get_statistics(),
            'recent_activity': self.get_recent_activity(),
            'health': self.health_check(),
            'export_timestamp': __import__('time').time()
        }
    
    def __repr__(self) -> str:
        """String representation of AIWAF instance."""
        enabled_features = [
            feature for feature in ['header_validation', 'rate_limiting']
            if self.config.is_enabled(feature)
        ]
        
        return f"AIWAF(features={enabled_features}, storage={self.config.get('storage.backend')})"
