# Flask AIWAF master middleware loader
from .ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from .rate_limit_middleware import RateLimitMiddleware
from .honeypot_timing_middleware import HoneypotTimingMiddleware
from .header_validation_middleware import HeaderValidationMiddleware
from .anomaly_middleware import AIAnomalyMiddleware
from .uuid_tamper_middleware import UUIDTamperMiddleware
from .logging_middleware import AIWAFLoggingMiddleware
from .geo_block_middleware import GeoBlockMiddleware

all = "all"

def register_aiwaf_middlewares(app, use_database=None, middlewares=None, disable_middlewares=None):
    """
    Register AIWAF middlewares with the Flask app.
    
    Args:
        app: Flask application instance
        use_database: Optional boolean to force database usage.
                     If None, auto-detects based on configuration.
        middlewares: List of middleware names to enable (if None, enables all)
        disable_middlewares: List of middleware names to disable
        
    Available middlewares:
        - ip_keyword_block: IP and keyword blocking
        - rate_limit: Rate limiting protection
        - honeypot: Honeypot timing protection
        - header_validation: HTTP header validation
        - geo_block: Geo-blocking by country
        - ai_anomaly: AI-powered anomaly detection
        - uuid_tamper: UUID tampering protection
        - logging: Request/response logging
    """
    # Import the main AIWAF class to handle registration
    from . import AIWAF
    
    # Create AIWAF instance with specified configuration
    aiwaf = AIWAF()
    aiwaf.init_app(
        app,
        middlewares=middlewares,
        disable_middlewares=disable_middlewares,
        use_database=use_database,
    )
    
    return aiwaf

def _should_use_database(app):
    """Check if the app has database configuration."""
    # If CSV is explicitly enabled, don't use database
    if app.config.get('AIWAF_USE_CSV', True):
        return False
    
    # Only use database if SQLAlchemy URI is configured and CSV is disabled
    return (hasattr(app.config, 'get') and 
            app.config.get('SQLALCHEMY_DATABASE_URI') is not None)

def _init_database(app):
    """Initialize database if not already done."""
    try:
        from .db_models import db
        
        # Only initialize if not already done
        if not hasattr(app, 'extensions') or 'sqlalchemy' not in app.extensions:
            db.init_app(app)
        
        # Create tables within app context
        with app.app_context():
            db.create_all()
    except Exception as e:
        # If database setup fails, continue with CSV/memory storage
        app.logger.warning(f"Database setup failed, using CSV/memory storage: {e}")
