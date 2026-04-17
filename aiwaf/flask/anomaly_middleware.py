"""
AI Anomaly Detection Middleware for Flask

This middleware uses machine learning to detect anomalous behavior patterns
and automatically blocks malicious IPs based on request characteristics.
"""

import re
import time
import logging
from flask import request, jsonify, g, current_app
from .utils import get_ip, is_exempt, is_path_exempt
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware
from aiwaf.core import rust_backend

# Try to import numpy and ML dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import dependencies for model loading
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False
    pickle = None

# Static malicious keywords (similar to Django implementation)
STATIC_KEYWORDS = {
    'admin', 'wp-admin', 'wp-content', 'wp-includes', 'wp-config', 'xmlrpc',
    'phpmyadmin', 'adminer', 'config', 'configuration', 'settings', 'setup',
    'install', 'installer', 'backup', 'database', 'mysql', 'sql', 'dump',
    '.env', '.git', '.htaccess', '.htpasswd', 'passwd', 'shadow', 'robots',
    'cgi-bin', 'scripts', 'shell', 'cmd', 'exec', 'system', 'eval', 'base64',
    'union', 'select', 'drop', 'delete', 'insert', 'update', 'script', 'javascript',
    'onload', 'onerror', 'onclick', 'document', 'cookie', 'alert'
}

# Status code mapping for ML features
STATUS_CODES = ['200', '201', '204', '301', '302', '400', '401', '403', '404', '405', '500', '502', '503']

class AIAnomalyMiddleware:
    """
    AI-powered anomaly detection middleware for Flask.
    
    Analyzes request patterns and uses machine learning to detect
    and block malicious behavior automatically.
    """
    
    def __init__(self, app=None):
        self.app = app
        self.model = None
        self.model_is_rust = False
        self.malicious_keywords = set(STATIC_KEYWORDS)
        self.request_cache = {}  # Simple in-memory cache for request history
        self.window_seconds = 60
        self.top_n = 10
        
        # Periodic AI check
        self.last_ai_check = 0
        self.ai_check_interval = app.config.get('AIWAF_AI_CHECK_INTERVAL', 3600) if app else 3600  # Default: Check every hour
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize the middleware with Flask app."""
        self.app = app
        
        # Configuration
        self.window_seconds = app.config.get('AIWAF_WINDOW_SECONDS', 60)
        self.top_n = app.config.get('AIWAF_DYNAMIC_TOP_N', 10)
        
        # Try to load ML model
        self._load_model(app)
        
        # Register middleware hooks
        app.before_request(self.before_request)
        app.after_request(self.after_request)

    def _get_default_model_path(self):
        """Get the default model path relative to the package."""
        from pathlib import Path
        
        # Get the directory where this file is located
        middleware_dir = Path(__file__).parent
        resources_dir = middleware_dir / 'resources'
        
        # Ensure resources directory exists
        resources_dir.mkdir(exist_ok=True)
        
        return str(resources_dir / 'model.pkl')

    def _check_log_data_sufficiency(self, app):
        """Check if there's enough log data to justify AI model usage."""
        min_ai_threshold = app.config.get('AIWAF_MIN_AI_LOGS', 10000)
        log_dir = app.config.get('AIWAF_LOG_DIR', 'logs')
        
        total_lines = 0
        try:
            import os
            import glob
            
            # Count lines in all log files
            log_patterns = [
                os.path.join(log_dir, '*.log'),
                os.path.join(log_dir, '*.csv'),
                os.path.join(log_dir, '*.json'),
                os.path.join(log_dir, '*.jsonl'),
            ]
            
            for pattern in log_patterns:
                for log_file in glob.glob(pattern):
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for _ in f)
                    except Exception as e:
                        self.logger.debug(f"Error reading {log_file}: {e}")
                        continue
            
            self.logger.info(f"Found {total_lines} total log lines, threshold: {min_ai_threshold}")
            return total_lines >= min_ai_threshold
            
        except Exception as e:
            self.logger.warning(f"Could not check log data sufficiency: {e}")
            return True  # Default to allowing AI if we can't check

    def _check_ai_status_periodically(self, app):
        """Periodically re-evaluate whether AI should be enabled based on current log data."""
        import time
        
        current_time = time.time()
        
        # Only check periodically to avoid performance impact
        if current_time - self.last_ai_check < self.ai_check_interval:
            return
            
        self.last_ai_check = current_time
        
        # Re-evaluate log data sufficiency
        force_ai = app.config.get('AIWAF_FORCE_AI', False)
        
        if force_ai:
            # If forcing AI, ensure model is loaded if available
            if self.model is None:
                self.logger.info("Force AI enabled - attempting to reload model")
                self._load_model(app)
            return
        
        # Check current log data sufficiency
        has_sufficient_data = self._check_log_data_sufficiency(app)
        
        if has_sufficient_data and self.model is None:
            # We now have enough data but no model - try to load it
            self.logger.info("Sufficient log data detected - attempting to load AI model")
            self._load_model(app)
        elif not has_sufficient_data and self.model is not None:
            # We no longer have enough data - disable AI
            self.logger.info("Insufficient log data detected - disabling AI model")
            self.model = None

    def _load_model(self, app):
        """Load the machine learning model if available and sufficient data exists."""
        self.model = None
        self.model_is_rust = False

        def _assign_model(model_data, source_label):
            if isinstance(model_data, dict):
                if model_data.get("model_type") == "aiwaf_rust.IsolationForest":
                    rust_model = rust_backend.rust_isolation_forest_from_json(
                        model_data.get("model_state")
                    )
                    if rust_model is not None:
                        self.model = rust_model
                        self.model_is_rust = True
                        self.logger.info(f"Loaded AI model from {model_path} (aiwaf-rust, {source_label})")
                        return True
                if 'model' in model_data:
                    self.model = model_data['model']
                    self.logger.info(f"Loaded AI model from {model_path} (with metadata, {source_label})")
                    return True
            self.model = model_data
            self.logger.info(f"Loaded AI model from {model_path} (legacy format, {source_label})")
            return True
        # Check if we have enough log data for AI to be effective (unless forced)
        force_ai = app.config.get('AIWAF_FORCE_AI', False)
        if not force_ai and not self._check_log_data_sufficiency(app):
            self.logger.info("Insufficient log data for AI anomaly detection - using keyword-only mode")
            self.logger.info("Use AIWAF_FORCE_AI=True to override this behavior")
            self.model = None
            return
        elif force_ai:
            self.logger.info("AI model loading forced despite potentially insufficient log data")
        
        # Get model path - use package-relative path by default
        default_model_path = self._get_default_model_path()
        model_path = app.config.get('AIWAF_MODEL_PATH', default_model_path)
        
        if JOBLIB_AVAILABLE:
            try:
                import os
                import warnings
                if os.path.exists(model_path):
                    # Try joblib first (preferred format from trainer)
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                            model_data = joblib.load(model_path)
                            _assign_model(model_data, "joblib")
                    
                    except Exception as joblib_error:
                        # Fallback to pickle if joblib fails
                        if PICKLE_AVAILABLE:
                            self.logger.warning(f"Joblib failed, trying pickle fallback: {joblib_error}")
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                                with open(model_path, 'rb') as f:
                                    model_data = pickle.load(f)
                                _assign_model(model_data, "pickle")
                        else:
                            raise joblib_error
                            
                else:
                    self.logger.warning(f"AI model not found at {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load AI model: {e}")
                self.logger.info("AI anomaly detection will continue without ML model (keyword-based only)")
                self.model = None
                self.model_is_rust = False
        else:
            if not JOBLIB_AVAILABLE:
                self.logger.warning("Joblib not available - trying pickle fallback")
                if PICKLE_AVAILABLE:
                    try:
                        import os
                        import warnings
                        if os.path.exists(model_path):
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                                with open(model_path, 'rb') as f:
                                    model_data = pickle.load(f)
                                _assign_model(model_data, "pickle only")
                        else:
                            self.logger.warning(f"AI model not found at {model_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to load AI model with pickle: {e}")
                        self.model = None
                        self.model_is_rust = False
                else:
                    self.logger.warning("Neither joblib nor pickle available - AI anomaly detection disabled")
            if not NUMPY_AVAILABLE:
                self.logger.warning("NumPy not available - AI anomaly detection disabled")

        if self.model is not None and not self.model_is_rust and not NUMPY_AVAILABLE:
            self.logger.warning("NumPy not available - disabling sklearn-based AI model")
            self.model = None
            self.model_is_rust = False

    def _is_malicious_context(self, request_obj, keyword):
        """
        Determine if a keyword appears in a malicious context.
        Only learn keywords when we have strong indicators of malicious intent.
        """
        path = request_obj.path.lower()
        query_string = request_obj.query_string.decode('utf-8', errors='ignore').lower()
        
        # Strong malicious indicators
        malicious_indicators = [
            # Multiple consecutive suspicious segments
            len([seg for seg in re.split(r"\W+", path) if seg in self.malicious_keywords]) > 1,
            
            # Common attack patterns in path
            any(pattern in path for pattern in [
                '../', '..\\', '.env', 'wp-admin', 'phpmyadmin', 'config',
                'backup', 'database', 'mysql', 'passwd', 'shadow', 'admin',
                'shell', 'cmd', 'exec', 'system'
            ]),
            
            # Suspicious query parameters
            any(param in request_obj.args for param in ['cmd', 'exec', 'system', 'shell', 'eval']),
            
            # Multiple directory traversal attempts
            path.count('../') > 2 or path.count('..\\') > 2,
            
            # Encoded attack patterns
            any(encoded in path for encoded in ['%2e%2e', '%252e', '%c0%ae', '%2f', '%5c']),
            
            # SQL injection patterns
            any(sql_pattern in query_string for sql_pattern in [
                'union select', 'drop table', 'insert into', 'delete from', 'update set'
            ]),
            
            # XSS patterns
            any(xss_pattern in query_string for xss_pattern in [
                '<script', 'javascript:', 'onload=', 'onerror=', 'document.cookie'
            ])
        ]
        
        return any(malicious_indicators)

    def _route_exists(self, path):
        """
        Check if a route exists in the Flask application.
        This is the Flask equivalent of Django's path_exists_in_django.
        """
        try:
            from flask import current_app
            from werkzeug.routing import RequestRedirect, MethodNotAllowed
            
            # Test if the path matches any route
            adapter = current_app.url_map.bind('localhost')
            try:
                adapter.match(path)
                return True
            except (RequestRedirect, MethodNotAllowed):
                # Route exists but wrong method or redirect
                return True
            except:
                # No route found
                return False
        except:
            return False

    def _calculate_features(self, request_obj, ip, response_time=0):
        """Calculate ML features for the current request."""
        path = request_obj.path
        path_len = len(path)
        
        # Check if route exists
        known_path = self._route_exists(path)
        
        # Count keyword hits
        kw_hits = 0
        if not known_path and not is_path_exempt(path):
            kw_hits = sum(1 for kw in self.malicious_keywords if kw in path.lower())
        
        # Get request history for this IP
        now = time.time()
        key = f"aiwaf:{ip}"
        data = self.request_cache.get(key, [])
        
        # Calculate burst count (requests within last 10 seconds)
        burst_count = sum(1 for (t, _, _, _) in data if now - t <= 10)
        
        # Calculate total 404s
        total_404 = sum(1 for (_, _, st, _) in data if st == 404)
        
        # Status code index (default to -1 for unknown)
        status_idx = -1  # Will be set when we know the response status
        
        return [path_len, kw_hits, response_time, status_idx, burst_count, total_404]

    def before_request(self):
        """Process request before it reaches the route handler."""
        # Check exemption status first - skip if exempt from AI anomaly detection
        if not should_apply_middleware('ai_anomaly'):
            return None  # Allow request to proceed without AI anomaly checking
        
        # Periodically check if AI should be enabled/disabled
        self._check_ai_status_periodically(current_app)
        
        # Legacy exemption check for backward compatibility
        if is_exempt(request):
            return None
            
        # Record start time for response time calculation
        g.aiwaf_start_time = time.time()
        
        # Get client IP
        ip = get_ip()
        
        # Check if IP is already blocked
        if BlacklistManager.is_blocked(ip):
            return jsonify({"error": "blocked"}), 403
            
        return None

    def after_request(self, response):
        """Process response after request completion."""
        # Check if request should be exempt
        if is_exempt(request):
            return response
            
        ip = get_ip()
        now = time.time()
        
        # Calculate response time
        start_time = getattr(g, 'aiwaf_start_time', now)
        resp_time = now - start_time
        
        # Get request cache for this IP
        key = f"aiwaf:{ip}"
        data = self.request_cache.get(key, [])
        
        # Calculate features for ML model
        features = self._calculate_features(request, ip, resp_time)
        
        # Update status code index in features
        status_code = str(response.status_code)
        if status_code in STATUS_CODES:
            features[3] = STATUS_CODES.index(status_code)
        
        # Only use AI model if it's available (NumPy required for sklearn, not for rust)
        if self.model is not None and (self.model_is_rust or NUMPY_AVAILABLE):
            try:
                if self.model_is_rust:
                    X = [list(map(float, features))]
                else:
                    X = np.array(features, dtype=float).reshape(1, -1)

                if self.model.predict(X)[0] == -1:  # -1 indicates anomaly
                    self.logger.info(f"AI detected anomaly for IP {ip}: {features}")
                    
                    # Analyze patterns before blocking (like Django implementation)
                    recent_data = [d for d in data if now - d[0] <= 300]  # Last 5 minutes
                    
                    if recent_data:
                        use_rust = (
                            self.app.config.get("AIWAF_USE_RUST", False)
                            and rust_backend.rust_available()
                        )
                        rust_result = None
                        if use_rust:
                            rust_entries = []
                            for entry_time, entry_path, entry_status, entry_resp_time in recent_data:
                                entry_known_path = self._route_exists(entry_path)
                                kw_check = not entry_known_path and not is_path_exempt(entry_path)
                                rust_entries.append({
                                    "path_lower": entry_path.lower(),
                                    "timestamp": float(entry_time),
                                    "status": int(entry_status),
                                    "kw_check": kw_check,
                                })
                            rust_result = rust_backend.analyze_recent_behavior(
                                rust_entries,
                                list(self.malicious_keywords),
                            )

                        if rust_result:
                            avg_kw_hits = rust_result.get("avg_kw_hits", 0.0)
                            max_404s = rust_result.get("max_404s", 0)
                            avg_burst = rust_result.get("avg_burst", 0.0)
                            total_requests = rust_result.get("total_requests", len(recent_data))
                            scanning_404s = rust_result.get("scanning_404s", 0)
                            legitimate_404s = rust_result.get("legitimate_404s", max_404s - scanning_404s)
                            should_block = rust_result.get("should_block", False)
                        else:
                            # Calculate behavior metrics (Python fallback)
                            recent_kw_hits = []
                            recent_404s = 0
                            recent_burst_counts = []
                            scanning_404s = 0
                            
                            for entry_time, entry_path, entry_status, entry_resp_time in recent_data:
                                # Calculate keyword hits for this entry
                                entry_known_path = self._route_exists(entry_path)
                                entry_kw_hits = 0
                                if not entry_known_path and not is_path_exempt(entry_path):
                                    entry_kw_hits = sum(1 for kw in self.malicious_keywords if kw in entry_path.lower())
                                recent_kw_hits.append(entry_kw_hits)
                                
                                # Count 404s and scanning 404s
                                if entry_status == 404:
                                    recent_404s += 1
                                    from aiwaf.core.training_logic import is_scanning_path
                                    if is_scanning_path(entry_path):
                                        scanning_404s += 1
                                
                                # Calculate burst for this entry
                                entry_burst = sum(1 for (t, _, _, _) in recent_data if abs(entry_time - t) <= 10)
                                recent_burst_counts.append(entry_burst)
                            
                            # Calculate averages and metrics
                            avg_kw_hits = sum(recent_kw_hits) / len(recent_kw_hits) if recent_kw_hits else 0
                            max_404s = recent_404s
                            avg_burst = sum(recent_burst_counts) / len(recent_burst_counts) if recent_burst_counts else 0
                            total_requests = len(recent_data)
                            legitimate_404s = max_404s - scanning_404s
                            
                            # Enhanced blocking logic - don't block legitimate behavior
                            should_block = not (
                                avg_kw_hits < 3 and           # Allow some keyword hits
                                scanning_404s < 5 and        # Focus on scanning 404s
                                legitimate_404s < 20 and     # Allow legitimate 404s
                                avg_burst < 25 and           # Allow higher burst
                                total_requests < 150         # Allow more total requests
                            )

                            # High burst alone is not enough to block if there are no signals of abuse
                            if avg_kw_hits == 0 and max_404s == 0:
                                should_block = False
                        
                        if should_block:
                            reason = f"AI anomaly + scanning behavior (404s:{max_404s}, scanning:{scanning_404s}, kw:{avg_kw_hits:.1f}, burst:{avg_burst:.1f})"
                            BlacklistManager.block(ip, reason)
                            self.logger.warning(f"Blocked IP {ip}: {reason}")
                            
                            if BlacklistManager.is_blocked(ip):
                                return jsonify({"error": "blocked"}), 403
                    else:
                        # No recent data - be more conservative
                        from aiwaf.core.training_logic import is_scanning_path
                        current_scanning = is_scanning_path(request.path)
                        current_kw_hits = sum(1 for kw in self.malicious_keywords if kw in request.path.lower())
                        
                        if current_kw_hits >= 3 and current_scanning:
                            reason = f"AI anomaly + scanning behavior (kw:{current_kw_hits}, scanning_path:{request.path})"
                            BlacklistManager.block(ip, reason)
                            self.logger.warning(f"Blocked IP {ip}: {reason}")
                            
                            if BlacklistManager.is_blocked(ip):
                                return jsonify({"error": "blocked"}), 403
                                
            except Exception as e:
                self.logger.error(f"Error in AI anomaly detection: {e}")
        
        # Update request cache
        data.append((now, request.path, response.status_code, resp_time))
        # Keep only recent data within the window
        data = [d for d in data if now - d[0] < self.window_seconds]
        self.request_cache[key] = data
        
        # Learn keywords from 404 responses on non-existent paths
        if (response.status_code == 404 and 
            not self._route_exists(request.path) and 
            not is_path_exempt(request.path)):
            
            try:
                from .storage import get_keyword_store
                keyword_store = get_keyword_store()
                
                # Extract and learn potential malicious keywords
                for seg in re.split(r"\W+", request.path.lower()):
                    if (len(seg) > 3 and 
                        seg not in self.malicious_keywords and
                        self._is_malicious_context(request, seg)):
                        keyword_store.add_keyword(seg)
                        self.malicious_keywords.add(seg)  # Update local cache
                        self.logger.info(f"Learned new malicious keyword: {seg}")
            except Exception as e:
                self.logger.error(f"Error learning keywords: {e}")
        
        return response

    def get_stats(self):
        """Get statistics about the anomaly detection middleware."""
        return {
            'model_loaded': self.model is not None,
            'numpy_available': NUMPY_AVAILABLE,
            'joblib_available': JOBLIB_AVAILABLE,
            'pickle_available': PICKLE_AVAILABLE,
            'cached_ips': len(self.request_cache),
            'malicious_keywords': len(self.malicious_keywords),
            'window_seconds': self.window_seconds
        }
