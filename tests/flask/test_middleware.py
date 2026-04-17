import pytest
import time
from unittest.mock import patch
from flask import Flask
from aiwaf.flask.middleware import register_aiwaf_middlewares
from aiwaf.flask.db_models import db
from aiwaf.flask.storage import add_ip_blacklist, add_keyword, add_geo_blocked_country

@pytest.fixture
def middleware_app():
    """Create Flask app with AIWAF middleware."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['AIWAF_RATE_WINDOW'] = 60
    app.config['AIWAF_RATE_MAX'] = 2
    app.config['AIWAF_RATE_FLOOD'] = 3
    app.config['AIWAF_MIN_FORM_TIME'] = 0.5
    
    # Force database mode for tests and disable path exemptions
    app.config['AIWAF_USE_CSV'] = False
    app.config['AIWAF_EXEMPT_PATHS'] = set()
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        register_aiwaf_middlewares(app)
        
        @app.route('/test')
        def test_route():
            return 'OK'
            
        @app.route('/test_post', methods=['POST'])
        def test_post():
            return 'POST OK'
            
        yield app
        db.drop_all()

def test_blacklisted_ip_blocked(app):
    """Test that blacklisted IP is blocked."""
    from aiwaf.flask.ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
    
    # Only register IP blocking middleware for this test
    IPAndKeywordBlockMiddleware(app)
    
    with app.app_context():
        add_ip_blacklist('127.0.0.1', 'test')
    
    @app.route('/test')
    def test_route():
        return 'OK'
    
    client = app.test_client()
    response = client.get('/test', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 403

def test_keyword_blocking(middleware_app):
    """Test keyword blocking in URL path."""
    client = middleware_app.test_client()
    response = client.get('/test.php')  # Should be blocked
    assert response.status_code == 403

def test_rate_limiting(app):
    """Test rate limiting functionality."""
    from aiwaf.flask.rate_limit_middleware import RateLimitMiddleware, _aiwaf_cache
    
    # Clear the cache before testing
    _aiwaf_cache.clear()
    
    # Configure for testing
    app.config['AIWAF_RATE_WINDOW'] = 2
    app.config['AIWAF_RATE_MAX'] = 2
    app.config['AIWAF_RATE_FLOOD'] = 3
    
    # Only register rate limiting middleware for this test
    RateLimitMiddleware(app)
    
    @app.route('/test')
    def test_route():
        return 'OK'
    
    client = app.test_client()
    
    # First requests should succeed
    response1 = client.get('/test', headers={'User-Agent': 'Test Browser 1.0'})
    response2 = client.get('/test', headers={'User-Agent': 'Test Browser 1.0'})
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    # Third request should be rate limited
    response3 = client.get('/test', headers={'User-Agent': 'Test Browser 1.0'})
    assert response3.status_code == 429

def test_header_validation(middleware_app):
    """Test header validation."""
    client = middleware_app.test_client()
    
    # Request without User-Agent should be blocked
    response = client.get('/test', headers={'User-Agent': ''})
    assert response.status_code == 403
    
    # Request with short User-Agent should be blocked
    response = client.get('/test', headers={'User-Agent': 'short'})
    assert response.status_code == 403

def test_uuid_tampering(app):
    """Test UUID tampering detection."""
    from aiwaf.flask.uuid_tamper_middleware import UUIDTamperMiddleware
    
    # Only register UUID tampering middleware for this test
    UUIDTamperMiddleware(app)
    
    @app.route('/test')
    def test_route():
        return 'OK'
    
    client = app.test_client()
    
    # Invalid UUID should be blocked
    response = client.get('/test?uuid=invalid-uuid', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 403
    
    # Valid UUID should pass
    response = client.get('/test?uuid=550e8400-e29b-41d4-a716-446655440000', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 200

def test_geo_block_allowlist(app, monkeypatch):
    """Test geo blocking allowlist."""
    from aiwaf.flask.geo_block_middleware import GeoBlockMiddleware

    app.config['AIWAF_GEO_BLOCK_ENABLED'] = True
    app.config['AIWAF_GEO_ALLOW_COUNTRIES'] = ['US']
    app.config['AIWAF_GEO_BLOCK_COUNTRIES'] = []
    app.config['AIWAF_EXEMPT_PATHS'] = set()

    monkeypatch.setattr(
        'aiwaf.flask.geo_block_middleware.get_country_for_ip',
        lambda ip, config: 'US'
    )

    GeoBlockMiddleware(app)

    @app.route('/geo-allow')
    def geo_allow():
        return 'OK'

    client = app.test_client()
    response = client.get('/geo-allow', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 200

def test_geo_block_blocklist(app, monkeypatch):
    """Test geo blocking blocklist."""
    from aiwaf.flask.geo_block_middleware import GeoBlockMiddleware

    app.config['AIWAF_GEO_BLOCK_ENABLED'] = True
    app.config['AIWAF_GEO_ALLOW_COUNTRIES'] = []
    app.config['AIWAF_GEO_BLOCK_COUNTRIES'] = ['US']
    app.config['AIWAF_EXEMPT_PATHS'] = set()

    monkeypatch.setattr(
        'aiwaf.flask.geo_block_middleware.get_country_for_ip',
        lambda ip, config: 'US'
    )

    GeoBlockMiddleware(app)

    @app.route('/geo-block')
    def geo_block():
        return 'OK'

    client = app.test_client()
    response = client.get('/geo-block', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 403

def test_geo_block_dynamic_blocked(app, monkeypatch):
    """Test dynamic geo blocked countries."""
    from aiwaf.flask.geo_block_middleware import GeoBlockMiddleware

    app.config['AIWAF_GEO_BLOCK_ENABLED'] = True
    app.config['AIWAF_GEO_ALLOW_COUNTRIES'] = []
    app.config['AIWAF_GEO_BLOCK_COUNTRIES'] = []
    app.config['AIWAF_EXEMPT_PATHS'] = set()

    add_geo_blocked_country('FR')

    monkeypatch.setattr(
        'aiwaf.flask.geo_block_middleware.get_country_for_ip',
        lambda ip, config: 'FR'
    )

    GeoBlockMiddleware(app)

    @app.route('/geo-dynamic')
    def geo_dynamic():
        return 'OK'

    client = app.test_client()
    response = client.get('/geo-dynamic', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 403

@patch('time.time')
def test_honeypot_timing(mock_time, middleware_app):
    """Test honeypot timing detection."""
    client = middleware_app.test_client()
    
    # Simulate GET request
    mock_time.return_value = 1000.0
    client.get('/test_post')
    
    # Simulate POST too quickly after GET
    mock_time.return_value = 1000.3  # 0.3 seconds later (< 0.5 min time)
    response = client.post('/test_post')
    assert response.status_code == 403


@patch('time.time')
def test_honeypot_timing_skips_authenticated_requests(mock_time, middleware_app, monkeypatch):
    """Authenticated requests can bypass honeypot timing checks when enabled."""
    from aiwaf.flask import honeypot_timing_middleware as hm

    middleware_app.config['AIWAF_HONEYPOT_SKIP_AUTHENTICATED'] = True
    monkeypatch.setattr(hm, "_is_authenticated_request", lambda: True)

    client = middleware_app.test_client()

    mock_time.return_value = 2000.0
    client.get('/test_post')

    mock_time.return_value = 2000.1
    response = client.post('/test_post')
    assert response.status_code == 200


@patch('time.time')
def test_honeypot_timing_auth_bypass_can_be_disabled(mock_time, middleware_app, monkeypatch):
    """When auth bypass is disabled, honeypot timing still blocks fast posts."""
    from aiwaf.flask import honeypot_timing_middleware as hm

    middleware_app.config['AIWAF_HONEYPOT_SKIP_AUTHENTICATED'] = False
    monkeypatch.setattr(hm, "_is_authenticated_request", lambda: True)

    client = middleware_app.test_client()

    mock_time.return_value = 3000.0
    client.get('/test_post')

    mock_time.return_value = 3000.1
    response = client.post('/test_post')
    assert response.status_code == 403
