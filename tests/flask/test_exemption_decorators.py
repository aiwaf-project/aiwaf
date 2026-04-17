#!/usr/bin/env python3
"""
Comprehensive tests for AIWAF Flask exemption decorators

Tests all exemption scenarios:
- @aiwaf_exempt: Full exemption from all middlewares
- @aiwaf_exempt_from: Partial exemption from specific middlewares
- @aiwaf_only: Apply only specific middlewares
- @aiwaf_require_protection: Force protection even if exempted elsewhere
"""

import time
import tempfile
import shutil
from flask import Flask, jsonify, g
from aiwaf.flask import (
    AIWAF, 
    aiwaf_exempt, 
    aiwaf_exempt_from, 
    aiwaf_only,
    aiwaf_require_protection,
    should_apply_middleware
)

def _setup_csv_storage(app):
    temp_dir = tempfile.mkdtemp()
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = temp_dir
    return temp_dir

def _cleanup_csv_storage(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)

def reset_rate_limit_cache():
    import aiwaf.flask.rate_limit_middleware as rl_mod
    rl_mod._aiwaf_cache.clear()


def test_full_exemption():
    """Test @aiwaf_exempt decorator bypasses all middlewares"""
    print(" Testing @aiwaf_exempt (full exemption)")
    
    app = Flask(__name__)
    app.config.update({
        'AIWAF_LOG_DIR': 'aiwaf_logs',
        'AIWAF_RATE_MAX': 1,        # Very low limit to trigger easily
        'AIWAF_MIN_FORM_TIME': 10,  # High time to trigger honeypot
        'AIWAF_MIN_AI_LOGS': 10,    # Low threshold for AI
    })
    temp_dir = _setup_csv_storage(app)
    
    # Register AIWAF with all middlewares
    aiwaf = AIWAF()
    aiwaf.init_app(app)
    
    @app.route('/health')
    @aiwaf_exempt
    def health_check():
        return jsonify({'status': 'ok', 'protected': False})
    
    @app.route('/protected')
    def protected_endpoint():
        return jsonify({'status': 'ok', 'protected': True})
    
    headers = {
        'User-Agent': 'Test Browser 1.0',
        'X-Forwarded-For': '203.0.113.10'
    }
    with app.test_client() as client:
        # Test exempt endpoint - should work even with aggressive limits
        for i in range(5):  # Way over rate limit
            response = client.get('/health', headers=headers)
            assert response.status_code == 200
            data = response.get_json()
            assert data['protected'] == False
        
        print("    Exempt endpoint bypassed rate limiting")
        
        # Test protected endpoint - should be blocked by rate limiting
        response1 = client.get('/protected', headers=headers)
        assert response1.status_code == 200  # First request OK
        
        response2 = client.get('/protected', headers=headers)
        assert response2.status_code == 429  # Second request blocked by rate limit
        
        print("    Protected endpoint enforced rate limiting")
    
    _cleanup_csv_storage(temp_dir)
    print("    Full exemption test passed!\n")


def test_partial_exemption():
    """Test @aiwaf_exempt_from decorator exempts specific middlewares"""
    print(" Testing @aiwaf_exempt_from (partial exemption)")
    
    app = Flask(__name__)
    app.config.update({
        'AIWAF_LOG_DIR': 'aiwaf_logs',
        'AIWAF_RATE_MAX': 1,        # Very low rate limit
        'AIWAF_MIN_AI_LOGS': 10,    # Low threshold for AI
    })
    temp_dir = _setup_csv_storage(app)
    
    aiwaf = AIWAF()
    aiwaf.init_app(app)
    
    @app.route('/webhook')
    @aiwaf_exempt_from('rate_limit', 'ai_anomaly')
    def webhook():
        # This should bypass rate limiting and AI but still check other things
        return jsonify({'received': True})
    
    @app.route('/normal')
    def normal_endpoint():
        return jsonify({'normal': True})
    
    headers = {
        'User-Agent': 'Test Browser 1.0',
        'X-Forwarded-For': '203.0.113.10'
    }
    with app.test_client() as client:
        # Test webhook - should bypass rate limiting
        for i in range(3):  # Over rate limit
            response = client.get('/webhook', headers=headers)
            assert response.status_code == 200
        
        print("    Webhook bypassed rate limiting")
        
        # Test normal endpoint - should be rate limited
        response1 = client.get('/normal', headers=headers)
        assert response1.status_code == 200  # First OK
        
        response2 = client.get('/normal', headers=headers)
        assert response2.status_code == 429  # Blocked by rate limit
        
        print("    Normal endpoint enforced rate limiting")
    
    _cleanup_csv_storage(temp_dir)
    print("    Partial exemption test passed!\n")


def test_middleware_only():
    """Test @aiwaf_only decorator applies only specific middlewares"""
    print(" Testing @aiwaf_only (selective protection)")
    
    app = Flask(__name__)
    app.config.update({
        'AIWAF_LOG_DIR': 'aiwaf_logs',
        'AIWAF_RATE_MAX': 1,        # Very low rate limit
        'AIWAF_MIN_AI_LOGS': 10,    # Low threshold for AI
    })
    temp_dir = _setup_csv_storage(app)
    
    aiwaf = AIWAF()
    aiwaf.init_app(app)
    
    @app.route('/api/sensitive')
    @aiwaf_only('ip_keyword_block', 'rate_limit')  # Only IP blocking and rate limiting
    def sensitive_api():
        return jsonify({'sensitive': True})
    
    headers = {'User-Agent': 'Test Browser 1.0'}
    with app.test_client() as client:
        # Test that rate limiting still works
        response1 = client.get('/api/sensitive', headers=headers)
        assert response1.status_code == 200
        
        response2 = client.get('/api/sensitive', headers=headers)
        assert response2.status_code == 429  # Rate limited
        
        print("    Rate limiting applied as expected")
        
        # Test malicious path - should be blocked by IP/keyword blocking
        keyword_headers = {'User-Agent': 'Test Browser 1.0', 'X-Forwarded-For': '198.51.100.10'}
        response3 = client.get('/api/sensitive/.env', headers=keyword_headers)
        assert response3.status_code == 403  # Blocked by keyword detection
        
        print("    Keyword blocking applied as expected")
    
    _cleanup_csv_storage(temp_dir)
    print("    Selective protection test passed!\n")


def test_required_protection():
    """Test @aiwaf_require_protection forces middlewares even if exempted"""
    print(" Testing @aiwaf_require_protection (forced protection)")
    
    app = Flask(__name__)
    app.config.update({
        'AIWAF_LOG_DIR': 'aiwaf_logs',
        'AIWAF_RATE_MAX': 1,
        'AIWAF_MIN_AI_LOGS': 10,
    })
    temp_dir = _setup_csv_storage(app)
    
    aiwaf = AIWAF()
    aiwaf.init_app(app)
    
    @app.route('/admin/critical')
    @aiwaf_exempt_from('rate_limit')  # Try to exempt from rate limiting
    @aiwaf_require_protection('rate_limit')  # But force it anyway
    def critical_admin():
        return jsonify({'admin': True})
    
    headers = {'User-Agent': 'Test Browser 1.0'}
    with app.test_client() as client:
        # Should still be rate limited despite exemption
        response1 = client.get('/admin/critical', headers=headers)
        assert response1.status_code == 200
        
        response2 = client.get('/admin/critical', headers=headers)
        assert response2.status_code == 429  # Still rate limited!
        
        print("    Required protection overrode exemption")
    
    _cleanup_csv_storage(temp_dir)
    print("    Required protection test passed!\n")


def test_exemption_utilities():
    """Test exemption utility functions work correctly"""
    print(" Testing exemption utility functions")
    
    app = Flask(__name__)
    temp_dir = _setup_csv_storage(app)
    
    @app.route('/test-utils')
    @aiwaf_exempt_from('rate_limit', 'ai_anomaly')
    def test_utils():
        # Test the utility functions
        results = {
            'should_apply_rate_limit': should_apply_middleware('rate_limit'),
            'should_apply_ip_block': should_apply_middleware('ip_keyword_block'),
            'should_apply_ai': should_apply_middleware('ai_anomaly'),
        }
        return jsonify(results)
    
    @app.route('/test-full-exempt')
    @aiwaf_exempt
    def test_full_exempt():
        results = {
            'should_apply_rate_limit': should_apply_middleware('rate_limit'),
            'should_apply_ip_block': should_apply_middleware('ip_keyword_block'),
        }
        return jsonify(results)
    
    headers = {'User-Agent': 'Test Browser 1.0'}
    with app.test_client() as client:
        # Test partial exemption utilities
        response = client.get('/test-utils', headers=headers)
        data = response.get_json()
        
        assert data['should_apply_rate_limit'] == False  # Exempt
        assert data['should_apply_ip_block'] == True     # Not exempt
        assert data['should_apply_ai'] == False          # Exempt

        print("    Partial exemption utilities work correctly")
        
        # Test full exemption utilities
        response = client.get('/test-full-exempt', headers=headers)
        data = response.get_json()
        
        assert data['should_apply_rate_limit'] == False  # Fully exempt
        assert data['should_apply_ip_block'] == False    # Fully exempt
        
        print("    Full exemption utilities work correctly")

    _cleanup_csv_storage(temp_dir)
    
    print("    Utility functions test passed!\n")


def test_complex_exemption_combinations():
    """Test complex combinations of exemption decorators"""
    print(" Testing complex exemption combinations")
    reset_rate_limit_cache()
    
    app = Flask(__name__)
    app.config.update({
        'AIWAF_LOG_DIR': 'aiwaf_logs',
        'AIWAF_RATE_MAX': 1,
        'AIWAF_MIN_AI_LOGS': 10,
    })
    
    temp_dir = _setup_csv_storage(app)
    aiwaf = AIWAF()
    aiwaf.init_app(app)
    
    @app.route('/complex1')
    @aiwaf_only('rate_limit', 'ip_keyword_block')      # Only these two
    @aiwaf_require_protection('header_validation')     # Plus force this one
    def complex_endpoint1():
        return jsonify({'endpoint': 'complex1'})
    
    @app.route('/complex2') 
    @aiwaf_exempt_from('rate_limit')                   # Exempt from rate limiting
    @aiwaf_require_protection('rate_limit')            # But force it anyway
    def complex_endpoint2():
        return jsonify({'endpoint': 'complex2'})
    
    headers = {'User-Agent': 'Test Browser 1.0'}
    with app.test_client() as client:
        # Test complex1 - should have rate limiting + IP blocking + forced header validation
        response1 = client.get('/complex1', headers=headers)
        assert response1.status_code == 200
        
        response2 = client.get('/complex1', headers=headers)
        assert response2.status_code == 429  # Rate limited
        
        print("    Complex combination 1 works correctly")
        
        # Test complex2 - exemption should be overridden by requirement
        response3 = client.get('/complex2', headers=headers)
        assert response3.status_code == 200
        
        response4 = client.get('/complex2', headers=headers)
        assert response4.status_code == 429  # Still rate limited despite exemption
        
        print("    Complex combination 2 works correctly")
    
    _cleanup_csv_storage(temp_dir)
    print("    Complex combinations test passed!\n")


def run_all_exemption_tests():
    """Run all exemption decorator tests"""
    print(" AIWAF Flask Exemption Decorators - Comprehensive Test Suite")
    print("=" * 70)
    
    test_full_exemption()
    test_partial_exemption() 
    test_middleware_only()
    test_required_protection()
    test_exemption_utilities()
    test_complex_exemption_combinations()
    
    print(" All exemption decorator tests passed!")
    print("\n Key Features Validated:")
    print("    @aiwaf_exempt - Complete bypass of all middlewares")
    print("    @aiwaf_exempt_from - Selective middleware exemption") 
    print("    @aiwaf_only - Apply only specific middlewares")
    print("    @aiwaf_require_protection - Force critical protection")
    print("    should_apply_middleware() - Runtime exemption checking")
    print("    Complex decorator combinations work correctly")
    
    print("\n  Use Cases Covered:")
    print("    Health checks and monitoring endpoints (@aiwaf_exempt)")
    print("    Webhooks and API callbacks (@aiwaf_exempt_from)")
    print("    High-security endpoints (@aiwaf_only + critical middlewares)")
    print("    Admin functions (@aiwaf_require_protection)")
    print("    Custom exemption logic (utility functions)")


if __name__ == '__main__':
    run_all_exemption_tests()
