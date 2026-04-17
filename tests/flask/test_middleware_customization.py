#!/usr/bin/env python3
"""
Test AIWAF middleware customization features.
Demonstrates how to selectively enable/disable specific middlewares.
"""

from flask import Flask, jsonify, request
from aiwaf.flask import AIWAF, register_aiwaf_middlewares
import time

def test_full_aiwaf():
    """Test AIWAF with all middlewares enabled (default)."""
    print("\n Testing AIWAF with ALL middlewares enabled:")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key-12345'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Enable all middlewares (default behavior)
    aiwaf = AIWAF(app)
    
    @app.route('/test')
    def test_route():
        return jsonify({"message": "Hello from full AIWAF!"})
    
    # Check which middlewares are enabled
    enabled = aiwaf.get_enabled_middlewares()
    print(f" Enabled middlewares: {sorted(enabled)}")
    print(f" Total middlewares: {len(enabled)}")
    
    # Test a request
    with app.test_client() as client:
        response = client.get('/test')
        print(f" Test request status: {response.status_code}")
    
    return aiwaf

def test_selective_middlewares():
    """Test AIWAF with only specific middlewares enabled."""
    print("\n Testing AIWAF with SELECTIVE middlewares:")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key-67890'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Enable only specific middlewares
    selected_middlewares = ['rate_limit', 'header_validation', 'ai_anomaly', 'logging']
    aiwaf = AIWAF(app, middlewares=selected_middlewares)
    
    @app.route('/selective')
    def selective_route():
        return jsonify({"message": "Selective protection active!"})
    
    enabled = aiwaf.get_enabled_middlewares()
    print(f" Enabled middlewares: {sorted(enabled)}")
    print(f" Selected {len(enabled)} out of {len(AIWAF.AVAILABLE_MIDDLEWARES)} middlewares")
    
    # Verify specific middlewares
    for middleware in selected_middlewares:
        status = " ENABLED" if aiwaf.is_middleware_enabled(middleware) else " DISABLED"
        print(f"  {middleware}: {status}")
    
    # Test a request
    with app.test_client() as client:
        response = client.get('/selective')
        print(f" Test request status: {response.status_code}")
    
    return aiwaf

def test_disabled_middlewares():
    """Test AIWAF with specific middlewares disabled."""
    print("\n Testing AIWAF with DISABLED middlewares:")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key-disabled'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Disable specific middlewares (enable all others)
    disabled_middlewares = ['honeypot', 'uuid_tamper']
    aiwaf = AIWAF(app, disable_middlewares=disabled_middlewares)
    
    @app.route('/disabled')
    def disabled_route():
        return jsonify({"message": "Some protections disabled!"})
    
    enabled = aiwaf.get_enabled_middlewares()
    all_middlewares = set(AIWAF.AVAILABLE_MIDDLEWARES.keys())
    disabled = all_middlewares - set(enabled)
    
    print(f" Enabled middlewares: {sorted(enabled)}")
    print(f" Disabled middlewares: {sorted(disabled)}")
    
    # Verify disabled middlewares
    for middleware in disabled_middlewares:
        status = " DISABLED" if not aiwaf.is_middleware_enabled(middleware) else "  STILL ENABLED"
        print(f"  {middleware}: {status}")
    
    # Test a request
    with app.test_client() as client:
        response = client.get('/disabled')
        print(f" Test request status: {response.status_code}")
    
    return aiwaf

def test_minimal_aiwaf():
    """Test AIWAF with minimal middlewares (security essentials only)."""
    print("\n  Testing MINIMAL AIWAF (security essentials):")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key-minimal'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Enable only essential security middlewares
    essential_middlewares = ['ip_keyword_block', 'rate_limit', 'logging']
    aiwaf = AIWAF(app, middlewares=essential_middlewares)
    
    @app.route('/minimal')
    def minimal_route():
        return jsonify({"message": "Minimal but secure!"})
    
    enabled = aiwaf.get_enabled_middlewares()
    print(f" Essential middlewares: {sorted(enabled)}")
    print(f" Running {len(enabled)} core protection middlewares")
    
    # Test a request
    with app.test_client() as client:
        response = client.get('/minimal')
        print(f" Test request status: {response.status_code}")
    
    return aiwaf

def test_ai_only():
    """Test AIWAF with AI anomaly detection only."""
    print("\n Testing AI-ONLY AIWAF:")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key-ai'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Enable only AI anomaly detection + logging
    ai_middlewares = ['ai_anomaly', 'logging']
    aiwaf = AIWAF(app, middlewares=ai_middlewares)
    
    @app.route('/ai-only')
    def ai_only_route():
        return jsonify({"message": "AI protection only!"})
    
    enabled = aiwaf.get_enabled_middlewares()
    print(f" AI middlewares: {sorted(enabled)}")
    
    # Test normal request
    with app.test_client() as client:
        response = client.get('/ai-only')
        print(f" Normal request status: {response.status_code}")
        
        # Test suspicious request that should trigger AI
        response = client.get('/ai-only?cmd=whoami&union=select')
        print(f"  Suspicious request status: {response.status_code}")
    
    return aiwaf

def test_backward_compatibility():
    """Test backward compatibility with old registration method."""
    print("\n Testing BACKWARD COMPATIBILITY:")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-key-compat'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Use old registration method with new customization
    aiwaf = register_aiwaf_middlewares(
        app, 
        middlewares=['rate_limit', 'logging'],
        disable_middlewares=['honeypot']
    )
    
    @app.route('/compat')
    def compat_route():
        return jsonify({"message": "Backward compatible!"})
    
    enabled = aiwaf.get_enabled_middlewares()
    print(f" Enabled via old method: {sorted(enabled)}")
    
    # Test a request
    with app.test_client() as client:
        response = client.get('/compat')
        print(f" Test request status: {response.status_code}")
    
    return aiwaf

def main():
    """Run all middleware customization tests."""
    print(" AIWAF Middleware Customization Tests")
    print("=" * 50)
    
    # List all available middlewares
    available = AIWAF.list_available_middlewares()
    print(f"\n Available middlewares: {sorted(available)}")
    print(f" Total available: {len(available)}")
    
    # Run tests
    test_full_aiwaf()
    test_selective_middlewares()
    test_disabled_middlewares()
    test_minimal_aiwaf()
    test_ai_only()
    test_backward_compatibility()
    
    print("\n All middleware customization tests completed!")
    print("\n Usage Examples:")
    print("   # Enable all middlewares (default)")
    print("   aiwaf = AIWAF(app)")
    print("")
    print("   # Enable specific middlewares only")
    print("   aiwaf = AIWAF(app, middlewares=['rate_limit', 'ai_anomaly'])")
    print("")
    print("   # Disable specific middlewares")
    print("   aiwaf = AIWAF(app, disable_middlewares=['honeypot', 'uuid_tamper'])")
    print("")
    print("   # Check what's enabled")
    print("   enabled = aiwaf.get_enabled_middlewares()")
    print("   is_ai_enabled = aiwaf.is_middleware_enabled('ai_anomaly')")

if __name__ == '__main__':
    main()