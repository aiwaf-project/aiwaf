#!/usr/bin/env python3
"""
Test AIWAF default behavior - when no middlewares specified, enable all.
"""

from flask import Flask, jsonify
from aiwaf.flask import AIWAF

def test_default_behavior():
    """Test that AIWAF() with no arguments enables all middlewares."""
    print(" Testing AIWAF Default Behavior (No Arguments)")
    print("=" * 50)
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-default-key'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Just AIWAF() with no arguments - should enable ALL middlewares
    aiwaf = AIWAF(app)
    
    @app.route('/test')
    def test_route():
        return jsonify({"message": "Default AIWAF behavior test"})
    
    # Check results
    enabled = aiwaf.get_enabled_middlewares()
    available = AIWAF.list_available_middlewares()
    
    print(f" Available middlewares: {len(available)}")
    print(f" Enabled middlewares: {len(enabled)}")
    print(f" All enabled: {len(enabled) == len(available)}")
    
    print(f"\n Enabled middlewares:")
    for middleware in sorted(enabled):
        print(f"   {middleware}")
    
    # Test a request
    with app.test_client() as client:
        response = client.get('/test')
        print(f"\n Test request status: {response.status_code}")
    
    return aiwaf

def test_app_only_initialization():
    """Test AIWAF(app) pattern - should also enable all."""
    print("\n Testing AIWAF(app) Pattern")
    print("=" * 30)
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-app-only'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Pass app directly to constructor - should enable ALL middlewares
    aiwaf = AIWAF(app)
    
    enabled = aiwaf.get_enabled_middlewares()
    available = AIWAF.list_available_middlewares()
    
    print(f" All middlewares enabled: {len(enabled) == len(available)}")
    print(f" Count: {len(enabled)}/{len(available)}")
    
    return aiwaf

def test_no_app_then_init():
    """Test AIWAF() then init_app() pattern."""
    print("\n Testing AIWAF() then init_app() Pattern")
    print("=" * 40)
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-init-app'
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    # Create AIWAF instance without app
    aiwaf = AIWAF()
    
    # Then initialize with app - should enable ALL middlewares
    aiwaf.init_app(app)
    
    enabled = aiwaf.get_enabled_middlewares()
    available = AIWAF.list_available_middlewares()
    
    print(f" All middlewares enabled: {len(enabled) == len(available)}")
    print(f" Count: {len(enabled)}/{len(available)}")
    
    return aiwaf

def test_explicit_vs_default():
    """Compare explicit all vs default behavior."""
    print("\n Comparing Explicit vs Default Behavior")
    print("=" * 42)
    
    app1 = Flask(__name__)
    app1.config['SECRET_KEY'] = 'test-explicit'
    app1.config['AIWAF_USE_CSV'] = True
    
    app2 = Flask(__name__)
    app2.config['SECRET_KEY'] = 'test-default'
    app2.config['AIWAF_USE_CSV'] = True
    
    # Explicit: specify all middlewares
    all_middlewares = AIWAF.list_available_middlewares()
    aiwaf_explicit = AIWAF(app1, middlewares=all_middlewares)
    
    # Default: no arguments
    aiwaf_default = AIWAF(app2)
    
    explicit_enabled = set(aiwaf_explicit.get_enabled_middlewares())
    default_enabled = set(aiwaf_default.get_enabled_middlewares())
    
    print(f" Explicit middlewares: {len(explicit_enabled)}")
    print(f" Default middlewares: {len(default_enabled)}")
    print(f" Same result: {explicit_enabled == default_enabled}")
    
    return aiwaf_explicit, aiwaf_default

def main():
    """Run all default behavior tests."""
    print(" AIWAF Default Behavior Tests")
    print("When no middlewares specified, AIWAF enables ALL middlewares")
    print("")
    
    # Test different initialization patterns
    test_default_behavior()
    test_app_only_initialization()
    test_no_app_then_init()
    test_explicit_vs_default()
    
    print("\n All tests completed!")
    print("\n Summary:")
    print("   AIWAF()               Enables ALL middlewares")
    print("   AIWAF(app)            Enables ALL middlewares")
    print("   AIWAF().init_app()    Enables ALL middlewares")
    print("   AIWAF(app, middlewares=['specific'])  Only specified")
    print("   AIWAF(app, disable_middlewares=[...])  All except disabled")

if __name__ == '__main__':
    main()