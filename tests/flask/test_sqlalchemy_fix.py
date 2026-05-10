#!/usr/bin/env python3
"""
Test the exact scenario that was causing the SQLAlchemy error
"""

from flask import Flask, jsonify
from aiwaf.flask.middleware import register_aiwaf_middlewares

def test_website_scenario():
    """Test the exact scenario that was failing in the AIWAF website."""
    
    app = Flask(__name__)
    
    # Configure for CSV storage (like the website should)
    app.config.update({
        'AIWAF_USE_CSV': True,
        'AIWAF_DATA_DIR': 'aiwaf_data',
        'AIWAF_RATE_WINDOW': 60,
        'AIWAF_RATE_MAX': 100,
        'AIWAF_RATE_FLOOD': 200,
        'AIWAF_MIN_FORM_TIME': 1.0,
    })
    
    # Register AIWAF middlewares (this should work without database errors)
    register_aiwaf_middlewares(app)
    print(" AIWAF middlewares registered successfully")
    
    # Add a test route like /aiwaf/status
    @app.route('/aiwaf/status')
    def aiwaf_status():
        return jsonify({
            "status": "ok",
            "protection": "enabled",
            "storage": "csv"
        })

    @app.route('/test')
    def test_route():
        return jsonify({"message": "test route works"})
    
    # Test the route that was failing
    with app.test_client() as client:
        print(" Testing /aiwaf/status route...")
        response = client.get('/aiwaf/status')
        
        if response.status_code == 200:
            print(" Route works successfully!")
            print(f"Response: {response.get_json()}")
        else:
            print(f" Route failed with status {response.status_code}")
            print(f"Response: {response.get_data()}")
            raise AssertionError("Status route failed")
    
    # Test other routes to ensure middleware works
    with app.test_client() as client:
        print(" Testing /test route...")
        response = client.get('/test')
        
        if response.status_code == 200:
            print(" Test route works!")
        else:
            print(f" Test route failed: {response.status_code}")
            raise AssertionError("Test route failed")
    
    print(" All tests passed! The SQLAlchemy error should be fixed.")
    assert True

if __name__ == "__main__":
    test_website_scenario()
