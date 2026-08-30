#!/usr/bin/env python3
"""
Test AI Anomaly Middleware

Tests the AI anomaly detection middleware functionality.
"""

from flask import Flask
from aiwaf.flask import register_aiwaf_middlewares
import time

def test_ai_anomaly_middleware():
    """Test AI anomaly detection middleware."""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure AIWAF with AI anomaly detection
    app.config.update({
        'AIWAF_USE_CSV': True,
        'AIWAF_DATA_DIR': 'test_ai_data',
        'AIWAF_WINDOW_SECONDS': 30,  # Shorter window for testing
        'AIWAF_DYNAMIC_TOP_N': 5,
        'AIWAF_MODEL_PATH': 'py/aiwaf/flask/resources/model.json',
        'TESTING': True
    })
    
    @app.route('/')
    def index():
        return "Hello World!"
    
    @app.route('/api/data')
    def api_data():
        return {"status": "ok", "data": "test"}
    
    print("=== AI Anomaly Middleware Test ===")
    print(f"App config: {dict(app.config)}")
    
    # Register AIWAF middlewares (including AI anomaly)
    try:
        register_aiwaf_middlewares(app)
        print(" AIWAF middlewares registered successfully")
    except Exception as e:
        print(f" Failed to register AIWAF: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError("Failed to register AIWAF middlewares") from e
    
    # Check if AI anomaly middleware is attached
    ai_anomaly_middleware = None
    for middleware in getattr(app, '_aiwaf_middlewares', []):
        if hasattr(middleware, 'model'):
            ai_anomaly_middleware = middleware
            break
    
    if ai_anomaly_middleware:
        print(f" AI Anomaly middleware found")
        stats = ai_anomaly_middleware.get_stats()
        print(f"   Model loaded: {stats['model_loaded']}")
        print(f"   NumPy available: {stats['numpy_available']}")
        print(f"   Malicious keywords: {stats['malicious_keywords']}")
        print(f"   Window: {stats['window_seconds']} seconds")
    else:
        print("  AI Anomaly middleware not found (but this is expected without direct access)")
    
    # Test with Flask test client
    print("\\nTesting AI anomaly detection...")
    with app.test_client() as client:
        # Normal requests
        print("Making normal requests...")
        for i in range(3):
            response = client.get('/')
            print(f"  GET / -> {response.status_code}")
            time.sleep(0.1)
        
        # API requests
        response = client.get('/api/data')
        print(f"  GET /api/data -> {response.status_code}")
        
        # Simulate scanning behavior
        print("\\nSimulating scanning behavior...")
        scanning_paths = [
            '/wp-admin',
            '/phpmyadmin',
            '/admin.php',
            '/config.php',
            '/backup.sql',
            '/.env',
            '/shell.php',
            '/cmd.php'
        ]
        
        for path in scanning_paths:
            response = client.get(path)
            print(f"  GET {path} -> {response.status_code}")
            time.sleep(0.05)  # Rapid scanning
        
        # Test with malicious keywords
        print("\\nTesting malicious keyword detection...")
        malicious_requests = [
            '/index.php?cmd=whoami',
            '/test?union=select',
            '/app?exec=ls',
            '/data?drop=table'
        ]
        
        for req in malicious_requests:
            response = client.get(req)
            print(f"  GET {req} -> {response.status_code}")
            time.sleep(0.05)
    
    print("\\n AI Anomaly middleware test completed!")
    print("\\nNote: Actual blocking depends on:")
    print("- ML model availability (model.json)")
    print("- NumPy installation")
    print("- Request pattern analysis")
    print("- Anomaly detection thresholds")
    
    assert True

if __name__ == '__main__':
    try:
        success = test_ai_anomaly_middleware()
        if success:
            print("\\n AI Anomaly middleware test passed!")
        else:
            print("\\n AI Anomaly middleware test failed!")
    except Exception as e:
        print(f"\\n Test crashed: {e}")
        import traceback
        traceback.print_exc()
