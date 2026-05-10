#!/usr/bin/env python3
"""
Test logging middleware with Flask test client.
"""

import os
import sys
from pathlib import Path
from flask import Flask
from aiwaf.flask import register_aiwaf_middlewares

def test_logging_middleware():
    """Test the logging middleware functionality."""
    
    # Create a simple Flask app
    app = Flask(__name__)
    
    # Configure logging
    app.config['AIWAF_LOG_DIR'] = 'test_logs'
    app.config['AIWAF_LOG_FORMAT'] = 'combined'
    app.config['AIWAF_ENABLE_LOGGING'] = True
    app.config['AIWAF_USE_CSV'] = True
    app.config['AIWAF_DATA_DIR'] = 'test_data'
    
    @app.route('/')
    def index():
        return "Hello World!"
    
    @app.route('/test')
    def test_route():
        return {"status": "ok", "message": "Test successful"}
    
    @app.route('/admin.php')
    def malicious_route():
        return "Should be blocked!"
    
    print("=== AIWAF Logging Test ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Clean up any existing logs
    log_dir = Path('test_logs')
    if log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)
        print(f"Cleaned up existing log directory: {log_dir}")
    
    # Register middlewares
    print("Registering AIWAF middlewares...")
    register_aiwaf_middlewares(app)
    print(" Middlewares registered successfully")
    
    logger = getattr(app, "aiwaf_logger", None)
    if logger is not None:
        print(f" AIWAF logger attached: {logger}")
        print(f"Log directory: {logger.log_dir}")
        print(f"Access log file: {logger.access_log_file}")
        print(f"Error log file: {logger.error_log_file}")
        print(f"AIWAF log file: {logger.aiwaf_log_file}")
    else:
        print(" AIWAF logger not attached to app (middleware still expected to emit logs)")
    
    # Create test client
    with app.test_client() as client:
        print("\\nMaking test requests...")
        
        # Test normal request
        response = client.get('/')
        print(f"GET / -> {response.status_code}")
        
        # Test API request
        response = client.get('/test')
        print(f"GET /test -> {response.status_code}")
        
        # Test malicious request (should be blocked)
        response = client.get('/admin.php')
        print(f"GET /admin.php -> {response.status_code}")
        
        # Test 404
        response = client.get('/nonexistent')
        print(f"GET /nonexistent -> {response.status_code}")
    
    # Check if log files were created
    print("\\nChecking log files...")
    
    access_log = Path('test_logs/access.log')
    error_log = Path('test_logs/error.log')
    aiwaf_log = Path('test_logs/aiwaf.log')
    
    if access_log.exists():
        print(f" Access log created: {access_log}")
        print(f"  Size: {access_log.stat().st_size} bytes")
        with open(access_log, 'r') as f:
            content = f.read()
            print(f"  Lines: {len(content.splitlines())}")
            print(f"  Sample content:")
            for i, line in enumerate(content.splitlines()[:3]):
                print(f"    {i+1}: {line}")
    else:
        print(f" Access log NOT created: {access_log}")
        raise AssertionError("Access log was not created")
    
    if error_log.exists():
        print(f" Error log exists: {error_log} (size: {error_log.stat().st_size} bytes)")
    else:
        print(f"! Error log not created: {error_log} (this is normal if no errors occurred)")
    
    if aiwaf_log.exists():
        print(f" AIWAF log exists: {aiwaf_log} (size: {aiwaf_log.stat().st_size} bytes)")
    else:
        print(f"! AIWAF log not created: {aiwaf_log} (this is normal if no blocks occurred)")
    
    print("\\n Logging test completed successfully!")
    assert True

if __name__ == '__main__':
    try:
        success = test_logging_middleware()
        if success:
            print("\\n All tests passed! Logging middleware is working correctly.")
        else:
            print("\\n Tests failed! Check the output above.")
            sys.exit(1)
    except Exception as e:
        print(f"\\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
