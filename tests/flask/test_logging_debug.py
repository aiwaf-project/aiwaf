#!/usr/bin/env python3
"""
Debug logging middleware - test what's actually happening.
"""

import os
import sys
from pathlib import Path
from flask import Flask
from aiwaf.flask import register_aiwaf_middlewares

# Create a simple Flask app
app = Flask(__name__)

# Configure logging
app.config['AIWAF_LOG_DIR'] = 'debug_logs'
app.config['AIWAF_LOG_FORMAT'] = 'combined'
app.config['AIWAF_ENABLE_LOGGING'] = True
app.config['AIWAF_USE_CSV'] = True
app.config['AIWAF_DATA_DIR'] = 'debug_data'

print("=== AIWAF Logging Debug Test ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Check if aiwaf.flask is properly imported
try:
    from aiwaf.flask.logging_middleware import AIWAFLoggingMiddleware
    print(" AIWAFLoggingMiddleware imported successfully")
except ImportError as e:
    print(f" Failed to import AIWAFLoggingMiddleware: {e}")
    sys.exit(1)

@app.route('/')
def index():
    return "Hello, this is a test!"

@app.route('/test')
def test_route():
    return {"status": "ok", "message": "Test successful"}

if __name__ == '__main__':
    print(f"App config: {dict(app.config)}")
    
    # Register middlewares
    print("Registering AIWAF middlewares...")
    try:
        register_aiwaf_middlewares(app)
        print(" Middlewares registered successfully")
    except Exception as e:
        print(f" Failed to register middlewares: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check if log directory will be created
    log_dir = Path('debug_logs')
    print(f"Log directory: {log_dir.absolute()}")
    
    # Check if middleware is properly attached
    if hasattr(app, 'aiwaf_logger'):
        print(f" AIWAF logger attached: {app.aiwaf_logger}")
        print(f"Log files will be created in: {app.aiwaf_logger.log_dir}")
        print(f"Access log: {app.aiwaf_logger.access_log_file}")
        print(f"Error log: {app.aiwaf_logger.error_log_file}")
        print(f"AIWAF log: {app.aiwaf_logger.aiwaf_log_file}")
    else:
        print(" AIWAF logger not attached to app")
    
    print("\nStarting test server...")
    print("Visit http://127.0.0.1:5000/ and http://127.0.0.1:5000/test")
    print("Then check the debug_logs directory for log files.")
    
    # Run the app
    app.run(debug=True, port=5000)