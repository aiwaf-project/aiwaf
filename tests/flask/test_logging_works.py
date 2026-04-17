#!/usr/bin/env python3
"""
AIWAF Logging Test - Minimal Working Example

Copy this file and run it to test if AIWAF logging works on your system.
If this works but your app doesn't, compare your setup to this example.
"""

from flask import Flask
from aiwaf.flask import register_aiwaf_middlewares
import os
import glob
from pathlib import Path

def main():
    print(" AIWAF Logging Test")
    print("=" * 50)
    
    # Clean up any existing test logs
    test_log_dir = Path('test_aiwaf_logs')
    if test_log_dir.exists():
        import shutil
        shutil.rmtree(test_log_dir)
        print(" Cleaned up previous test logs")
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure AIWAF logging
    app.config.update({
        'AIWAF_LOG_DIR': 'test_aiwaf_logs',
        'AIWAF_LOG_FORMAT': 'combined',
        'AIWAF_ENABLE_LOGGING': True,
        'AIWAF_USE_CSV': True,
        'AIWAF_DATA_DIR': 'test_aiwaf_data'
    })
    
    @app.route('/')
    def index():
        return '<h1>AIWAF Test App</h1><p>Logging test successful!</p>'
    
    @app.route('/api/data')
    def api_data():
        return {'status': 'ok', 'message': 'API endpoint working'}
    
    @app.route('/admin.php')  # This should be blocked
    def malicious():
        return 'This should be blocked!'
    
    print(" App Configuration:")
    print(f"  Flask app: {app}")
    print(f"  Log directory: {app.config['AIWAF_LOG_DIR']}")
    print(f"  Log format: {app.config['AIWAF_LOG_FORMAT']}")
    print(f"  Logging enabled: {app.config['AIWAF_ENABLE_LOGGING']}")
    
    # Register AIWAF middlewares
    print("\\n Registering AIWAF middlewares...")
    try:
        register_aiwaf_middlewares(app)
        print(" AIWAF middlewares registered successfully")
    except Exception as e:
        print(f" Failed to register AIWAF: {e}")
        return False
    
    # Verify logger attachment
    if hasattr(app, 'aiwaf_logger'):
        logger = app.aiwaf_logger
        print(f" Logger attached: {type(logger).__name__}")
        print(f"   Log directory: {logger.log_dir}")
        print(f"   Access log: {logger.access_log_file}")
        print(f"   Error log: {logger.error_log_file}")
        print(f"   AIWAF log: {logger.aiwaf_log_file}")
    else:
        print(" AIWAF logger not attached to app")
        return False
    
    # Test with Flask test client
    print("\\n Making test requests...")
    with app.test_client() as client:
        test_requests = [
            ('/', 'Normal page'),
            ('/api/data', 'API endpoint'),
            ('/admin.php', 'Malicious request (should be blocked)'),
            ('/nonexistent', '404 test')
        ]
        
        for url, description in test_requests:
            try:
                response = client.get(url)
                print(f"  {description}: GET {url} -> {response.status_code}")
            except Exception as e:
                print(f"  {description}: GET {url} -> ERROR: {e}")
    
    # Check if log files were created
    print("\\n Checking log files...")
    log_dir = Path('test_aiwaf_logs')
    
    if log_dir.exists():
        print(f" Log directory created: {log_dir.absolute()}")
        
        log_files = list(log_dir.glob('*.log'))
        if log_files:
            print(f" Found {len(log_files)} log files:")
            
            for log_file in log_files:
                size = log_file.stat().st_size
                print(f"   {log_file.name}: {size} bytes")
                
                if size > 0:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        print(f"     Lines: {len(lines)}")
                        if lines:
                            # Show first line as sample
                            sample = lines[0].strip()
                            if len(sample) > 80:
                                sample = sample[:77] + "..."
                            print(f"     Sample: {sample}")
                else:
                    print(f"      File is empty")
        else:
            print(" No log files found in directory")
            return False
    else:
        print(f" Log directory not created: {log_dir.absolute()}")
        return False
    
    # Success summary
    print("\\n Test Results:")
    print(" AIWAF logging is working correctly!")
    print(" Log files are being generated")
    print(" Requests are being logged properly")
    
    print("\\n Next Steps:")
    print("1. Check the 'test_aiwaf_logs' directory for the generated log files")
    print("2. Compare your app setup to this working example")
    print("3. Make sure you call register_aiwaf_middlewares(app) in your app")
    print("4. Verify your AIWAF_LOG_DIR configuration")
    
    # Show actual log content
    access_log = log_dir / 'access.log'
    if access_log.exists() and access_log.stat().st_size > 0:
        print("\\n Sample Access Log Content:")
        with open(access_log, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()[:3]):
                print(f"  {i+1}: {line.strip()}")
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("\\n Test completed successfully!")
            print("\\nIf your app still doesn't create logs, check:")
            print("- Are you calling register_aiwaf_middlewares(app)?")
            print("- Is AIWAF_ENABLE_LOGGING set to True?")
            print("- Do you have write permissions in the log directory?")
            print("- Are you making actual HTTP requests to your app?")
        else:
            print("\\n Test failed - see output above for details")
            
    except Exception as e:
        print(f"\\n Test crashed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\\nThis indicates a problem with your AIWAF installation.")
        print("Try reinstalling: pip install -e .")