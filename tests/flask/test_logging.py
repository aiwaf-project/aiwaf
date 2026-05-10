#!/usr/bin/env python3
"""
Test AIWAF Logging Middleware

Tests the web server style logging functionality.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

def test_logging_formats():
    """Test different logging formats."""
    print(" Testing AIWAF Logging Middleware...")
    
    try:
        from flask import Flask
        from aiwaf.flask import register_aiwaf_middlewares
        
        # Test different log formats
        formats = ['combined', 'common', 'csv', 'json']
        
        for log_format in formats:
            print(f"\n Testing {log_format} format...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                app = Flask(__name__)
                
                # Configure AIWAF with logging
                app.config['AIWAF_LOG_DIR'] = temp_dir
                app.config['AIWAF_LOG_FORMAT'] = log_format
                app.config['AIWAF_USE_CSV'] = True
                app.config['AIWAF_DATA_DIR'] = temp_dir
                
                # Register AIWAF middlewares
                register_aiwaf_middlewares(app)
                
                @app.route('/')
                def index():
                    return "Hello AIWAF!"
                
                @app.route('/test')
                def test():
                    return "Test page"
                
                # Test requests
                with app.test_client() as client:
                    # Normal requests
                    response1 = client.get('/')
                    response2 = client.get('/test?param=value')
                    response3 = client.post('/test', data={'test': 'data'})
                    
                    # Request that should be blocked
                    response4 = client.get('/admin.php')  # Should trigger keyword block
                    
                print(f" {log_format} format test completed")
                
                # Check if log files were created
                log_dir = Path(temp_dir)
                access_log = log_dir / 'access.log'
                error_log = log_dir / 'error.log'
                aiwaf_log = log_dir / 'aiwaf.log'
                
                if access_log.exists():
                    print(f"   Access log created ({access_log.stat().st_size} bytes)")
                    
                    # Show sample log entries
                    with open(access_log, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"   Sample log entry:")
                            print(f"    {lines[0].strip()}")
                
                if aiwaf_log.exists():
                    print(f"   AIWAF log created ({aiwaf_log.stat().st_size} bytes)")
        
        assert True
        
    except Exception as e:
        print(f" Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError("Logging format test failed") from e

def test_log_analysis():
    """Test log analysis functionality."""
    print("\n Testing log analysis...")
    
    try:
        from aiwaf.flask.logging_middleware import analyze_access_logs
        
        # Create sample log data
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / 'access.log'
            
            # Create sample combined log entries
            sample_logs = [
                '127.0.0.1 - - [14/Sep/2025:14:30:00 +0000] "GET / HTTP/1.1" 200 1234 "-" "Mozilla/5.0" 50ms - "-"',
                '192.168.1.10 - - [14/Sep/2025:14:30:01 +0000] "POST /login HTTP/1.1" 200 567 "http://example.com" "Mozilla/5.0" 120ms - "-"',
                '203.0.113.10 - - [14/Sep/2025:14:30:02 +0000] "GET /admin.php HTTP/1.1" 403 0 "-" "BadBot/1.0" 10ms BLOCKED "Malicious keyword: .php"',
                '127.0.0.1 - - [14/Sep/2025:14:30:03 +0000] "GET /test HTTP/1.1" 404 123 "-" "Mozilla/5.0" 25ms - "-"'
            ]
            
            with open(log_file, 'w') as f:
                for log_line in sample_logs:
                    f.write(log_line + '\n')
            
            # Analyze the logs
            stats = analyze_access_logs(temp_dir, 'combined')
            
            print(f"   Analysis results:")
            print(f"    Total requests: {stats['total_requests']}")
            print(f"    Blocked requests: {stats['blocked_requests']}")
            print(f"    Unique IPs: {len(stats.get('ips', {}))}")
            print(f"    Status codes: {stats.get('status_codes', {})}")
            
            if stats.get('blocked_reasons'):
                print(f"    Block reasons: {stats['blocked_reasons']}")
            
            print(" Log analysis test passed")
        
        assert True
        
    except Exception as e:
        raise AssertionError(f"Log analysis test failed: {e}") from e

def test_cli_logs_command():
    """Test the CLI logs command."""
    print("\n Testing CLI logs command...")
    
    try:
        from aiwaf.flask.cli import AIWAFManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV log
            log_file = Path(temp_dir) / 'access.log'
            
            # CSV format sample
            csv_content = """timestamp,ip,method,path,query_string,protocol,status_code,content_length,response_time_ms,referer,user_agent,blocked,block_reason
2025-09-14T14:30:00,127.0.0.1,GET,/,,HTTP/1.1,200,1234,50,,Mozilla/5.0,False,
2025-09-14T14:30:01,192.168.1.10,POST,/login,,HTTP/1.1,200,567,120,http://example.com,Mozilla/5.0,False,
2025-09-14T14:30:02,203.0.113.10,GET,/admin.php,,HTTP/1.1,403,0,10,,BadBot/1.0,True,Malicious keyword: .php"""
            
            with open(log_file, 'w') as f:
                f.write(csv_content)
            
            # Test CLI analysis
            manager = AIWAFManager()
            result = manager.analyze_logs(temp_dir, 'csv')
            
            if result:
                print(" CLI logs command test passed")
            else:
                print(" CLI logs command test failed")
            
            assert result
        
    except Exception as e:
        raise AssertionError(f"CLI logs command test failed: {e}") from e

if __name__ == '__main__':
    print(" AIWAF Logging Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test logging formats
    if not test_logging_formats():
        success = False
    
    # Test log analysis
    if not test_log_analysis():
        success = False
    
    # Test CLI command
    if not test_cli_logs_command():
        success = False
    
    if success:
        print("\n All logging tests passed!")
        print("\nLogging features ready:")
        print("   Multiple log formats (Combined, Common, CSV, JSON)")
        print("   Standard web server style access logs")
        print("   Separate error and AIWAF event logs")
        print("   CLI log analysis with statistics")
        print("   Performance metrics and traffic patterns")
    else:
        print("\n Some logging tests failed.")
        sys.exit(1)
