#!/usr/bin/env python3
"""
Quick test to verify middleware integration with path exemptions
"""

from flask import Flask, jsonify, request
from aiwaf.flask.utils import is_exempt, is_path_exempt

def test_middleware_integration():
    """Test that middleware correctly handles exemptions."""
    
    app = Flask(__name__)
    
    # Configure some exempt paths
    app.config['AIWAF_EXEMPT_PATHS'] = {
        '/favicon.ico',
        '/robots.txt',
        '/health',
        '*.css',
        '/static/',
    }
    
    with app.test_request_context('/favicon.ico'):
        print("Testing /favicon.ico exemption:")
        print(f"  is_path_exempt(): {is_path_exempt('/favicon.ico')}")
        print(f"  is_exempt(request): {is_exempt(request)}")
    
    with app.test_request_context('/admin/login'):
        print("\nTesting /admin/login (should NOT be exempt):")
        print(f"  is_path_exempt(): {is_path_exempt('/admin/login')}")
        print(f"  is_exempt(request): {is_exempt(request)}")
    
    with app.test_request_context('/static/style.css'):
        print("\nTesting /static/style.css (directory + wildcard):")
        print(f"  is_path_exempt(): {is_path_exempt('/static/style.css')}")
        print(f"  is_exempt(request): {is_exempt(request)}")
    
    print("\n Middleware integration test complete!")

if __name__ == "__main__":
    test_middleware_integration()