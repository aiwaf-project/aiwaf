#!/usr/bin/env python3
"""
Test Path-Based Exemptions for AIWAF Flask

This test demonstrates how the new path exemption system works
to prevent false positives for legitimate resources like favicon.ico.
"""

from flask import Flask
from aiwaf.flask.utils import is_path_exempt, get_default_exempt_paths

def test_path_exemptions():
    """Test various path exemption scenarios."""
    print("Testing AIWAF Path Exemptions")
    print("=" * 50)
    
    # Test cases: (path, should_be_exempt, description)
    test_cases = [
        # Common static files
        ('/favicon.ico', True, 'Favicon should be exempt'),
        ('/robots.txt', True, 'Robots.txt should be exempt'),
        ('/sitemap.xml', True, 'Sitemap should be exempt'),
        ('/ads.txt', True, 'Ads.txt should be exempt'),
        
        # Apple touch icons
        ('/apple-touch-icon.png', True, 'Apple touch icon should be exempt'),
        ('/apple-touch-icon-precomposed.png', True, 'Apple touch icon precomposed should be exempt'),
        
        # Manifest and config files
        ('/manifest.json', True, 'Manifest should be exempt'),
        ('/browserconfig.xml', True, 'Browser config should be exempt'),
        
        # Health check endpoints
        ('/health', True, 'Health check should be exempt'),
        ('/ping', True, 'Ping endpoint should be exempt'),
        ('/status', True, 'Status endpoint should be exempt'),
        
        # Static file extensions with wildcards
        ('/assets/style.css', True, 'CSS files should be exempt'),
        ('/js/app.js', True, 'JS files should be exempt'),
        ('/images/logo.png', True, 'PNG files should be exempt'),
        ('/uploads/photo.jpg', True, 'JPG files should be exempt'),
        ('/fonts/font.woff2', True, 'Font files should be exempt'),
        
        # Static directories
        ('/static/anything/here.txt', True, 'Static directory should be exempt'),
        ('/assets/deep/nested/file.css', True, 'Assets directory should be exempt'),
        ('/css/bootstrap.min.css', True, 'CSS directory should be exempt'),
        
        # Well-known URIs
        ('/.well-known/security.txt', True, 'Well-known URIs should be exempt'),
        ('/.well-known/acme-challenge/abc123', True, 'ACME challenge should be exempt'),
        
        # Case insensitive
        ('/FAVICON.ICO', True, 'Uppercase paths should be exempt'),
        ('/RoBots.TxT', True, 'Mixed case should be exempt'),
        
        # Non-exempt paths (should trigger AIWAF)
        ('/admin/login', False, 'Admin paths should NOT be exempt'),
        ('/api/users', False, 'API paths should NOT be exempt'),
        ('/login', False, 'Login should NOT be exempt'),
        ('/wp-admin/', False, 'WordPress admin should NOT be exempt'),
        ('/phpmyadmin/', False, 'phpMyAdmin should NOT be exempt'),
        ('/.env', False, 'Environment files should NOT be exempt'),
        ('/config.php', False, 'Config files should NOT be exempt'),
    ]
    
    passed = 0
    failed = 0
    
    for path, expected, description in test_cases:
        result = is_path_exempt(path)
        status = " PASS" if result == expected else " FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} {path:<35} | {description}")
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    # Show default exempt paths
    print(f"\nDefault Exempt Paths ({len(get_default_exempt_paths())} total):")
    print("-" * 50)
    for path in sorted(get_default_exempt_paths()):
        print(f"  {path}")

def test_custom_exempt_paths():
    """Test custom exempt paths configuration."""
    print(f"\n\nTesting Custom Exempt Paths")
    print("=" * 50)
    
    app = Flask(__name__)
    
    # Custom exempt paths for a specific application
    custom_exempt_paths = {
        '/favicon.ico',
        '/special-endpoint',
        '/custom-health-check',
        '*.pdf',
        '/downloads/',
    }
    
    app.config['AIWAF_EXEMPT_PATHS'] = custom_exempt_paths
    
    with app.app_context():
        test_cases = [
            ('/favicon.ico', True, 'Standard favicon'),
            ('/special-endpoint', True, 'Custom endpoint'),
            ('/custom-health-check', True, 'Custom health check'),
            ('/document.pdf', True, 'PDF file should match *.pdf'),
            ('/downloads/file.zip', True, 'Downloads directory'),
            ('/robots.txt', False, 'Not in custom list'),
            ('/admin', False, 'Admin not exempt'),
        ]
        
        for path, expected, description in test_cases:
            result = is_path_exempt(path)
            status = " PASS" if result == expected else " FAIL"
            print(f"{status} {path:<25} | {description}")

def demo_real_world_usage():
    """Demonstrate real-world usage scenarios."""
    print(f"\n\nReal-World Usage Examples")
    print("=" * 50)
    
    scenarios = [
        ("SEO crawler requesting /robots.txt", "/robots.txt"),
        ("Browser requesting favicon", "/favicon.ico"),
        ("SSL certificate validation", "/.well-known/acme-challenge/xyz123"),
        ("Apple device requesting touch icon", "/apple-touch-icon.png"),
        ("Google Ads verification", "/ads.txt"),
        ("Static CSS file", "/static/css/bootstrap.min.css"),
        ("Font file request", "/assets/fonts/roboto.woff2"),
        ("Health check from load balancer", "/health"),
        ("Sitemap for search engines", "/sitemap.xml"),
        ("Security policy file", "/.well-known/security.txt"),
    ]
    
    print("These requests would be EXEMPT from AIWAF protection:")
    for description, path in scenarios:
        exempt = is_path_exempt(path)
        status = " EXEMPT" if exempt else "  PROTECTED"
        print(f"{status} {description:<35} | {path}")

if __name__ == "__main__":
    test_path_exemptions()
    test_custom_exempt_paths()
    demo_real_world_usage()
    
    print(f"\n\n Path exemption system ready!")
    print("Configure custom exempt paths in your Flask app:")
    print("app.config['AIWAF_EXEMPT_PATHS'] = {'/custom-path', '*.pdf', '/api/public/'}")