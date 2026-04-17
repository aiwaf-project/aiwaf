#!/usr/bin/env python3
"""Test AIWAF Flask dynamic AI status checking"""

import time
from flask import Flask
from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware

def test_dynamic_ai_status():
    """Test that middleware can dynamically enable/disable AI based on runtime conditions"""
    
    print(" Testing AIWAF Flask Dynamic AI Status Checking")
    print("=" * 60)
    
    # Create app with high threshold initially (AI disabled)
    app = Flask(__name__)
    app.config.update({
        'AIWAF_DATA_DIR': 'aiwaf_data',
        'AIWAF_LOG_DIR': 'aiwaf_logs',
        'AIWAF_MIN_AI_LOGS': 50000,  # High threshold - AI disabled
        'AIWAF_FORCE_AI': False
    })
    
    middleware = AIAnomalyMiddleware()
    middleware.ai_check_interval = 1  # Check every second for testing
    middleware.init_app(app)
    
    with app.app_context():
        print(f"\n Initial state (threshold: {app.config['AIWAF_MIN_AI_LOGS']})")
        print(f"   AI Model: {'Loaded' if middleware.model else 'Disabled'}")
        
        # Simulate periodic check - should remain disabled
        middleware._check_ai_status_periodically(app)
        print(f"   After check: {'Loaded' if middleware.model else 'Disabled'}")
        
        # Now lower the threshold (simulate config change)
        print(f"\n Lowering threshold to 20...")
        app.config['AIWAF_MIN_AI_LOGS'] = 20
        
        # Force a check
        middleware.last_ai_check = 0  # Reset timer
        middleware._check_ai_status_periodically(app)
        print(f"   After check: {'Loaded' if middleware.model else 'Disabled'}")
        
        # Now test force AI
        print(f"\n Enabling force AI...")
        app.config['AIWAF_FORCE_AI'] = True
        app.config['AIWAF_MIN_AI_LOGS'] = 100000  # Very high threshold
        
        # Force a check
        middleware.last_ai_check = 0  # Reset timer
        middleware._check_ai_status_periodically(app)
        print(f"   After check: {'Loaded' if middleware.model else 'Disabled'}")
        
        # Test disabling force AI with high threshold
        print(f"\n Disabling force AI with high threshold...")
        app.config['AIWAF_FORCE_AI'] = False
        
        # Force a check
        middleware.last_ai_check = 0  # Reset timer
        middleware._check_ai_status_periodically(app)
        print(f"   After check: {'Loaded' if middleware.model else 'Disabled'}")
    
    print("\n Dynamic AI status checking tests completed!")

if __name__ == '__main__':
    test_dynamic_ai_status()