#!/usr/bin/env python3
"""Test AIWAF Flask log threshold behavior"""

import os
from flask import Flask
from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware

def test_log_threshold():
    """Test that middleware respects log data threshold"""
    
    print(" Testing AIWAF Flask Log Threshold Behavior")
    print("=" * 60)
    
    # Test 1: Default threshold (should disable AI)
    print("\n Test 1: Default threshold (10,000 logs)")
    app1 = Flask(__name__)
    app1.config['AIWAF_DATA_DIR'] = 'aiwaf_data'
    app1.config['AIWAF_LOG_DIR'] = 'aiwaf_logs'
    # Default AIWAF_MIN_AI_LOGS = 10000
    
    middleware1 = AIAnomalyMiddleware()
    middleware1.init_app(app1)
    
    with app1.app_context():
        if hasattr(middleware1, 'model') and middleware1.model is not None:
            print(" AI model loaded despite insufficient data")
            print(f"   Model type: {type(middleware1.model).__name__}")
        else:
            print(" AI model correctly disabled due to insufficient data")
    
    # Test 2: Low threshold (should enable AI)
    print("\n Test 2: Low threshold (20 logs)")
    app2 = Flask(__name__)
    app2.config['AIWAF_DATA_DIR'] = 'aiwaf_data'
    app2.config['AIWAF_LOG_DIR'] = 'aiwaf_logs'
    app2.config['AIWAF_MIN_AI_LOGS'] = 20  # Low threshold
    
    middleware2 = AIAnomalyMiddleware()
    middleware2.init_app(app2)
    
    with app2.app_context():
        if hasattr(middleware2, 'model') and middleware2.model is not None:
            print(" AI model loaded with sufficient data (low threshold)")
            print(f"   Model type: {type(middleware2.model).__name__}")
        else:
            print(" AI model disabled even with sufficient data")
    
    # Test 3: Force AI (should enable despite high threshold)
    print("\n Test 3: Force AI enabled")
    app3 = Flask(__name__)
    app3.config['AIWAF_DATA_DIR'] = 'aiwaf_data'
    app3.config['AIWAF_LOG_DIR'] = 'aiwaf_logs'
    app3.config['AIWAF_MIN_AI_LOGS'] = 10000  # High threshold
    app3.config['AIWAF_FORCE_AI'] = True  # Force it
    
    middleware3 = AIAnomalyMiddleware()
    middleware3.init_app(app3)
    
    with app3.app_context():
        if hasattr(middleware3, 'model') and middleware3.model is not None:
            print(" AI model loaded due to force flag")
            print(f"   Model type: {type(middleware3.model).__name__}")
        else:
            print(" AI model disabled even with force flag")
    
    # Test 4: Very high threshold (should disable AI even if model exists)
    print("\n Test 4: Very high threshold (100,000 logs)")
    app4 = Flask(__name__)
    app4.config['AIWAF_DATA_DIR'] = 'aiwaf_data'
    app4.config['AIWAF_LOG_DIR'] = 'aiwaf_logs'
    app4.config['AIWAF_MIN_AI_LOGS'] = 100000  # Very high threshold
    app4.config['AIWAF_FORCE_AI'] = False
    
    middleware4 = AIAnomalyMiddleware()
    middleware4.init_app(app4)
    
    with app4.app_context():
        if hasattr(middleware4, 'model') and middleware4.model is not None:
            print(" AI model loaded despite very high threshold")
            print(f"   Model type: {type(middleware4.model).__name__}")
        else:
            print(" AI model correctly disabled due to very high threshold")
    
    print("\n Log threshold behavior tests completed!")

if __name__ == '__main__':
    test_log_threshold()