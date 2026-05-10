#!/usr/bin/env python3
"""
Test script to demonstrate AIWAF Flask comprehensive training
"""

import os
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from flask import Flask
from aiwaf.flask.trainer import train_from_logs

def test_training():
    """Test the comprehensive training system"""
    
    print(" Testing AIWAF Flask Training System")
    print("=" * 50)
    
    # Create minimal Flask app
    app = Flask(__name__)
    
    # Add some routes for testing
    @app.route('/')
    def home():
        return "Home Page"
    
    @app.route('/about')
    def about():
        return "About Page"
    
    @app.route('/contact')
    def contact():
        return "Contact Page"
    
    @app.route('/api/users')
    def api_users():
        return {"users": []}
    
    # Configure for testing
    app.config.update({
        'AIWAF_LOG_DIR': 'test_logs',
        'AIWAF_DYNAMIC_TOP_N': 5,
        'AIWAF_AI_CONTAMINATION': 0.1,
        'AIWAF_EXEMPT_PATHS': {'/favicon.ico', '/robots.txt'},
        'AIWAF_EXEMPT_KEYWORDS': ['api'],
        'SECRET_KEY': 'test-key'
    })
    
    # Create app context
    with app.app_context():
        try:
            print(" Testing with keyword learning only (no AI dependencies required)")
            train_from_logs(app, disable_ai=True)
            print(" Keyword-only training completed successfully!")
            
        except Exception as e:
            print(f" Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise AssertionError("Training failed") from e
    
    assert True

if __name__ == '__main__':
    success = test_training()
    if success:
        print("\n Training test completed successfully!")
        print(" Next steps:")
        print("   1. Install AI dependencies: pip install aiwaf-flask[ai]")
        print("   2. Run with AI: python train_aiwaf.py --log-dir test_logs")
        print("   3. Check generated model: aiwaf/flask/resources/model.pkl")
    else:
        print("\n Training test failed")
        sys.exit(1)
