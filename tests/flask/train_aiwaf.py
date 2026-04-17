#!/usr/bin/env python3
"""
Standalone training script for AIWAF Flask

Usage:
    python train_aiwaf.py [--disable-ai] [--log-dir PATH]
"""

import os
import sys
import argparse
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from flask import Flask
from aiwaf.flask.trainer import train_from_logs

def main():
    parser = argparse.ArgumentParser(description='Train AIWAF Flask from access logs')
    parser.add_argument('--disable-ai', action='store_true', 
                       help='Disable AI model training (keyword learning only)')
    parser.add_argument('--log-dir', type=str, default='aiwaf_logs',
                       help='Directory containing log files (default: aiwaf_logs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(" AIWAF Flask Training Tool")
        print("="*40)
        print(f"Log directory: {args.log_dir}")
        print(f"AI training: {'disabled' if args.disable_ai else 'enabled'}")
        print("="*40)
    
    # Create minimal Flask app for training
    app = Flask(__name__)
    
    # Configure AIWAF settings
    app.config['AIWAF_LOG_DIR'] = args.log_dir
    app.config['AIWAF_DYNAMIC_TOP_N'] = 15
    app.config['AIWAF_AI_CONTAMINATION'] = 0.05
    
    # Optional settings (can be customized)
    app.config['AIWAF_EXEMPT_PATHS'] = {'/health', '/status', '/favicon.ico'}
    app.config['AIWAF_EXEMPT_KEYWORDS'] = ['health', 'status', 'ping', 'check']
    
    try:
        # Run training
        train_from_logs(app, disable_ai=args.disable_ai)
        print("\n Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
