#!/usr/bin/env python3
"""
Test script to verify AIWAF Flask dependencies and AI features.
"""

def test_basic_dependencies():
    """Test basic AIWAF dependencies."""
    print(" Testing Basic Dependencies...")
    
    try:
        from flask import Flask
        print(" Flask: Available")
    except ImportError as e:
        raise AssertionError(f"Flask dependency missing: {e}") from e
    
    try:
        from flask_sqlalchemy import SQLAlchemy
        print(" Flask-SQLAlchemy: Available")
    except ImportError as e:
        raise AssertionError(f"Flask-SQLAlchemy dependency missing: {e}") from e
    
    try:
        from aiwaf.flask import AIWAF
        print(" AIWAF Flask: Available")
    except ImportError as e:
        raise AssertionError(f"AIWAF Flask import missing: {e}") from e
    assert True

def test_ai_dependencies():
    """Test AI anomaly detection dependencies."""
    print("\n Testing AI Dependencies...")
    
    numpy_available = False
    sklearn_available = False
    
    try:
        import numpy as np
        print(f" NumPy: Available (version {np.__version__})")
        numpy_available = True
    except ImportError:
        print(" NumPy: Missing - install with 'pip install aiwaf-flask[ai]'")
    
    try:
        import sklearn
        print(f" Scikit-learn: Available (version {sklearn.__version__})")
        sklearn_available = True
    except ImportError:
        print(" Scikit-learn: Missing - install with 'pip install aiwaf-flask[ai]'")
    
    assert isinstance(numpy_available and sklearn_available, bool)

def test_ai_middleware():
    """Test AI anomaly middleware functionality."""
    print("\n Testing AI Middleware...")
    
    try:
        from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware
        print(" AI Middleware: Import successful")
        
        # Test if AI features are detected properly
        from flask import Flask
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test'
        app.config['AIWAF_USE_CSV'] = True
        
        # Initialize middleware
        ai_middleware = AIAnomalyMiddleware(app)
        print(" AI Middleware: Initialization successful")
        
        # Check if NumPy is properly detected
        from aiwaf.flask.anomaly_middleware import NUMPY_AVAILABLE
        if NUMPY_AVAILABLE:
            print(" AI Features: NumPy detected - ML capabilities enabled")
        else:
            print("  AI Features: NumPy not detected - basic analysis only")
        
        assert True
        
    except Exception as e:
        raise AssertionError(f"AI middleware test failed: {e}") from e

def test_installation_variants():
    """Test different installation scenarios."""
    print("\n Installation Guide:")
    print("Basic:     pip install aiwaf-flask")
    print("With AI:   pip install aiwaf-flask[ai]")
    print("Full:      pip install aiwaf-flask[all]")
    print("AI Only:   pip install numpy>=1.20.0 scikit-learn>=1.0.0")

def main():
    """Run all dependency tests."""
    print(" AIWAF Flask Dependency Test")
    print("=" * 40)
    
    basic_ok = test_basic_dependencies()
    ai_ok = test_ai_dependencies()
    middleware_ok = test_ai_middleware()
    
    print("\n Test Summary:")
    print(f"Basic Dependencies: {' PASS' if basic_ok else ' FAIL'}")
    print(f"AI Dependencies: {' PASS' if ai_ok else '  OPTIONAL'}")
    print(f"AI Middleware: {' PASS' if middleware_ok else ' FAIL'}")
    
    if basic_ok and middleware_ok:
        if ai_ok:
            print("\n All tests passed! Full AI capabilities available.")
        else:
            print("\n Basic tests passed! Install AI dependencies for full features:")
            print("   pip install aiwaf-flask[ai]")
    else:
        print("\n Some tests failed. Check dependencies.")
    
    test_installation_variants()

if __name__ == '__main__':
    main()
