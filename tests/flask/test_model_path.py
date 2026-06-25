#!/usr/bin/env python3
"""
Test script to verify model path resolution works correctly
"""

import os
import sys
from pathlib import Path

import pytest

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def test_flask_middleware_loads_json_model_artifact(tmp_path, monkeypatch):
    from flask import Flask

    from aiwaf.core.model_serialization import dump_model_artifact
    from aiwaf.core import rust_backend
    from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware

    class StubModel:
        def predict(self, rows):
            return [1 for _ in rows]

    artifact = {
        "model_type": "aiwaf_rust.IsolationForest",
        "model_state": {"stub": True},
        "model_backend": "aiwaf_rust",
        "framework": "flask",
    }
    model_path = tmp_path / "model.json"
    dump_model_artifact(artifact, model_path)
    monkeypatch.setattr(rust_backend, "rust_isolation_forest_from_json", lambda state: StubModel())

    app = Flask(__name__)
    app.config.update(
        AIWAF_FORCE_AI=True,
        AIWAF_MODEL_PATH=str(model_path),
        AIWAF_ALLOW_CUSTOM_MODEL_PATH=True,
    )

    middleware = AIAnomalyMiddleware()
    middleware._load_model(app)

    assert middleware.model is not None
    assert middleware.model_is_rust is True
    assert middleware.model.predict([[1], [2]]) == [1, 1]

def test_model_path_resolution():
    """Test that the model path resolves correctly"""
    
    print(" Testing AIWAF Model Path Resolution")
    print("=" * 50)
    
    try:
        # Test trainer path resolution
        from aiwaf.flask.trainer import get_default_model_path
        trainer_model_path = get_default_model_path()
        print(f" Trainer model path: {trainer_model_path}")
        print(f" Trainer model exists: {os.path.exists(trainer_model_path)}")
        
        # Test anomaly middleware path resolution
        from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware
        from flask import Flask
        
        app = Flask(__name__)
        middleware = AIAnomalyMiddleware()
        
        middleware_model_path = middleware._get_default_model_path()
        print(f" Middleware model path: {middleware_model_path}")
        print(f" Middleware model exists: {os.path.exists(middleware_model_path)}")
        
        # Test if paths are the same
        if trainer_model_path == middleware_model_path:
            print(" Model paths match between trainer and middleware")
        else:
            print("  Model paths differ between trainer and middleware")
            print(f"   Trainer: {trainer_model_path}")
            print(f"   Middleware: {middleware_model_path}")
        
        # Test absolute vs relative paths
        abs_path = os.path.abspath(trainer_model_path)
        print(f" Absolute model path: {abs_path}")
        
        # Check if it's in the package directory
        package_dir = Path(__file__).resolve().parents[2] / 'aiwaf' / 'flask'
        expected_path = package_dir / 'resources' / 'model.json'
        print(f" Expected package path: {expected_path}")
        print(f" Package model exists: {expected_path.exists()}")
        
        # Test model loading
        if os.path.exists(trainer_model_path):
            try:
                # Try to load the model to verify it's valid
                from aiwaf.core.model_serialization import load_model_artifact
                model_data = load_model_artifact(trainer_model_path)
                print(" Model loaded successfully!")
                    
                # Check if it's the expected format
                if isinstance(model_data, dict) and 'model' in model_data:
                    print(f" Model metadata: {model_data.get('framework', 'Unknown')} framework")
                    print(f" Created: {model_data.get('created_at', 'Unknown')}")
                    print(f" Features: {model_data.get('feature_count', 'Unknown')}")
                else:
                    print(" Model format: Direct model object")
                        
            except Exception as e:
                print(f" Error loading model: {e}")
        else:
            print("  Model file not found - run training first")
        
        assert True
        
    except Exception as e:
        print(f" Error testing model path resolution: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError("Model path resolution test failed") from e

if __name__ == '__main__':
    success = test_model_path_resolution()
    if success:
        print("\n Model path resolution test completed!")
        print(" Next steps:")
        print("   1. Install package: pip install -e .")
        print("   2. Test in your app: from aiwaf.flask import AIWAF")
        print("   3. Verify model loads: Check logs for 'Loaded AI model'")
    else:
        print("\n Model path resolution test failed")
        sys.exit(1)
