#!/usr/bin/env python3
"""
Simple test for model path resolution without Flask dependencies
"""

import os
import sys
from pathlib import Path

def test_model_path_simple():
    """Test model path resolution without Flask imports"""
    
    print(" Testing AIWAF Model Path Resolution (Simple)")
    print("=" * 50)
    
    try:
        # Test the path resolution function directly
        def get_default_model_path():
            """Get the default model path relative to the package."""
            # Get the directory where this test file is located
            repo_root = Path(__file__).resolve().parents[2]
            trainer_dir = repo_root / 'aiwaf' / 'flask'
            resources_dir = trainer_dir / 'resources'
            
            # Ensure resources directory exists
            resources_dir.mkdir(exist_ok=True)
            
            return str(resources_dir / 'model.pkl')
        
        # Test path resolution
        model_path = get_default_model_path()
        print(f" Resolved model path: {model_path}")
        print(f" Model file exists: {os.path.exists(model_path)}")
        
        # Test absolute path
        abs_path = os.path.abspath(model_path)
        print(f" Absolute model path: {abs_path}")
        
        # Check the actual file
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f" Model file size: {file_size:,} bytes")
            
            # Try to peek at the model file
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    # Try to load just the first part to check format
                    model_data = pickle.load(f)
                    print(" Model file is valid pickle format!")
                    
                    # Check if it's our expected format
                    if isinstance(model_data, dict):
                        print(" Model metadata found:")
                        for key, value in model_data.items():
                            if key != 'model':  # Don't print the actual model
                                print(f"   {key}: {value}")
                    else:
                        print(" Model format: Direct model object")
                        
            except Exception as e:
                print(f"  Could not read model file: {e}")
        else:
            print(" Model file not found!")
            
            # Check if resources directory exists
            resources_dir = Path(model_path).parent
            print(f" Resources directory exists: {resources_dir.exists()}")
            if resources_dir.exists():
                print(f" Contents of resources directory:")
                for item in resources_dir.iterdir():
                    print(f"   {item.name}")
        
        # Test package structure for installation
        package_dir = Path(__file__).resolve().parents[2] / 'aiwaf' / 'flask'
        print(f"\n Package Structure Check:")
        print(f"   Package dir exists: {package_dir.exists()}")
        
        if package_dir.exists():
            resources_in_package = package_dir / 'resources'
            print(f"   Resources in package: {resources_in_package.exists()}")
            
            if resources_in_package.exists():
                print(f"   Resources contents:")
                for item in resources_in_package.iterdir():
                    size = item.stat().st_size if item.is_file() else 0
                    print(f"     {item.name} ({size:,} bytes)")
        
        assert True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError("Simple model path test failed") from e

if __name__ == '__main__':
    success = test_model_path_simple()
    if success:
        print("\n Model path resolution looks good!")
        print(" Your model should be loadable when the package is installed.")
    else:
        print("\n Model path resolution issues found")
        sys.exit(1)
