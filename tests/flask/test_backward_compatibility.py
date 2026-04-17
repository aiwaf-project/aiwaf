#!/usr/bin/env python3
"""
Test backward compatibility for register_aiwaf_protection function name.
"""

from flask import Flask
from aiwaf.flask import register_aiwaf_protection, register_aiwaf_middlewares

def test_backward_compatibility():
    """Test that both function names work and are the same."""
    
    # Test 1: Both functions should be importable
    assert callable(register_aiwaf_protection)
    assert callable(register_aiwaf_middlewares)
    
    # Test 2: They should be the same function object
    assert register_aiwaf_protection is register_aiwaf_middlewares
    
    # Test 3: Both should work with Flask apps
    app1 = Flask(__name__ + '_1')
    app1.config['AIWAF_USE_CSV'] = True
    
    app2 = Flask(__name__ + '_2')
    app2.config['AIWAF_USE_CSV'] = True
    
    # This should not raise any exceptions
    register_aiwaf_middlewares(app1)
    register_aiwaf_protection(app2)
    
    print(" All backward compatibility tests passed!")

if __name__ == '__main__':
    test_backward_compatibility()