# Flask-adapted decorators (stub)
from functools import wraps

def aiwaf_exempt(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Mark this endpoint as exempt from AIWAF
        return f(*args, **kwargs)
    decorated._aiwaf_exempt = True
    return decorated
