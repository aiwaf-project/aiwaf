"""
Flask compatibility wrapper around shared core Rust backend helpers.
"""

from aiwaf.core.rust_backend import (  # noqa: F401
    aiwaf_rust,
    analyze_recent_behavior,
    extract_features,
    extract_features_batch,
    finalize_feature_state,
    rust_available,
    rust_isolation_forest_class as get_isolation_forest_class,
    rust_isolation_forest_from_json as load_isolation_forest,
    supports_chunked_feature_extraction,
)
from aiwaf.core import rust_backend as _core_rust_backend


def validate_headers(headers, required_headers=None, min_score=None):
    """Pass arguments through unchanged, unlike core, which normalizes header names.

    Core expects a list of required headers and would misread this function's
    method-keyed form, such as {"GET": ["User-Agent"]}.
    """
    backend = _core_rust_backend.aiwaf_rust
    if backend is None:
        return None
    try:
        if hasattr(backend, "validate_headers_with_config"):
            return backend.validate_headers_with_config(headers, required_headers, min_score)
        return backend.validate_headers(headers)
    except Exception:
        return None


def write_csv_log(csv_file, headers, row):
    """Rust CSV logging is deprecated; always use Python fallback."""
    return False
