"""
FastAPI compatibility wrapper around shared core Rust backend helpers.
"""

from aiwaf.core.rust_backend import (  # noqa: F401
    analyze_recent_behavior,
    extract_features,
    extract_features_batch,
    finalize_feature_state,
    rust_available,
    supports_chunked_feature_extraction,
    validate_headers,
)

__all__ = [
    "rust_available",
    "validate_headers",
    "extract_features",
    "supports_chunked_feature_extraction",
    "extract_features_batch",
    "finalize_feature_state",
    "analyze_recent_behavior",
]
