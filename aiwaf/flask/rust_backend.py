"""
Backward-compatible shim for Flask integrations.
Prefer aiwaf.core.rust_backend for shared functionality.
"""

from __future__ import annotations

try:
    import aiwaf_rust  # noqa: F401
except Exception:
    aiwaf_rust = None


def rust_available() -> bool:
    return aiwaf_rust is not None


def get_isolation_forest_class():
    if aiwaf_rust is None:
        return None
    return getattr(aiwaf_rust, "IsolationForest", None)


def load_isolation_forest(state):
    if aiwaf_rust is None or not hasattr(aiwaf_rust, "IsolationForest"):
        return None
    try:
        return aiwaf_rust.IsolationForest.from_json(state)
    except Exception:
        return None


def validate_headers(headers, required_headers=None, min_score=None):
    if aiwaf_rust is None:
        return None
    try:
        if hasattr(aiwaf_rust, "validate_headers_with_config"):
            return aiwaf_rust.validate_headers_with_config(headers, required_headers, min_score)
        return aiwaf_rust.validate_headers(headers)
    except Exception:
        return None


def extract_features(records, static_keywords):
    if aiwaf_rust is None:
        return None
    try:
        return aiwaf_rust.extract_features(records, static_keywords)
    except Exception:
        return None


def supports_chunked_feature_extraction() -> bool:
    if aiwaf_rust is None:
        return False
    return hasattr(aiwaf_rust, "extract_features_batch_with_state") and hasattr(aiwaf_rust, "finalize_feature_state")


def extract_features_batch(records, static_keywords, state=None):
    if aiwaf_rust is None or not hasattr(aiwaf_rust, "extract_features_batch_with_state"):
        return None, state
    try:
        result = aiwaf_rust.extract_features_batch_with_state(records, static_keywords, state)
        if isinstance(result, dict):
            return result.get("features"), result.get("state")
        if isinstance(result, (list, tuple)) and len(result) == 2:
            return result[0], result[1]
        return None, state
    except Exception:
        return None, state


def finalize_feature_state(static_keywords, state=None):
    if aiwaf_rust is None or not hasattr(aiwaf_rust, "finalize_feature_state"):
        return None
    try:
        result = aiwaf_rust.finalize_feature_state(static_keywords, state)
        if isinstance(result, dict):
            return result.get("features")
        return result
    except Exception:
        return None


def analyze_recent_behavior(entries, static_keywords):
    if aiwaf_rust is None:
        return None
    try:
        return aiwaf_rust.analyze_recent_behavior(entries, static_keywords)
    except Exception:
        return None


def write_csv_log(csv_file, headers, row):
    """Rust CSV logging is deprecated; always use Python fallback."""
    return False
