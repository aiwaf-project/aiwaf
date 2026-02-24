"""
Optional Rust backend for header validation and CSV logging.
Falls back to Python if the Rust extension is unavailable.
"""

from __future__ import annotations

try:
    import aiwaf_rust  # Built via maturin/pyo3
except Exception:
    aiwaf_rust = None


def rust_available() -> bool:
    return aiwaf_rust is not None


def validate_headers(headers, required_headers=None, min_score=None) -> str | None:
    if aiwaf_rust is None:
        return None
    try:
        if hasattr(aiwaf_rust, "validate_headers_with_config"):
            return aiwaf_rust.validate_headers_with_config(
                headers,
                required_headers,
                min_score,
            )
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
    return (
        hasattr(aiwaf_rust, "extract_features_batch_with_state")
        and hasattr(aiwaf_rust, "finalize_feature_state")
    )


def extract_features_batch(records, static_keywords, state=None):
    """Chunk-aware feature extraction for Rust backends that support stateful batches.

    Expected extension contract:
    - extract_features_batch_with_state(records, static_keywords, state)
      returns either:
      - {"features": [...], "state": ...}
      - (features, state)
    """
    if aiwaf_rust is None:
        return None, state
    if not hasattr(aiwaf_rust, "extract_features_batch_with_state"):
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
    """Finalize pending state for chunked Rust feature extraction."""
    if aiwaf_rust is None:
        return None
    if not hasattr(aiwaf_rust, "finalize_feature_state"):
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
