"""
Framework-agnostic core utilities for AIWAF.
"""

from .constants import STATUS_IDX  # noqa: F401
from .defaults import DEFAULT_EXEMPT_PATHS_DJANGO, DEFAULT_EXEMPT_PATHS_FLASK  # noqa: F401
from . import exemptions  # noqa: F401
from . import header_validation # noqa: F401
from . import geoip  # noqa: F401
from . import logs  # noqa: F401
from . import storage_csv  # noqa: F401
from . import storage_schema  # noqa: F401
from . import storage_ops  # noqa: F401
from . import keyword_fallback  # noqa: F401
from . import storage_interfaces  # noqa: F401
from . import storage_csv_impl  # noqa: F401
from . import model_artifacts  # noqa: F401
from . import training  # noqa: F401
from . import training_logic  # noqa: F401
from . import training_features  # noqa: F401
from . import whois  # noqa: F401
from . import utils  # noqa: F401
from . import uuid_tamper  # noqa: F401
from . import rate_limit  # noqa: F401
from . import honeypot  # noqa: F401
from . import ip_keyword  # noqa: F401
from . import method_validation  # noqa: F401
from . import geo_policy  # noqa: F401
from . import block_responses  # noqa: F401
from . import request_context  # noqa: F401
from .rust_backend import (  # noqa: F401
    rust_available,
    rust_isolation_forest_available,
    rust_isolation_forest_class,
    is_rust_isolation_forest,
    rust_isolation_forest_from_json,
    validate_headers,
    extract_features,
    supports_chunked_feature_extraction,
    extract_features_batch,
    finalize_feature_state,
    analyze_recent_behavior,
)


def __getattr__(name):
    if name == "AIWAF":
        from aiwaf.fast.core import AIWAF

        return AIWAF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "STATUS_IDX",
    "DEFAULT_EXEMPT_PATHS_DJANGO",
    "DEFAULT_EXEMPT_PATHS_FLASK",
    "exemptions",
    "header_validation",
    "geoip",
    "logs",
    "storage_csv",
    "storage_schema",
    "storage_ops",
    "keyword_fallback",
    "storage_interfaces",
    "storage_csv_impl",
    "model_artifacts",
    "training",
    "training_logic",
    "training_features",
    "whois",
    "utils",
    "uuid_tamper",
    "rate_limit",
    "honeypot",
    "ip_keyword",
    "method_validation",
    "geo_policy",
    "block_responses",
    "request_context",
    "rust_available",
    "rust_isolation_forest_available",
    "rust_isolation_forest_class",
    "is_rust_isolation_forest",
    "rust_isolation_forest_from_json",
    "validate_headers",
    "extract_features",
    "supports_chunked_feature_extraction",
    "extract_features_batch",
    "finalize_feature_state",
    "analyze_recent_behavior",
    "AIWAF",
]
