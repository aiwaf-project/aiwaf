"""
Shared helpers to build model artifact payloads.
"""

from __future__ import annotations

from datetime import datetime


def rust_model_artifact(model, feature_cols, samples_count, framework: str):
    return {
        "model_type": "aiwaf_rust.IsolationForest",
        "model_state": model.to_json(),
        "created_at": str(datetime.now()),
        "feature_count": len(feature_cols),
        "samples_count": samples_count,
        "framework": framework,
        "backend": "rust",
        "model_backend": "aiwaf_rust",
    }


def sklearn_model_artifact(model, sklearn_version: str, feature_cols, samples_count, framework: str):
    return {
        "model": model,
        "sklearn_version": sklearn_version,
        "created_at": str(datetime.now()),
        "feature_count": len(feature_cols),
        "samples_count": samples_count,
        "framework": framework,
        "backend": "sklearn",
        "model_backend": "sklearn",
    }
