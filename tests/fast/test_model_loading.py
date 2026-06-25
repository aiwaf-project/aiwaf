from aiwaf.core.model_artifacts import sklearn_model_artifact
from aiwaf.core.model_serialization import dump_model_artifact, load_model_artifact

import pytest


class _DummyModel:
    pass


def test_model_artifact_contains_backend_metadata():
    artifact = sklearn_model_artifact(_DummyModel(), "1.0", ["f1"], 10, framework="fastapi")
    assert artifact["backend"] == "sklearn"
    assert artifact["framework"] == "fastapi"


def test_fastapi_json_model_artifact_roundtrips(tmp_path):
    artifact = {
        "model_type": "aiwaf_rust.IsolationForest",
        "model_state": {"trees": [], "threshold": 0.0},
        "model_backend": "aiwaf_rust",
        "framework": "fastapi",
    }
    model_path = tmp_path / "model.json"
    dump_model_artifact(artifact, model_path)
    loaded = load_model_artifact(model_path)

    assert loaded["model_backend"] == "aiwaf_rust"
    assert loaded["framework"] == "fastapi"
    assert loaded["model_state"] == artifact["model_state"]


def test_fastapi_rejects_python_object_model_artifact(tmp_path):
    class UnsafeModelObject:
        pass

    artifact = sklearn_model_artifact(
        UnsafeModelObject(),
        "test",
        ["f1"],
        1,
        framework="fastapi",
    )

    with pytest.raises(RuntimeError, match="JSON serializable"):
        dump_model_artifact(artifact, tmp_path / "model.json")
