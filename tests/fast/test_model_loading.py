from aiwaf.core.model_artifacts import sklearn_model_artifact
from aiwaf.core.model_serialization import dump_model_artifact, load_model_artifact

import pytest


class _DummyModel:
    pass


def test_model_artifact_contains_backend_metadata():
    artifact = sklearn_model_artifact(_DummyModel(), "1.0", ["f1"], 10, framework="fastapi")
    assert artifact["backend"] == "sklearn"
    assert artifact["framework"] == "fastapi"


def test_fastapi_sklearn_model_artifact_roundtrips_with_skops(tmp_path):
    pytest.importorskip("skops")
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import IsolationForest

    samples = [[0.0, 0.0], [0.1, 0.2], [10.0, 10.0], [0.2, 0.1]]
    model = IsolationForest(contamination=0.25, random_state=42)
    model.fit(samples)
    expected = list(model.predict(samples))
    artifact = sklearn_model_artifact(
        model,
        sklearn.__version__,
        ["f1", "f2"],
        len(samples),
        framework="fastapi",
    )

    model_path = tmp_path / "model.skops"
    dump_model_artifact(artifact, model_path)
    loaded = load_model_artifact(model_path)

    assert loaded["model_backend"] == "sklearn"
    assert loaded["framework"] == "fastapi"
    assert list(loaded["model"].predict(samples)) == expected
