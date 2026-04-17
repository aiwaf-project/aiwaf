from aiwaf.core.model_artifacts import sklearn_model_artifact


class _DummyModel:
    pass


def test_model_artifact_contains_backend_metadata():
    artifact = sklearn_model_artifact(_DummyModel(), "1.0", ["f1"], 10, framework="fastapi")
    assert artifact["backend"] == "sklearn"
    assert artifact["framework"] == "fastapi"

