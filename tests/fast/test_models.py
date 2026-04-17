from aiwaf.core.model_artifacts import sklearn_model_artifact


def test_model_metadata_sample_count():
    artifact = sklearn_model_artifact(object(), "1.0", ["a", "b", "c"], 99, framework="fastapi")
    assert artifact["samples_count"] == 99

