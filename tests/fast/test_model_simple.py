from aiwaf.core.model_artifacts import rust_model_artifact


class _DummyRustModel:
    def to_json(self):
        return '{"ok":true}'


def test_rust_model_artifact_shape():
    artifact = rust_model_artifact(_DummyRustModel(), ["f1", "f2"], 5, framework="fastapi")
    assert artifact["backend"] == "rust"
    assert artifact["feature_count"] == 2

