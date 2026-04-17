from pathlib import Path

from aiwaf.core import model_artifacts


def test_model_artifacts_module_is_importable():
    assert Path(model_artifacts.__file__).exists()

