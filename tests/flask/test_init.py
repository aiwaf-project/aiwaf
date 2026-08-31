from aiwaf import flask as flask_adapter


def test_lazy_logging_modules_load():
    assert flask_adapter._load_logging()
    assert flask_adapter._load_logger()
