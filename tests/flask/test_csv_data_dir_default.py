from flask import Flask

from aiwaf.core.storage_schema import DEFAULT_DATA_DIR
from aiwaf.flask.storage import _get_data_dir


def test_flask_csv_data_dir_defaults_to_aiwaf_data_when_unset(monkeypatch):
    monkeypatch.delenv("AIWAF_DATA_DIR", raising=False)

    app = Flask(__name__)
    app.config["AIWAF_USE_CSV"] = True

    with app.app_context():
        app.config.pop("AIWAF_DATA_DIR", None)
        assert _get_data_dir() == DEFAULT_DATA_DIR


def test_flask_csv_data_dir_uses_app_config_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AIWAF_DATA_DIR", str(tmp_path / "env_data"))
    configured = tmp_path / "configured_data"

    app = Flask(__name__)
    app.config["AIWAF_USE_CSV"] = True
    app.config["AIWAF_DATA_DIR"] = str(configured)

    with app.app_context():
        assert _get_data_dir() == str(configured)
