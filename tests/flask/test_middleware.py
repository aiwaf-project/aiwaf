from flask import Flask

from aiwaf.flask import middleware


def test_database_selection_and_initialization_fallback():
    app = Flask(__name__)
    app.config["AIWAF_USE_CSV"] = True
    assert not middleware._should_use_database(app)
    app.config["AIWAF_USE_CSV"] = False
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    assert middleware._should_use_database(app)
    middleware._init_database(app)
