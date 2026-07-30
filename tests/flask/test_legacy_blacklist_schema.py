from flask import Flask
from sqlalchemy import text

from aiwaf.flask.db_models import db
from aiwaf.flask.storage import (
    add_ip_blacklist,
    is_ip_blacklisted,
    remove_ip_blacklist,
)


def test_unmigrated_sqlalchemy_blacklist_keeps_legacy_behavior():
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        AIWAF_USE_CSV=False,
        TESTING=True,
    )
    db.init_app(app)

    with app.app_context():
        db.session.execute(text(
            "CREATE TABLE blacklisted_ip ("
            "id INTEGER PRIMARY KEY, "
            "ip VARCHAR(45) UNIQUE NOT NULL, "
            "reason VARCHAR(255)"
            ")"
        ))
        db.session.commit()

        add_ip_blacklist("203.0.113.20", "legacy scanner")
        assert is_ip_blacklisted("203.0.113.20")

        reason = db.session.execute(text(
            "SELECT reason FROM blacklisted_ip WHERE ip = :ip"
        ), {"ip": "203.0.113.20"}).scalar_one()
        assert reason == "legacy scanner"

        remove_ip_blacklist("203.0.113.20")
        assert not is_ip_blacklisted("203.0.113.20")
