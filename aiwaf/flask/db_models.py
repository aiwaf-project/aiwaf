# SQLAlchemy models for AIWAF Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class WhitelistedIP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.String(45), unique=True, nullable=False)

class BlacklistedIP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.String(45), unique=True, nullable=False)
    reason = db.Column(db.String(255))
    extended_request_info = db.Column(db.JSON, nullable=True)

class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    keyword = db.Column(db.String(255), unique=True, nullable=False)

class GeoBlockedCountry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country_code = db.Column(db.String(8), unique=True, nullable=False)
