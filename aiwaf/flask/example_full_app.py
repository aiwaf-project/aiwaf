# Full Flask AIWAF example app with database integration
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from aiwaf.flask.db_models import db
from aiwaf.flask.middleware import register_aiwaf_middlewares
from aiwaf.flask.storage import add_ip_whitelist, add_ip_blacklist, add_keyword

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aiwaf.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['AIWAF_RATE_WINDOW'] = 10
app.config['AIWAF_RATE_MAX'] = 20
app.config['AIWAF_RATE_FLOOD'] = 40
app.config['AIWAF_MIN_FORM_TIME'] = 1.0

db.init_app(app)

with app.app_context():
    db.create_all()

register_aiwaf_middlewares(app)

@app.route('/')
def index():
    return "AIWAF Flask full integration OK"

@app.route('/whitelist/<ip>')
def whitelist(ip):
    add_ip_whitelist(ip)
    return jsonify({'whitelisted': ip})

@app.route('/blacklist/<ip>')
def blacklist(ip):
    add_ip_blacklist(ip, reason='manual')
    return jsonify({'blacklisted': ip})

@app.route('/add_keyword/<kw>')
def add_kw(kw):
    add_keyword(kw)
    return jsonify({'added_keyword': kw})

if __name__ == '__main__':
    app.run(debug=True)
