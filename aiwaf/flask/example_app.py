from flask import Flask, request, jsonify
from aiwaf.flask.middleware import register_aiwaf_middlewares

app = Flask(__name__)

# Configure AIWAF settings
app.config['AIWAF_RATE_WINDOW'] = 60
app.config['AIWAF_RATE_MAX'] = 100
app.config['AIWAF_USE_CSV'] = True

# Initialize AIWAF middleware
register_aiwaf_middlewares(app)

@app.route('/')
def index():
    return "AIWAF Flask integration is running!"

@app.route('/protected', methods=['GET', 'POST'])
def protected():
    # Example endpoint protected by AIWAF
    return jsonify({"message": "Protected endpoint accessed."})

if __name__ == '__main__':
    app.run(debug=True)
