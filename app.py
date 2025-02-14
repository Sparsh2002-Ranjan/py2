import os
from flask import Flask, jsonify
from flask_cors import CORS
from TakeATest import take_a_test_bp
from Summarization import summarization_bp

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "https://studywave-sr.netlify.app"}})

app.register_blueprint(take_a_test_bp)
app.register_blueprint(summarization_bp)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server is running!"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Running Flask app on http://0.0.0.0:{port}")  
    app.run(host="0.0.0.0", port=port, debug=True)
