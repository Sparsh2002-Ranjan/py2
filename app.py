import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://studywave-sr.netlify.app"}})

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",  # Lighter model
    device=-1
)

def preprocess_text(text, max_tokens=150):  # Reduce max tokens to lower memory usage
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(text.split()[:max_tokens])

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://studywave-sr.netlify.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Summarization API!"})

@app.route('/summarize', methods=['OPTIONS'])
def handle_options():
    return '', 204

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        processed_text = preprocess_text(text)
        min_length = max(30, len(processed_text.split()) // 5)
        max_length = min(80, len(processed_text.split()) // 2)

        summary = summarizer(
            processed_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return jsonify({"summary": summary[0]['summary_text']})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
