import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://studywave-sr.netlify.app"}})

def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-base",  # Lighter model
        device=-1  # Forces CPU for stability
    )

def preprocess_text(text, max_tokens=200):
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(text.split()[:max_tokens])

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Optimized Summarization API!"})

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

        summarizer = load_summarizer()  # Load model only when needed
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
