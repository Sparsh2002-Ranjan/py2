import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://studywave-sr.netlify.app"}})

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def preprocess_text(text, max_tokens=300):
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(text.split()[:max_tokens])

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Summarization API!"})

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        processed_text = preprocess_text(text)
        min_length = max(50, len(processed_text.split()) // 4)
        max_length = min(100, len(processed_text.split()) // 2)

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
    port = int(os.environ.get('PORT', 5000))  # Use Render's assigned port
    app.run(host='0.0.0.0', port=port)
