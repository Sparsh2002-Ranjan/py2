from flask import Flask, Blueprint, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import re
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://studywave-sr.netlify.app"}})

summarization_bp = Blueprint('summarization', __name__)

# Load summarization model with optimized settings
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",  
    device=-1  # Keep inference on CPU to prevent Render crashes
)

def preprocess_text(text, max_tokens=512):
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return " ".join(words[:max_tokens])

@summarization_bp.route('/summarize', methods=['POST', 'GET'])
def summarize_text():
    if request.method == 'GET':
        return jsonify({"message": "Send a POST request with text to summarize."})
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        processed_text = preprocess_text(text)
        min_length = max(80, len(processed_text.split()) // 3)
        max_length = min(150, len(processed_text.split()) // 2)

        summary = summarizer(
            processed_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            batch_size=1
        )

        return jsonify({"summary": summary[0]['summary_text']})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

app.register_blueprint(summarization_bp)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
