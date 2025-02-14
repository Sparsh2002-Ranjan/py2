from flask import Blueprint, request, jsonify
from transformers import pipeline
import re

summarization_bp = Blueprint('summarization', __name__)

# Load summarization model with optimized settings
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",  
    device=-1  # Keep inference on CPU to prevent Render crashes
)

def preprocess_text(text, max_tokens=512):
    """ Clean and truncate text for efficient summarization. """
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return " ".join(words[:max_tokens])  # Limit input length

@summarization_bp.route('/summarize', methods=['POST'])
def summarize_text():
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
            batch_size=1  # Reduces memory usage
        )

        return jsonify({"summary": summary[0]['summary_text']})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
