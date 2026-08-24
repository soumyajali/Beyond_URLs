import os
import torch
from flask import Flask, request, jsonify, render_template
from src.preprocess import TextPreprocessor
from src.models.distilbert_classifier import FinancialPhishingDistilBERT
from src.xai.explainer import TextExplainer

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Configuration & Model Initialization ---
# Setup device (MPS for Mac M-series, or CPU/CUDA)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize Preprocessor
preprocessor = TextPreprocessor()

# Initialize Model and Load Weights
model = FinancialPhishingDistilBERT(num_classes=3)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models_saved', 'distilbert_phishing.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Initialize XAI Explainer
explainer = TextExplainer(model, preprocessor.tokenizer, device=device)

# Classes mapping
CLASS_MAP = {
    0: "Legitimate",
    1: "Suspicious",
    2: "Financially Deceptive Phishing"
}

@app.route('/', methods=['GET'])
def index():
    """Serve the lightweight web interface."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({"status": "Healthy", "device": str(device)})

@app.route('/api/v1/analyze_text', methods=['POST'])
def analyze_text():
    """
    Main endpoint for analyzing incoming text messages for financial phishing.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in JSON payload"}), 400
            
        raw_text = data['text']
        
        # 1. Clean the text (Remove URLs to adhere to constraint)
        cleaned_text = preprocessor.clean_text(raw_text)
        
        # Check if empty after cleaning
        if not cleaned_text.strip():
             return jsonify({
                 "status": "error",
                 "message": "Text is empty after removing URLs/noise. No semantic context found."
             }), 400
        
        # 2. Get XAI attribution and prediction
        explanation = explainer.explain(cleaned_text)
        
        # 3. Format Response
        predicted_class_id = explanation["predicted_class"]
        
        # Format probabilities dictionary
        prob_dict = {
            CLASS_MAP[0]: round(explanation["probabilities"][0] * 100, 2),
            CLASS_MAP[1]: round(explanation["probabilities"][1] * 100, 2),
            CLASS_MAP[2]: round(explanation["probabilities"][2] * 100, 2)
        }
        
        response = {
            "status": "success",
            "original_text": raw_text,
            "analyzed_text": cleaned_text, # Shows URLs were converted
            "prediction": CLASS_MAP[predicted_class_id],
            "confidence": round(explanation["confidence"] * 100, 2),
            "probabilities": prob_dict,
            "indicators": explanation["indicators"],
            "explanation": explanation["explanation"],
            "xai_explanations": explanation["attributions"][:10] # Top 10 meaningful words
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask development server on a unique port
    app.run(host='0.0.0.0', port=8431, debug=True)
