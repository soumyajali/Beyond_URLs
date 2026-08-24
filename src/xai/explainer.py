import shap
import torch
import numpy as np
import re

class TextExplainer:
    """
    Explainable AI (XAI) Module for Financial Phishing Detection.
    Provides local interpretability via SHAP and heuristic context indicators.
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        def predict_fn(texts):
            inputs = self.tokenizer(
                texts.tolist() if isinstance(texts, np.ndarray) else texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()
            return probs
            
        self.predict_fn = predict_fn
        self.explainer = shap.Explainer(self.predict_fn, self.tokenizer)

    def extract_indicators(self, text: str):
        """
        Rule-based context extractor to augment the AI explanation.
        """
        indicators = []
        text_lower = text.lower()
        
        # Categories (Context-Aware)
        
        # Financial context vs Deception
        if re.search(r'\b(pay the|transfer|wire|send money|fee|urgent payment)\b', text_lower):
            indicators.append({"type": "warning", "text": "Suspicious financial transfer request"})
        elif re.search(r'\b(payment|refund|account|bank|money|₹|\$)\b', text_lower):
            indicators.append({"type": "safe", "text": "Financial context detected"})
            
        # Urgency
        if re.search(r'\b(urgent|immediately|act now|right now|deadline|within a few minutes)\b', text_lower):
            indicators.append({"type": "warning", "text": "Urgent language detected"})
            
        # Credentials
        if re.search(r'\b(otp|pin|password|cvv|verification code|verify your account)\b', text_lower):
            indicators.append({"type": "warning", "text": "Sensitive information or verification requested"})
            
        # Authority
        if re.search(r'\b(manager|ceo|government|security department|officer|irs)\b', text_lower):
            indicators.append({"type": "warning", "text": "Possible authority impersonation"})
            
        # Threat
        if re.search(r'\b(suspended|blocked|closed|penalty|arrested|prevent suspension)\b', text_lower):
            indicators.append({"type": "warning", "text": "Threatening language (e.g., account suspension)"})
            
        # Normal operations
        if re.search(r'\b(successfully processed|credited|shipped|available in your|completed|received)\b', text_lower):
            indicators.append({"type": "safe", "text": "Normal transaction notification"})
            
        if not any(indicator['type'] == 'warning' for indicator in indicators):
            indicators.append({"type": "safe", "text": "No threatening language or credential requests"})
            
        return indicators

    def generate_explanation_text(self, predicted_class, indicators):
        """Generates a human readable explanation based on indicators."""
        warnings = [i for i in indicators if i['type'] == 'warning']
        safes = [i for i in indicators if i['type'] == 'safe']
        
        if predicted_class == 2: # Phishing
            if warnings:
                traits = ", ".join([w['text'].lower() for w in warnings])
                return f"The message combines {traits}. This combination is strongly associated with financially deceptive phishing."
            return "The model detected deceptive linguistic patterns associated with phishing."
        elif predicted_class == 1: # Suspicious
            return "The message contains unusual requests, but lacks explicit financial threats or strong impersonation."
        else: # Legitimate
            return "This message appears to report normal activity. It does not request an OTP, password, PIN, payment, transfer, or other sensitive action."

    def explain(self, text: str):
        """
        Generates SHAP attribution values and context indicators.
        """
        shap_values = self.explainer([text])
        probs = self.predict_fn([text])[0]
        predicted_class_idx = int(np.argmax(probs))
        
        values_for_class = shap_values.values[0][:, predicted_class_idx]
        tokens = shap_values.data[0]
        
        word_weights = {}
        for token, val in zip(tokens, values_for_class):
            clean_token = token.strip()
            # Filter out subword artifacts, punctuation, and zero values
            if clean_token and clean_token not in ["[CLS]", "[SEP]", "[PAD]"] and not clean_token.startswith("##") and len(clean_token) > 1:
                # Only care about words with some actual weight
                if abs(val) > 0.01:
                    word_weights[clean_token] = float(val)
                
        sorted_weights = sorted(word_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        indicators = self.extract_indicators(text)
        explanation_text = self.generate_explanation_text(predicted_class_idx, indicators)
        
        return {
            "predicted_class": predicted_class_idx,
            "confidence": float(probs[predicted_class_idx]),
            "probabilities": probs.tolist(),
            "attributions": sorted_weights,
            "indicators": indicators,
            "explanation": explanation_text
        }
