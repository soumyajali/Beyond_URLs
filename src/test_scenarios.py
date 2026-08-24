import torch
from transformers import DistilBertTokenizer
from src.models.distilbert_classifier import FinancialPhishingDistilBERT
from src.preprocess import TextPreprocessor

def test_scenarios():
    print("Loading retrained model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = FinancialPhishingDistilBERT(num_classes=3)
    model.load_state_dict(torch.load('models_saved/distilbert_phishing.pth', map_location=device))
    model.to(device)
    model.eval()
    
    preprocessor = TextPreprocessor()
    
    test_cases = [
        {
            "category": "Legitimate financial message",
            "input": "Your payment of ₹2,500 has been successfully processed. Transaction reference number is TXN458921.",
            "expected": 0 # Legitimate
        },
        {
            "category": "Normal account notification",
            "input": "Your monthly account statement is now available in your official banking application. Please log in to view your transactions.",
            "expected": 0 # Legitimate
        },
        {
            "category": "Suspicious verification message",
            "input": "Your account requires verification to continue using the service. Please review your account information.",
            "expected": 1 # Suspicious
        },
        {
            "category": "Financial phishing message (No URL)",
            "input": "URGENT! Your bank account will be suspended today. Send the OTP received on your phone to the verification officer immediately.",
            "expected": 2 # Phishing
        },
        {
            "category": "Fake refund",
            "input": "Your refund of ₹8,500 is ready to be processed. Pay the ₹499 verification fee immediately to receive your refund.",
            "expected": 2 # Phishing
        },
        {
            "category": "Bank impersonation",
            "input": "This is the bank security department. Your account has been flagged. Provide your OTP and account verification code immediately to prevent suspension.",
            "expected": 2 # Phishing
        }
    ]
    
    class_map = {0: "Legitimate", 1: "Suspicious", 2: "Phishing"}
    
    print("\n--- RUNNING SCENARIO TESTS ---\n")
    passed = 0
    
    for i, tc in enumerate(test_cases):
        text = preprocessor.clean_text(tc["input"])
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=-1).squeeze().cpu().numpy()
            
        pred_class = int(torch.argmax(outputs, dim=1).item())
        confidence = probs[pred_class] * 100
        
        is_pass = pred_class == tc["expected"]
        if is_pass: passed += 1
        
        print(f"Test {i+1}: {tc['category']}")
        print(f"Input: {tc['input']}")
        print(f"Expected: {class_map[tc['expected']]}")
        print(f"Actual: {class_map[pred_class]} ({confidence:.2f}%)")
        print(f"Status: {'✅ PASS' if is_pass else '❌ FAIL'}\n")
        
    print(f"Total Passed: {passed}/{len(test_cases)}")
    
if __name__ == "__main__":
    test_scenarios()
