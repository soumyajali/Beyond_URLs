import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.preprocess import TextPreprocessor
from src.dataset import PhishingDataset, generate_dummy_data
from src.models.distilbert_classifier import FinancialPhishingDistilBERT

def train():
    # 1. Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # 2. Load Data (Using Mock Data for now)
    print("Generating mock dataset...")
    df = generate_dummy_data(num_samples=1500)
    
    # 3. Preprocessing
    preprocessor = TextPreprocessor()
    print("Cleaning text (removing URLs, special chars)...")
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    # 5. Tokenization
    print("Tokenizing data for DistilBERT...")
    train_encodings = preprocessor.tokenize_for_bert(X_train)
    test_encodings = preprocessor.tokenize_for_bert(X_test)
    
    # 6. Create PyTorch Datasets & DataLoaders
    train_dataset = PhishingDataset(train_encodings, y_train)
    test_dataset = PhishingDataset(test_encodings, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 7. Initialize Model, Optimizer, Loss Function
    model = FinancialPhishingDistilBERT(num_classes=3)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 3
    
    # 8. Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Training Loss: {avg_train_loss:.4f}")
        
    # 9. Evaluation Loop
    print("\nStarting Evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 10. Metrics Output for VTU Report
    print("\n=== Classification Report ===")
    target_names = ['Legitimate', 'Suspicious', 'Financially Deceptive']
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")

    # 11. Save model
    os.makedirs('models_saved', exist_ok=True)
    torch.save(model.state_dict(), 'models_saved/distilbert_phishing.pth')
    print("\nModel saved to models_saved/distilbert_phishing.pth")

if __name__ == "__main__":
    train()
