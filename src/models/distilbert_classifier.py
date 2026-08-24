import torch
from torch import nn
from transformers import DistilBertModel

class FinancialPhishingDistilBERT(nn.Module):
    """
    Context-Aware Sequence Classifier for Financial Phishing.
    Uses a pretrained DistilBERT backbone followed by a custom classification head.
    
    Classes:
    0: Legitimate
    1: Suspicious
    2: Financially Deceptive Phishing
    """
    def __init__(self, model_name="distilbert-base-uncased", num_classes=3, dropout_rate=0.3):
        super(FinancialPhishingDistilBERT, self).__init__()
        
        # Load the base contextualized embedding model
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Dropout for regularization during training
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head.
        # DistilBERT outputs a hidden state of dimension 768
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tokenized input sequences. Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask indicating padded tokens. Shape: (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Raw logits for each class. Shape: (batch_size, num_classes)
        """
        # Pass through DistilBERT
        # Returns a BaseModelOutput object where last_hidden_state is index 0
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the representation of the [CLS] token (the first token of the sequence)
        # This token aggregates the sequence-level semantic context.
        # Shape: (batch_size, hidden_size)
        pooled_output = outputs[0][:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Produce logits mapping to our 3 classes
        logits = self.classifier(pooled_output)
        
        return logits
