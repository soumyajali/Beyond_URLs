import torch
from torch import nn

class FinancialPhishingBiLSTM(nn.Module):
    """
    Bidirectional LSTM Classifier for Financial Phishing Detection.
    Serves as an alternative sequential model to capture deep semantic dependencies.
    
    Classes:
    0: Legitimate
    1: Suspicious
    2: Financially Deceptive Phishing
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_layers=2, num_classes=3, dropout_rate=0.4):
        super(FinancialPhishingBiLSTM, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM to capture context from both directions (left-to-right and right-to-left)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        # Multiply hidden_dim by 2 because it's a Bi-directional LSTM
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Padded token indices. Shape: (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Class logits. Shape: (batch_size, num_classes)
        """
        # 1. Map tokens to embeddings. Shape: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(input_ids)
        
        # 2. Pass through Bi-LSTM.
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        # hidden_state shape: (num_layers * 2, batch_size, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 3. Concatenate the final forward and backward hidden states
        # hidden[-2, :, :] is the last forward state, hidden[-1, :, :] is the last backward state
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # 4. Apply dropout and pass to dense classification layer
        hidden_cat = self.dropout(hidden_cat)
        logits = self.fc(hidden_cat)
        
        return logits
