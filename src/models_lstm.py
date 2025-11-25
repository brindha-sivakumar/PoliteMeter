"""
models_lstm.py
LSTM-based politeness classifier using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

class PolitenessDataset(Dataset):
    """
    PyTorch Dataset for politeness classification
    Handles text → ID conversion and padding
    """
    
    def __init__(self, texts, labels, vocab=None, max_length=100):
        """
        Args:
            texts: List of text strings
            labels: List of labels (strings: 'Polite', 'Neutral', 'Impolite')
            vocab: Dictionary {word: id}. If None, build from texts
            max_length: Maximum sequence length (pad/truncate to this)
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Label mapping
        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab
    
    def _build_vocab(self, texts):
        
        word_counts = Counter()
        
        # Count words in all texts
        for text in texts:
            words = text.lower().split()  # Simple tokenization
            word_counts.update(words)
        
        # Get most common words
        vocab_size = 5000
        most_common = word_counts.most_common(vocab_size - 2)  # -2 for PAD and UNK
        
        # Build vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, count) in enumerate(most_common, start=2):
            vocab[word] = idx
        
        return vocab
    
    def _text_to_ids(self, text):
        
        words = text.lower().split()
        
        # Convert words to IDs
        ids = []
        for word in words:
            word_id = self.vocab.get(word, 1)  # 1 = <UNK>
            ids.append(word_id)
        
        # Pad or truncate to max_length
        if len(ids) < self.max_length:
            # Pad with 0s
            ids = ids + [0] * (self.max_length - len(ids))
        else:
            # Truncate
            ids = ids[:self.max_length]
        
        return ids
    
    def __len__(self):
        """Return number of samples"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to IDs
        text_ids = self._text_to_ids(text)
        
        # Convert label to number
        label_id = self.label_map[label]
        
        # Convert to tensors
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        
        return text_tensor, label_tensor
    
class LSTMClassifier(nn.Module):
    """
    LSTM Neural Network for text classification
    
    Architecture:
    Input → Embedding → LSTM → Linear → Output
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=3, 
                num_layers=2, dropout=0.5):
        """
        TODO: Initialize the network layers
        
        Args:
            vocab_size: Size of vocabulary (number of unique words)
            embedding_dim: Dimension of word embeddings (e.g., 100)
            hidden_dim: Dimension of LSTM hidden state (e.g., 256)
            output_dim: Number of output classes (3 for our case)
            num_layers: Number of LSTM layers (2 = bidirectional-like)
            dropout: Dropout probability for regularization
        """
        super(LSTMClassifier, self).__init__()
        
        # YOUR CODE HERE
        # Hint 1: Create embedding layer
        # nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # padding_idx=0 means embeddings for <PAD> won't be updated
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Don't learn embeddings for padding
        )
        
        # YOUR CODE HERE
        # Hint 2: Create LSTM layer
        # nn.LSTM(embedding_dim, hidden_dim, num_layers, 
        #         batch_first=True, dropout=dropout, bidirectional=False)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,  # Only dropout between layers
            bidirectional=False
        )
        
        # YOUR CODE HERE
        # Hint 3: Create dropout layer for regularization
        # nn.Dropout(dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # YOUR CODE HERE
        # Hint 4: Create final linear layer to map hidden state to classes
        # nn.Linear(hidden_dim, output_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        """
        TODO: Define forward pass (how data flows through network)
        
        Args:
            text: Tensor of word IDs, shape (batch_size, seq_length)
        
        Returns:
            output: Class scores, shape (batch_size, output_dim)
        
        Steps:
        1. Pass text through embedding layer
        2. Pass embeddings through LSTM
        3. Extract final hidden state from LSTM output
        4. Apply dropout
        5. Pass through linear layer
        """
        
        
        # Step 1: Embedding
        # Shape: (batch, seq_length) → (batch, seq_length, embedding_dim)
        embedded = self.embedding(text)
        
        # Step 2: LSTM
        # lstm_out shape: (batch, seq_length, hidden_dim)
        # hidden shape: (num_layers, batch, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Step 3: Extract final hidden state
        # We want the last timestep's output for each sequence
        # Option A: Use lstm_out[:, -1, :] (last timestep)
        # Option B: Use hidden[-1, :, :] (last layer's hidden state)
        # Let's use Option A (more common for classification)
        
        # Get last timestep output
        # Shape: (batch, hidden_dim)
        last_output = lstm_out[:, -1, :]
        
        # Step 4: Apply dropout
        dropped = self.dropout(last_output)
        
        # Step 5: Linear layer to get class scores
        # Shape: (batch, output_dim)
        output = self.fc(dropped)
        
        return output
    
    # Test the LSTM network (add to __main__ section)
if __name__ == "__main__":
    # ... (previous dataset test code) ...
    
    print("\n" + "="*60)
    print("🧪 TESTING LSTM NETWORK")
    print("="*60)
        
    # Create network
    vocab_size = len(dataset.vocab)
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=50,  # Small for testing
        hidden_dim=128,
        output_dim=3
    )
    
    print(f"\n✅ Model created!")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Model architecture:")
    print(model)
    
    # Test forward pass with one sample
    text_tensor, label_tensor = dataset[0]
    
    # Add batch dimension: (seq_length) → (1, seq_length)
    text_batch = text_tensor.unsqueeze(0)
    
    print(f"\n✅ Testing forward pass:")
    print(f"   Input shape: {text_batch.shape}")
    
    # Forward pass
    output = model(text_batch)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Raw scores: {output}")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"   Probabilities: {probabilities}")
    
    # Get prediction
    predicted_class = torch.argmax(probabilities, dim=1)
    print(f"   Predicted class: {predicted_class.item()} (0=Impolite, 1=Neutral, 2=Polite)")
    print(f"   Actual class: {label_tensor.item()}")
    
    # Test with a batch
    print(f"\n✅ Testing with batch of 3:")
    batch_texts = torch.stack([dataset[i][0] for i in range(3)])
    batch_labels = torch.stack([dataset[i][1] for i in range(3)])
    
    print(f"   Batch input shape: {batch_texts.shape}")
    
    batch_output = model(batch_texts)
    print(f"   Batch output shape: {batch_output.shape}")
    
    batch_probs = torch.softmax(batch_output, dim=1)
    batch_preds = torch.argmax(batch_probs, dim=1)
    
    print(f"   Predictions: {batch_preds}")
    print(f"   Actual:      {batch_labels}")
    
    print("\n✅ LSTM Network test complete!")