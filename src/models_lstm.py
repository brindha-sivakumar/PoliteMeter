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