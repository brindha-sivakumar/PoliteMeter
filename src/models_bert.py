"""
models_bert.py
BERT-based politeness classifier using Hugging Face Transformers
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import os
from torch.optim import AdamW


class PolitenessDatasetBERT(Dataset):
    """
    Dataset for BERT - uses BertTokenizer instead of custom vocabulary
    
    Key Differences from LSTM Dataset:
    1. Uses pre-trained tokenizer (no vocabulary building)
    2. Returns attention_mask (tells BERT which tokens are padding)
    3. Uses token_type_ids (for sentence pairs, not needed here)
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of text strings
            labels: List of labels ('Polite', 'Neutral', 'Impolite')
            tokenizer: Pre-trained BertTokenizer
            max_length: Maximum sequence length (128 is good for BERT)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label mapping
        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        YOUR CHALLENGE: Understand BERT tokenization
        
        The tokenizer does a LOT automatically:
        1. Splits into WordPiece tokens
        2. Adds [CLS] and [SEP] tokens
        3. Converts to IDs
        4. Creates attention_mask (1 for real tokens, 0 for padding)
        5. Pads to max_length
        
        All in one call!
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # BERT tokenizer does everything!
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding='max_length',         # Pad to max_length
            truncation=True,              # Truncate if too long
            return_tensors='pt'           # Return PyTorch tensors
        )
        
        # Extract components
        input_ids = encoding['input_ids'].squeeze()           # Token IDs
        attention_mask = encoding['attention_mask'].squeeze()  # Mask (1=real, 0=pad)
        
        # Convert label to number
        label_id = self.label_map[label]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_id, dtype=torch.long)
        }