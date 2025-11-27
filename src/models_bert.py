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
    

class PolitenessClassifierBERT:
    """
    BERT-based politeness classifier with fine-tuning
    
    Key Differences from LSTM:
    1. Uses pre-trained BERT (110M parameters already trained!)
    2. Only fine-tunes top layers
    3. Much faster convergence (3-5 epochs vs 20-50)
    4. Better performance on small datasets
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=128, learning_rate=2e-5):
        """
        Args:
            model_name: Pre-trained BERT model to use
            max_length: Maximum sequence length
            learning_rate: Learning rate (BERT uses small LR: 2e-5 to 5e-5)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        
        self.tokenizer = None
        self.model = None
        
        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
        self.reverse_label_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
    
    def train(self, texts, labels, epochs=5, batch_size=16, validation_split=0.1):
        """
        Fine-tune BERT on politeness data
        
        Args:
            texts: List of text strings
            labels: List of labels
            epochs: Number of epochs (3-5 is typical for BERT)
            batch_size: Batch size (16 or 32 for BERT)
            validation_split: Fraction for validation
        """
        print(f"\n{'='*80}")
        print(f"🔄 FINE-TUNING BERT MODEL")
        print(f"{'='*80}")
        print(f"   Model: {self.model_name}")
        print(f"   Training samples: {len(texts)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        
        # Load tokenizer
        print("\n📦 Loading pre-trained tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        print(f"   ✅ Tokenizer loaded")
        
        # Split train/validation
        from sklearn.model_selection import train_test_split
        
        if validation_split > 0:
            texts_train, texts_val, labels_train, labels_val = train_test_split(
                texts, labels, test_size=validation_split, random_state=42, stratify=labels
            )
            print(f"\n✂️  Data split:")
            print(f"   Training: {len(texts_train)}")
            print(f"   Validation: {len(texts_val)}")
        else:
            texts_train, labels_train = texts, labels
            texts_val, labels_val = None, None
            print(f"\n   Training on all {len(texts_train)} samples (no validation)")
        
        # Create datasets
        print("\n📊 Creating datasets...")
        train_dataset = PolitenessDatasetBERT(
            texts_train, labels_train, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if texts_val is not None:
            val_dataset = PolitenessDatasetBERT(
                texts_val, labels_val, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        print(f"   ✅ Training batches: {len(train_loader)}")
        if val_loader:
            print(f"   ✅ Validation batches: {len(val_loader)}")
        
        # Load pre-trained BERT for classification
        print("\n🏗️  Loading pre-trained BERT model...")
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,  # 3 classes: Impolite, Neutral, Polite
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)
        print(f"   ✅ BERT model loaded ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # No warmup
            num_training_steps=total_steps
        )
        
        print(f"   ✅ Optimizer configured (AdamW)")
        print(f"   ✅ Scheduler configured ({total_steps} total steps)")
        
        # Training loop
        print(f"\n{'='*80}")
        print("   TRAINING PROGRESS")
        print(f"{'='*80}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                loss.backward()
                
                # Clip gradients (prevent exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                total_train_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels_batch).sum().item()
                train_total += labels_batch.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate epoch metrics
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels_batch = batch['labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels_batch
                        )
                        
                        loss = outputs.loss
                        logits = outputs.logits
                        
                        total_val_loss += loss.item()
                        
                        predictions = torch.argmax(logits, dim=1)
                        val_correct += (predictions == labels_batch).sum().item()
                        val_total += labels_batch.size(0)
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                
                print(f"   Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}% | "
                      f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.2f}%")
            else:
                print(f"   Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}%")
        
        print(f"\n{'='*80}")
        print("✅ FINE-TUNING COMPLETE!")
        print(f"{'='*80}")
        
        return self
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained! Call train() first.")
        
        self.model.eval()
        
        # Create dataset
        dummy_labels = ['Neutral'] * len(texts)
        dataset = PolitenessDatasetBERT(texts, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                batch_predictions = torch.argmax(logits, dim=1)
                
                # Convert to labels
                for pred in batch_predictions:
                    predictions.append(self.reverse_label_map[pred.item()])
        
        return predictions
    
    def evaluate(self, texts, true_labels):
        """Evaluate model performance"""
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(
            true_labels,
            predictions,
            labels=['Impolite', 'Neutral', 'Polite']
        )
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }
    
    def save_model(self, filepath='models/bert_model'):
        """Save fine-tuned model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(filepath, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        print(f"💾 Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='models/bert_model'):
        """Load fine-tuned model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        # Create instance
        classifier = PolitenessClassifierBERT()
        
        # Load model and tokenizer
        classifier.tokenizer = BertTokenizer.from_pretrained(filepath)
        classifier.model = BertForSequenceClassification.from_pretrained(filepath)
        classifier.model.to(classifier.device)
        classifier.model.eval()
        
        print(f"✅ Model loaded from {filepath}")
        return classifier