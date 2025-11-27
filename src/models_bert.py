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
        
        #self.tokenizer = None
        #self.model = None

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 💡 Critical: This line might be missing or failing!
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3 # Assuming 3 classes: Polite, Neutral, Impolite
        )
        self.model.to(self.device)

        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
        self.reverse_label_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
    
    def train(self, texts, labels, epochs=5, batch_size=16, validation_split=0.1):
        """
        Fine-tune BERT on politeness data with early stopping
        """
        # Training loop with early stopping
        print(f"\n{'='*80}")
        print("   TRAINING PROGRESS (with Early Stopping)")
        print(f"{'='*80}")

        best_val_accuracy = 0
        best_epoch = 0
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels_batch).sum().item()
                train_total += labels_batch.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
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
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    patience_counter = 0
                    print(f"   Epoch [{epoch+1}/{epochs}] "
                        f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}% | "
                        f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.2f}% ⭐ NEW BEST!")
                else:
                    patience_counter += 1
                    print(f"   Epoch [{epoch+1}/{epochs}] "
                        f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}% | "
                        f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.2f}% "
                        f"(No improvement: {patience_counter}/{patience})")
                    
                    # THIS IS THE KEY FIX - Actually break out of the loop!
                    if patience_counter >= patience:
                        print(f"\n   🛑 Early stopping triggered!")
                        print(f"   Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")
                        print(f"   Stopping training to prevent overfitting.")
                        break  # ← CRITICAL: This actually stops the loop
            else:
                print(f"   Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}%")

        print(f"\n{'='*80}")
        if val_loader is not None and best_epoch > 0:
            print(f"✅ TRAINING COMPLETE!")
            print(f"   Best Model: Epoch {best_epoch} with {best_val_accuracy:.2f}% validation accuracy")
        else:
            print("✅ FINE-TUNING COMPLETE!")
        print(f"{'='*80}")
    
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