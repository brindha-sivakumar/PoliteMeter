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
        
        # Create embedding layer
        # nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # padding_idx=0 means embeddings for <PAD> won't be updated
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Don't learn embeddings for padding
        )
        
        # Create LSTM layer
        # nn.LSTM(embedding_dim, hidden_dim, num_layers, 
        #         batch_first=True, dropout=dropout, bidirectional=False)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,  # Only dropout between layers
            bidirectional=True
        )
        
        # Create dropout layer for regularization
        # nn.Dropout(dropout)
        
        self.dropout = nn.Dropout(0.5)
        
        # Create final linear layer to map hidden state to classes
        # nn.Linear(hidden_dim, output_dim)
        # Input must be 2 * hidden_dim for a bidirectional LSTM
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

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
    

class PolitenessClassifierLSTM:
    """
    Main LSTM classifier - Handles training, prediction, and evaluation
    """
    
    def __init__(self, embedding_dim=100, hidden_dim=256, max_length=100, 
                 num_layers=2, dropout=0.3):
        """
        Initialize the LSTM classifier
        
        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden state dimension
            max_length: Maximum sequence length
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.vocab = None
        self.model = None
        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
        self.reverse_label_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
        
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
    
    def _build_vocab(self, texts):
        """
        Build vocabulary from training texts
        
        Args:
            texts: List of text strings
        
        Returns:
            vocab: Dictionary {word: id}
        """
        print("   Building vocabulary...")
        word_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Keep top 5000 words
        vocab_size = 5000
        most_common = word_counts.most_common(vocab_size - 2)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, count) in enumerate(most_common, start=2):
            vocab[word] = idx
        
        print(f"   ✅ Vocabulary built: {len(vocab)} words")
        return vocab
    
    def train(self, texts, labels, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the LSTM model with class weighting for imbalanced data"""
        print(f"\n{'='*80}")
        print(f"🔄 TRAINING LSTM MODEL WITH CLASS WEIGHTS")
        print(f"{'='*80}")
        print(f"   Training samples: {len(texts)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Step 1: Build vocabulary
        self.vocab = self._build_vocab(texts)
        
        # Step 2: Create dataset and dataloader
        print("   Creating dataset...")
        train_dataset = PolitenessDataset(texts, labels, self.vocab, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"   ✅ DataLoader created: {len(train_loader)} batches")
        
        # Step 3: Initialize model
        print("   Initializing model...")
        vocab_size = len(self.vocab)
        self.model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=3,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        self.model = self.model.to(self.device)
        print(f"   ✅ Model initialized")
        
        # Step 4: Calculate class weights to handle imbalanced data
        print("\n   🎯 Calculating class weights to handle imbalance...")
        label_counts = Counter(labels)
        
        total_samples = len(labels)
        class_weights = []
        
        print(f"   Class distribution:")
        for class_name in ['Impolite', 'Neutral', 'Polite']:
            count = label_counts[class_name]
            # Calculate inverse frequency
            #raw_weight = total_samples / (3 * count)
            raw_weight = total_samples / (len(label_counts) * count)
            # Soften using square root (reduces extreme values)
            # Use a hyper-aggressive exponent (2.0) to force learning
            # CHANGE EXPONENT FROM 2.0 to 1.2 for better balance
            softened_weight = raw_weight ** 1.2
            class_weights.append(softened_weight)
            print(f"      {class_name}: {count} samples | "
                  f"raw: {raw_weight:.3f} → softened: {softened_weight:.3f}")
        
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # Use WEIGHTED loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"   ✅ Using weighted CrossEntropyLoss")
        
        # Step 5: Training loop
        print(f"\n{'='*80}")
        print("   TRAINING PROGRESS (with per-class tracking)")
        print(f"{'='*80}")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Track per-class predictions
            class_correct = {0: 0, 1: 0, 2: 0}
            class_total = {0: 0, 1: 0, 2: 0}
            
            for batch_texts, batch_labels in train_loader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                # Track per-class accuracy
                for i in range(len(batch_labels)):
                    label_idx = batch_labels[i].item()
                    class_total[label_idx] += 1
                    if predicted[i] == batch_labels[i]:
                        class_correct[label_idx] += 1
            
            # Calculate metrics
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            # Per-class accuracies
            imp_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
            neu_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
            pol_acc = 100 * class_correct[2] / class_total[2] if class_total[2] > 0 else 0
            
            print(f"   Epoch [{epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:.2f}% | Imp: {imp_acc:.1f}% Neu: {neu_acc:.1f}% Pol: {pol_acc:.1f}%")
        
        print(f"\n{'='*80}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*80}")
        
        return self
        
    def predict(self, texts):
        """
        TODO: Make predictions on new texts
        
        Args:
            texts: List of text strings
        
        Returns:
            predictions: List of predicted labels
        """
        if self.model is None or self.vocab is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dataset (no labels needed for prediction)
        # Use dummy labels
        dummy_labels = ['Neutral'] * len(texts)
        pred_dataset = PolitenessDataset(texts, dummy_labels, self.vocab, self.max_length)
        pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        
        # YOUR CODE HERE
        # Steps:
        # 1. No gradient calculation needed: use torch.no_grad()
        # 2. For each batch, get predictions
        # 3. Convert predictions to labels using reverse_label_map
        
        with torch.no_grad():  # Don't calculate gradients (faster)
            for batch_texts, _ in pred_loader:
                batch_texts = batch_texts.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_texts)
                
                # Get predicted class
                _, predicted = torch.max(outputs, 1)
                
                # Convert to labels
                batch_predictions = [self.reverse_label_map[p.item()] for p in predicted]
                predictions.extend(batch_predictions)
        
        return predictions
    
    def evaluate(self, texts, true_labels):
        """
        TODO: Evaluate model performance
        
        Args:
            texts: List of text strings
            true_labels: List of true labels
        
        Returns:
            dict with accuracy, report, confusion_matrix
        """
        # Get predictions
        predictions = self.predict(texts)
        
        # Calculate metrics
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
    
    def save_model(self, filepath='models/lstm_model.pkl'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save! Train first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state, vocabulary, and config
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_length': self.max_length,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        
        torch.save(save_dict, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='models/lstm_model.pkl'):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load saved dict
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Create new instance
        model = PolitenessClassifierLSTM(
            embedding_dim=save_dict['embedding_dim'],
            hidden_dim=save_dict['hidden_dim'],
            max_length=save_dict['max_length'],
            num_layers=save_dict['num_layers'],
            dropout=save_dict['dropout']
        )
        
        # Restore vocabulary
        model.vocab = save_dict['vocab']
        
        # Restore model
        vocab_size = len(model.vocab)
        model.model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=model.embedding_dim,
            hidden_dim=model.hidden_dim,
            output_dim=3,
            num_layers=model.num_layers,
            dropout=model.dropout
        )
        model.model.load_state_dict(save_dict['model_state_dict'])
        model.model = model.model.to(model.device)
        model.model.eval()
        
        print(f"✅ Model loaded from {filepath}")
        return model
    
if __name__ == "__main__":
    test_texts = [
        "Could you please help me?",
        "This is stupid!",
        "I need help with this."
    ]
    test_labels = ['Polite', 'Impolite', 'Neutral']
    
    # Create dataset
    dataset = PolitenessDataset(test_texts, test_labels, max_length=10)
    
    print("✅ Dataset created!")
    print(f"   Vocabulary size: {len(dataset.vocab)}")
    print(f"   Number of samples: {len(dataset)}")
    
    # Test __getitem__
    text_tensor, label_tensor = dataset[0]
    print(f"\n✅ First sample:")
    print(f"   Text IDs shape: {text_tensor.shape}")
    print(f"   Text IDs: {text_tensor}")
    print(f"   Label: {label_tensor} (0=Impolite, 1=Neutral, 2=Polite)")
    
    # Check vocabulary
    print(f"\n✅ Sample vocabulary entries:")
    for word in ['<PAD>', '<UNK>', 'please', 'help']:
        word_id = dataset.vocab.get(word, 'NOT FOUND')
        print(f"   '{word}': {word_id}")

    
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

    print("\n" + "="*80)
    print("🧪 TESTING FULL LSTM CLASSIFIER")
    print("="*80)
    
    # Small training set
    train_texts = [
        "Could you please help me with this?",
        "Thank you so much for your assistance!",
        "I really appreciate your help.",
        "This is completely stupid and wrong!",
        "Fix this garbage immediately!",
        "You have no idea what you're doing.",
        "I need help with this problem.",
        "Can you explain this to me?",
        "How does this work?"
    ]
    
    train_labels = [
        'Polite', 'Polite', 'Polite',
        'Impolite', 'Impolite', 'Impolite',
        'Neutral', 'Neutral', 'Neutral'
    ]
    
    # Create and train classifier
    lstm_classifier = PolitenessClassifierLSTM(
        embedding_dim=50,
        hidden_dim=64,
        max_length=20
    )
    
    # Train for just 3 epochs (quick test)
    lstm_classifier.train(train_texts, train_labels, epochs=3, batch_size=3)
    
    # Test predictions
    test_texts = [
        "Could you please help?",
        "This is terrible!",
        "I need assistance."
    ]
    
    predictions = lstm_classifier.predict(test_texts)
    
    print(f"\n✅ Test Predictions:")
    for text, pred in zip(test_texts, predictions):
        print(f"   '{text}' → {pred}")
    
    print("\n✅ Full LSTM Classifier test complete!")