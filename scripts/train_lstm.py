"""
train_lstm.py
Train LSTM model on full Stanford Politeness Corpus
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.models_lstm import PolitenessClassifierLSTM
from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

def main():
    print("="*80)
    print("🚀 TRAINING LSTM ON FULL DATASET")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data = load_processed_data()
    print(f"✅ Loaded {len(data)} samples")
    
    # Show distribution
    print(f"\n📊 Class Distribution:")
    print(data['Politeness'].value_counts())
    
    # Prepare data
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Split data (same split as SVM for fair comparison)
    print("\n✂️  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create LSTM classifier
    print("\n🏗️  Creating LSTM classifier...")
    lstm_model = PolitenessClassifierLSTM(
        embedding_dim=100,      # Size of word embeddings
        hidden_dim=256,         # LSTM hidden state size
        max_length=100,         # Maximum sequence length
        num_layers=2,           # Number of LSTM layers
        dropout=0.3             # Dropout for regularization
    )
    
    # Train model
    print("\n" + "="*80)
    print("🔄 STARTING TRAINING")
    print("="*80)
    
    lstm_model.train(
        texts=X_train,
        labels=y_train,
        epochs=50,              # Number of training epochs
        batch_size=32,          # Batch size
        learning_rate=0.005   # Learning rate
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("📊 EVALUATING ON TEST SET")
    print("="*80)
    
    results = lstm_model.evaluate(X_test, y_test)
    
    print(f"\n🎯 Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(results['report'])
    print(f"\n🔢 Confusion Matrix:")
    print("              Predicted")
    print("           Imp  Neu  Pol")
    cm = results['confusion_matrix']
    print(f"Actual Imp {cm[0]}")
    print(f"       Neu {cm[1]}")
    print(f"       Pol {cm[2]}")
    
    # Save model
    print("\n💾 Saving model...")
    lstm_model.save_model('models/lstm_model.pkl')
    
    # Save results
    import json
    os.makedirs('results', exist_ok=True)
    
    results_summary = {
        'model': 'LSTM',
        'accuracy': float(results['accuracy']),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'epochs': 10,
        'embedding_dim': 100,
        'hidden_dim': 256
    }
    
    with open('results/lstm_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("✅ Results saved to results/lstm_results.json")
    
    # Quick test on custom examples
    print("\n" + "="*80)
    print("🧪 TESTING ON CUSTOM EXAMPLES")
    print("="*80)
    
    test_examples = [
        "Could you please help me with this? I would really appreciate it!",
        "This is completely wrong and stupid!",
        "I need help with this problem.",
        "Thank you so much for your assistance!",
        "Fix this immediately!"
    ]
    
    predictions = lstm_model.predict(test_examples)
    
    for text, pred in zip(test_examples, predictions):
        print(f"\n📝 Text: {text}")
        print(f"🎯 Prediction: {pred}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    # Compare with SVM baseline
    print("\n📊 COMPARISON WITH SVM BASELINE:")
    print(f"   SVM Accuracy:  68.43%")
    print(f"   LSTM Accuracy: {results['accuracy']*100:.2f}%")
    
    if results['accuracy'] > 0.6843:
        improvement = (results['accuracy'] - 0.6843) * 100
        print(f"   🎉 LSTM improved by {improvement:.2f} percentage points!")
    else:
        print(f"   📊 SVM still performs better (this can happen with small training)")
    
    return lstm_model, results

if __name__ == "__main__":
    model, results = main() 