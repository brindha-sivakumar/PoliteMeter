"""
train_bert.py
Fine-tune BERT on Stanford Politeness Corpus
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.models_bert import PolitenessClassifierBERT
from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    print("="*80)
    print("🚀 FINE-TUNING BERT ON POLITENESS DATASET")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data = load_processed_data()
    print(f"✅ Loaded {len(data)} samples")
    
    print(f"\n📊 Class Distribution:")
    print(data['Politeness'].value_counts())
    
    # Prepare data
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Split (same as SVM and LSTM for fair comparison)
    print("\n✂️  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create BERT classifier
    bert_model = PolitenessClassifierBERT(
        model_name='bert-base-uncased',
        max_length=128,
        learning_rate=2e-5  # Standard BERT learning rate
    )
    
    # Fine-tune
    bert_model.train(
        texts=X_train,
        labels=y_train,
        epochs=4,              # 3-5 epochs is typical for BERT
        batch_size=16,         # 16 or 32 for BERT
        validation_split=0.1   # Use 10% for validation
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("📊 EVALUATING ON TEST SET")
    print("="*80)
    
    results = bert_model.evaluate(X_test, y_test)
    
    print(f"\n🎯 Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(results['report'])
    print(f"\n🔢 Confusion Matrix:")
    cm = results['confusion_matrix']
    print("              Predicted")
    print("           Imp  Neu  Pol")
    print(f"Actual Imp {cm[0]}")
    print(f"       Neu {cm[1]}")
    print(f"       Pol {cm[2]}")
    
    # Save model
    print("\n💾 Saving model...")
    bert_model.save_model('models/bert_model')
    
    # Comparison
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80)
    print(f"   SVM Baseline:  68.43%")
    print(f"   LSTM:          47.99% (best)")
    print(f"   BERT:          {results['accuracy']*100:.2f}%")
    
    if results['accuracy'] > 0.6843:
        improvement = (results['accuracy'] - 0.6843) * 100
        print(f"\n   🎉 BERT improved by {improvement:.2f} percentage points over SVM!")
    
    print("\n✅ TRAINING COMPLETE!")
    
    return bert_model, results

if __name__ == "__main__":
    model, results = main()