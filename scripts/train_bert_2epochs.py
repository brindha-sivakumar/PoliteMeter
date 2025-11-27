"""
train_bert_2epochs.py
Fine-tune BERT for exactly 2 epochs (known optimal)
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
    print("🚀 FINE-TUNING BERT (2 EPOCHS - OPTIMAL)")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data = load_processed_data()
    
    # Prepare data
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Create BERT classifier
    bert_model = PolitenessClassifierBERT(
        model_name='bert-base-uncased',
        max_length=128,
        learning_rate=2e-5
    )
    
    # Train for ONLY 2 epochs
    print("\n💡 Training for 2 epochs (optimal based on previous run)")
    
    bert_model.train(
        texts=X_train,
        labels=y_train,
        epochs=2,              # EXACTLY 2 epochs
        batch_size=16,
        validation_split=0.1
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("📊 FINAL EVALUATION")
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
    
    # Save
    bert_model.save_model('models/bert_2epochs')
    
    # Comparison
    print("\n" + "="*80)
    print("📊 FINAL MODEL COMPARISON")
    print("="*80)
    print(f"   SVM Baseline:      68.43%")
    print(f"   BERT (2 epochs):   {results['accuracy']*100:.2f}%")
    print(f"   BERT (4 epochs):   64.05%")
    print(f"   BERT (10 epochs):  63.05%")
    print(f"   LSTM (best):       47.99%")
    
    if results['accuracy'] > 0.6843:
        improvement = (results['accuracy'] - 0.6843) * 100
        print(f"\n   🎉 BERT beats SVM by {improvement:.2f} percentage points!")
    elif results['accuracy'] >= 0.68:
        diff = abs(results['accuracy'] - 0.6843) * 100
        print(f"\n   ✅ BERT matches SVM (within {diff:.2f} pp)")
    else:
        diff = (0.6843 - results['accuracy']) * 100
        print(f"\n   📊 SVM leads by {diff:.2f} pp, but BERT avoided overfitting")
    
    print("\n✅ TRAINING COMPLETE!")
    
    return bert_model, results

if __name__ == "__main__":
    model, results = main()