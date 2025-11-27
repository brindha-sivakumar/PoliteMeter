"""
train_bert.py
Fine-tune BERT with early stopping
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
    print("🚀 FINE-TUNING BERT WITH EARLY STOPPING")
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
    
    # Split
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
        learning_rate=2e-5
    )
    
    # Fine-tune with early stopping
    print("\n💡 Training with early stopping (patience=2)")
    print("   Will stop if validation accuracy doesn't improve for 2 epochs")
    
    bert_model.train(
        texts=X_train,
        labels=y_train,
        epochs=10,             # Set high, but early stopping will kick in
        batch_size=16,
        validation_split=0.1   # 10% validation for early stopping
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
    
    # Per-class analysis
    print(f"\n📈 Per-Class Performance:")
    classes = ['Impolite', 'Neutral', 'Polite']
    for i, label in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        print(f"   {label:10s}: {correct:4d}/{total:4d} correct ({accuracy:6.2%})")
    
    # Save model
    print("\n💾 Saving model...")
    bert_model.save_model('models/bert_early_stop')
    
    # Save results
    import json
    os.makedirs('results', exist_ok=True)
    
    results_summary = {
        'model': 'BERT with Early Stopping',
        'accuracy': float(results['accuracy']),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'learning_rate': 2e-5,
        'early_stopping': True
    }
    
    with open('results/bert_early_stop_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("✅ Results saved to results/bert_early_stop_results.json")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80)
    print(f"   SVM Baseline:        68.43%")
    print(f"   LSTM (best):         47.99%")
    print(f"   BERT (no ES):        64.05%")
    print(f"   BERT (Early Stop):   {results['accuracy']*100:.2f}%")
    
    if results['accuracy'] > 0.6843:
        improvement = (results['accuracy'] - 0.6843) * 100
        print(f"\n   🎉 BERT with early stopping improved by {improvement:.2f} pp over SVM!")
    elif results['accuracy'] > 0.6405:
        improvement = (results['accuracy'] - 0.6405) * 100
        print(f"\n   ✅ Early stopping improved BERT by {improvement:.2f} percentage points!")
    else:
        print(f"\n   📊 SVM still leads, but early stopping prevented overfitting")
    
    print("\n✅ TRAINING COMPLETE!")
    
    return bert_model, results

if __name__ == "__main__":
    model, results = main()