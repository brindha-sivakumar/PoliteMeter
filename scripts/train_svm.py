"""
train_svm.py
Train SVM baseline model on politeness dataset
"""

import sys
import os

# Setup paths
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.models import PolitenessClassifierSVM
from src.data_loader import load_processed_data, download_official_corpus, save_processed_data
from src.preprocessing import preprocess_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    print("="*80)
    print("🚀 TRAINING SVM BASELINE MODEL")
    print("="*80)
    
    # Load or download data
    print("\n📊 Loading data...")
    try:
        data = load_processed_data()
        print(f"✅ Loaded {len(data)} samples from disk")
    except FileNotFoundError:
        print("⚠️  Processed data not found. Downloading...")
        wiki_data, se_data = download_official_corpus()
        combined = pd.concat([wiki_data, se_data], ignore_index=True)
        data = preprocess_dataset(combined)
        save_processed_data(data)
        print(f"✅ Downloaded and processed {len(data)} samples")
    
    # Show distribution
    print(f"\n📊 Class Distribution:")
    print(data['Politeness'].value_counts())
    
    # Split data
    print("\n✂️  Splitting data (80/20 train/test)...")
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n🔄 Training SVM classifier...")
    print("-"*80)
    model = PolitenessClassifierSVM()
    model.train(X_train, y_train)
    
    # Quick evaluation
    print("\n📊 Quick Evaluation on Test Set...")
    results = model.evaluate(X_test, y_test)
    
    print(f"\n🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(results['report'])
    
    # Save model
    print("\n💾 Saving model...")
    model.save_model('models/svm_baseline.pkl')
    
    # Save results
    import json
    os.makedirs('results', exist_ok=True)
    
    results_summary = {
        'model': 'SVM Baseline',
        'accuracy': float(results['accuracy']),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'class_distribution': data['Politeness'].value_counts().to_dict()
    }
    
    with open('results/svm_baseline_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("✅ Results saved to results/svm_baseline_results.json")
    
    print("\n"+"="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    return model, results

if __name__ == "__main__":
    try:
        model, results = main()
    except FileNotFoundError:
        print("⚠️  Processed data not found. Downloading all raw corpora...")
        
        # 1. Download official corpus (returns wiki_data, se_data)
        wiki_data, se_data = download_official_corpus()
        
        # 2. Download the new corpus
        from src.data_loader import download_new_politeness_data # Make sure to import this!
        new_data = download_new_politeness_data()
        
        # 3. Combine all three DataFrames (CRITICAL STEP)
        combined = pd.concat([wiki_data, se_data, new_data], ignore_index=True)
        
        # 4. Preprocess and save
        data = preprocess_dataset(combined)