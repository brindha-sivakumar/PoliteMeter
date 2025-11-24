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
    model = PolitenessClassi