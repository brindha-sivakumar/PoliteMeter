"""
evaluate_bert.py
Detailed evaluation of trained BERT model
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import numpy as np

# Utility function for tokenizing test data (copied from train_bert_simple.py)
def tokenize_data(texts, labels, tokenizer, label_map):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
    labels_numeric = [label_map[l] for l in labels]
    
    dataset = []
    for i in range(len(texts)):
        dataset.append({
            'input_ids': torch.tensor(encodings['input_ids'][i]),
            'attention_mask': torch.tensor(encodings['attention_mask'][i]),
            'labels': torch.tensor(labels_numeric[i])
        })
    return dataset

def main():
    print("="*80)
    print("📊 EVALUATING BERT MODEL (2-epoch simple)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'models/bert_2ep_simple'
    
    # Load model and tokenizer
    print("\n📦 Loading trained model and tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
        model.to(device)
    except OSError:
        print("❌ BERT Model or tokenizer not found! Train the model first using train_bert_simple.py")
        return
    
    # Load data
    print("📊 Loading test data...")
    data = load_processed_data()
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tokenize test data
    label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
    test_dataset = tokenize_data(X_test, y_test, tokenizer, label_map)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Evaluate
    print("\n🔍 Running evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert back to labels
    reverse_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
    pred_labels = [reverse_map[p] for p in all_preds]
    true_labels = [reverse_map[l] for l in all_labels]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=['Impolite', 'Neutral', 'Polite'])
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS: ACCURACY, PRECISION, RECALL, F1-SCORE")
    print(f"{'='*80}")
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📋 Detailed Classification Report (Contains Precision, Recall, F1-Score per class):")
    print(report)
    
    # Per-class analysis
    print(f"\n📈 Per-Class Performance:")
    classes = ['Impolite', 'Neutral', 'Polite']
    
    for i, label in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy_class = correct / total if total > 0 else 0
        print(f"   {label:10s}: {correct:4d}/{total:4d} correct ({accuracy_class:6.2%})")

    print("\n"+"="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()