"""
evaluate_ensemble.py
Detailed evaluation of trained Ensemble classifier
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.ensemble import EnsembleClassifier # Assumes RuleBasedClassifier is implicitly handled
from src.models import PolitenessClassifierSVM
from src.models_bert import PolitenessClassifierBERT # Assumes this exists for loading
from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*80)
    print("📊 EVALUATING ENSEMBLE CLASSIFIER")
    print("="*80)
    
    # Load configuration from best training run (from train_ensemble.py)
    try:
        with open('results/ensemble_results.json', 'r') as f:
            ensemble_summary = json.load(f)
        best_weights = ensemble_summary['best_weights']
        print(f"\nLoaded best weights: SVM={best_weights['svm']:.2f}, BERT={best_weights['bert']:.2f}, Rules={best_weights['rules']:.2f}")
    except FileNotFoundError:
        print("❌ Ensemble results not found! Train the model first using train_ensemble.py")
        return

    # Load trained models
    print("\n📦 Loading trained component models...")
    try:
        svm_model = PolitenessClassifierSVM.load_model('models/svm_baseline.pkl')
        # PolitenessClassifierBERT.load_model is assumed to handle the BERT model loading
        bert_model = PolitenessClassifierBERT.load_model('models/bert_2ep_simple') 
    except (FileNotFoundError, AttributeError) as e:
        print(f"❌ One or more component models not found: {e}. Ensure SVM and BERT models are trained.")
        return

    # Initialize Ensemble
    ensemble = EnsembleClassifier(
        svm_model=svm_model,
        bert_model=bert_model,
        weights=best_weights
    )

    # Load data
    print("📊 Loading test data...")
    data = load_processed_data()
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Evaluate
    print("\n🔍 Running evaluation...")
    results = ensemble.evaluate(X_test, y_test)
    cm = results['confusion_matrix']
    
    # --- VISUALIZATION BLOCK ---
    print("\n🎨 Generating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = ['Impolite', 'Neutral', 'Polite']

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes,
                ax=axes[0])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title('Ensemble Confusion Matrix')

    # Per-class accuracy
    # Normalize by row to get per-class accuracy
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_accs = [cm_normalized[i, i] for i in range(3)]
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    axes[1].bar(classes, class_accs, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Ensemble Per-Class Accuracy')
    axes[1].set_ylim([0, 1])

    for i, v in enumerate(class_accs):
        axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ensemble_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to results/ensemble_evaluation.png")
    # ---------------------------

    # Display results
    print(f"\n{'='*80}")
    print("RESULTS: ACCURACY, PRECISION, RECALL, F1-SCORE")
    print(f"{'='*80}")
    print(f"\n🎯 Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\n📋 Detailed Classification Report (Contains Precision, Recall, F1-Score per class):")
    print(results['report'])
    
    # Per-class analysis
    print(f"\n📈 Per-Class Performance:")
    
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