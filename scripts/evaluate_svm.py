"""
evaluate.py
Detailed evaluation of trained SVM model
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.models import PolitenessClassifierSVM
from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    print("="*80)
    print("📊 EVALUATING SVM BASELINE MODEL")
    print("="*80)
    
    # Load model
    print("\n📦 Loading trained model...")
    try:
        model = PolitenessClassifierSVM.load_model('models/svm_baseline.pkl')
    except FileNotFoundError:
        print("❌ Model not found! Train the model first using train_svm.py")
        return
    
    # Load data
    print("📊 Loading test data...")
    data = load_processed_data()
    
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Same split as training (important!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Evaluate
    print("\n🔍 Running evaluation...")
    results = model.evaluate(X_test, y_test)
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS: ACCURACY, PRECISION, RECALL, F1-SCORE")
    print(f"{'='*80}")
    print(f"\n🎯 Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\n📋 Detailed Classification Report:")
    print(results['report'])
    
    # Per-class analysis
    print(f"\n📈 Per-Class Performance:")
    cm = results['confusion_matrix']
    classes = ['Impolite', 'Neutral', 'Polite']
    
    for i, label in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        print(f"   {label:10s}: {correct:4d}/{total:4d} correct ({accuracy:6.2%})")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=axes[0])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title('Confusion Matrix')
    
    # Per-class accuracy
    class_accs = [cm[i,i]/cm[i].sum() for i in range(3)]
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    axes[1].bar(classes, class_accs, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Class Accuracy')
    axes[1].set_ylim([0, 1])
    
    for i, v in enumerate(class_accs):
        axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/svm_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to results/svm_evaluation.png")
    plt.show()
    
    # Sample predictions
    print("\n🔮 Sample Predictions (First 10):")
    print("-"*80)
    
    for i in range(10):
        text = X_test[i][:60] + "..." if len(X_test[i]) > 60 else X_test[i]
        pred = model.predict([X_test[i]])[0]
        actual = y_test[i]
        match = "✅" if pred == actual else "❌"
        print(f"{match} {pred:10s} | {actual:10s} | {text}")
    
    print("\n"+"="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()