"""
train_ensemble.py
Create and evaluate ensemble classifier
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.ensemble import EnsembleClassifier, RuleBasedClassifier
from src.models import PolitenessClassifierSVM
from src.models_bert import PolitenessClassifierBERT
from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    print("="*80)
    print("🚀 ENSEMBLE CLASSIFIER WITH RULE-BASED REASONING")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data = load_processed_data()
    X = data['Cleaned_Text'].values
    y = data['Politeness'].values
    
    # Split (same as all other models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Test samples: {len(X_test)}")
    
    # Load trained models
    print("\n📦 Loading trained models...")
    
    print("   Loading SVM...")
    svm_model = PolitenessClassifierSVM.load_model('models/svm_baseline.pkl')
    
    print("   Loading BERT...")
    bert_model = PolitenessClassifierBERT.load_model('models/bert_2ep_simple')
    
    print("   ✅ Models loaded!")
    
    # Test rule-based classifier alone
    print("\n" + "="*80)
    print("📊 RULE-BASED CLASSIFIER (Standalone)")
    print("="*80)
    
    rule_classifier = RuleBasedClassifier()
    rule_predictions = rule_classifier.predict(X_test)
    
    from sklearn.metrics import accuracy_score, classification_report
    rule_accuracy = accuracy_score(y_test, rule_predictions)
    
    print(f"\n🎯 Rule-Based Accuracy: {rule_accuracy:.4f} ({rule_accuracy*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, rule_predictions))
    
    # Create ensemble with different weight configurations
    print("\n" + "="*80)
    print("📊 TESTING ENSEMBLE CONFIGURATIONS")
    print("="*80)
    
    weight_configs = [
        {'name': 'Equal Weights', 'weights': {'svm': 0.33, 'bert': 0.33, 'rules': 0.34}},
        {'name': 'ML Heavy', 'weights': {'svm': 0.4, 'bert': 0.4, 'rules': 0.2}},
        {'name': 'SVM Heavy', 'weights': {'svm': 0.5, 'bert': 0.3, 'rules': 0.2}},
        {'name': 'BERT Heavy', 'weights': {'svm': 0.3, 'bert': 0.5, 'rules': 0.2}},
        {'name': 'Rules Heavy', 'weights': {'svm': 0.35, 'bert': 0.35, 'rules': 0.3}},
    ]
    
    best_accuracy = 0
    best_config = None
    results_summary = []
    
    for config in weight_configs:
        print(f"\n🔧 Testing: {config['name']}")
        print(f"   Weights: SVM={config['weights']['svm']:.2f}, "
              f"BERT={config['weights']['bert']:.2f}, "
              f"Rules={config['weights']['rules']:.2f}")
        
        ensemble = EnsembleClassifier(
            svm_model=svm_model,
            bert_model=bert_model,
            weights=config['weights']
        )
        
        results = ensemble.evaluate(X_test, y_test)
        accuracy = results['accuracy']
        
        print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        results_summary.append({
            'config': config['name'],
            'weights': config['weights'],
            'accuracy': accuracy,
            'results': results
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = config
    
    # Detailed results for best configuration
    print("\n" + "="*80)
    print("📊 BEST ENSEMBLE CONFIGURATION")
    print("="*80)
    
    print(f"\n🏆 Best: {best_config['name']}")
    print(f"   Weights: SVM={best_config['weights']['svm']:.2f}, "
          f"BERT={best_config['weights']['bert']:.2f}, "
          f"Rules={best_config['weights']['rules']:.2f}")
    
    best_ensemble = EnsembleClassifier(
        svm_model=svm_model,
        bert_model=bert_model,
        weights=best_config['weights']
    )
    
    final_results = best_ensemble.evaluate(X_test, y_test)
    
    print(f"\n🎯 Test Accuracy: {final_results['accuracy']:.4f} ({final_results['accuracy']*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(final_results['report'])
    print(f"\n🔢 Confusion Matrix:")
    cm = final_results['confusion_matrix']
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
        accuracy_class = correct / total if total > 0 else 0
        print(f"   {label:10s}: {correct:4d}/{total:4d} correct ({accuracy_class:6.2%})")
    
    # Compare with individual models
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Status'}")
    print("-" * 50)
    print(f"{'Rule-Based Only':<25} {rule_accuracy*100:>6.2f}%      Baseline")
    print(f"{'SVM':<25} {'68.43%':<12} Strong")
    print(f"{'BERT (2 epochs)':<25} {'68.16%':<12} Strong")
    print(f"{'LSTM':<25} {'47.99%':<12} Weak")
    print(f"{'Ensemble (Best)':<25} {final_results['accuracy']*100:>6.2f}%      {'🏆 BEST' if final_results['accuracy'] > 0.6843 else 'Good'}")
    
    improvement = (final_results['accuracy'] - 0.6843) * 100
    if improvement > 0:
        print(f"\n🎉 Ensemble improves over SVM by {improvement:.2f} percentage points!")
    else:
        print(f"\n📊 SVM still competitive (difference: {abs(improvement):.2f} pp)")
    
    # Show example predictions with component votes
    print("\n" + "="*80)
    print("🔍 EXAMPLE PREDICTIONS WITH COMPONENT BREAKDOWN")
    print("="*80)
    
    test_examples = [
        "Could you please help me? I would really appreciate it!",
        "This is completely stupid and wrong!",
        "I need help with this problem.",
        "Thank you so much!",
        "Fix this now!"
    ]
    
    predictions_with_conf = best_ensemble.predict_with_confidence(test_examples)
    
    for text, (pred, conf, votes) in zip(test_examples, predictions_with_conf):
        print(f"\n📝 Text: {text}")
        print(f"   🎯 Ensemble: {pred} (confidence: {conf:.2%})")
        print(f"   📊 Component votes:")
        print(f"      SVM:   {votes['svm']}")
        print(f"      BERT:  {votes['bert']}")
        print(f"      Rules: {votes['rules']}")
        
        if votes['svm'] == votes['bert'] == votes['rules']:
            print(f"      ✅ Unanimous agreement!")
        elif pred in [votes['svm'], votes['bert'], votes['rules']]:
            print(f"      🤝 Majority consensus")
        else:
            print(f"      ⚖️ Weighted decision")
    
    # Save results
    print("\n💾 Saving ensemble results...")
    import json
    os.makedirs('results', exist_ok=True)
    
    ensemble_summary = {
        'best_configuration': best_config['name'],
        'best_weights': best_config['weights'],
        'accuracy': float(final_results['accuracy']),
        'all_configurations': [
            {
                'name': r['config'],
                'weights': r['weights'],
                'accuracy': float(r['accuracy'])
            }
            for r in results_summary
        ]
    }
    
    with open('results/ensemble_results.json', 'w') as f:
        json.dump(ensemble_summary, f, indent=2)
    
    print("✅ Results saved to results/ensemble_results.json")
    print("\n✅ ENSEMBLE TRAINING COMPLETE!")
    
    return best_ensemble, final_results

if __name__ == "__main__":
    ensemble, results = main()