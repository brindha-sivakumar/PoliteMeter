from src.models import PolitenessClassifierSVM
from src.data_loader import load_processed_data
from sklearn.model_selection import train_test_split

# Load data
data = load_processed_data()
X = data['Cleaned_Text'].values
y = data['Politeness'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = PolitenessClassifierSVM()
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test[:5])  # Test on first 5
print(f"Predictions: {predictions}")
print(f"Actual: {y_test[:5]}")

# Full evaluation on all test data
print("\n" + "="*80)
print("📊 FULL TEST SET EVALUATION")
print("="*80)

results = model.evaluate(X_test, y_test)

print(f"\n🎯 Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

print(f"\n📋 Detailed Classification Report:")
print(results['report'])

print(f"\n🔢 Confusion Matrix:")
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Impolite', 'Neutral', 'Polite'],
            yticklabels=['Impolite', 'Neutral', 'Polite'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('SVM Baseline - Confusion Matrix')
plt.show()

# Calculate per-class accuracy
print("\n📈 Per-Class Analysis:")
cm = results['confusion_matrix']
for i, label in enumerate(['Impolite', 'Neutral', 'Polite']):
    class_total = cm[i].sum()
    class_correct = cm[i, i]
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"   {label:10s}: {class_correct:4d}/{class_total:4d} correct ({class_accuracy:.2%})")