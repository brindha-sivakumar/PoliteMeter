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