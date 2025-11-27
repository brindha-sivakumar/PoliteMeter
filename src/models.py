"""
models.py
Machine learning models for politeness classification
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


class PolitenessClassifierSVM:
    """SVM-based politeness classifier using TF-IDF features"""
    
    def __init__(self):
        """Initialize TF-IDF vectorizer and SVM classifier"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.model = SVC(
            kernel='linear',
            class_weight='balanced',
            random_state=42
        )
    
    def train(self, texts, labels):
        """
        Train the model
        
        Args:
            texts: array/list of text strings
            labels: array/list of labels
        
        Returns:
            self (for method chaining)
        """
        print(f"   Training on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        print(f"   Feature matrix shape: {X.shape}")
        self.model.fit(X, labels)
        print(f"   ✅ Model trained!")
        return self
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts: array/list of text strings
        
        Returns:
            predictions: array of predicted labels
        """
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, texts, true_labels):
        """
        Evaluate model performance
        
        Args:
            texts: array/list of text strings
            true_labels: array/list of true labels
        
        Returns:
            dict with accuracy, report, confusion_matrix
        """
        predictions = self.predict(texts)
        
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'report': classification_report(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'predictions': predictions
        }

    def save_model(self, filepath='models/svm_baseline.pkl'):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"💾 Model saved to {filepath}")

    @staticmethod
    def load_model(filepath='models/svm_baseline.pkl'):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded from {filepath}")
        return model