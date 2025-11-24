"""
models.py
Machine learning models for politeness classification
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np


class PolitenessClassifierSVM:
    """SVM-based politeness classifier using TF-IDF features"""
    
    def __init__(self):
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)

        self.model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    
    def train(self, texts, labels):
        
        print(f"   Training on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        print(f"   Feature matrix shape: {X.shape}")
        self.model.fit(X, labels)
        print(f"   ✅ Model trained!")
        return self
    
    def predict(self, texts):
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, texts, true_labels):
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }