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
        
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
    
    def predict(self, texts):
        
        X = self.vectorizer.transform(texts)
        self.model.predict(X)
    
    def evaluate(self, texts, true_labels):
        """
        YOUR CHALLENGE: Evaluate model performance
        
        Args:
            texts: array/list of text strings
            true_labels: array/list of true labels
        
        TODO:
        1. Get predictions
        2. Calculate accuracy
        3. Get detailed classification report
        4. Generate confusion matrix
        
        Returns:
            dict with 'accuracy', 'report', 'confusion_matrix'
        """
        # YOUR CODE HERE
        pass