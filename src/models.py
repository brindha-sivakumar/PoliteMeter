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
import re
from nltk.corpus import stopwords
import string
from scipy.sparse import hstack

class PolitenessClassifierSVM:
    """SVM-based politeness classifier using TF-IDF features AUGMENTED with Rule-Based Score"""
    
    def __init__(self):
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        # Instantiate the new rule scorer
        self.rule_scorer = PolitenessRuleScorer() 
        self.model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    
    def _augment_features(self, texts, tfidf_matrix):
        """Calculates rule scores and horizontally stacks them onto the TF-IDF matrix."""
        
        # Calculate the rule scores for the texts
        rule_scores = self.rule_scorer.calculate_scores(texts)
        
        # Reshape the 1D numpy array into a 2D column vector
        rule_scores_column = rule_scores.reshape(-1, 1)
        
        # Horizontally stack the TF-IDF matrix (sparse) and the rule score column (dense)
        # This returns a sparse matrix where the rule score is the final column
        augmented_matrix = hstack([tfidf_matrix, rule_scores_column])
        
        return augmented_matrix
    
    def train(self, texts, labels):
        
        print(f"   Training on {len(texts)} samples...")
        # 1. Fit and transform TF-IDF
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        # 2. Augment features with rule scores
        X_augmented = self._augment_features(texts, X_tfidf)
        
        print(f"   Feature matrix shape: {X_augmented.shape} (TF-IDF + 1 Rule Feature)")
        self.model.fit(X_augmented, labels)
        print(f"   ✅ Model trained!")
        return self
    
    def predict(self, texts):
        
        # 1. Transform texts using the *fitted* vectorizer
        X_tfidf = self.vectorizer.transform(texts)
        
        # 2. Augment features (using the same rule scorer)
        X_augmented = self._augment_features(texts, X_tfidf)
        
        predictions = self.model.predict(X_augmented)
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
    
    def save_model(self, filepath='models/svm_baseline.pkl'):
        
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"💾 Model saved to {filepath}")

    @staticmethod
    def load_model(filepath='models/svm_baseline.pkl'):
        import pickle
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded from {filepath}")
        return model
    

STOP_WORDS = set(stopwords.words('english'))

class PolitenessRuleScorer:
    """
    Calculates a numerical score based on the presence of politeness/impoliteness lexicons.
    """
    
    # --- Define Lexicons ---
    # Politeness markers (words/phrases that strongly imply politeness)
    POLITE_LEXICON = {
        'please', 'thank you', 'thanks', 'appreciate', 
        'could you', 'would you mind', 'apologies', 'sorry',
        'kindly', 'respectfully', 'best regards', 'thank'
    }

    # Impoliteness markers (words/phrases that strongly imply impoliteness)
    IMPOLITE_LEXICON = {
        'stupid', 'wrong', 'garbage', 'must', 'immediately', 
        'fix this', 'demand', 'ridiculous', 'terrible', 'idiot',
        'never', 'useless', 'do it', 'i want'
    }

    def __init__(self):
        # Create a combined set of all keywords for faster lookup
        self.all_keywords = self.POLITE_LEXICON.union(self.IMPOLITE_LEXICON)

    def _clean_text(self, text):
        """Standard cleaning: lowercase, remove punctuation."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def score_text(self, text):
        """
        Calculates the Rule-Based Score for a given text.
        Score > 0 implies Polite. Score < 0 implies Impolite.
        """
        score = 0
        words = self._clean_text(text)
        
        # 1. Keyword Scoring
        for word in words:
            if word in self.POLITE_LEXICON:
                score += 1.0
            elif word in self.IMPOLITE_LEXICON:
                score -= 1.0

        # 2. Heuristic Bonus/Penalty (for strong signals)
        # Check for excessive capitalization (strong impoliteness marker)
        # We check the original text before lowercasing
        if any(word.isupper() and len(word) > 2 for word in text.split()):
            score -= 1.5 # Strong penalty for all caps
            
        # Check for multiple question/exclamation marks
        if re.search(r'[!?]{2,}', text):
            score -= 0.5 # Penalty for strong emotion/demand

        # Check for polite salutation at the start (strong politeness marker)
        if words and words[0] in ['please', 'could', 'would', 'thank']:
            score += 0.5

        return score

    def calculate_scores(self, texts):
        """Applies scoring to a list of texts."""
        return np.array([self.score_text(text) for text in texts])
