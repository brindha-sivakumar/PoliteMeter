"""
ensemble.py
Ensemble classifier combining SVM, BERT, and rule-based reasoning
"""

import numpy as np
from collections import Counter
from src.preprocessing import clean_text, count_politeness_features, count_impoliteness_features


class RuleBasedClassifier:
    """
    Rule-based politeness classifier using linguistic features
    """
    
    def __init__(self):
        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
        self.reverse_label_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
    
    def _calculate_politeness_score(self, text):
        """
        Calculate politeness score based on linguistic features
        
        Returns:
            tuple: (politeness_score, confidence)
                politeness_score: -1 to 1 (-1=impolite, 0=neutral, 1=polite)
                confidence: 0 to 1 (how confident we are)
        """
        cleaned = clean_text(text)
        polite_features = count_politeness_features(cleaned)
        impolite_features = count_impoliteness_features(cleaned)
        
        # Count positive indicators
        polite_count = sum([
            polite_features.get('please', 0),
            polite_features.get('thank', 0),
            polite_features.get('could you', 0),
            polite_features.get('would you', 0),
            polite_features.get('sorry', 0),
            polite_features.get('appreciate', 0),
            polite_features.get('kindly', 0)
        ])
        
        # Questions are somewhat polite
        question_count = polite_features.get('?', 0)
        
        # Count negative indicators
        impolite_count = (
            impolite_features.get('exclamation', 0) +
            impolite_features.get('imperative', 0) * 2  # Weight imperatives more
        )
        
        # Calculate raw score
        raw_score = polite_count + (question_count * 0.5) - impolite_count
        
        # Normalize to -1 to 1 range
        if raw_score > 0:
            score = min(raw_score / 3.0, 1.0)  # Cap at 1.0
        elif raw_score < 0:
            score = max(raw_score / 3.0, -1.0)  # Cap at -1.0
        else:
            score = 0.0
        
        # Calculate confidence (how many features we found)
        total_features = polite_count + impolite_count + question_count
        confidence = min(total_features / 3.0, 1.0)  # Cap at 1.0
        
        return score, confidence
    
    def predict_proba(self, texts):
        """
        Predict class probabilities for texts
        
        Returns:
            np.array: Shape (n_samples, 3) with probabilities for [Impolite, Neutral, Polite]
        """
        probabilities = []
        
        for text in texts:
            score, confidence = self._calculate_politeness_score(text)
            
            # Convert score to probabilities
            if score > 0.3:  # Polite
                # High polite score = high polite probability
                polite_prob = 0.4 + (score * 0.6)  # 0.4 to 1.0
                neutral_prob = 0.3 - (score * 0.2)
                impolite_prob = 0.3 - (score * 0.4)
            elif score < -0.3:  # Impolite
                # High impolite score = high impolite probability
                impolite_prob = 0.4 + (abs(score) * 0.6)  # 0.4 to 1.0
                neutral_prob = 0.3 - (abs(score) * 0.2)
                polite_prob = 0.3 - (abs(score) * 0.4)
            else:  # Neutral
                # Low score = high neutral probability
                neutral_prob = 0.6
                polite_prob = 0.2 + (score * 0.2)
                impolite_prob = 0.2 - (score * 0.2)
            
            # Ensure probabilities are valid
            probs = np.array([
                max(0, impolite_prob),
                max(0, neutral_prob),
                max(0, polite_prob)
            ])
            
            # Normalize to sum to 1
            probs = probs / probs.sum()
            
            # Apply confidence weighting
            # Low confidence = more uniform distribution
            if confidence < 0.5:
                uniform = np.array([0.33, 0.34, 0.33])
                probs = probs * confidence + uniform * (1 - confidence)
            
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def predict(self, texts):
        """Predict class labels"""
        probas = self.predict_proba(texts)
        predictions = np.argmax(probas, axis=1)
        return [self.reverse_label_map[p] for p in predictions]


class EnsembleClassifier:
    """
    Ensemble classifier combining SVM, BERT, and rule-based reasoning
    """
    
    def __init__(self, svm_model=None, bert_model=None, 
                 weights={'svm': 0.4, 'bert': 0.4, 'rules': 0.2}):
        """
        Args:
            svm_model: Trained SVM model (PolitenessClassifierSVM)
            bert_model: Trained BERT model (PolitenessClassifierBERT)
            weights: Dictionary with weights for each component (must sum to 1.0)
        """
        self.svm_model = svm_model
        self.bert_model = bert_model
        self.rule_classifier = RuleBasedClassifier()
        
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights = weights
        
        self.label_map = {'Impolite': 0, 'Neutral': 1, 'Polite': 2}
        self.reverse_label_map = {0: 'Impolite', 1: 'Neutral', 2: 'Polite'}
    
    def predict_proba(self, texts):
        """
        Get ensemble probability predictions
        
        Returns:
            np.array: Shape (n_samples, 3) with probabilities for [Impolite, Neutral, Polite]
        """
        n_samples = len(texts)
        ensemble_probs = np.zeros((n_samples, 3))
        
        # Get predictions from each component
        print("   🔮 Getting SVM predictions...")
        if self.svm_model is not None:
            svm_probs = self._get_svm_probabilities(texts)
            ensemble_probs += svm_probs * self.weights['svm']
        
        print("   🔮 Getting BERT predictions...")
        if self.bert_model is not None:
            bert_probs = self._get_bert_probabilities(texts)
            ensemble_probs += bert_probs * self.weights['bert']
        
        print("   🔮 Getting rule-based predictions...")
        rule_probs = self.rule_classifier.predict_proba(texts)
        ensemble_probs += rule_probs * self.weights['rules']
        
        return ensemble_probs
    
    def _get_svm_probabilities(self, texts):
        """Get probability estimates from SVM"""
        # SVM doesn't naturally give probabilities, so we'll use decision function
        # and convert to probabilities
        try:
            # If SVM has predict_proba (with probability=True)
            if hasattr(self.svm_model.model, 'predict_proba'):
                probs = self.svm_model.model.predict_proba(
                    self.svm_model.vectorizer.transform(texts)
                )
                return probs
        except:
            pass
        
        # Fallback: Convert predictions to one-hot probabilities
        predictions = self.svm_model.predict(texts)
        probs = np.zeros((len(texts), 3))
        for i, pred in enumerate(predictions):
            probs[i, self.label_map[pred]] = 1.0
        return probs
    
    def _get_bert_probabilities(self, texts):
        """Get probabilities from BERT"""
        import torch
        from torch.nn.functional import softmax
        
        self.bert_model.model.eval()
        
        # Create dataset
        dummy_labels = ['Neutral'] * len(texts)
        from src.models_bert import PolitenessDatasetBERT
        from torch.utils.data import DataLoader
        
        dataset = PolitenessDatasetBERT(
            texts, dummy_labels, 
            self.bert_model.tokenizer, 
            self.bert_model.max_length
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.bert_model.device)
                attention_mask = batch['attention_mask'].to(self.bert_model.device)
                
                outputs = self.bert_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get probabilities using softmax
                probs = softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def predict(self, texts):
        """Predict class labels using ensemble"""
        probs = self.predict_proba(texts)
        predictions = np.argmax(probs, axis=1)
        return [self.reverse_label_map[p] for p in predictions]
    
    def predict_with_confidence(self, texts):
        """
        Predict with confidence scores
        
        Returns:
            list of tuples: (prediction, confidence, component_votes)
        """
        probs = self.predict_proba(texts)
        
        results = []
        for i, prob in enumerate(probs):
            prediction_idx = np.argmax(prob)
            prediction = self.reverse_label_map[prediction_idx]
            confidence = prob[prediction_idx]
            
            # Get individual component predictions for transparency
            component_votes = {
                'svm': self.svm_model.predict([texts[i]])[0] if self.svm_model else None,
                'bert': self.bert_model.predict([texts[i]])[0] if self.bert_model else None,
                'rules': self.rule_classifier.predict([texts[i]])[0]
            }
            
            results.append((prediction, confidence, component_votes))
        
        return results
    
    def evaluate(self, texts, true_labels):
        """Evaluate ensemble performance"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(
            true_labels,
            predictions,
            labels=['Impolite', 'Neutral', 'Polite']
        )
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }