"""
preprocessing.py
Text preprocessing and feature extraction for PoliteMeter
"""

import re
import pandas as pd
from . import config


def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text (str): Raw text to clean
    
    Returns:
        str: Cleaned text (lowercase, normalized whitespace)
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text


def count_politeness_features(text):
    """
    Count politeness markers in text
    
    Args:
        text (str): Text to analyze (should be cleaned first)
    
    Returns:
        dict: Dictionary with counts of each politeness feature
    """
    # Clean the text first
    text = clean_text(text)
    
    # Count each politeness marker
    features = {}
    for marker in config.POLITENESS_MARKERS:
        features[marker] = text.count(marker)
    
    # Add question mark count
    features['?'] = text.count('?')
    
    return features


def count_impoliteness_features(text):
    """
    Count impoliteness indicators in text
    
    Args:
        text (str): Text to analyze (should be cleaned first)
    
    Returns:
        dict: Dictionary with counts of impoliteness features
    """
    # Clean the text first
    text = clean_text(text)
    
    features = {
        'exclamation': text.count('!')
    }
    
    # Check for imperative verbs at start
    words = text.split()
    if len(words) > 0:
        first_word = words[0]
        imperative_verbs = ['fix', 'change', 'do', 'stop', 'delete', 'remove']
        features['imperative'] = 1 if first_word in imperative_verbs else 0
    else:
        features['imperative'] = 0
    
    return features


def categorize_politeness(score):
    """
    Convert continuous politeness score to category
    
    Args:
        score (float): Normalized politeness score (0-1)
    
    Returns:
        str: 'Polite', 'Neutral', or 'Impolite'
    """
    if score < config.IMPOLITE_THRESHOLD:
        return 'Impolite'
    elif score < config.POLITE_THRESHOLD:
        return 'Neutral'
    else:
        return 'Polite'


def preprocess_dataset(df):
    """
    Apply all preprocessing steps to a dataframe
    
    Args:
        df (pd.DataFrame): Raw dataframe with 'Request' and 'Normalized Score' columns
    
    Returns:
        pd.DataFrame: Processed dataframe with additional feature columns
    """
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Apply cleaning
    processed_df['Cleaned_Text'] = processed_df['Request'].apply(clean_text)
    
    # Extract features
    processed_df['Politeness_Features'] = processed_df['Cleaned_Text'].apply(
        count_politeness_features
    )
    processed_df['Impoliteness_Features'] = processed_df['Cleaned_Text'].apply(
        count_impoliteness_features
    )
    
    # Add category labels
    processed_df['Politeness'] = processed_df['Normalized Score'].apply(
        categorize_politeness
    )
    
    return processed_df


def extract_feature_vector(features_dict):
    """
    Convert feature dictionary to flat feature vector
    Useful for traditional ML models
    
    Args:
        features_dict (dict): Dictionary of features
    
    Returns:
        list: Flat list of feature values
    """
    return list(features_dict.values())
