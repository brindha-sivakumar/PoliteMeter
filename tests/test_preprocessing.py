"""
test_preprocessing.py
Unit tests for preprocessing functions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import (
    clean_text, 
    count_politeness_features,
    count_impoliteness_features,
    categorize_politeness
)


def test_clean_text():
    """Test text cleaning function"""
    # Test case 1: Basic cleaning
    text = "  Could You PLEASE   help me?  "
    result = clean_text(text)
    expected = "could you please help me?"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✅ test_clean_text: Basic cleaning passed")
    
    # Test case 2: Multiple spaces
    text = "Hello    world"
    result = clean_text(text)
    expected = "hello world"
    assert result == expected
    print("✅ test_clean_text: Multiple spaces passed")


def test_count_politeness_features():
    """Test politeness feature counting"""
    text = "Could you please help? Thank you!"
    features = count_politeness_features(text)
    
    assert features['please'] == 1, "Should find 'please' once"
    assert features['thank'] == 1, "Should find 'thank' once"
    assert features['could you'] == 1, "Should find 'could you' once"
    assert features['?'] == 1, "Should find one question mark"
    
    print("✅ test_count_politeness_features passed")


def test_count_impoliteness_features():
    """Test impoliteness feature counting"""
    text = "Fix this now!"
    features = count_impoliteness_features(text)
    
    assert features['exclamation'] == 1, "Should find one exclamation"
    assert features['imperative'] == 1, "Should detect imperative"
    
    print("✅ test_count_impoliteness_features passed")


def test_categorize_politeness():
    """Test politeness categorization"""
    assert categorize_politeness(0.1) == 'Impolite'
    assert categorize_politeness(0.5) == 'Neutral'
    assert categorize_politeness(0.9) == 'Polite'
    
    print("✅ test_categorize_politeness passed")


if __name__ == "__main__":
    print("\n🧪 Running tests...\n")
    test_clean_text()
    test_count_politeness_features()
    test_count_impoliteness_features()
    test_categorize_politeness()
    print("\n✅ All tests passed!\n")
