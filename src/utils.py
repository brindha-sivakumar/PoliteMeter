
"""
utils.py
Utility functions for testing and evaluation
"""

from src.preprocessing import clean_text


def test_politeness(model, text, verbose=True):
    
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    
    if verbose:
        print(f"Text:       {text}")
        print(f"Prediction: {prediction}")
        print("-" * 60)
    
    return prediction


def batch_test_examples(model, examples_dict):
    
    print("\n🧪 Testing Custom Examples:")
    print("="*80)
    
    for expected_label, examples in examples_dict.items():
        print(f"\n🎯 EXPECTED: {expected_label.upper()}")
        for text in examples:
            test_politeness(model, text)