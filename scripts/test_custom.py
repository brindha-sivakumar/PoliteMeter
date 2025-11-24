"""
test_custom.py
Test the trained model on custom examples
"""

import sys
import os
sys.path.insert(0, '/content/PoliteMeter')
os.chdir('/content/PoliteMeter')

from src.models import PolitenessClassifierSVM
from src.utils import test_politeness, batch_test_examples

def main():
    print("="*80)
    print("🧪 TESTING SVM MODEL ON CUSTOM EXAMPLES")
    print("="*80)
    
    # Load model
    print("\n📦 Loading trained model...")
    try:
        model = PolitenessClassifierSVM.load_model('models/svm_baseline.pkl')
    except FileNotFoundError:
        print("❌ Model not found! Train the model first using train_svm.py")
        return
    
    # Define test examples
    examples = {
        'Polite': [
            "Could you please help me with this? I would really appreciate it!",
            "Thank you so much for your time and consideration.",
            "I apologize for any inconvenience this may have caused.",
            "Would you mind taking a look at this when you have a moment?",
            "I'm very grateful for your assistance on this matter."
        ],
        'Neutral': [
            "I need help with this problem.",
            "Can you explain how this works?",
            "I don't understand this concept.",
            "What's the solution to this?",
            "How do I fix this issue?"
        ],
        'Impolite': [
            "Fix this now!",
            "This is completely wrong and stupid.",
            "You have no idea what you're talking about.",
            "This is a waste of time.",
            "Why did you even bother posting this garbage?"
        ]
    }
    
    # Test examples
    batch_test_examples(model, examples)
    
    # Interactive testing
    print("\n" + "="*80)
    print("🎯 TRY YOUR OWN TEXT")
    print("="*80)
    print("\nTest a few custom sentences:")
    
    custom_tests = [
        "I really appreciate your help with this!",
        "This doesn't make any sense.",
        "Can you help me understand?",
        "Delete this immediately!",
        "Thank you for being so patient with me."
    ]
    
    for text in custom_tests:
        test_politeness(model, text)
    
    print("\n"+"="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()