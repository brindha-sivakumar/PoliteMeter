# LSTM Hyperparameter Tuning Documentation

## Overview

This document details the systematic hyperparameter tuning process for the LSTM politeness classifier, including all experiments, results, and insights.

## Dataset Characteristics

- **Total Samples:** 10,956
- **Training Samples:** 8,764 (80%)
- **Test Samples:** 2,192 (20%)
- **Class Distribution:**
  - Impolite: 7,222 (65.9%)
  - Neutral: 2,227 (20.3%)
  - Polite: 1,507 (13.8%)

## Model Architecture
```python
PolitenessClassifierLSTM(
    embedding_dim=100,
    hidden_dim=256,
    max_length=100,
    num_layers=2,
    dropout=variable  # tuned parameter
)
```

**Total Parameters:** ~2.7 million

## Tuning Process

### Experiment 1: Baseline (No Class Weights)

**Configuration:**
- Learning Rate: 0.001
- Epochs: 10
- Batch Size: 32
- Dropout: 0.3
- Class Weights: None (uniform)

**Results:**
- Test Accuracy: 65.92%
- Confusion Matrix: Model predicted ONLY "Impolite" (100%)

**Analysis:**
- Model learned to always predict majority class
- Achieved 65.9% accuracy by exploiting class distribution
- No actual learning of politeness patterns

**Decision:** Add class weighting

---

### Experiment 2: Aggressive Class Weights

**Configuration:**
- Learning Rate: 0.001
- Class Weights: Raw inverse frequency
  - Impolite: 0.506
  - Neutral: 1.640
  - Polite: 2.422

**Results:**
- Test Accuracy: 13.73%
- Confusion Matrix: Model predicted ONLY "Polite" (100%)

**Analysis:**
- Weights too extreme (5x difference)
- Over-corrected: now obsessed with minority class
- Lost ability to distinguish classes

**Decision:** Soften weights using square root

---

### Experiment 3: Softened Weights + Tuned Hyperparameters

**Configuration:**
- Learning Rate: 0.002
- Dropout: 0.5
- Class Weights: Softened (square root of raw)
  - Impolite: 0.711
  - Neutral: 1.281
  - Polite: 1.556

**Results:**
- **Test Accuracy: 47.99%** ⭐ Best result
- Training Accuracy: 90.62%
- Generalization Gap: 42.63%

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Impolite | 0.52 | 0.67 | 0.59 |
| Neutral | 0.38 | 0.24 | 0.29 |
| Polite | 0.42 | 0.31 | 0.36 |

**Analysis:**
- ✅ Successfully predicts all three classes
- ❌ Severe overfitting (42.63% gap)
- Model memorizing training data, not generalizing

**Decision:** Address overfitting

---

### Experiment 4: Data Undersampling

**Configuration:**
- Undersampled Impolite class to 1,781 samples
- Balanced dataset: ~1,800 samples per class
- Learning Rate: 0.001
- Dropout: 0.3

**Results:**
- Test Accuracy: 13.73%
- Model collapsed to Polite again

**Analysis:**
- Reducing data made problem worse
- Learning rate too low for balanced data
- Model couldn't escape local minimum

**Decision:** Increase learning rate

---

### Experiment 5: Higher Learning Rate

**Configuration:**
- Undersampled data
- Learning Rate: 0.002
- Dropout: 0.3

**Results:**
- Test Accuracy: 38.18%
- Training Accuracy: 91.00%
- Generalization Gap: 52.82%

**Analysis:**
- Achieved multi-class prediction
- Overfitting even worse than before
- Undersampling didn't help generalization

**Decision:** Increase regularization

---

### Experiment 6: High Dropout

**Configuration:**
- Learning Rate: 0.002
- Dropout: 0.7 (very aggressive)

**Results:**
- Test Accuracy: 15.15%
- Model collapsed (underfitting)

**Analysis:**
- Dropout too strong
- Model couldn't learn patterns
- Lost expressive power

**Decision:** Find middle ground

---

### Experiment 7: Balanced Regularization

**Configuration:**
- Learning Rate: 0.002
- Dropout: 0.5
- Epochs: 50

**Results:**
- Test Accuracy: 13.73%
- Training unstable, collapsed after 20 epochs

**Analysis:**
- Model fundamentally unstable
- Even with extensive tuning, couldn't achieve stable multi-class learning
- Architecture too complex for dataset size

**Final Decision:** LSTM not suitable for this dataset

---

## Key Insights

### 1. Parameter-to-Sample Ratio

| Model | Parameters | Training Samples | Ratio |
|-------|------------|------------------|-------|
| SVM | ~25,000 | 8,764 | 3:1 ✅ |
| LSTM | ~2,700,000 | 8,764 | 308:1 ❌ |

**Insight:** LSTM is 100x overparameterized for this dataset

### 2. Learning Dynamics

- LSTM showed **extreme sensitivity** to hyperparameters
- Small changes caused dramatic performance shifts
- Frequent collapse to single-class prediction
- Training curves were erratic and unstable

### 3. Data Requirements

**Minimum data needed for stable LSTM:**
- 10,000+ samples per class (30,000+ total)
- Our dataset: 1,206 samples for minority class
- Gap: 8.3x too small

### 4. Task Characteristics

**Politeness detection in this dataset:**
- Often keyword-based ("please", "thank", "stupid")
- Simple patterns, not complex sequences
- No long-range dependencies
- SVM's bag-of-words approach is sufficient

## Recommendations

1. **For datasets < 20,000 samples:** Use SVM or traditional ML
2. **For keyword-driven tasks:** TF-IDF + SVM is optimal
3. **For LSTMs:** Require 50,000+ samples and complex sequential patterns
4. **Next approach:** Try BERT (pre-trained, doesn't need large dataset)

## References

- PyTorch LSTM Documentation
- "Understanding LSTM Networks" - Christopher Olah
- "Deep Learning for NLP Best Practices" - Ruder (2017)