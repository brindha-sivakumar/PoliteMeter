# PoliteMeter: Automated Politeness Detection System

An AI-powered system that classifies text as polite, neutral, or impolite using a hybrid approach combining machine learning, deep learning, and transformer models with rule-based reasoning.

## 📊 Project Overview

**Goal:** Automatically detect and classify politeness levels in text for applications like:
- Online content moderation
- Customer service quality analysis
- Professional communication tools

**Approach:** Hybrid system combining:
- Traditional ML (SVM)
- Deep Learning (LSTM)
- Transformers (BERT)
- Rule-based linguistic features

---

## 📁 Dataset

**Stanford Politeness Corpus**
- **Source:** [ConvoKit - Cornell University](https://convokit.cornell.edu/)
- **Size:** 10,956 samples
  - Wikipedia Talk Pages: 4,353 requests
  - Stack Exchange: 6,603 posts
- **Labels:** Continuous politeness scores (0-1)
- **Categories:**
  - Impolite: 7,222 samples (65.9%)
  - Neutral: 2,227 samples (20.3%)
  - Polite: 1,507 samples (13.8%)

**Note:** Dataset is imbalanced - will need to address in modeling phase.

---

## 🎯 Progress Tracker

### ✅ Phase 1: Data Collection & Preprocessing (COMPLETED)

**Achievements:**
1. ✅ Successfully downloaded official Stanford Politeness Corpus (10,956 samples)
2. ✅ Implemented text cleaning pipeline
3. ✅ Engineered rule-based politeness features:
   - Politeness markers: "please", "thank", "could you", "would you", "sorry", "appreciate", "kindly"
   - Impoliteness markers: exclamation marks, imperative commands
4. ✅ Created categorical labels from continuous scores
5. ✅ Processed and saved complete dataset

**Key Files:**
- `src/preprocessing.py` - Text cleaning and feature extraction
- `src/data_loader.py` - Dataset download utilities
- `data/processed/politeness_corpus_processed.csv` - Processed dataset

### 🔄 Phase 2: Model Development (IN PROGRESS)
- [ ] Baseline: SVM with TF-IDF features
- [ ] Deep Learning: LSTM for sequence modeling
- [ ] Transformers: Fine-tune BERT for politeness classification
- [ ] Rule-based: Integrate linguistic features
- [ ] Ensemble: Combine all approaches

### 📋 Phase 3: Evaluation & Deployment (UPCOMING)
- [ ] Split data: 80% train / 20% test
- [ ] Evaluate with accuracy, precision, recall, F1-score
- [ ] Handle class imbalance (SMOTE, class weights)
- [ ] Case study: Real-world validation
- [ ] Implement feedback loop for adaptive learning

## 📊 Results

### SVM Baseline (Phase 2.1 - COMPLETED)

**Model:** Support Vector Machine with TF-IDF features
**Accuracy:** 68.43% on test set

**Performance by Class:**
- Impolite: Precision 0.72, Recall 0.89, F1 0.80
- Neutral: Precision 0.58, Recall 0.31, F1 0.40  
- Polite: Precision 0.61, Recall 0.48, F1 0.54

**Key Findings:**
- Model performs best on Impolite class (66% of data)
- Struggles with Neutral class (often confused with Impolite)
- Class imbalance affects performance

**Next Steps:**
- Implement SMOTE for class balancing
- Try LSTM for sequence understanding
- Fine-tune BERT for better context
---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/PoliteMeter.git
cd PoliteMeter

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Quick Start

### Download Dataset

```python
from src.data_loader import download_official_corpus

# Download Stanford Politeness Corpus
wiki_data, se_data = download_official_corpus()
print(f"Downloaded {len(wiki_data)} Wikipedia samples")
print(f"Downloaded {len(se_data)} Stack Exchange samples")
```

### Preprocess Text

```python
from src.preprocessing import clean_text, count_politeness_features
import pandas as pd

# Clean text
text = "Could you PLEASE help me?  Thank you!"
cleaned = clean_text(text)
print(f"Cleaned: {cleaned}")

# Extract features
features = count_politeness_features(cleaned)
print(f"Features: {features}")
```

### Load Processed Data

```python
from src.data_loader import load_processed_data

# Load pre-processed dataset
data = load_processed_data()
print(data.head())
```

---

## 📊 Project Structure

```
PoliteMeter/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory
│   ├── raw/                     # Original downloaded data
│   └── processed/               # Cleaned & featured data
│       └── politeness_corpus_processed.csv
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration settings
│   ├── data_loader.py           # Dataset utilities
│   └── preprocessing.py         # Text preprocessing
│
├── models/                      # Saved trained models
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_data_exploration.ipynb
│
├── results/                     # Outputs and visualizations
│
└── tests/                       # Unit tests
```

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocessing.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/01_data_exploration.ipynb
```

---

## 📈 Results (Coming Soon)

Model performance metrics will be added as models are trained.

---

## 👥 Contributing

This is a learning project. Suggestions and feedback are welcome!

---

## 📚 References

1. Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). 
   *A computational approach to politeness with application to social factors.* ACL 2013.
   
2. ConvoKit: https://convokit.cornell.edu/

3. Stanford Politeness Corpus: https://convokit.cornell.edu/documentation/politeness.html

---

## 📝 License

This project is for educational purposes. Dataset credit: Stanford NLP Group.

---

## 🙏 Acknowledgments

- Stanford NLP Group for the Politeness Corpus
- Cornell University for ConvoKit
- The open-source ML community
