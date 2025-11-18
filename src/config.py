"""
config.py
Configuration settings for PoliteMeter project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Data files
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'politeness_corpus_processed.csv')

# Cornell ConvoKit URLs
CONVOKIT_BASE_URL = "https://zissou.infosci.cornell.edu/convokit/datasets/"
WIKI_CORPUS_ID = "wikipedia-politeness-corpus"
SE_CORPUS_ID = "stack-exchange-politeness-corpus"

# Politeness thresholds
IMPOLITE_THRESHOLD = 0.3
POLITE_THRESHOLD = 0.7

# Feature lists
POLITENESS_MARKERS = [
    'please',
    'thank',
    'could you',
    'would you',
    'sorry',
    'appreciate',
    'kindly'
]

IMPOLITENESS_INDICATORS = [
    '!',
    'fix',
    'change',
    'do',
    'stop'
]

# Model parameters (will be used in Phase 2)
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
MAX_LENGTH = 128

# Class labels
LABEL_MAP = {
    'Impolite': 0,
    'Neutral': 1,
    'Polite': 2
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
