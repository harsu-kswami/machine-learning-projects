"""Test suite for the positional encoding visualizer."""

import sys
import os
import warnings

# Add the src directory to the Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Test configuration
TEST_DEVICE = 'cpu'  # Use CPU for consistent testing
TEST_SEED = 42

# Common test parameters
TEST_CONFIGS = {
    'small': {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'max_seq_len': 32
    },
    'medium': {
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'max_seq_len': 64
    },
    'large': {
        'd_model': 256,
        'n_heads': 16,
        'n_layers': 6,
        'max_seq_len': 128
    }
}

# Test data
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world! This is a test sentence.",
    "Artificial intelligence and machine learning are transforming technology.",
    "Transformers use attention mechanisms to process sequences.",
    "PyTorch is a popular deep learning framework."
]

def set_test_seed(seed=TEST_SEED):
    """Set random seeds for reproducible testing."""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Set seed on import
set_test_seed()
