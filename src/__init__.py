"""Positional Encoding Visualizer - Educational Transformer Implementation.

This package provides interactive tools for understanding transformer architectures
and positional encoding mechanisms through hands-on visualization.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import (
    TransformerEncoder,
    MultiHeadAttention,
    FeedForward,
    TokenEmbedding,
    LayerNorm
)

from .positional_encoding import (
    SinusoidalEncoding,
    RelativePositionalEncoding,
    RoPEEncoding
)

from .visualization import (
    AttentionVisualizer,
    PositionalEncodingVisualizer,
    InteractiveDashboard
)

__all__ = [
    # Models
    "TransformerEncoder",
    "MultiHeadAttention", 
    "FeedForward",
    "TokenEmbedding",
    "LayerNorm",
    
    # Positional Encodings
    "SinusoidalEncoding",
    "RelativePositionalEncoding", 
    "RoPEEncoding",
    
    # Visualization
    "AttentionVisualizer",
    "PositionalEncodingVisualizer",
    "InteractiveDashboard",
]
