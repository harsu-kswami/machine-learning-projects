"""Model configuration settings for transformer components."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch


@dataclass
class ModelConfig:
    """Configuration class for transformer model parameters."""
    
    # Model Architecture
    d_model: int = 512  # Model dimension
    n_heads: int = 8    # Number of attention heads
    n_layers: int = 6   # Number of encoder layers
    d_ff: int = 2048    # Feed-forward dimension
    
    # Sequence Parameters
    max_seq_len: int = 512      # Maximum sequence length
    vocab_size: int = 10000     # Vocabulary size
    pad_token_id: int = 0       # Padding token ID
    
    # Positional Encoding
    encoding_type: str = "sinusoidal"  # "sinusoidal", "relative", "rope"
    rope_theta: float = 10000.0        # RoPE base frequency
    relative_attention_num_buckets: int = 32  # Relative encoding buckets
    
    # Training Parameters
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    activation: str = "relu"  # "relu", "gelu", "swish"
    
    # Visualization Parameters
    attention_temperature: float = 1.0  # Temperature for attention softmax
    visualize_gradients: bool = True    # Whether to track gradients
    store_attention_weights: bool = True # Store attention for visualization
    
    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # Experimental Features
    use_flash_attention: bool = False   # Flash attention optimization
    checkpoint_activations: bool = False # Gradient checkpointing
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.encoding_type in ["sinusoidal", "relative", "rope"], \
            f"Unknown encoding type: {self.encoding_type}"
        assert self.activation in ["relu", "gelu", "swish"], \
            f"Unknown activation: {self.activation}"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
    
    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    @property 
    def scale_factor(self) -> float:
        """Scaling factor for attention scores."""
        return 1.0 / (self.d_k ** 0.5)


@dataclass
class ExperimentConfig:
    """Configuration for interactive experiments."""
    
    # Sequence Length Experiments
    seq_lengths: List[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])
    
    # Model Size Experiments
    model_dimensions: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    attention_head_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Temperature Scaling
    temperature_range: tuple = (0.1, 2.0)
    temperature_steps: int = 20
    
    # Comparison Settings
    encoding_types_to_compare: List[str] = field(
        default_factory=lambda: ["sinusoidal", "relative", "rope"]
    )
    
    # Benchmark Settings
    benchmark_tasks: List[str] = field(
        default_factory=lambda: ["position_prediction", "sequence_classification", "copying"]
    )
    
    # Visualization Settings
    max_tokens_to_display: int = 50
    heatmap_resolution: tuple = (800, 600)
    animation_fps: int = 10
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg", "html"])


# Predefined configurations for different use cases
CONFIGS = {
    "tiny": ModelConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
        vocab_size=1000
    ),
    
    "small": ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_seq_len=256,
        vocab_size=5000
    ),
    
    "medium": ModelConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        max_seq_len=512,
        vocab_size=10000
    ),
    
    "large": ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=512,
        vocab_size=30000
    )
}


def get_model_config(config_name: str = "medium") -> ModelConfig:
    """Get predefined model configuration.
    
    Args:
        config_name: Name of configuration ("tiny", "small", "medium", "large")
        
    Returns:
        ModelConfig instance
        
    Raises:
        KeyError: If config_name not found
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise KeyError(f"Config '{config_name}' not found. Available: {available}")
    
    return CONFIGS[config_name]


def create_custom_config(**kwargs) -> ModelConfig:
    """Create custom model configuration.
    
    Args:
        **kwargs: ModelConfig parameters to override
        
    Returns:
        Custom ModelConfig instance
    """
    base_config = get_model_config("medium")
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown parameter: {key}")
    
    return base_config


def validate_config(config: ModelConfig) -> List[str]:
    """Validate configuration and return warnings.
    
    Args:
        config: ModelConfig to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Performance warnings
    if config.d_model > 512 and config.device == "cpu":
        warnings.append("Large model on CPU may be slow")
    
    if config.max_seq_len > 1024 and not config.use_flash_attention:
        warnings.append("Long sequences without flash attention may cause memory issues")
    
    # Architecture warnings
    if config.n_heads > 16:
        warnings.append("Very large number of heads may hurt performance")
    
    if config.d_ff < config.d_model:
        warnings.append("Feed-forward dimension smaller than model dimension is unusual")
    
    # Visualization warnings
    if config.max_seq_len > 100 and config.store_attention_weights:
        warnings.append("Storing attention weights for long sequences uses significant memory")
    
    return warnings
