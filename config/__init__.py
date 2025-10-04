"""Configuration package for Positional Encoding Visualizer."""

from .model_config import ModelConfig, get_model_config
from .visualization_config import VisualizationConfig, get_viz_config

__all__ = [
    "ModelConfig",
    "VisualizationConfig", 
    "get_model_config",
    "get_viz_config"
]
