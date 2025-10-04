"""Visualization configuration settings and themes."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ColorScheme:
    """Color scheme for visualizations."""
    
    primary: str = "#2E86C1"      # Main accent color
    secondary: str = "#F39C12"    # Secondary accent color  
    background: str = "#FFFFFF"   # Background color
    text: str = "#2C3E50"        # Text color
    grid: str = "#ECF0F1"        # Grid lines
    attention_high: str = "#E74C3C"  # High attention values
    attention_low: str = "#3498DB"   # Low attention values
    encoding_sin: str = "#9B59B6"    # Sinusoidal encoding
    encoding_rel: str = "#E67E22"    # Relative encoding
    encoding_rope: str = "#27AE60"   # RoPE encoding


@dataclass 
class VisualizationConfig:
    """Configuration for visualization settings and themes."""
    
    # Figure Settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    font_size: int = 12
    title_size: int = 16
    
    # Color Settings
    color_scheme: ColorScheme = field(default_factory=ColorScheme)
    colormap_attention: str = "viridis"
    colormap_encoding: str = "plasma"
    alpha_transparency: float = 0.8
    
    # Attention Visualization
    attention_heatmap_size: Tuple[int, int] = (800, 600)
    show_attention_values: bool = True
    attention_value_precision: int = 3
    max_attention_heads_display: int = 8
    
    # Position Encoding Plots
    encoding_plot_3d: bool = True
    encoding_dimensions_to_show: int = 8
    position_range_display: Tuple[int, int] = (0, 100)
    frequency_analysis_bins: int = 50
    
    # Interactive Elements
    enable_hover_tooltips: bool = True
    enable_click_selection: bool = True
    enable_zoom_pan: bool = True
    animation_duration: int = 1000  # milliseconds
    
    # Export Settings
    export_format: str = "png"  # "png", "svg", "pdf", "html"
    export_quality: int = 300   # DPI for raster formats
    export_transparent: bool = False
    
    # Performance Settings
    max_sequence_length_display: int = 50
    downsample_long_sequences: bool = True
    use_progressive_rendering: bool = True
    cache_visualizations: bool = True
    
    # Layout Settings
    subplot_spacing: float = 0.3
    margin_size: float = 0.1
    legend_position: str = "right"  # "right", "bottom", "top"
    show_grid: bool = True
    grid_alpha: float = 0.3
    
    # Text and Annotations
    show_token_labels: bool = True
    token_label_rotation: int = 45
    max_token_label_length: int = 10
    show_dimension_labels: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.export_format in ["png", "svg", "pdf", "html"], \
            f"Unsupported export format: {self.export_format}"
        assert 0 <= self.alpha_transparency <= 1, \
            "Alpha transparency must be between 0 and 1"
        assert self.legend_position in ["right", "bottom", "top", "none"], \
            f"Invalid legend position: {self.legend_position}"


@dataclass
class DashboardConfig:
    """Configuration for interactive dashboard."""
    
    # Layout
    sidebar_width: int = 300
    main_content_width: int = 1000  
    header_height: int = 80
    footer_height: int = 50
    
    # Components
    show_model_architecture_diagram: bool = True
    show_parameter_controls: bool = True
    show_comparison_tools: bool = True
    show_export_options: bool = True
    
    # Performance
    update_delay_ms: int = 100  # Debounce delay for parameter updates
    max_concurrent_visualizations: int = 4
    enable_caching: bool = True
    
    # User Interface
    theme: str = "light"  # "light", "dark", "auto"
    enable_keyboard_shortcuts: bool = True
    show_tooltips: bool = True
    enable_tutorial_mode: bool = True


# Predefined themes
THEMES = {
    "default": VisualizationConfig(),
    
    "dark": VisualizationConfig(
        color_scheme=ColorScheme(
            primary="#3498DB",
            secondary="#F39C12", 
            background="#2C3E50",
            text="#ECF0F1",
            grid="#34495E",
            attention_high="#E74C3C",
            attention_low="#3498DB"
        ),
        colormap_attention="plasma",
        colormap_encoding="inferno"
    ),
    
    "academic": VisualizationConfig(
        figure_size=(10, 6),
        font_size=14,
        color_scheme=ColorScheme(
            primary="#2C3E50",
            secondary="#E74C3C",
            background="#FFFFFF", 
            text="#2C3E50",
            grid="#BDC3C7"
        ),
        colormap_attention="Blues",
        colormap_encoding="Reds",
        export_format="svg",
        export_quality=300
    ),
    
    "presentation": VisualizationConfig(
        figure_size=(16, 10),
        font_size=18,
        title_size=24,
        color_scheme=ColorScheme(
            primary="#3498DB",
            secondary="#E67E22",
            background="#FFFFFF",
            text="#2C3E50", 
            grid="#ECF0F1"
        ),
        show_attention_values=False,  # Cleaner for presentations
        export_format="png",
        export_quality=300
    ),
    
    "colorblind_friendly": VisualizationConfig(
        color_scheme=ColorScheme(
            primary="#0173B2",
            secondary="#DE8F05", 
            background="#FFFFFF",
            text="#000000",
            attention_high="#CC78BC",
            attention_low="#56B4E9",
            encoding_sin="#009E73",
            encoding_rel="#F0E442", 
            encoding_rope="#D55E00"
        ),
        colormap_attention="viridis",
        colormap_encoding="cividis"
    )
}


def get_viz_config(theme_name: str = "default") -> VisualizationConfig:
    """Get predefined visualization configuration.
    
    Args:
        theme_name: Name of theme ("default", "dark", "academic", etc.)
        
    Returns:
        VisualizationConfig instance
        
    Raises:
        KeyError: If theme_name not found
    """
    if theme_name not in THEMES:
        available = ", ".join(THEMES.keys())
        raise KeyError(f"Theme '{theme_name}' not found. Available: {available}")
    
    return THEMES[theme_name]


def setup_matplotlib_style(config: VisualizationConfig) -> None:
    """Configure matplotlib with visualization settings.
    
    Args:
        config: VisualizationConfig instance
    """
    plt.rcParams.update({
        'figure.figsize': config.figure_size,
        'figure.dpi': config.dpi,
        'font.size': config.font_size,
        'axes.titlesize': config.title_size,
        'axes.labelsize': config.font_size,
        'xtick.labelsize': config.font_size - 2,
        'ytick.labelsize': config.font_size - 2,
        'legend.fontsize': config.font_size - 2,
        'figure.facecolor': config.color_scheme.background,
        'axes.facecolor': config.color_scheme.background,
        'text.color': config.color_scheme.text,
        'axes.labelcolor': config.color_scheme.text,
        'xtick.color': config.color_scheme.text,
        'ytick.color': config.color_scheme.text,
        'axes.grid': config.show_grid,
        'grid.alpha': config.grid_alpha,
        'grid.color': config.color_scheme.grid,
    })


def create_custom_viz_config(**kwargs) -> VisualizationConfig:
    """Create custom visualization configuration.
    
    Args:
        **kwargs: VisualizationConfig parameters to override
        
    Returns:
        Custom VisualizationConfig instance  
    """
    base_config = get_viz_config("default")
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown parameter: {key}")
    
    return base_config


def get_attention_colormap(config: VisualizationConfig, head_idx: int) -> str:
    """Get colormap for specific attention head.
    
    Args:
        config: VisualizationConfig instance
        head_idx: Attention head index
        
    Returns:
        Colormap name for the head
    """
    colormaps = ["viridis", "plasma", "inferno", "magma", "Blues", "Reds", "Greens", "Oranges"]
    return colormaps[head_idx % len(colormaps)]


def validate_viz_config(config: VisualizationConfig) -> List[str]:
    """Validate visualization configuration.
    
    Args:
        config: VisualizationConfig to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Performance warnings
    if config.max_sequence_length_display > 100:
        warnings.append("Displaying very long sequences may be slow")
    
    if not config.downsample_long_sequences and config.max_sequence_length_display > 50:
        warnings.append("Consider enabling downsampling for long sequences")
    
    # Visual clarity warnings
    if config.font_size < 8:
        warnings.append("Very small font size may be hard to read")
    
    if config.alpha_transparency < 0.3:
        warnings.append("Very low transparency may make overlapping elements invisible")
    
    return warnings
