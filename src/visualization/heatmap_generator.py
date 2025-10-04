"""Advanced heatmap generation for attention and encoding patterns."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import math

from config import VisualizationConfig


class HeatmapGenerator:
    """Advanced heatmap generation with customization options."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.setup_colormaps()
    
    def setup_colormaps(self):
        """Setup custom colormaps for different visualization types."""
        # Custom colormap for attention (emphasizes high values)
        attention_colors = ['#000033', '#000066', '#0000CC', '#3366FF', '#66CCFF', '#FFFF66', '#FF6600', '#FF0000']
        self.attention_cmap = mcolors.LinearSegmentedColormap.from_list('attention', attention_colors)
        
        # Custom colormap for encodings
        encoding_colors = ['#2C3E50', '#3498DB', '#E74C3C', '#F39C12', '#27AE60']
        self.encoding_cmap = mcolors.LinearSegmentedColormap.from_list('encoding', encoding_colors)
        
        # Register colormaps
        plt.cm.register_cmap('attention_custom', self.attention_cmap)
        plt.cm.register_cmap('encoding_custom', self.encoding_cmap)
    
    def create_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        mask: Optional[torch.Tensor] = None,
        title: str = "Attention Heatmap",
        show_values: bool = False,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create sophisticated attention heatmap with advanced features.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings for labeling
            mask: Optional attention mask
            title: Heatmap title
            show_values: Whether to show numerical values
            threshold: Threshold for highlighting strong connections
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with attention heatmap
        """
        # Process attention weights
        if attention_weights.dim() == 4:
            attn_matrix = attention_weights[0, 0].detach().cpu().numpy()  # First batch, first head
        elif attention_weights.dim() == 3:
            attn_matrix = attention_weights[0].detach().cpu().numpy()  # First head
        else:
            attn_matrix = attention_weights.detach().cpu().numpy()
        
        seq_len = attn_matrix.shape[0]
        
        # Create figure with custom styling
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Apply threshold if specified
        display_matrix = attn_matrix.copy()
        if threshold is not None:
            display_matrix = np.where(display_matrix >= threshold, display_matrix, 0)
        
        # Create heatmap
        im = ax.imshow(
            display_matrix,
            cmap='attention_custom' if threshold else self.config.colormap_attention,
            aspect='equal',
            interpolation='nearest',
            alpha=self.config.alpha_transparency
        )
        
        # Customize colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Attention Weight', fontsize=self.config.font_size, labelpad=15)
        cbar.ax.tick_params(labelsize=self.config.font_size - 2)
        
        # Set labels and ticks
        if tokens is not None:
            # Prepare token labels
            display_tokens = self._prepare_token_labels(tokens)
            
            ax.set_xticks(range(len(display_tokens)))
            ax.set_yticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=self.config.token_label_rotation, ha='right')
            ax.set_yticklabels(display_tokens)
            
            # Adjust font size based on number of tokens
            label_fontsize = max(8, self.config.font_size - len(tokens) // 4)
            ax.tick_params(axis='both', labelsize=label_fontsize)
        else:
            # Use position indices
            tick_step = max(1, seq_len // 10)
            ticks = list(range(0, seq_len, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_yticklabels([str(t) for t in ticks])
        
        # Add grid for better readability
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, color=self.config.color_scheme.grid)
        
        # Add attention values as text
        if show_values and seq_len <= 16:
            self._add_attention_values(ax, display_matrix)
        
        # Highlight strong connections if threshold is set
        if threshold is not None:
            self._highlight_strong_connections(ax, attn_matrix, threshold)
        
        # Apply mask visualization if provided
        if mask is not None:
            self._apply_mask_overlay(ax, mask)
        
        # Set labels and title
        ax.set_xlabel('Key Position', fontsize=self.config.font_size, labelpad=10)
        ax.set_ylabel('Query Position', fontsize=self.config.font_size, labelpad=10)
        ax.set_title(title, fontsize=self.config.title_size, pad=20)
        
        # Add statistics box
        self._add_statistics_box(ax, attn_matrix)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight', 
                       transparent=self.config.export_transparent)
        
        return fig
    
    def _prepare_token_labels(self, tokens: List[str]) -> List[str]:
        """Prepare token labels for display."""
        display_tokens = []
        for token in tokens:
            if len(token) > self.config.max_token_label_length:
                display_token = token[:self.config.max_token_label_length-3] + '...'
            else:
                display_token = token
            display_tokens.append(display_token)
        return display_tokens
    
    def _add_attention_values(self, ax: plt.Axes, matrix: np.ndarray):
        """Add attention values as text overlay."""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value > 0.01:  # Only show significant values
                    # Choose text color based on background
                    text_color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.{self.config.attention_value_precision}f}',
                           ha='center', va='center', color=text_color, 
                           fontsize=8, weight='bold')
    
    def _highlight_strong_connections(self, ax: plt.Axes, matrix: np.ndarray, threshold: float):
        """Highlight strong attention connections with borders."""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] >= threshold:
                    # Add border rectangle
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=2, edgecolor='red', 
                                           facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
    
    def _apply_mask_overlay(self, ax: plt.Axes, mask: torch.Tensor):
        """Apply mask overlay to show masked positions."""
        mask_np = mask.detach().cpu().numpy()
        
        # Create overlay for masked positions
        overlay = np.zeros_like(mask_np, dtype=float)
        overlay[mask_np == 0] = 0.8  # High alpha for masked positions
        
        ax.imshow(overlay, cmap='gray', alpha=0.6, aspect='equal')
    
    def _add_statistics_box(self, ax: plt.Axes, matrix: np.ndarray):
        """Add statistics information box."""
        stats_text = (
            f'Max: {matrix.max():.3f}\n'
            f'Mean: {matrix.mean():.3f}\n'
            f'Std: {matrix.std():.3f}'
        )
        
        # Position box in top-right corner
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    def create_multi_layer_heatmap(
        self,
        layer_attentions: List[torch.Tensor],
        tokens: Optional[List[str]] = None,
        head_idx: int = 0,
        title: str = "Multi-Layer Attention Evolution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create heatmap showing attention evolution across layers.
        
        Args:
            layer_attentions: List of attention weights for each layer
            tokens: Token strings
            head_idx: Attention head index
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            Multi-layer attention heatmap figure
        """
        n_layers = len(layer_attentions)
        
        # Calculate subplot arrangement
        cols = min(4, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Global colorbar limits for consistency
        all_values = []
        for attention in layer_attentions:
            if attention.dim() == 4:
                matrix = attention[0, head_idx].detach().cpu().numpy()
            elif attention.dim() == 3:
                matrix = attention[head_idx].detach().cpu().numpy()
            else:
                matrix = attention.detach().cpu().numpy()
            all_values.extend(matrix.flatten())
        
        vmin, vmax = min(all_values), max(all_values)
        
        for layer_idx, attention in enumerate(layer_attentions):
            ax = axes[layer_idx]
            
            # Extract attention matrix
            if attention.dim() == 4:
                matrix = attention[0, head_idx].detach().cpu().numpy()
            elif attention.dim() == 3:
                matrix = attention[head_idx].detach().cpu().numpy()
            else:
                matrix = attention.detach().cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(matrix, cmap=self.config.colormap_attention,
                          aspect='equal', vmin=vmin, vmax=vmax)
            
            ax.set_title(f'Layer {layer_idx}', fontsize=self.config.font_size)
            
            # Set labels for edge subplots only
            if layer_idx >= n_layers - cols:  # Bottom row
                if tokens and len(tokens) <= 10:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(self._prepare_token_labels(tokens), 
                                      rotation=45, fontsize=8)
                ax.set_xlabel('Key Position', fontsize=10)
            else:
                ax.set_xticks([])
            
            if layer_idx % cols == 0:  # Left column
                if tokens and len(tokens) <= 10:
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(self._prepare_token_labels(tokens), fontsize=8)
                ax.set_ylabel('Query Position', fontsize=10)
            else:
                ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].set_visible(False)
        
        # Add shared colorbar
        if n_layers > 0:
            cbar = fig.colorbar(im, ax=axes[:n_layers], shrink=0.8, pad=0.02)
            cbar.set_label('Attention Weight', fontsize=self.config.font_size)
        
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_encoding_heatmap(
        self,
        encoding_matrix: torch.Tensor,
        encoding_name: str = "Positional Encoding",
        max_dimensions: int = 64,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create heatmap for positional encodings.
        
        Args:
            encoding_matrix: Encoding matrix of shape (seq_len, d_model)
            encoding_name: Name of the encoding method
            max_dimensions: Maximum dimensions to display
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Encoding heatmap figure
        """
        matrix = encoding_matrix.detach().cpu().numpy()
        seq_len, d_model = matrix.shape
        
        # Limit dimensions for visualization
        display_dims = min(d_model, max_dimensions)
        display_matrix = matrix[:, :display_dims]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main encoding heatmap
        im1 = ax1.imshow(display_matrix.T, cmap=self.config.colormap_encoding,
                        aspect='auto', interpolation='nearest')
        
        ax1.set_title(f'{encoding_name} Pattern')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Dimension')
        
        # Set ticks
        pos_step = max(1, seq_len // 10)
        dim_step = max(1, display_dims // 10)
        
        ax1.set_xticks(range(0, seq_len, pos_step))
        ax1.set_yticks(range(0, display_dims, dim_step))
        ax1.set_xticklabels(range(0, seq_len, pos_step))
        ax1.set_yticklabels(range(0, display_dims, dim_step))
        
        # Colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Encoding Value', fontsize=self.config.font_size)
        
        # Position similarity matrix
        similarities = np.dot(display_matrix, display_matrix.T) / display_dims
        
        im2 = ax2.imshow(similarities, cmap='coolwarm', aspect='equal',
                        vmin=-1, vmax=1)
        
        ax2.set_title('Position Similarities')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Position')
        
        ax2.set_xticks(range(0, seq_len, pos_step))
        ax2.set_yticks(range(0, seq_len, pos_step))
        ax2.set_xticklabels(range(0, seq_len, pos_step))
        ax2.set_yticklabels(range(0, seq_len, pos_step))
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Cosine Similarity', fontsize=self.config.font_size)
        
        if title is None:
            title = f'{encoding_name} Visualization'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_comparison_heatmap(
        self,
        encoding_matrices: Dict[str, torch.Tensor],
        title: str = "Encoding Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comparison heatmap for multiple encoding methods.
        
        Args:
            encoding_matrices: Dictionary of encoding name to matrix
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            Comparison heatmap figure
        """
        n_encodings = len(encoding_matrices)
        fig, axes = plt.subplots(2, n_encodings, figsize=(4*n_encodings, 8))
        
        if n_encodings == 1:
            axes = axes.reshape(-1, 1)
        
        # Global color limits for consistency
        all_values = []
        for matrix in encoding_matrices.values():
            all_values.extend(matrix.detach().cpu().numpy().flatten())
        vmin, vmax = min(all_values), max(all_values)
        
        for col, (enc_name, matrix) in enumerate(encoding_matrices.items()):
            matrix_np = matrix.detach().cpu().numpy()
            
            # Encoding pattern
            im1 = axes[0, col].imshow(matrix_np.T, cmap=self.config.colormap_encoding,
                                     aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'{enc_name.title()} Encoding')
            axes[0, col].set_xlabel('Position')
            if col == 0:
                axes[0, col].set_ylabel('Dimension')
            
            # Position similarities
            similarities = np.dot(matrix_np, matrix_np.T) / matrix_np.shape[1]
            im2 = axes[1, col].imshow(similarities, cmap='coolwarm', aspect='equal',
                                     vmin=-1, vmax=1)
            axes[1, col].set_title(f'{enc_name.title()} Similarities')
            axes[1, col].set_xlabel('Position')
            if col == 0:
                axes[1, col].set_ylabel('Position')
            
            # Add colorbar to last column
            if col == n_encodings - 1:
                cbar1 = plt.colorbar(im1, ax=axes[0, col], shrink=0.8)
                cbar1.set_label('Encoding Value')
                cbar2 = plt.colorbar(im2, ax=axes[1, col], shrink=0.8)
                cbar2.set_label('Similarity')
        
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig


class AttentionHeatmapGenerator(HeatmapGenerator):
    """Specialized heatmap generator for attention patterns."""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
    
    def create_head_specialization_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        specialization_metrics: Dict[str, torch.Tensor],
        title: str = "Attention Head Specialization",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create heatmap showing head specialization patterns.
        
        Args:
            attention_weights: Multi-head attention weights
            tokens: Token strings
            specialization_metrics: Computed specialization metrics
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            Head specialization heatmap figure
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        n_heads, seq_len, _ = attention_weights.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Individual head patterns (sample)
        sample_heads = min(4, n_heads)
        for i in range(sample_heads):
            row, col = i // 2, i % 2
            if row < 2 and col < 2:
                matrix = attention_weights[i].detach().cpu().numpy()
                
                im = axes[row, col].imshow(matrix, cmap=self.config.colormap_attention,
                                         aspect='equal')
                axes[row, col].set_title(f'Head {i} Pattern')
                
                if tokens and len(tokens) <= 8:
                    axes[row, col].set_xticks(range(len(tokens)))
                    axes[row, col].set_yticks(range(len(tokens)))
                    axes[row, col].set_xticklabels(tokens, rotation=45, fontsize=8)
                    axes[row, col].set_yticklabels(tokens, fontsize=8)
        
        # If we have specialization metrics, overlay them
        if 'head_entropies' in specialization_metrics:
            entropies = specialization_metrics['head_entropies'].cpu().numpy()
            
            # Overlay entropy information
            for i in range(min(4, len(entropies))):
                row, col = i // 2, i % 2
                if row < 2 and col < 2:
                    entropy_text = f'Entropy: {entropies[i]:.3f}'
                    axes[row, col].text(0.02, 0.98, entropy_text, 
                                       transform=axes[row, col].transAxes,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_attention_rollout_heatmap(
        self,
        rollout_matrix: torch.Tensor,
        tokens: List[str],
        title: str = "Attention Rollout",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create heatmap for attention rollout visualization.
        
        Args:
            rollout_matrix: Attention rollout matrix
            tokens: Token strings
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            Attention rollout heatmap figure
        """
        rollout_np = rollout_matrix.detach().cpu().numpy()
        
        # Handle batch dimension
        if rollout_np.ndim == 3:
            rollout_np = rollout_np[0]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create heatmap with special colormap for rollout
        im = ax.imshow(rollout_np, cmap='Reds', aspect='equal', 
                      interpolation='nearest')
        
        # Customize appearance
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Flow', fontsize=self.config.font_size)
        
        # Set labels
        if tokens and len(tokens) <= 16:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=self.config.token_label_rotation)
            ax.set_yticklabels(tokens)
        
        ax.set_xlabel('Source Token', fontsize=self.config.font_size)
        ax.set_ylabel('Target Token', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.title_size, pad=20)
        
        # Add interpretation text
        interpretation = ("Attention rollout shows information flow paths.\n"
                         "Darker colors indicate stronger information transfer.")
        ax.text(0.02, -0.1, interpretation, transform=ax.transAxes,
                fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_interactive_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_names: Optional[List[str]] = None
    ) -> go.Figure:
        """Create interactive Plotly heatmap for attention exploration.
        
        Args:
            attention_weights: Multi-layer attention weights
            tokens: Token strings
            layer_names: Names for each layer
            
        Returns:
            Interactive Plotly figure
        """
        if not isinstance(attention_weights, list):
            attention_weights = [attention_weights]
        
        n_layers = len(attention_weights)
        layer_names = layer_names or [f'Layer {i}' for i in range(n_layers)]
        
        # Create dropdown options for layers and heads
        buttons = []
        
        # Initialize with first layer, first head
        initial_attention = attention_weights[0]
        if initial_attention.dim() == 4:
            initial_matrix = initial_attention[0, 0].detach().cpu().numpy()
            n_heads = initial_attention.shape[1]
        elif initial_attention.dim() == 3:
            initial_matrix = initial_attention[0].detach().cpu().numpy()
            n_heads = initial_attention.shape[0]
        else:
            initial_matrix = initial_attention.detach().cpu().numpy()
            n_heads = 1
        
        # Create base heatmap
        fig = go.Figure(data=go.Heatmap(
            z=initial_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Attention Weight")
        ))
        
        # Create dropdown for layer selection
        layer_buttons = []
        for layer_idx, layer_name in enumerate(layer_names):
            attention = attention_weights[layer_idx]
            if attention.dim() == 4:
                matrix = attention[0, 0].detach().cpu().numpy()
            elif attention.dim() == 3:
                matrix = attention[0].detach().cpu().numpy()
            else:
                matrix = attention.detach().cpu().numpy()
            
            layer_buttons.append(dict(
                label=layer_name,
                method="restyle",
                args=[{"z": [matrix]}]
            ))
        
        # Add dropdown menu
        fig.update_layout(
            title="Interactive Attention Heatmap",
            xaxis_title="Key Position",
            yaxis_title="Query Position",
            updatemenus=[
                dict(
                    buttons=layer_buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ],
            annotations=[
                dict(text="Layer:", x=0.05, xref="paper", y=1.18, yref="paper",
                     align="left", showarrow=False)
            ]
        )
        
        return fig


# Additional utility functions for heatmap customization

def create_custom_colormap(colors: List[str], name: str) -> mcolors.LinearSegmentedColormap:
    """Create custom colormap from color list.
    
    Args:
        colors: List of color strings (hex or named colors)
        name: Name for the colormap
        
    Returns:
        Custom colormap
    """
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


def apply_attention_threshold(
    attention_matrix: np.ndarray, 
    threshold: float, 
    mode: str = 'hard'
) -> np.ndarray:
    """Apply thresholding to attention matrix.
    
    Args:
        attention_matrix: Input attention matrix
        threshold: Threshold value
        mode: 'hard' (zero below threshold) or 'soft' (scale values)
        
    Returns:
        Thresholded attention matrix
    """
    if mode == 'hard':
        return np.where(attention_matrix >= threshold, attention_matrix, 0)
    elif mode == 'soft':
        return np.maximum(attention_matrix - threshold, 0) / (1 - threshold)
    else:
        raise ValueError("Mode must be 'hard' or 'soft'")


def compute_attention_statistics(attention_matrix: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics for attention matrix.
    
    Args:
        attention_matrix: Input attention matrix
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': float(attention_matrix.mean()),
        'std': float(attention_matrix.std()),
        'min': float(attention_matrix.min()),
        'max': float(attention_matrix.max()),
        'entropy': float(-np.sum(attention_matrix * np.log(attention_matrix + 1e-8))),
        'sparsity': float(np.sum(attention_matrix < 0.01) / attention_matrix.size),
        'concentration': float(np.sum(attention_matrix > 0.1) / attention_matrix.size)
    }

