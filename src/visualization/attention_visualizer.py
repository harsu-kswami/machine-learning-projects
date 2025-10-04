"""Attention pattern visualization with comprehensive analysis tools."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from config import VisualizationConfig
from src.utils.tokenizer import SimpleTokenizer


class AttentionVisualizer:
    """Comprehensive attention pattern visualization and analysis."""
    
    def __init__(self, config: VisualizationConfig, tokenizer: Optional[SimpleTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.setup_style()
        
        # Cache for computed visualizations
        self.visualization_cache = {}
        
    def setup_style(self):
        """Setup matplotlib style based on configuration."""
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'figure.facecolor': self.config.color_scheme.background,
            'axes.facecolor': self.config.color_scheme.background,
            'text.color': self.config.color_scheme.text,
            'axes.labelcolor': self.config.color_scheme.text,
            'xtick.color': self.config.color_scheme.text,
            'ytick.color': self.config.color_scheme.text,
        })
    
    def visualize_attention_matrix(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        head_idx: int = 0,
        layer_idx: int = 0,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create attention matrix heatmap visualization.
        
        Args:
            attention_weights: Attention weights of shape (batch, heads, seq_len, seq_len)
            tokens: List of token strings for labeling
            head_idx: Attention head to visualize
            layer_idx: Layer index for title
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract single attention matrix
        if attention_weights.dim() == 4:
            attn_matrix = attention_weights[0, head_idx].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attn_matrix = attention_weights[head_idx].detach().cpu().numpy()
        else:
            attn_matrix = attention_weights.detach().cpu().numpy()
        
        seq_len = attn_matrix.shape[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create heatmap
        im = ax.imshow(
            attn_matrix, 
            cmap=self.config.colormap_attention,
            aspect='equal',
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=self.config.font_size)
        
        # Set labels
        if tokens is not None:
            # Truncate long tokens for display
            display_tokens = [
                token[:self.config.max_token_label_length] + '...' 
                if len(token) > self.config.max_token_label_length else token
                for token in tokens
            ]
            
            ax.set_xticks(range(len(display_tokens)))
            ax.set_yticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=self.config.token_label_rotation)
            ax.set_yticklabels(display_tokens)
        else:
            ax.set_xticks(range(0, seq_len, max(1, seq_len // 10)))
            ax.set_yticks(range(0, seq_len, max(1, seq_len // 10)))
        
        # Set title
        if title is None:
            title = f'Attention Pattern - Layer {layer_idx}, Head {head_idx}'
        ax.set_title(title, fontsize=self.config.title_size, pad=20)
        
        ax.set_xlabel('Key Position', fontsize=self.config.font_size)
        ax.set_ylabel('Query Position', fontsize=self.config.font_size)
        
        # Add attention values as text if enabled and matrix is small
        if self.config.show_attention_values and seq_len <= 16:
            for i in range(seq_len):
                for j in range(seq_len):
                    value = attn_matrix[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.{self.config.attention_value_precision}f}',
                           ha='center', va='center', color=color, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def visualize_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        max_heads: Optional[int] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention patterns across multiple heads.
        
        Args:
            attention_weights: Attention weights of shape (batch, heads, seq_len, seq_len)
            tokens: List of token strings
            layer_idx: Layer index
            max_heads: Maximum number of heads to display
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure with subplots for each head
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        n_heads = attention_weights.shape[0]
        max_heads = max_heads or min(n_heads, self.config.max_attention_heads_display)
        
        # Calculate subplot grid
        cols = min(4, max_heads)
        rows = (max_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for head_idx in range(max_heads):
            ax = axes[head_idx] if max_heads > 1 else axes[0]
            
            # Get attention matrix for this head
            attn_matrix = attention_weights[head_idx].detach().cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(
                attn_matrix,
                cmap=self.config.colormap_attention,
                aspect='equal',
                interpolation='nearest'
            )
            
            # Set title for each head
            ax.set_title(f'Head {head_idx}', fontsize=self.config.font_size)
            
            # Set labels for first and last heads only to reduce clutter
            if head_idx == 0 or head_idx == max_heads - 1:
                if tokens is not None and len(tokens) <= 10:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_yticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45, fontsize=8)
                    ax.set_yticklabels(tokens, fontsize=8)
                else:
                    # Show position indices for longer sequences
                    seq_len = attn_matrix.shape[0]
                    step = max(1, seq_len // 5)
                    ax.set_xticks(range(0, seq_len, step))
                    ax.set_yticks(range(0, seq_len, step))
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add colorbar to last subplot
            if head_idx == max_heads - 1:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Attention', fontsize=10)
        
        # Hide unused subplots
        for idx in range(max_heads, len(axes)):
            axes[idx].set_visible(False)
        
        # Set main title
        if title is None:
            title = f'Multi-Head Attention Patterns - Layer {layer_idx}'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_patterns_over_layers(
        self,
        layer_attention_weights: List[torch.Tensor],
        tokens: Optional[List[str]] = None,
        head_idx: int = 0,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize how attention patterns evolve across layers.
        
        Args:
            layer_attention_weights: List of attention weights for each layer
            tokens: List of token strings
            head_idx: Attention head to track
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Figure showing attention evolution across layers
        """
        n_layers = len(layer_attention_weights)
        
        # Calculate grid dimensions
        cols = min(4, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for layer_idx, attention_weights in enumerate(layer_attention_weights):
            ax = axes[layer_idx]
            
            # Extract attention matrix
            if attention_weights.dim() == 4:
                attn_matrix = attention_weights[0, head_idx].detach().cpu().numpy()
            elif attention_weights.dim() == 3:
                attn_matrix = attention_weights[head_idx].detach().cpu().numpy()
            else:
                attn_matrix = attention_weights.detach().cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(
                attn_matrix,
                cmap=self.config.colormap_attention,
                aspect='equal',
                interpolation='nearest'
            )
            
            ax.set_title(f'Layer {layer_idx}', fontsize=self.config.font_size)
            
            # Set labels only for bottom row
            if layer_idx >= n_layers - cols:
                if tokens is not None and len(tokens) <= 8:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45, fontsize=8)
                else:
                    seq_len = attn_matrix.shape[0]
                    step = max(1, seq_len // 4)
                    ax.set_xticks(range(0, seq_len, step))
                ax.set_xlabel('Key Position', fontsize=10)
            else:
                ax.set_xticks([])
            
            # Set y-labels only for leftmost column
            if layer_idx % cols == 0:
                if tokens is not None and len(tokens) <= 8:
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=8)
                else:
                    seq_len = attn_matrix.shape[0]
                    step = max(1, seq_len // 4)
                    ax.set_yticks(range(0, seq_len, step))
                ax.set_ylabel('Query Position', fontsize=10)
            else:
                ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].set_visible(False)
        
        # Add colorbar to the last subplot
        if n_layers > 0:
            cbar = plt.colorbar(im, ax=axes[n_layers-1], shrink=0.8)
            cbar.set_label('Attention Weight', fontsize=10)
        
        if title is None:
            title = f'Attention Evolution Across Layers (Head {head_idx})'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def analyze_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze attention patterns and compute statistics.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Optional token strings
            
        Returns:
            Dictionary with attention analysis results
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        analysis = {}
        n_heads, seq_len, _ = attention_weights.shape
        
        # Basic statistics
        analysis['basic_stats'] = {
            'mean_attention': attention_weights.mean().item(),
            'std_attention': attention_weights.std().item(),
            'max_attention': attention_weights.max().item(),
            'min_attention': attention_weights.min().item()
        }
        
        # Attention entropy (measure of attention spread)
        entropies = []
        for head in range(n_heads):
            head_entropy = []
            for pos in range(seq_len):
                attn_dist = attention_weights[head, pos]
                entropy = -(attn_dist * torch.log(attn_dist + 1e-8)).sum().item()
                head_entropy.append(entropy)
            entropies.append(torch.tensor(head_entropy))
        
        analysis['attention_entropy'] = {
            'per_head': entropies,
            'mean_per_head': [entropy.mean().item() for entropy in entropies],
            'overall_mean': torch.stack(entropies).mean().item()
        }
        
        # Attention distance (how far each position looks on average)
        attention_distances = []
        position_indices = torch.arange(seq_len, dtype=torch.float32)
        
        for head in range(n_heads):
            head_distances = []
            for query_pos in range(seq_len):
                attn_dist = attention_weights[head, query_pos]
                # Weighted average of key positions
                avg_distance = (attn_dist * torch.abs(position_indices - query_pos)).sum().item()
                head_distances.append(avg_distance)
            attention_distances.append(head_distances)
        
        analysis['attention_distance'] = {
            'per_head': attention_distances,
            'mean_per_head': [np.mean(distances) for distances in attention_distances]
        }
        
        # Head similarity (correlation between heads)
        head_similarities = torch.zeros(n_heads, n_heads)
        for i in range(n_heads):
            for j in range(n_heads):
                if i != j:
                    attn_i = attention_weights[i].flatten()
                    attn_j = attention_weights[j].flatten()
                    correlation = torch.corrcoef(torch.stack([attn_i, attn_j]))[0, 1]
                    head_similarities[i, j] = correlation
        
        analysis['head_similarity'] = {
            'correlation_matrix': head_similarities,
            'mean_similarity': head_similarities.mean().item(),
            'max_similarity': head_similarities.max().item()
        }
        
        # Attention sparsity (concentration measure)
        sparsity_measures = []
        for head in range(n_heads):
            # Gini coefficient as sparsity measure
            head_attn = attention_weights[head].flatten().sort().values
            n = len(head_attn)
            index = torch.arange(1, n + 1, dtype=torch.float32)
            gini = (2 * (index * head_attn).sum()) / (n * head_attn.sum()) - (n + 1) / n
            sparsity_measures.append(gini.item())
        
        analysis['attention_sparsity'] = {
            'gini_coefficients': sparsity_measures,
            'mean_gini': np.mean(sparsity_measures)
        }
        
        return analysis
    
    def create_attention_flow_diagram(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        threshold: float = 0.1,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create attention flow diagram showing strong connections.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings
            threshold: Minimum attention weight to show
            save_path: Path to save figure
            
        Returns:
            Figure with attention flow diagram
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0, 0]  # Use first head
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights[0]  # Use first head
        
        attn_matrix = attention_weights.detach().cpu().numpy()
        seq_len = len(tokens)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Position tokens in a circle
        angles = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)
        radius = 3.0
        positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
        
        # Draw tokens
        for i, (token, (x, y)) in enumerate(zip(tokens, positions)):
            circle = plt.Circle((x, y), 0.3, color=self.config.color_scheme.primary, alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, token, ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw attention connections
        for i in range(seq_len):
            for j in range(seq_len):
                attention_weight = attn_matrix[i, j]
                if attention_weight > threshold and i != j:
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    
                    # Arrow properties based on attention weight
                    alpha = min(1.0, attention_weight / 0.5)
                    width = attention_weight * 3
                    
                    # Draw arrow
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(
                                   arrowstyle='->', 
                                   color=self.config.color_scheme.attention_high,
                                   alpha=alpha,
                                   lw=width,
                                   shrinkA=15, shrinkB=15
                               ))
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Attention Flow Diagram (threshold > {threshold})', 
                    fontsize=self.config.title_size, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.config.color_scheme.attention_high, 
                      lw=2, label=f'Attention > {threshold}'),
            plt.Circle((0, 0), 0, facecolor=self.config.color_scheme.primary, 
                      alpha=0.7, label='Tokens')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig


class MultiHeadAttentionVisualizer(AttentionVisualizer):
    """Specialized visualizer for multi-head attention analysis."""
    
    def __init__(self, config: VisualizationConfig, tokenizer: Optional[SimpleTokenizer] = None):
        super().__init__(config, tokenizer)
    
    def create_head_comparison_plot(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        query_position: int,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare attention patterns across heads for a specific query position.
        
        Args:
            attention_weights: Attention weights tensor (heads, seq_len, seq_len)
            tokens: Token strings
            query_position: Query position to analyze
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Figure comparing attention across heads
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        n_heads = attention_weights.shape[0]
        seq_len = len(tokens)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot for each head
        x_positions = np.arange(seq_len)
        width = 0.8 / n_heads
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_heads))
        
        for head_idx in range(n_heads):
            attn_values = attention_weights[head_idx, query_position].detach().cpu().numpy()
            offset = (head_idx - n_heads/2 + 0.5) * width
            
            bars = ax.bar(x_positions + offset, attn_values, width, 
                         label=f'Head {head_idx}', color=colors[head_idx], alpha=0.7)
        
        ax.set_xlabel('Key Position (Tokens)', fontsize=self.config.font_size)
        ax.set_ylabel('Attention Weight', fontsize=self.config.font_size)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if title is None:
            title = f'Attention Patterns for Query Position: "{tokens[query_position]}"'
        ax.set_title(title, fontsize=self.config.title_size, pad=20)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_head_specialization_analysis(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Analyze and visualize head specialization patterns.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings
            save_path: Path to save figure
            
        Returns:
            Figure showing head specialization analysis
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        n_heads, seq_len, _ = attention_weights.shape
        
        # Analyze different attention patterns
        patterns = {}
        
        # 1. Local vs Global attention
        for head_idx in range(n_heads):
            local_attention = 0
            global_attention = 0
            
            for i in range(seq_len):
                attn_dist = attention_weights[head_idx, i]
                
                # Local: attention to nearby positions
                local_mask = torch.abs(torch.arange(seq_len) - i) <= 2
                local_attention += attn_dist[local_mask].sum().item()
                
                # Global: attention to distant positions
                global_mask = torch.abs(torch.arange(seq_len) - i) > 2
                global_attention += attn_dist[global_mask].sum().item()
            
            patterns[head_idx] = {
                'local': local_attention / seq_len,
                'global': global_attention / seq_len
            }
        
        # 2. Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Local vs Global attention
        head_indices = list(range(n_heads))
        local_scores = [patterns[h]['local'] for h in head_indices]
        global_scores = [patterns[h]['global'] for h in head_indices]
        
        x = np.arange(n_heads)
        width = 0.35
        
        axes[0, 0].bar(x - width/2, local_scores, width, label='Local', alpha=0.7)
        axes[0, 0].bar(x + width/2, global_scores, width, label='Global', alpha=0.7)
        axes[0, 0].set_xlabel('Head Index')
        axes[0, 0].set_ylabel('Attention Score')
        axes[0, 0].set_title('Local vs Global Attention by Head')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Attention entropy by head
        entropies = []
        for head_idx in range(n_heads):
            head_entropy = []
            for pos in range(seq_len):
                attn_dist = attention_weights[head_idx, pos]
                entropy = -(attn_dist * torch.log(attn_dist + 1e-8)).sum().item()
                head_entropy.append(entropy)
            entropies.append(np.mean(head_entropy))
        
        axes[0, 1].bar(range(n_heads), entropies, alpha=0.7)
        axes[0, 1].set_xlabel('Head Index')
        axes[0, 1].set_ylabel('Average Entropy')
        axes[0, 1].set_title('Attention Entropy by Head')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Head similarity matrix
        head_similarities = torch.zeros(n_heads, n_heads)
        for i in range(n_heads):
            for j in range(n_heads):
                if i != j:
                    attn_i = attention_weights[i].flatten()
                    attn_j = attention_weights[j].flatten()
                    correlation = torch.corrcoef(torch.stack([attn_i, attn_j]))[0, 1]
                    head_similarities[i, j] = correlation.item()
        
        im = axes[1, 0].imshow(head_similarities.numpy(), cmap='coolwarm', 
                              vmin=-1, vmax=1, aspect='equal')
        axes[1, 0].set_title('Head Similarity Matrix')
        axes[1, 0].set_xlabel('Head Index')
        axes[1, 0].set_ylabel('Head Index')
        plt.colorbar(im, ax=axes[1, 0], label='Correlation')
        
        # Plot 4: Attention distance distribution
        attention_distances = []
        for head_idx in range(n_heads):
            head_distances = []
            for i in range(seq_len):
                attn_dist = attention_weights[head_idx, i]
                distances = torch.abs(torch.arange(seq_len, dtype=torch.float32) - i)
                avg_distance = (attn_dist * distances).sum().item()
                head_distances.append(avg_distance)
            attention_distances.append(head_distances)
        
        axes[1, 1].boxplot(attention_distances, labels=[f'H{i}' for i in range(n_heads)])
        axes[1, 1].set_xlabel('Head Index')
        axes[1, 1].set_ylabel('Average Attention Distance')
        axes[1, 1].set_title('Attention Distance Distribution by Head')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_interactive_attention_explorer(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str]
    ) -> go.Figure:
        """Create interactive Plotly visualization for exploring attention patterns.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings
            
        Returns:
            Interactive Plotly figure
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dimension
        
        n_heads, seq_len, _ = attention_weights.shape
        
        # Create subplots for each head
        fig = make_subplots(
            rows=(n_heads + 3) // 4,
            cols=4,
            subplot_titles=[f'Head {i}' for i in range(n_heads)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for head_idx in range(n_heads):
            row = head_idx // 4 + 1
            col = head_idx % 4 + 1
            
            attn_matrix = attention_weights[head_idx].detach().cpu().numpy()
            
            heatmap = go.Heatmap(
                z=attn_matrix,
                x=tokens,
                y=tokens,
                colorscale='Viridis',
                showscale=(head_idx == 0),  # Show scale only for first head
                hovetemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>',
                name=f'Head {head_idx}'
            )
            
            fig.add_trace(heatmap, row=row, col=col)
            
            # Update axis labels
            fig.update_xaxes(title_text="Key Position", row=row, col=col)
            fig.update_yaxes(title_text="Query Position", row=row, col=col)
        
        fig.update_layout(
            title="Interactive Multi-Head Attention Patterns",
            height=300 * ((n_heads + 3) // 4),
            showlegend=False
        )
        
        return fig
