"""Positional encoding visualization with comprehensive plotting capabilities."""

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
import math

from config import VisualizationConfig
from src.positional_encoding import (
    SinusoidalEncoding, RelativePositionalEncoding, 
    RoPEEncoding, encoding_utils
)


class EncodingPlotter:
    """Comprehensive positional encoding visualization toolkit."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.setup_style()
    
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
    
    def plot_sinusoidal_patterns(
        self,
        encoding: SinusoidalEncoding,
        seq_len: int = 64,
        dimensions: Optional[List[int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot sinusoidal encoding patterns across positions and dimensions.
        
        Args:
            encoding: Sinusoidal encoding instance
            seq_len: Sequence length to visualize
            dimensions: Specific dimensions to plot
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if dimensions is None:
            dimensions = list(range(0, min(encoding.d_model, 16), 2))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get encoding data
        encoding_data = encoding.forward(seq_len, encoding.d_model).squeeze(0)
        positions = torch.arange(seq_len)
        
        # Plot 1: Encoding heatmap
        im = axes[0, 0].imshow(
            encoding_data.T.cpu().numpy(),
            aspect='auto',
            cmap=self.config.colormap_encoding,
            extent=[0, seq_len, encoding.d_model, 0]
        )
        axes[0, 0].set_title('Sinusoidal Encoding Heatmap')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[0, 0], label='Encoding Value')
        
        # Plot 2: Selected dimension patterns
        for i, dim in enumerate(dimensions[:8]):  # Limit to 8 lines for clarity
            if dim < encoding.d_model:
                pattern = encoding_data[:, dim].cpu().numpy()
                axes[0, 1].plot(positions.numpy(), pattern, 
                               label=f'Dim {dim}', alpha=0.7, linewidth=2)
        
        axes[0, 1].set_title('Encoding Patterns by Dimension')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Encoding Value')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Frequency analysis
        freq_analysis = encoding.analyze_frequency_components()
        frequencies = freq_analysis['frequencies'].cpu().numpy()
        wavelengths = freq_analysis['wavelengths'].cpu().numpy()
        
        dim_indices = np.arange(len(frequencies))
        axes[1, 0].semilogy(dim_indices, frequencies, 'o-', markersize=4)
        axes[1, 0].set_title('Frequency Components by Dimension')
        axes[1, 0].set_xlabel('Dimension Index')
        axes[1, 0].set_ylabel('Frequency (log scale)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Position similarity matrix
        similarities = torch.mm(encoding_data, encoding_data.t()) / encoding.d_model
        im2 = axes[1, 1].imshow(
            similarities.cpu().numpy(),
            cmap='coolwarm',
            aspect='equal'
        )
        axes[1, 1].set_title('Position Similarity Matrix')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Position')
        plt.colorbar(im2, ax=axes[1, 1], label='Cosine Similarity')
        
        if title is None:
            title = 'Sinusoidal Positional Encoding Analysis'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def plot_rope_rotations(
        self,
        encoding: RoPEEncoding,
        seq_len: int = 32,
        sample_positions: Optional[List[int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize RoPE rotation patterns in 2D space.
        
        Args:
            encoding: RoPE encoding instance
            seq_len: Sequence length
            sample_positions: Positions to visualize
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Figure showing RoPE rotations
        """
        if sample_positions is None:
            sample_positions = list(range(0, seq_len, max(1, seq_len // 8)))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get RoPE data
        rope_data = encoding.forward(seq_len, encoding.d_model)
        cos_values = rope_data['cos'].squeeze(0)
        sin_values = rope_data['sin'].squeeze(0)
        
        # Plot 1: Cos and Sin patterns for first few dimensions
        positions = torch.arange(seq_len)
        for dim_pair in range(min(4, encoding.d_model // 2)):
            cos_pattern = cos_values[:, dim_pair * 2].cpu().numpy()
            sin_pattern = sin_values[:, dim_pair * 2].cpu().numpy()
            
            axes[0, 0].plot(positions.numpy(), cos_pattern, 
                           label=f'Cos Dim {dim_pair*2}', linestyle='-')
            axes[0, 0].plot(positions.numpy(), sin_pattern, 
                           label=f'Sin Dim {dim_pair*2}', linestyle='--')
        
        axes[0, 0].set_title('RoPE Cos/Sin Patterns')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: 2D rotation visualization for different positions
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_positions)))
        
        for i, pos in enumerate(sample_positions):
            # Show rotation for first dimension pair
            if encoding.d_model >= 2:
                cos_val = cos_values[pos, 0].item()
                sin_val = sin_values[pos, 0].item()
                
                # Unit vector rotation
                original = np.array([1, 0])
                rotated = np.array([cos_val, sin_val])
                
                # Draw rotation
                axes[0, 1].arrow(0, 0, original[0]*0.8, original[1]*0.8,
                                head_width=0.05, head_length=0.05, 
                                fc='gray', ec='gray', alpha=0.3)
                axes[0, 1].arrow(0, 0, rotated[0]*0.8, rotated[1]*0.8,
                                head_width=0.05, head_length=0.05, 
                                fc=colors[i], ec=colors[i],
                                label=f'Pos {pos}')
        
        axes[0, 1].set_title('2D Rotation Visualization (First Dim Pair)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].set_xlim(-1.2, 1.2)
        axes[0, 1].set_ylim(-1.2, 1.2)
        axes[0, 1].add_patch(plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.3))
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_aspect('equal')
        
        # Plot 3: Frequency analysis
        freq_analysis = encoding.analyze_rotation_frequencies()
        frequencies = freq_analysis['frequencies'].cpu().numpy()
        wavelengths = freq_analysis['wavelengths'].cpu().numpy()
        
        dim_indices = np.arange(len(frequencies))
        axes[1, 0].semilogy(dim_indices, frequencies, 'o-', markersize=6, 
                           color=self.config.color_scheme.encoding_rope)
        axes[1, 0].set_title('RoPE Rotation Frequencies')
        axes[1, 0].set_xlabel('Dimension Pair Index')
        axes[1, 0].set_ylabel('Frequency (log scale)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Position similarities using RoPE
        similarities = encoding.compute_position_similarities(seq_len, 'cosine')
        im = axes[1, 1].imshow(
            similarities.cpu().numpy(),
            cmap='coolwarm',
            aspect='equal'
        )
        axes[1, 1].set_title('RoPE Position Similarities')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Position')
        plt.colorbar(im, ax=axes[1, 1], label='Similarity')
        
        if title is None:
            title = 'Rotary Position Embedding (RoPE) Analysis'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def plot_encoding_comparison(
        self,
        encodings: Dict[str, Any],
        seq_len: int = 64,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare different positional encoding methods side by side.
        
        Args:
            encodings: Dictionary of encoding name to encoding instance
            seq_len: Sequence length
            title: Custom title
            save_path: Path to save figure
            
Returns:
            Comparison figure
        """
        n_encodings = len(encodings)
        fig, axes = plt.subplots(3, n_encodings, figsize=(5*n_encodings, 12))
        
        if n_encodings == 1:
            axes = axes.reshape(-1, 1)
        
        encoding_matrices = {}
        
        # Process each encoding
        for col, (enc_name, encoding) in enumerate(encodings.items()):
            
            # Get encoding representation
            if hasattr(encoding, 'forward'):
                if enc_name in ['sinusoidal', 'learned']:
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    matrix = enc_output.squeeze(0)
                elif enc_name == 'rope':
                    rope_data = encoding.forward(seq_len, encoding.d_model)
                    matrix = rope_data['cos'].squeeze(0)  # Use cos component
                else:
                    # Skip unsupported encoding types
                    continue
            else:
                continue
            
            encoding_matrices[enc_name] = matrix
            
            # Row 1: Encoding heatmap
            im1 = axes[0, col].imshow(
                matrix.T.cpu().numpy(),
                aspect='auto',
                cmap=self.config.colormap_encoding
            )
            axes[0, col].set_title(f'{enc_name.title()} Encoding')
            axes[0, col].set_xlabel('Position')
            axes[0, col].set_ylabel('Dimension')
            
            # Add colorbar to last column
            if col == n_encodings - 1:
                plt.colorbar(im1, ax=axes[0, col], label='Encoding Value')
            
            # Row 2: Position similarity matrix
            similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
            im2 = axes[1, col].imshow(
                similarities.cpu().numpy(),
                cmap='coolwarm',
                aspect='equal'
            )
            axes[1, col].set_title(f'{enc_name.title()} Similarities')
            axes[1, col].set_xlabel('Position')
            axes[1, col].set_ylabel('Position')
            
            if col == n_encodings - 1:
                plt.colorbar(im2, ax=axes[1, col], label='Cosine Similarity')
            
            # Row 3: Sample dimension patterns
            for dim in range(0, min(matrix.size(-1), 8), 2):
                pattern = matrix[:, dim].cpu().numpy()
                axes[2, col].plot(range(seq_len), pattern, 
                                 alpha=0.7, linewidth=1.5)
            
            axes[2, col].set_title(f'{enc_name.title()} Patterns')
            axes[2, col].set_xlabel('Position')
            axes[2, col].set_ylabel('Encoding Value')
            axes[2, col].grid(True, alpha=0.3)
        
        if title is None:
            title = 'Positional Encoding Comparison'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def plot_sequence_length_analysis(
        self,
        encoding: Any,
        sequence_lengths: List[int],
        encoding_name: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Analyze how encoding patterns change with sequence length.
        
        Args:
            encoding: Positional encoding instance
            sequence_lengths: List of sequence lengths to analyze
            encoding_name: Name of the encoding method
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Sequence length analysis figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect data for different sequence lengths
        similarity_decay = {}
        max_similarities = []
        min_similarities = []
        
        for seq_len in sequence_lengths:
            if hasattr(encoding, 'forward'):
                if encoding_name in ['sinusoidal', 'learned']:
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    matrix = enc_output.squeeze(0)
                elif encoding_name == 'rope':
                    rope_data = encoding.forward(seq_len, encoding.d_model)
                    matrix = rope_data['cos'].squeeze(0)
                else:
                    continue
            else:
                continue
            
            # Compute similarities
            similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
            
            # Analyze similarity decay with distance
            distances = torch.arange(seq_len).float()
            decay_pattern = []
            
            for dist in range(1, min(seq_len, 20)):  # Analyze up to distance 20
                mask = torch.abs(
                    torch.arange(seq_len).unsqueeze(0) - 
                    torch.arange(seq_len).unsqueeze(1)
                ) == dist
                
                if mask.sum() > 0:
                    avg_similarity = similarities[mask].mean().item()
                    decay_pattern.append(avg_similarity)
                else:
                    decay_pattern.append(0.0)
            
            similarity_decay[seq_len] = decay_pattern
            max_similarities.append(similarities.max().item())
            min_similarities.append(similarities.min().item())
        
        # Plot 1: Similarity decay patterns
        colors = plt.cm.viridis(np.linspace(0, 1, len(sequence_lengths)))
        
        for i, seq_len in enumerate(sequence_lengths):
            if seq_len in similarity_decay:
                decay = similarity_decay[seq_len]
                distances = range(1, len(decay) + 1)
                axes[0, 0].plot(distances, decay, 'o-', 
                               color=colors[i], label=f'Seq Len {seq_len}')
        
        axes[0, 0].set_title('Similarity Decay with Distance')
        axes[0, 0].set_xlabel('Distance')
        axes[0, 0].set_ylabel('Average Similarity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Max/Min similarity vs sequence length
        axes[0, 1].plot(sequence_lengths, max_similarities, 'o-', 
                       label='Max Similarity', color='red')
        axes[0, 1].plot(sequence_lengths, min_similarities, 'o-', 
                       label='Min Similarity', color='blue')
        axes[0, 1].set_title('Similarity Range vs Sequence Length')
        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Similarity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Encoding visualization for different lengths
        sample_lengths = sequence_lengths[:4]  # Show first 4 lengths
        
        for i, seq_len in enumerate(sample_lengths):
            if hasattr(encoding, 'forward'):
                if encoding_name in ['sinusoidal', 'learned']:
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    matrix = enc_output.squeeze(0)
                elif encoding_name == 'rope':
                    rope_data = encoding.forward(seq_len, encoding.d_model)
                    matrix = rope_data['cos'].squeeze(0)
                else:
                    continue
            else:
                continue
            
            # Plot first dimension pattern
            if matrix.size(-1) > 0:
                pattern = matrix[:, 0].cpu().numpy()
                axes[1, 0].plot(range(seq_len), pattern, 
                               label=f'Len {seq_len}', alpha=0.7)
        
        axes[1, 0].set_title(f'{encoding_name.title()} - First Dimension')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Encoding Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Extrapolation quality (if applicable)
        if encoding_name in ['sinusoidal', 'rope']:
            base_len = min(sequence_lengths)
            extrapolation_quality = []
            
            for seq_len in sequence_lengths:
                if seq_len > base_len:
                    if encoding_name == 'sinusoidal':
                        base_enc = encoding.forward(base_len, encoding.d_model).squeeze(0)
                        ext_enc = encoding.forward(seq_len, encoding.d_model).squeeze(0)
                        
                        similarity = F.cosine_similarity(
                            base_enc, ext_enc[:base_len], dim=-1
                        ).mean().item()
                        
                    elif encoding_name == 'rope':
                        base_data = encoding.forward(base_len, encoding.d_model)
                        ext_data = encoding.forward(seq_len, encoding.d_model)
                        
                        base_cos = base_data['cos'].squeeze(0)
                        ext_cos = ext_data['cos'].squeeze(0)[:base_len]
                        
                        similarity = F.cosine_similarity(
                            base_cos, ext_cos, dim=-1
                        ).mean().item()
                    
                    extrapolation_quality.append((seq_len, similarity))
            
            if extrapolation_quality:
                ext_lengths, ext_qualities = zip(*extrapolation_quality)
                axes[1, 1].plot(ext_lengths, ext_qualities, 'o-')
                axes[1, 1].set_title('Extrapolation Quality')
                axes[1, 1].set_xlabel('Sequence Length')
                axes[1, 1].set_ylabel('Similarity to Base')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No extrapolation data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Extrapolation not supported', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        if title is None:
            title = f'{encoding_name.title()} - Sequence Length Analysis'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
    
    def create_interactive_encoding_explorer(
        self,
        encodings: Dict[str, Any],
        seq_len: int = 64
    ) -> go.Figure:
        """Create interactive Plotly visualization for encoding exploration.
        
        Args:
            encodings: Dictionary of encoding instances
            seq_len: Sequence length
            
        Returns:
            Interactive Plotly figure
        """
        # Create subplots
        n_encodings = len(encodings)
        fig = make_subplots(
            rows=2, cols=n_encodings,
            subplot_titles=[f'{name.title()} Encoding' for name in encodings.keys()] + 
                          [f'{name.title()} Similarities' for name in encodings.keys()],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for col, (enc_name, encoding) in enumerate(encodings.items(), 1):
            # Get encoding matrix
            if hasattr(encoding, 'forward'):
                if enc_name in ['sinusoidal', 'learned']:
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    matrix = enc_output.squeeze(0)
                elif enc_name == 'rope':
                    rope_data = encoding.forward(seq_len, encoding.d_model)
                    matrix = rope_data['cos'].squeeze(0)
                else:
                    continue
            else:
                continue
            
            matrix_np = matrix.cpu().numpy()
            
            # Encoding heatmap
            heatmap1 = go.Heatmap(
                z=matrix_np.T,
                colorscale='Viridis',
                showscale=(col == 1),
                hovertemplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
            )
            fig.add_trace(heatmap1, row=1, col=col)
            
            # Similarity matrix
            similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
            similarity_np = similarities.cpu().numpy()
            
            heatmap2 = go.Heatmap(
                z=similarity_np,
                colorscale='RdBu',
                zmid=0,
                showscale=(col == 1),
                hovertemplate='Pos1: %{x}<br>Pos2: %{y}<br>Similarity: %{z:.3f}<extra></extra>'
            )
            fig.add_trace(heatmap2, row=2, col=col)
        
        fig.update_layout(
            title="Interactive Positional Encoding Explorer",
            height=600,
            showlegend=False
        )
        
        return fig


class PositionalEncodingVisualizer(EncodingPlotter):
    """Specialized class for advanced positional encoding visualizations."""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
    
    def create_frequency_spectrum_analysis(
        self,
        encodings: Dict[str, Any],
        seq_len: int = 128,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comprehensive frequency spectrum analysis.
        
        Args:
            encodings: Dictionary of encoding instances
            seq_len: Sequence length for analysis
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Frequency analysis figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (enc_name, encoding) in enumerate(encodings.items()):
            color = colors[i % len(colors)]
            
            if enc_name == 'sinusoidal' and hasattr(encoding, 'analyze_frequency_components'):
                freq_analysis = encoding.analyze_frequency_components()
                frequencies = freq_analysis['frequencies'].cpu().numpy()
                wavelengths = freq_analysis['wavelengths'].cpu().numpy()
                
                # Plot frequencies
                axes[0, 0].semilogy(frequencies, 'o-', label=enc_name, color=color)
                
                # Plot wavelengths
                axes[0, 1].semilogy(wavelengths, 's-', label=enc_name, color=color)
                
            elif enc_name == 'rope' and hasattr(encoding, 'analyze_rotation_frequencies'):
                freq_analysis = encoding.analyze_rotation_frequencies()
                frequencies = freq_analysis['frequencies'].cpu().numpy()
                wavelengths = freq_analysis['wavelengths'].cpu().numpy()
                
                axes[0, 0].semilogy(frequencies, '^-', label=enc_name, color=color)
                axes[0, 1].semilogy(wavelengths, 'v-', label=enc_name, color=color)
        
        axes[0, 0].set_title('Frequency Components')
        axes[0, 0].set_xlabel('Dimension Index')
        axes[0, 0].set_ylabel('Frequency (log)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Wavelengths')
        axes[0, 1].set_xlabel('Dimension Index')
        axes[0, 1].set_ylabel('Wavelength (log)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # FFT analysis for learned encodings
        for i, (enc_name, encoding) in enumerate(encodings.items()):
            if enc_name == 'learned' and hasattr(encoding, 'get_embedding_matrix'):
                matrix = encoding.get_embedding_matrix()[:seq_len]
                
                # Compute FFT for sample dimensions
                for dim in range(0, min(matrix.size(1), 4)):
                    signal = matrix[:, dim].cpu().numpy()
                    fft = np.fft.fft(signal)
                    freqs = np.fft.fftfreq(len(signal))
                    magnitude = np.abs(fft)
                    
                    axes[1, 0].plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2],
                                   label=f'{enc_name} Dim {dim}', alpha=0.7)
        
        axes[1, 0].set_title('FFT Analysis (Learned Encoding)')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Periodicity analysis
        for i, (enc_name, encoding) in enumerate(encodings.items()):
            color = colors[i % len(colors)]
            
            # Compute autocorrelation for position similarities
            if hasattr(encoding, 'forward'):
                if enc_name in ['sinusoidal', 'learned']:
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    matrix = enc_output.squeeze(0)
                elif enc_name == 'rope':
                    rope_data = encoding.forward(seq_len, encoding.d_model)
                    matrix = rope_data['cos'].squeeze(0)
                else:
                    continue
                
                # Compute position similarities
                similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
                
                # Autocorrelation of first row (position 0 similarities)
                signal = similarities[0].cpu().numpy()
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                axes[1, 1].plot(autocorr[:min(len(autocorr), 50)], 
                               label=enc_name, color=color, alpha=0.7)
        
        axes[1, 1].set_title('Autocorrelation Analysis')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        if title is None:
            title = 'Frequency Spectrum Analysis'
        fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.export_quality, bbox_inches='tight')
        
        return fig
