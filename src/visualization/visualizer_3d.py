"""3D visualization tools for positional encodings and attention patterns."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import math

from config import VisualizationConfig


class ThreeDVisualizer:
    """3D visualization tools for encoding and attention analysis."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def create_encoding_3d_surface(
        self,
        encoding_matrix: torch.Tensor,
        encoding_name: str = "Positional Encoding",
        max_dims: int = 32,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D surface plot of positional encoding.
        
        Args:
            encoding_matrix: Encoding matrix of shape (seq_len, d_model)
            encoding_name: Name of encoding method
            max_dims: Maximum dimensions to display
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Plotly 3D surface figure
        """
        matrix = encoding_matrix.detach().cpu().numpy()
        seq_len, d_model = matrix.shape
        
        # Limit dimensions for visualization
        display_dims = min(d_model, max_dims)
        display_matrix = matrix[:, :display_dims]
        
        # Create coordinate grids
        x = np.arange(seq_len)
        y = np.arange(display_dims)
        X, Y = np.meshgrid(x, y)
        Z = display_matrix.T
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            hovertemplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(title="Encoding Value")
        )])
        
        fig.update_layout(
            title=title or f"{encoding_name} 3D Surface",
            scene=dict(
                xaxis_title="Position",
                yaxis_title="Dimension",
                zaxis_title="Encoding Value",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_rope_rotation_3d(
        self,
        rope_encoding,
        seq_len: int = 32,
        dim_pairs: Optional[List[int]] = None,
        title: str = "RoPE 3D Rotations",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D visualization of RoPE rotations.
        
        Args:
            rope_encoding: RoPE encoding instance
            seq_len: Sequence length
            dim_pairs: Dimension pairs to visualize
            title: Figure title
            save_path: Path to save
            
        Returns:
            3D RoPE rotation figure
        """
        if dim_pairs is None:
            dim_pairs = [0, 1, 2, 3]
        
        rope_data = rope_encoding.forward(seq_len, rope_encoding.d_model)
        cos_values = rope_data['cos'].squeeze(0).detach().cpu().numpy()
        sin_values = rope_data['sin'].squeeze(0).detach().cpu().numpy()
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                   [{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[f'Dimension Pair {dp}' for dp in dim_pairs[:4]]
        )
        
        for i, dim_pair in enumerate(dim_pairs[:4]):
            if dim_pair < cos_values.shape[1]:
                row = i // 2 + 1
                col = i % 2 + 1
                
                # Get rotation data for this dimension pair
                cos_vals = cos_values[:, dim_pair]
                sin_vals = sin_values[:, dim_pair]
                
                # Create 3D trajectory showing rotation evolution
                positions = np.arange(seq_len)
                
                # Unit vectors rotated by position
                x_vals = cos_vals
                y_vals = sin_vals
                z_vals = positions
                
                # Add trajectory line
                fig.add_trace(
                    go.Scatter3d(
                        x=x_vals, y=y_vals, z=z_vals,
                        mode='lines+markers',
                        line=dict(color='blue', width=4),
                        marker=dict(size=3),
                        name=f'Dim Pair {dim_pair}',
                        hovertemplate='Position: %{z}<br>Cos: %{x:.3f}<br>Sin: %{y:.3f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Add unit circle at each z-level for reference
                if i == 0:  # Only for first subplot to avoid clutter
                    theta = np.linspace(0, 2*np.pi, 50)
                    for pos_sample in range(0, seq_len, seq_len//4):
                        circle_x = np.cos(theta)
                        circle_y = np.sin(theta)
                        circle_z = np.full_like(theta, pos_sample)
                        
                        fig.add_trace(
                            go.Scatter3d(
                                x=circle_x, y=circle_y, z=circle_z,
                                mode='lines',
                                line=dict(color='gray', width=1),
                                showlegend=False,
                                opacity=0.3
                            ),
                            row=row, col=col
                        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_attention_3d_landscape(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        head_idx: int = 0,
        title: str = "Attention 3D Landscape",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D landscape visualization of attention patterns.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings
            head_idx: Attention head index
            title: Figure title
            save_path: Path to save
            
        Returns:
            3D attention landscape figure
        """
        # Extract attention matrix
        if attention_weights.dim() == 4:
            attn_matrix = attention_weights[0, head_idx].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attn_matrix = attention_weights[head_idx].detach().cpu().numpy()
        else:
            attn_matrix = attention_weights.detach().cpu().numpy()
        
        seq_len = attn_matrix.shape[0]
        
        # Create coordinate grids
        x = np.arange(seq_len)
        y = np.arange(seq_len)
        X, Y = np.meshgrid(x, y)
        Z = attn_matrix
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(title="Attention Weight")
        )])
        
        # Add token labels if not too many
        if len(tokens) <= 16:
            # Create custom tick labels
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="Key Position",
                        tickmode='array',
                        tickvals=list(range(len(tokens))),
                        ticktext=tokens
                    ),
                    yaxis=dict(
                        title="Query Position", 
                        tickmode='array',
                        tickvals=list(range(len(tokens))),
                        ticktext=tokens
                    ),
                    zaxis_title="Attention Weight",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            )
        else:
            fig.update_layout(
                scene=dict(
                    xaxis_title="Key Position",
                    yaxis_title="Query Position",
                    zaxis_title="Attention Weight",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            )
        
        fig.update_layout(
            title=title,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_encoding_evolution_3d(
        self,
        encodings_dict: Dict[str, torch.Tensor],
        sequence_lengths: List[int],
        title: str = "Encoding Evolution with Sequence Length",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D visualization showing how encodings change with sequence length.
        
        Args:
            encodings_dict: Dictionary of encoding name to matrices for different lengths
            sequence_lengths: List of sequence lengths
            title: Figure title
            save_path: Path to save
            
        Returns:
            3D encoding evolution figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (enc_name, encoding_matrices) in enumerate(encodings_dict.items()):
            color = colors[i % len(colors)]
            
            # Process each sequence length
            for seq_len, matrix in zip(sequence_lengths, encoding_matrices):
                matrix_np = matrix.detach().cpu().numpy()
                
                # Use first few dimensions for visualization
                for dim in range(min(3, matrix_np.shape[1])):
                    positions = np.arange(seq_len)
                    values = matrix_np[:, dim]
                    
                    # Create 3D line plot
                    fig.add_trace(go.Scatter3d(
                        x=positions,
                        y=np.full_like(positions, dim),
                        z=values,
                        mode='lines+markers',
                        line=dict(color=color, width=3),
                        marker=dict(size=2),
                        name=f'{enc_name} L={seq_len} D={dim}',
                        hovertemplate=f'Encoding: {enc_name}<br>Position: %{{x}}<br>Dimension: %{{y}}<br>Value: %{{z:.3f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Position",
                yaxis_title="Dimension",
                zaxis_title="Encoding Value",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class PositionalEncoding3D:
    """Specialized 3D visualizations for positional encodings."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def create_sinusoidal_wave_3d(
        self,
        sinusoidal_encoding,
        seq_len: int = 64,
        dimensions: Optional[List[int]] = None,
        title: str = "Sinusoidal Encoding 3D Waves",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D wave visualization of sinusoidal encodings.
        
        Args:
            sinusoidal_encoding: Sinusoidal encoding instance
            seq_len: Sequence length
            dimensions: Dimensions to visualize
            title: Figure title
            save_path: Path to save
            
        Returns:
            3D sinusoidal waves figure
        """
        if dimensions is None:
            dimensions = list(range(0, min(sinusoidal_encoding.d_model, 8), 2))
        
        encoding_output = sinusoidal_encoding.forward(seq_len, sinusoidal_encoding.d_model)
        matrix = encoding_output.squeeze(0).detach().cpu().numpy()
        
        fig = go.Figure()
        
        positions = np.arange(seq_len)
        colors = px.colors.qualitative.Set1
        
        for i, dim in enumerate(dimensions):
            if dim < matrix.shape[1]:
                values = matrix[:, dim]
                color = colors[i % len(colors)]
                
                # Create 3D wave
                fig.add_trace(go.Scatter3d(
                    x=positions,
                    y=np.full_like(positions, dim),
                    z=values,
                    mode='lines+markers',
                    line=dict(color=color, width=4),
                    marker=dict(size=3),
                    name=f'Dimension {dim}',
                    hovertemplate=f'Position: %{{x}}<br>Dimension: {dim}<br>Value: %{{z:.3f}}<extra></extra>'
                ))
        
        # Add frequency analysis if available
        if hasattr(sinusoidal_encoding, 'analyze_frequency_components'):
            freq_analysis = sinusoidal_encoding.analyze_frequency_components()
            frequencies = freq_analysis['frequencies'].cpu().numpy()
            
            # Add frequency information as annotation
            freq_text = "Frequencies:<br>" + "<br>".join([
                f"Dim {dim}: {freq:.6f}" for dim, freq in 
                zip(dimensions, frequencies[dimensions]) if dim < len(frequencies)
            ])
            
            fig.add_annotation(
                text=freq_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Position",
                yaxis_title="Dimension",
                zaxis_title="Encoding Value",
                camera=dict(eye=dict(x=1.5, y=0.5, z=1.2))
            ),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_frequency_spectrum_3d(
        self,
        encoding_matrices: Dict[str, torch.Tensor],
        encoding_names: List[str],
        title: str = "3D Frequency Spectrum Comparison",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D frequency spectrum visualization.
        
        Args:
            encoding_matrices: Dictionary of encoding matrices
            encoding_names: Names of encodings
            title: Figure title
            save_path: Path to save
            
        Returns:
            3D frequency spectrum figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (enc_name, matrix) in enumerate(encoding_matrices.items()):
            matrix_np = matrix.detach().cpu().numpy()
            color = colors[i % len(colors)]
            
            # Compute FFT for each dimension
            for dim in range(min(4, matrix_np.shape[1])):  # Limit dimensions
                signal = matrix_np[:, dim]
                fft = np.fft.fft(signal)
                freqs = np.fft.fftfreq(len(signal))
                magnitude = np.abs(fft)
                
                # Only show positive frequencies
                pos_freqs = freqs[:len(freqs)//2]
                pos_magnitude = magnitude[:len(magnitude)//2]
                
                fig.add_trace(go.Scatter3d(
                    x=pos_freqs,
                    y=np.full_like(pos_freqs, dim),
                    z=pos_magnitude,
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(size=2),
                    name=f'{enc_name} Dim {dim}',
                    hovertemplate=f'Encoding: {enc_name}<br>Frequency: %{{x:.3f}}<br>Dimension: {dim}<br>Magnitude: %{{z:.3f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Frequency",
                yaxis_title="Dimension",
                zaxis_title="FFT Magnitude",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_position_similarity_3d(
        self,
        encoding_matrix: torch.Tensor,
        encoding_name: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D visualization of position similarities.
        
        Args:
            encoding_matrix: Encoding matrix
            encoding_name: Name of encoding
            title: Figure title
            save_path: Path to save
            
        Returns:
            3D position similarity figure
        """
        matrix = encoding_matrix.detach().cpu().numpy()
        
        # Compute similarity matrix
        similarities = np.dot(matrix, matrix.T) / matrix.shape[1]
        
        seq_len = similarities.shape[0]
        
        # Create coordinate grids
        x = np.arange(seq_len)
        y = np.arange(seq_len)
        X, Y = np.meshgrid(x, y)
        Z = similarities
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='Position 1: %{x}<br>Position 2: %{y}<br>Similarity: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(title="Cosine Similarity")
        )])
        
        # Add diagonal line for reference (self-similarity)
        diagonal_trace = go.Scatter3d(
            x=list(range(seq_len)),
            y=list(range(seq_len)),
            z=[similarities[i, i] for i in range(seq_len)],
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=4, color='red'),
            name='Self-similarity',
            hovertemplate='Position: %{x}<br>Self-similarity: %{z:.3f}<extra></extra>'
        )
        fig.add_trace(diagonal_trace)
        
        if title is None:
            title = f"{encoding_name} Position Similarity 3D"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Position 1",
                yaxis_title="Position 2",
                zaxis_title="Similarity",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
