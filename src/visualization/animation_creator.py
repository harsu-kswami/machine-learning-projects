"""Animation creation tools for encoding and attention pattern evolution."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import tempfile
import os
from PIL import Image
import io
import base64

from config import VisualizationConfig


class AnimationCreator:
    """Create animated visualizations for positional encodings and attention."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
    
    def create_attention_evolution_animation(
        self,
        layer_attentions: List[torch.Tensor],
        tokens: List[str],
        head_idx: int = 0,
        interval: int = 1000,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Create animation showing attention evolution across layers.
        
        Args:
            layer_attentions: List of attention weights for each layer
            tokens: Token strings
            head_idx: Attention head to animate
            interval: Animation interval in milliseconds
            save_path: Path to save animation
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Prepare data
        matrices = []
        for attention in layer_attentions:
            if attention.dim() == 4:
                matrix = attention[0, head_idx].detach().cpu().numpy()
            elif attention.dim() == 3:
                matrix = attention[head_idx].detach().cpu().numpy()
            else:
                matrix = attention.detach().cpu().numpy()
            matrices.append(matrix)
        
        # Find global min/max for consistent colorbar
        all_values = np.concatenate([m.flatten() for m in matrices])
        vmin, vmax = all_values.min(), all_values.max()
        
        # Initialize plot
        im = ax.imshow(matrices[0], cmap=self.config.colormap_attention,
                      aspect='equal', vmin=vmin, vmax=vmax)
        
        # Setup labels
        if tokens and len(tokens) <= 16:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=self.config.token_label_rotation)
            ax.set_yticklabels(tokens)
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        title = ax.set_title(f'Attention Evolution - Layer 0, Head {head_idx}')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight')
        
        def animate(frame):
            im.set_array(matrices[frame])
            title.set_text(f'Attention Evolution - Layer {frame}, Head {head_idx}')
            return [im, title]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(matrices),
                                     interval=interval, blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        return anim
    
    def create_encoding_parameter_animation(
        self,
        encoding_generator: Callable,
        parameter_name: str,
        parameter_values: List[float],
        seq_len: int = 32,
        interval: int = 500,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Create animation showing how encoding changes with parameter values.
        
        Args:
            encoding_generator: Function that generates encoding given parameter value
            parameter_name: Name of the parameter being varied
            parameter_values: List of parameter values to animate through
            seq_len: Sequence length
            interval: Animation interval in milliseconds
            save_path: Path to save animation
            
        Returns:
            Matplotlib animation object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generate all encoding matrices
        matrices = []
        similarities = []
        
        for param_val in parameter_values:
            encoding = encoding_generator(param_val)
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, encoding.d_model)
                matrix = enc_output.squeeze(0).detach().cpu().numpy()
            else:
                matrix = encoding.detach().cpu().numpy()
            
            matrices.append(matrix)
            
            # Compute similarities
            sim_matrix = np.dot(matrix, matrix.T) / matrix.shape[1]
            similarities.append(sim_matrix)
        
        # Find global limits
        all_enc_values = np.concatenate([m.flatten() for m in matrices])
        enc_vmin, enc_vmax = all_enc_values.min(), all_enc_values.max()
        
        all_sim_values = np.concatenate([s.flatten() for s in similarities])
        sim_vmin, sim_vmax = all_sim_values.min(), all_sim_values.max()
        
        # Initialize plots
        im1 = ax1.imshow(matrices[0].T, cmap=self.config.colormap_encoding,
                        aspect='auto', vmin=enc_vmin, vmax=enc_vmax)
        im2 = ax2.imshow(similarities[0], cmap='coolwarm', aspect='equal',
                        vmin=sim_vmin, vmax=sim_vmax)
        
        # Setup labels
        ax1.set_title('Encoding Pattern')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Dimension')
        
        ax2.set_title('Position Similarities')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Position')
        
        # Colorbars
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Encoding Value')
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Similarity')
        
        # Parameter value text
        param_text = ax1.text(0.02, 0.98, f'{parameter_name}: {parameter_values[0]:.3f}',
                             transform=ax1.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            im1.set_array(matrices[frame].T)
            im2.set_array(similarities[frame])
            param_text.set_text(f'{parameter_name}: {parameter_values[frame]:.3f}')
            return [im1, im2, param_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(parameter_values),
                                     interval=interval, blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        return anim
    
    def create_rope_rotation_animation(
        self,
        rope_encoding,
        seq_len: int = 16,
        dim_pair: int = 0,
        interval: int = 200,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Create animation of RoPE rotations evolving with position.
        
        Args:
            rope_encoding: RoPE encoding instance
            seq_len: Sequence length
            dim_pair: Dimension pair to visualize
            interval: Animation interval in milliseconds
            save_path: Path to save animation
            
        Returns:
            Matplotlib animation object
        """
        rope_data = rope_encoding.forward(seq_len, rope_encoding.d_model)
        cos_values = rope_data['cos'].squeeze(0).detach().cpu().numpy()
        sin_values = rope_data['sin'].squeeze(0).detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Setup plot
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Cosine')
        ax.set_ylabel('Sine')
        ax.set_title(f'RoPE Rotation Animation - Dimension Pair {dim_pair}')
        
        # Draw unit circle
        circle = Circle((0, 0), 1, fill=False, linestyle='--', color='gray', alpha=0.5)
        ax.add_patch(circle)
        
        # Initialize elements
        vector_line, = ax.plot([0, 1], [0, 0], 'b-', linewidth=3, label='Current Vector')
        vector_head = ax.scatter([1], [0], c='blue', s=100, zorder=5)
        
        # Trail of previous positions
        trail_points = ax.scatter([], [], c='lightblue', s=20, alpha=0.6, label='Trail')
        
        # Position text
        pos_text = ax.text(0.02, 0.98, f'Position: 0', transform=ax.transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Angle text
        angle_text = ax.text(0.02, 0.92, f'Angle: 0.000', transform=ax.transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend()
        
        # Store trail data
        trail_x, trail_y = [], []
        
        def animate(frame):
            if dim_pair < cos_values.shape[1]:
                cos_val = cos_values[frame, dim_pair]
                sin_val = sin_values[frame, dim_pair]
                
                # Update vector
                vector_line.set_data([0, cos_val], [0, sin_val])
                vector_head.set_offsets([[cos_val, sin_val]])
                
                # Update trail
                trail_x.append(cos_val)
                trail_y.append(sin_val)
                
                # Keep trail length manageable
                if len(trail_x) > seq_len:
                    trail_x.pop(0)
                    trail_y.pop(0)
                
                trail_points.set_offsets(list(zip(trail_x, trail_y)))
                
                # Update text
                pos_text.set_text(f'Position: {frame}')
                angle = np.arctan2(sin_val, cos_val)
                angle_text.set_text(f'Angle: {angle:.3f} rad')
            
            return [vector_line, vector_head, trail_points, pos_text, angle_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=seq_len,
                                     interval=interval, blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        return anim
    
    def create_interactive_plotly_animation(
        self,
        data_sequence: List[torch.Tensor],
        frame_labels: List[str],
        animation_type: str = "heatmap",
        title: str = "Interactive Animation",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive Plotly animation.
        
        Args:
            data_sequence: Sequence of data tensors to animate
            frame_labels: Labels for each frame
            animation_type: Type of animation ('heatmap', 'surface', 'scatter')
            title: Animation title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure with animation
        """
        # Convert tensors to numpy
        matrices = [data.detach().cpu().numpy() for data in data_sequence]
        
        # Create frames
        frames = []
        
        if animation_type == "heatmap":
            # Find global limits for consistent coloring
            all_values = np.concatenate([m.flatten() for m in matrices])
            zmin, zmax = all_values.min(), all_values.max()
            
            for i, (matrix, label) in enumerate(zip(matrices, frame_labels)):
                frame = go.Frame(
                    data=[go.Heatmap(
                        z=matrix,
                        colorscale='Viridis',
                        zmin=zmin, zmax=zmax,
                        hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>'
                    )],
                    name=str(i),
                    layout=go.Layout(title_text=f"{title} - {label}")
                )
                frames.append(frame)
            
            # Initial frame
            fig = go.Figure(
                data=[go.Heatmap(
                    z=matrices[0],
                    colorscale='Viridis',
                    zmin=zmin, zmax=zmax,
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>'
                )],
                frames=frames
            )
            
        elif animation_type == "surface":
            # 3D surface animation
            for i, (matrix, label) in enumerate(zip(matrices, frame_labels)):
                x = np.arange(matrix.shape[0])
                y = np.arange(matrix.shape[1])
                X, Y = np.meshgrid(x, y)
                
                frame = go.Frame(
                    data=[go.Surface(
                        x=X, y=Y, z=matrix,
                        colorscale='Viridis',
                        hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:.3f}<extra></extra>'
                    )],
                    name=str(i),
                    layout=go.Layout(title_text=f"{title} - {label}")
                )
                frames.append(frame)
            
            # Initial frame
            x = np.arange(matrices[0].shape[0])
            y = np.arange(matrices[0].shape[1])
            X, Y = np.meshgrid(x, y)
            
            fig = go.Figure(
                data=[go.Surface(
                    x=X, y=Y, z=matrices[0],
                    colorscale='Viridis',
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:.3f}<extra></extra>'
                )],
                frames=frames
            )
            
            fig.update_layout(scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Value"
            ))
        
        # Add animation controls
        fig.update_layout(
            title=title,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }],
                        "label": label,
                        "method": "animate"
                    } for f, label in zip(frames, frame_labels)
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class EncodingAnimator(AnimationCreator):
    """Specialized animator for positional encodings."""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
    
    def animate_sequence_length_effects(
        self,
        encoding_generator: Callable,
        sequence_lengths: List[int],
        encoding_name: str = "Encoding",
        interval: int = 1000,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Animate how encoding patterns change with sequence length.
        
        Args:
            encoding_generator: Function to generate encoding for given length
            sequence_lengths: List of sequence lengths
            encoding_name: Name of encoding method
            interval: Animation interval in milliseconds
            save_path: Path to save animation
            
        Returns:
            Sequence length animation
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Generate data for all sequence lengths
        matrices = []
        similarities = []
        first_dim_patterns = []
        
        max_seq_len = max(sequence_lengths)
        
        for seq_len in sequence_lengths:
            encoding = encoding_generator(seq_len)
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, encoding.d_model)
                matrix = enc_output.squeeze(0).detach().cpu().numpy()
            else:
                matrix = encoding.detach().cpu().numpy()
            
            matrices.append(matrix)
            
            # Similarities
            sim_matrix = np.dot(matrix, matrix.T) / matrix.shape[1]
            similarities.append(sim_matrix)
            
            # First dimension pattern
            first_dim_patterns.append(matrix[:, 0])
        
        # Initialize plots
        # Plot 1: Encoding heatmap
        im1 = ax1.imshow(matrices[0].T, cmap=self.config.colormap_encoding, aspect='auto')
        ax1.set_title(f'{encoding_name} Pattern')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Dimension')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot 2: Similarities
        im2 = ax2.imshow(similarities[0], cmap='coolwarm', aspect='equal')
        ax2.set_title('Position Similarities')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Position')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Plot 3: First dimension pattern
        line, = ax3.plot(range(len(first_dim_patterns[0])), first_dim_patterns[0], 'b-', linewidth=2)
        ax3.set_xlim(0, max_seq_len)
        ax3.set_ylim(min(min(p) for p in first_dim_patterns), 
                    max(max(p) for p in first_dim_patterns))
        ax3.set_title('First Dimension Pattern')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Encoding Value')
        ax3.grid(True, alpha=0.3)
        
        # Sequence length text
        seq_len_text = fig.suptitle(f'{encoding_name} - Sequence Length: {sequence_lengths[0]}',
                                   fontsize=16)
        
        def animate(frame):
            seq_len = sequence_lengths[frame]
            matrix = matrices[frame]
            similarity = similarities[frame]
            pattern = first_dim_patterns[frame]
            
            # Update heatmap
            im1.set_array(matrix.T)
            im1.set_extent([0, seq_len, matrix.shape[1], 0])
            
            # Update similarities
            im2.set_array(similarity)
            im2.set_extent([0, seq_len, seq_len, 0])
            
            # Update line plot
            line.set_data(range(len(pattern)), pattern)
            
            # Update title
            seq_len_text.set_text(f'{encoding_name} - Sequence Length: {seq_len}')
            
            return [im1, im2, line, seq_len_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(sequence_lengths),
                                     interval=interval, blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        return anim
    
    def create_encoding_comparison_animation(
        self,
        encoding_generators: Dict[str, Callable],
        parameter_values: List[float],
        parameter_name: str,
        seq_len: int = 32,
        interval: int = 800,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Create animation comparing different encodings as parameter varies.
        
        Args:
            encoding_generators: Dict of encoding name to generator function
            parameter_values: List of parameter values
            parameter_name: Name of parameter being varied
            seq_len: Sequence length
            interval: Animation interval
            save_path: Save path
            
        Returns:
            Comparison animation
        """
        n_encodings = len(encoding_generators)
        fig, axes = plt.subplots(2, n_encodings, figsize=(4*n_encodings, 8))
        
        if n_encodings == 1:
            axes = axes.reshape(-1, 1)
        
        encoding_names = list(encoding_generators.keys())
        
        # Generate all data
        all_matrices = {name: [] for name in encoding_names}
        all_similarities = {name: [] for name in encoding_names}
        
        for param_val in parameter_values:
            for enc_name, generator in encoding_generators.items():
                encoding = generator(param_val)
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    matrix = enc_output.squeeze(0).detach().cpu().numpy()
                else:
                    matrix = encoding.detach().cpu().numpy()
                
                all_matrices[enc_name].append(matrix)
                
                sim_matrix = np.dot(matrix, matrix.T) / matrix.shape[1]
                all_similarities[enc_name].append(sim_matrix)
        
        # Initialize plots
        ims = []
        titles = []
        
        for col, enc_name in enumerate(encoding_names):
            # Encoding pattern
            im1 = axes[0, col].imshow(all_matrices[enc_name][0].T, 
                                     cmap=self.config.colormap_encoding, aspect='auto')
            title1 = axes[0, col].set_title(f'{enc_name.title()} Encoding')
            ims.append(im1)
            titles.append(title1)
            
            if col == 0:
                axes[0, col].set_ylabel('Dimension')
            axes[0, col].set_xlabel('Position')
            
            # Similarities
            im2 = axes[1, col].imshow(all_similarities[enc_name][0], 
                                     cmap='coolwarm', aspect='equal')
            title2 = axes[1, col].set_title(f'{enc_name.title()} Similarities')
            ims.append(im2)
            titles.append(title2)
            
            if col == 0:
                axes[1, col].set_ylabel('Position')
            axes[1, col].set_xlabel('Position')
        
        # Parameter text
        param_text = fig.suptitle(f'{parameter_name}: {parameter_values[0]:.3f}',
                                 fontsize=16)
        
        def animate(frame):
            param_val = parameter_values[frame]
            
            for col, enc_name in enumerate(encoding_names):
                # Update encoding pattern
                ims[col*2].set_array(all_matrices[enc_name][frame].T)
                
                # Update similarities
                ims[col*2 + 1].set_array(all_similarities[enc_name][frame])
            
            # Update parameter text
            param_text.set_text(f'{parameter_name}: {param_val:.3f}')
            
            return ims + [param_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(parameter_values),
                                     interval=interval, blit=False, repeat=True)
        
        plt.tight_layout()
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        return anim
    
    def __del__(self):
        """Clean up temporary directory."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
