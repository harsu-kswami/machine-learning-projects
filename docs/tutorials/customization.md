#!/usr/bin/env python3
"""
Customization Tutorial - Create custom encodings, visualizations, and interfaces
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import math

from config import ModelConfig, VisualizationConfig
from positional_encoding import PositionalEncoding
from visualization import BaseVisualizer
from utils.metrics import EncodingMetrics

class CustomizationTutorial:
    """Tutorial for creating custom components"""
    
    def __init__(self):
        self.custom_components = {}
        self.examples = {}
        
    def create_custom_positional_encoding(self):
        """Create custom positional encoding implementations"""
        
        print("🛠️ Creating Custom Positional Encodings")
        
        class GaussianPositionalEncoding(PositionalEncoding):
            """Gaussian-based positional encoding"""
            
            def __init__(self, d_model: int, max_seq_len: int = 5000, sigma: float = 1.0):
                super().__init__(d_model, max_seq_len)
                self.sigma = sigma
                
            def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
                position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
                
                # Create Gaussian curves with different centers and widths
                encoding = torch.zeros(seq_len, d_model)
                
                for dim in range(d_model):
                    center = (dim / d_model) * seq_len
                    width = self.sigma * (1 + dim / d_model)
                    
                    # Gaussian function
                    encoding[:, dim] = torch.exp(-0.5 * ((position.squeeze() - center) / width) ** 2)
                
                return encoding.unsqueeze(0)
        
        class WaveletPositionalEncoding(PositionalEncoding):
            """Wavelet-based positional encoding"""
            
            def __init__(self, d_model: int, max_seq_len: int = 5000):
                super().__init__(d_model, max_seq_len)
                
            def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
                position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
                
                encoding = torch.zeros(seq_len, d_model)
                
                for dim in range(d_model):
                    # Different wavelet parameters for each dimension
                    scale = 1 + dim * 2
                    translation = dim * seq_len / d_model
                    
                    # Morlet wavelet
                    t = (position.squeeze() - translation) / scale
                    wavelet = torch.exp(-0.5 * t**2) * torch.cos(5 * t)
                    
                    encoding[:, dim] = wavelet
                
                return encoding.unsqueeze(0)
        
        class AdaptivePositionalEncoding(nn.Module):
            """Learnable adaptive positional encoding with constraints"""
            
            def __init__(self, d_model: int, max_seq_len: int = 5000):
                super().__init__()
                self.d_model = d_model
                self.max_seq_len = max_seq_len
                
                # Base sinusoidal encoding
                self.register_buffer('base_encoding', self._create_sinusoidal_base(max_seq_len, d_model))
                
                # Learnable adaptation parameters
                self.adaptation_weights = nn.Parameter(torch.ones(d_model) * 0.1)
                self.position_bias = nn.Parameter(torch.zeros(max_seq_len, d_model))
                
            def _create_sinusoidal_base(self, max_len: int, d_model: int) -> torch.Tensor:
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   (-math.log(10000.0) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                
                return pe
                
            def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
                base = self.base_encoding[:seq_len, :d_model]
                adaptation = self.position_bias[:seq_len, :d_model]
                
                # Combine base encoding with learned adaptation
                encoding = base * (1 + self.adaptation_weights) + adaptation
                
                return encoding.unsqueeze(0)
        
        # Create and test custom encodings
        custom_encodings = {
            'gaussian': GaussianPositionalEncoding(128, sigma=2.0),
            'wavelet': WaveletPositionalEncoding(128),
            'adaptive': AdaptivePositionalEncoding(128)
        }
        
        # Test encodings
        seq_len = 64
        d_model = 128
        
        encoding_results = {}
        
        for name, encoding in custom_encodings.items():
            try:
                output = encoding.forward(seq_len, d_model)
                encoding_results[name] = output
                print(f"  ✅ {name}: shape {output.shape}, range [{output.min():.3f}, {output.max():.3f}]")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        # Visualize custom encodings
        self._visualize_custom_encodings(encoding_results)
        
        self.custom_components['encodings'] = custom_encodings
        return custom_encodings
    
    def _visualize_custom_encodings(self, encoding_results: Dict[str, torch.Tensor]):
        """Visualize custom encodings"""
        
        if not encoding_results:
            return
        
        fig, axes = plt.subplots(2, len(encoding_results), figsize=(5 * len(encoding_results), 10))
        if len(encoding_results) == 1:
            axes = axes.reshape(-1, 1)
        
        for col, (name, encoding) in enumerate(encoding_results.items()):
            matrix = encoding.squeeze(0).detach().cpu().numpy()
            
            # Heatmap
            im1 = axes[0, col].imshow(matrix.T, aspect='auto', cmap='RdBu')
            axes[0, col].set_title(f'{name.title()} Encoding')
            axes[0, col].set_xlabel('Position')
            axes[0, col].set_ylabel('Dimension')
            plt.colorbar(im1, ax=axes[0, col])
            
            # Line plot of first few dimensions
            for dim in range(min(8, matrix.shape[1])):
                axes[1, col].plot(matrix[:, dim], label=f'Dim {dim}', alpha=0.7)
            axes[1, col].set_title(f'{name.title()} - Selected Dimensions')
            axes[1, col].set_xlabel('Position')
            axes[1, col].set_ylabel('Value')
            axes[1, col].legend()
            axes[1, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('custom_encodings_visualization.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved visualization to 'custom_encodings_visualization.png'")
    
    def create_custom_visualizers(self):
        """Create custom visualization components"""
        
        print("🎨 Creating Custom Visualizers")
        
        class CircularAttentionVisualizer(BaseVisualizer):
            """Visualize attention in circular/radial format"""
            
            def __init__(self, config: VisualizationConfig):
                super().__init__(config)
                
            def create_circular_attention_plot(self, 
                                             attention_weights: torch.Tensor,
                                             tokens: List[str]) -> plt.Figure:
                """Create circular attention visualization"""
                
                if attention_weights.dim() == 4:
                    attention_matrix = attention_weights[0, 0].detach().cpu().numpy()
                else:
                    attention_matrix = attention_weights[0].detach().cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                n_tokens = len(tokens)
                angles = np.linspace(0, 2 * np.pi, n_tokens, endpoint=False)
                
                # Plot attention as radial lines
                for i in range(n_tokens):
                    for j in range(n_tokens):
                        if attention_matrix[i, j] > 0.1:  # Threshold for visibility
                            # Draw arc from token i to token j
                            start_angle = angles[i]
                            end_angle = angles[j]
                            
                            # Arc intensity based on attention weight
                            intensity = attention_matrix[i, j]
                            
                            ax.annotate('', 
                                      xy=(end_angle, 1.0), 
                                      xytext=(start_angle, 1.0),
                                      arrowprops=dict(
                                          arrowstyle='->', 
                                          color='red', 
                                          alpha=intensity,
                                          lw=intensity * 3
                                      ))
                
                # Add token labels
                for i, (angle, token) in enumerate(zip(angles, tokens)):
                    ax.text(angle, 1.1, token, 
                           horizontalalignment='center',
                           verticalalignment='center')
                
                ax.set_ylim(0, 1.2)
                ax.set_title('Circular Attention Pattern')
                
                return fig
        
        class DynamicAttentionVisualizer:
            """Create animated attention visualizations"""
            
            def __init__(self, config: VisualizationConfig):
                self.config = config
                
            def create_attention_evolution_gif(self,
                                             layer_attentions: List[torch.Tensor],
                                             tokens: List[str],
                                             output_path: str = 'attention_evolution.gif'):
                """Create GIF showing attention evolution across layers"""
                
                import matplotlib.animation as animation
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Prepare data
                attention_matrices = []
                for layer_attn in layer_attentions:
                    if layer_attn.dim() == 4:
                        matrix = layer_attn[0, 0].detach().cpu().numpy()
                    else:
                        matrix = layer_attn[0].detach().cpu().numpy()
                    attention_matrices.append(matrix)
                
                # Animation function
                def animate(frame):
                    ax.clear()
                    
                    im = ax.imshow(attention_matrices[frame], cmap='viridis')
                    ax.set_title(f'Attention Pattern - Layer {frame}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    
                    if tokens and len(tokens) <= 20:  # Avoid overcrowding
                        ax.set_xticks(range(len(tokens)))
                        ax.set_yticks(range(len(tokens)))
                        ax.set_xticklabels(tokens, rotation=45, ha='right')
                        ax.set_yticklabels(tokens)
                    
                    return [im]
                
                # Create animation
                anim = animation.FuncAnimation(
                    fig, animate, frames=len(attention_matrices), 
                    interval=1000, blit=False, repeat=True
                )
                
                # Save as GIF
                try:
                    anim.save(output_path, writer='pillow', fps=1)
                    print(f"  ✅ Saved animation to '{output_path}'")
                except Exception as e:
                    print(f"  ⚠️ Could not save animation: {e}")
                
                return fig, anim
        
        class StatisticalVisualizer:
            """Create statistical analysis visualizations"""
            
            def __init__(self):
                pass
                
            def create_attention_distribution_plot(self, 
                                                 attention_weights: torch.Tensor) -> plt.Figure:
                """Analyze and visualize attention weight distributions"""
                
                if attention_weights.dim() == 4:
                    # Flatten across batch and heads
                    flat_attention = attention_weights.view(-1).detach().cpu().numpy()
                else:
                    flat_attention = attention_weights.view(-1).detach().cpu().numpy()
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Histogram
                axes[0, 0].hist(flat_attention, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Attention Weight Distribution')
                axes[0, 0].set_xlabel('Attention Weight')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Log histogram
                axes[0, 1].hist(flat_attention[flat_attention > 1e-6], 
                               bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].set_yscale('log')
                axes[0, 1].set_title('Attention Distribution (Log Scale)')
                axes[0, 1].set_xlabel('Attention Weight')
                axes[0, 1].set_ylabel('Frequency (log)')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Q-Q plot against uniform distribution
                from scipy import stats
                uniform_quantiles = np.random.uniform(0, 1, len(flat_attention))
                stats.probplot(flat_attention, dist=stats.uniform, plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot vs Uniform')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Sparsity analysis
                sparsity_thresholds = np.logspace(-4, -1, 20)
                sparsity_ratios = []
                
                for threshold in sparsity_thresholds:
                    ratio = (flat_attention < threshold).mean()
                    sparsity_ratios.append(ratio)
                
                axes[1, 1].semilogx(sparsity_thresholds, sparsity_ratios, 'o-')
                axes[1, 1].set_title('Attention Sparsity Analysis')
                axes[1, 1].set_xlabel('Threshold')
                axes[1, 1].set_ylabel('Fraction Below Threshold')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                return fig
        
        # Create and test custom visualizers
        viz_config = VisualizationConfig()
        custom_visualizers = {
            'circular': CircularAttentionVisualizer(viz_config),
            'dynamic': DynamicAttentionVisualizer(viz_config),
            'statistical': StatisticalVisualizer()
        }
        
        print("  ✅ Created custom visualizers:")
        for name in custom_visualizers:
            print(f"    - {name}")
        
        self.custom_components['visualizers'] = custom_visualizers
        return custom_visualizers
    
    def create_custom_metrics(self):
        """Create custom evaluation metrics"""
        
        print("📊 Creating Custom Metrics")
        
        class AdvancedEncodingMetrics:
            """Advanced metrics for positional encodings"""
            
            def __init__(self):
                pass
            
            def compute_frequency_coherence(self, encoding_matrix: torch.Tensor) -> float:
                """Measure how well encoding preserves frequency relationships"""
                
                # Compute FFT for each dimension
                fft_magnitudes = []
                for dim in range(min(32, encoding_matrix.shape[1])):  # Limit dimensions
                    signal = encoding_matrix[:, dim].cpu().numpy()
                    fft_mag = np.abs(np.fft.fft(signal))
                    fft_magnitudes.append(fft_mag)
                
                fft_matrix = np.array(fft_magnitudes)
                
                # Measure coherence as correlation between adjacent dimensions
                coherences = []
                for i in range(fft_matrix.shape[0] - 1):
                    corr = np.corrcoef(fft_matrix[i], fft_matrix[i+1])[0, 1]
                    if not np.isnan(corr):
                        coherences.append(abs(corr))
                
                return np.mean(coherences) if coherences else 0.0
            
            def compute_position_interpolation_error(self, 
                                                   encoding_matrix: torch.Tensor,
                                                   test_positions: List[float]) -> float:
                """Test how well positions can be interpolated"""
                
                errors = []
                
                for test_pos in test_positions:
                    if 0 <= test_pos <= encoding_matrix.shape[0] - 1:
                        # Get integer positions around test position
                        pos_low = int(np.floor(test_pos))
                        pos_high = int(np.ceil(test_pos))
                        
                        if pos_low == pos_high:
                            continue
                            
                        # Linear interpolation
                        alpha = test_pos - pos_low
                        interpolated = (1 - alpha) * encoding_matrix[pos_low] + alpha * encoding_matrix[pos_high]
                        
                        # Find closest actual encoding
                        distances = torch.norm(encoding_matrix - interpolated.unsqueeze(0), dim=1)
                        closest_pos = torch.argmin(distances).item()
                        
                        # Error is distance from expected position
                        position_error = abs(closest_pos - test_pos)
                        errors.append(position_error)
                
                return np.mean(errors) if errors else float('inf')
            
            def compute_translation_invariance(self, encoding_matrix: torch.Tensor) -> float:
                """Measure translation invariance properties"""
                
                # For relative encodings, nearby positions should have similar relationships
                seq_len = encoding_matrix.shape[0]
                similarities = []
                
                for offset in range(1, min(10, seq_len // 2)):
                    # Compare relationship patterns at different positions
                    pattern_similarities = []
                    
                    for pos in range(seq_len - offset):
                        if pos + offset < seq_len and pos + 2 * offset < seq_len:
                            # Compare relative patterns
                            rel1 = encoding_matrix[pos + offset] - encoding_matrix[pos]
                            rel2 = encoding_matrix[pos + 2 * offset] - encoding_matrix[pos + offset]
                            
                            similarity = F.cosine_similarity(rel1, rel2, dim=0).item()
                            pattern_similarities.append(abs(similarity))
                    
                    if pattern_similarities:
                        similarities.append(np.mean(pattern_similarities))
                
                return np.mean(similarities) if similarities else 0.0
        
        class AttentionQualityMetrics:
            """Specialized metrics for attention quality"""
            
            def __init__(self):
                pass
            
            def compute_attention_locality_score(self, attention_weights: torch.Tensor) -> Dict[str, float]:
                """Measure local vs global attention tendencies"""
                
                if attention_weights.dim() == 4:
                    attention_matrix = attention_weights[0, 0]  # First batch, first head
                else:
                    attention_matrix = attention_weights[0]
                
                seq_len = attention_matrix.shape[0]
                
                # Define local windows of different sizes
                local_scores = {}
                
                for window_size in [3, 5, 7]:
                    if window_size >= seq_len:
                        continue
                    
                    local_attention_sum = 0
                    total_queries = 0
                    
                    for query_pos in range(seq_len):
                        start_pos = max(0, query_pos - window_size // 2)
                        end_pos = min(seq_len, query_pos + window_size // 2 + 1)
                        
                        local_sum = attention_matrix[query_pos, start_pos:end_pos].sum().item()
                        local_attention_sum += local_sum
                        total_queries += 1
                    
                    local_scores[f'window_{window_size}'] = local_attention_sum / total_queries
                
                return local_scores
            
            def compute_attention_diversity_index(self, attention_weights: torch.Tensor) -> float:
                """Measure diversity of attention patterns across heads"""
                
                if attention_weights.dim() != 4:
                    return 0.0
                
                batch_size, n_heads, seq_len, _ = attention_weights.shape
                
                # Flatten attention matrices for each head
                head_patterns = attention_weights[0].view(n_heads, -1)  # First batch
                
                # Compute pairwise distances between heads
                distances = []
                for i in range(n_heads):
                    for j in range(i + 1, n_heads):
                        dist = torch.norm(head_patterns[i] - head_patterns[j]).item()
                        distances.append(dist)
                
                # Diversity is mean distance
                return np.mean(distances) if distances else 0.0
        
        # Create and test custom metrics
        custom_metrics = {
            'advanced_encoding': AdvancedEncodingMetrics(),
            'attention_quality': AttentionQualityMetrics()
        }
        
        print("  ✅ Created custom metrics:")
        for name in custom_metrics:
            print(f"    - {name}")
        
        self.custom_components['metrics'] = custom_metrics
        return custom_metrics
    
    def create_custom_interface_components(self):
        """Create custom interface components"""
        
        print("🖥️ Creating Custom Interface Components")
        
        class CustomDashboard:
            """Custom dashboard with specialized controls"""
            
            def __init__(self):
                self.state = {}
                
            def create_encoding_comparison_widget(self):
                """Widget for comparing multiple encodings simultaneously"""
                
                import ipywidgets as widgets
                from IPython.display import display
                
                # Multi-select for encodings
                encoding_selector = widgets.SelectMultiple(
                    options=['sinusoidal', 'learned', 'rope', 'custom_gaussian'],
                    value=['sinusoidal', 'rope'],
                    description='Encodings:',
                    rows=4
                )
                
                # Parameter controls
                seq_len_slider = widgets.IntSlider(
                    value=64, min=16, max=256, step=16,
                    description='Seq Length:'
                )
                
                d_model_dropdown = widgets.Dropdown(
                    options=[64, 128, 256, 512],
                    value=128,
                    description='D Model:'
                )
                
                # Analysis options
                analysis_options = widgets.SelectMultiple(
                    options=['Position Similarity', 'Quality Metrics', 'Frequency Analysis', 'Custom Metrics'],
                    value=['Position Similarity', 'Quality Metrics'],
                    description='Analyses:'
                )
                
                # Run button
                run_button = widgets.Button(
                    description='🔍 Run Comparison',
                    button_style='primary'
                )
                
                # Output area
                output_area = widgets.Output()
                
                def on_run_clicked(b):
                    with output_area:
                        output_area.clear_output()
                        print("Running custom comparison...")
                        
                        selected_encodings = list(encoding_selector.value)
                        selected_analyses = list(analysis_options.value)
                        
                        print(f"Encodings: {selected_encodings}")
                        print(f"Sequence Length: {seq_len_slider.value}")
                        print(f"Model Dimension: {d_model_dropdown.value}")
                        print(f"Analyses: {selected_analyses}")
                        
                        # Here you would run the actual comparison
                        # For demo, just show the configuration
                        
                run_button.on_click(on_run_clicked)
                
                # Layout
                controls = widgets.VBox([
                    widgets.HTML("<h3>🎛️ Comparison Controls</h3>"),
                    encoding_selector,
                    seq_len_slider,
                    d_model_dropdown,
                    analysis_options,
                    run_button
                ])
                
                dashboard = widgets.HBox([controls, output_area])
                
                return dashboard
        
        class ExperimentTracker:
            """Track and manage experiments"""
            
            def __init__(self):
                self.experiments = []
                self.results_database = {}
                
            def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
                """Create new experiment with unique ID"""
                
                import uuid
                from datetime import datetime
                
                experiment_id = str(uuid.uuid4())[:8]
                
                experiment = {
                    'id': experiment_id,
                    'name': name,
                    'config': config,
                    'created_at': datetime.now().isoformat(),
                    'status': 'created',
                    'results': {}
                }
                
                self.experiments.append(experiment)
                self.results_database[experiment_id] = experiment
                
                print(f"  ✅ Created experiment '{name}' with ID: {experiment_id}")
                return experiment_id
            
            def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
                """Execute experiment and store results"""
                
                if experiment_id not in self.results_database:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                experiment = self.results_database[experiment_id]
                experiment['status'] = 'running'
                
                try:
                    # Run experiment based on config
                    config = experiment['config']
                    
                    # This is a placeholder - implement actual experiment logic
                    results = {
                        'success': True,
                        'metrics': {'accuracy': 0.95, 'loss': 0.05},
                        'execution_time': 1.23
                    }
                    
                    experiment['results'] = results
                    experiment['status'] = 'completed'
                    
                    print(f"  ✅ Completed experiment {experiment_id}")
                    return results
                
                except Exception as e:
                    experiment['status'] = 'failed'
                    experiment['error'] = str(e)
                    print(f"  ❌ Experiment {experiment_id} failed: {e}")
                    return {'success': False, 'error': str(e)}
            
            def get_experiment_summary(self) -> Dict[str, Any]:
                """Get summary of all experiments"""
                
                summary = {
                    'total_experiments': len(self.experiments),
                    'completed': len([e for e in self.experiments if e['status'] == 'completed']),
                    'failed': len([e for e in self.experiments if e['status'] == 'failed']),
                    'running': len([e for e in self.experiments if e['status'] == 'running'])
                }
                
                return summary
        
        # Create and test custom interface components
        custom_interfaces = {
            'dashboard': CustomDashboard(),
            'experiment_tracker': ExperimentTracker()
        }
        
        print("  ✅ Created custom interface components:")
        for name in custom_interfaces:
            print(f"    - {name}")
        
        # Demo experiment tracker
        tracker = custom_interfaces['experiment_tracker']
        
        exp_id = tracker.create_experiment(
            "Custom Encoding Test",
            {'encoding': 'gaussian', 'seq_len': 64, 'd_model': 128}
        )
        
        results = tracker.run_experiment(exp_id)
        summary = tracker.get_experiment_summary()
        
        print(f"  Experiment results: {results}")
        print(f"  Tracker summary: {summary}")
        
        self.custom_components['interfaces'] = custom_interfaces
        return custom_interfaces
    
    def integration_example(self):
        """Example of integrating all custom components"""
        
        print("🔧 Integration Example")
        
        # Use custom encoding
        if 'encodings' in self.custom_components:
            gaussian_encoding = self.custom_components['encodings']['gaussian']
            
            # Generate encoding
            seq_len, d_model = 32, 128
            encoding_output = gaussian_encoding.forward(seq_len, d_model)
            
            print(f"  Custom encoding shape: {encoding_output.shape}")
            
            # Use custom metrics
            if 'metrics' in self.custom_components:
                advanced_metrics = self.custom_components['metrics']['advanced_encoding']
                
                encoding_matrix = encoding_output.squeeze(0)
                
                # Compute custom metrics
                coherence = advanced_metrics.compute_frequency_coherence(encoding_matrix)
                interpolation_error = advanced_metrics.compute_position_interpolation_error(
                    encoding_matrix, [1.5, 2.7, 15.3]
                )
                
                print(f"  Frequency coherence: {coherence:.3f}")
                print(f"  Interpolation error: {interpolation_error:.3f}")
            
            # Use custom visualizer
            if 'visualizers' in self.custom_components:
                statistical_viz = self.custom_components['visualizers']['statistical']
                
                # Create fake attention weights for demo
                fake_attention = torch.softmax(torch.randn(1, 4, seq_len, seq_len), dim=-1)
                
                fig = statistical_viz.create_attention_distribution_plot(fake_attention)
                plt.savefig('custom_statistical_analysis.png', dpi=300, bbox_inches='tight')
                print("  ✅ Saved statistical analysis to 'custom_statistical_analysis.png'")
            
            # Use experiment tracker
            if 'interfaces' in self.custom_components:
                tracker = self.custom_components['interfaces']['experiment_tracker']
                
                integration_exp = tracker.create_experiment(
                    "Custom Integration Test",
                    {
                        'encoding': 'gaussian',
                        'metrics': ['frequency_coherence', 'interpolation_error'],
                        'sequence_length': seq_len,
                        'd_model': d_model
                    }
                )
                
                # Simulate running experiment
                integration_results = tracker.run_experiment(integration_exp)
                print(f"  Integration experiment results: {integration_results}")
        
        print("  ✅ Integration example completed")
    
    def save_custom_components(self):
        """Save custom components for reuse"""
        
        print("💾 Saving Custom Components")
        
        import pickle
        from pathlib import Path
        
        # Create save directory
        save_dir = Path('custom_components')
        save_dir.mkdir(exist_ok=True)
        
        # Save configurations and results
        for component_type, components in self.custom_components.items():
            try:
                if component_type == 'encodings':
                    # Save encoding configurations
                    config_path = save_dir / f'{component_type}_configs.json'
                    configs = {}
                    for name, encoding in components.items():
                        if hasattr(encoding, '__dict__'):
                            configs[name] = {
                                'class': encoding.__class__.__name__,
                                'parameters': {k: v for k, v in encoding.__dict__.items() 
                                             if not k.startswith('_') and not callable(v)}
                            }
                    
                    import json
                    with open(config_path, 'w') as f:
                        json.dump(configs, f, indent=2, default=str)
                    
                    print(f"  ✅ Saved {component_type} configurations")
                
                elif component_type == 'interfaces':
                    # Save interface states
                    state_path = save_dir / f'{component_type}_states.json'
                    states = {}
                    
                    for name, interface in components.items():
                        if hasattr(interface, '__dict__'):
                            states[name] = {k: v for k, v in interface.__dict__.items() 
                                          if not callable(v)}
                    
                    with open(state_path, 'w') as f:
                        json.dump(states, f, indent=2, default=str)
                    
                    print(f"  ✅ Saved {component_type} states")
                
            except Exception as e:
                print(f"  ⚠️ Could not save {component_type}: {e}")
        
        print(f"  Custom components saved to {save_dir}")
        return save_dir

def main():
    """Run the customization tutorial"""
    
    print("🎨 Customization Tutorial")
    print("=" * 40)
    
    tutorial = CustomizationTutorial()
    
    try:
        # 1. Create custom positional encodings
        custom_encodings = tutorial.create_custom_positional_encoding()
        
        # 2. Create custom visualizers
        custom_visualizers = tutorial.create_custom_visualizers()
        
        # 3. Create custom metrics
        custom_metrics = tutorial.create_custom_metrics()
        
        # 4. Create custom interface components
        custom_interfaces = tutorial.create_custom_interface_components()
        
        # 5. Integration example
        tutorial.integration_example()
        
        # 6. Save components
        save_dir = tutorial.save_custom_components()
        
        print("\n🎉 Customization Tutorial Completed!")
        print("\nCreated custom components:")
        print(f"  - {len(custom_encodings)} custom encodings")
        print(f"  - {len(custom_visualizers)} custom visualizers")
        print(f"  - {len(custom_metrics)} custom metrics")
        print(f"  - {len(custom_interfaces)} custom interfaces")
        
        print("\nGenerated files:")
        print("  - custom_encodings_visualization.png")
        print("  - custom_statistical_analysis.png")
        print(f"  - {save_dir}/ (component configurations)")
        
        print("\n🚀 Next Steps:")
        print("  - Modify the custom components for your specific needs")
        print("  - Integrate custom components into your workflow")
        print("  - Create additional specialized visualizations")
        print("  - Build domain-specific metrics and encodings")
        
    except Exception as e:
        print(f"❌ Customization tutorial failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
