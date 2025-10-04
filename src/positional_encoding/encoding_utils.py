"""Utility functions for positional encoding analysis and comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math

from config import ModelConfig
from .absolute_encoding import SinusoidalEncoding, LearnedPositionalEncoding
from .relative_encoding import RelativePositionalEncoding, T5RelativeEncoding
from .rope_encoding import RoPEEncoding


def get_positional_encoding(config: ModelConfig) -> nn.Module:
    """Factory function to create positional encoding based on config.
    
    Args:
        config: Model configuration
        
    Returns:
        Positional encoding module
    """
    encoding_type = config.encoding_type.lower()
    
    if encoding_type == 'sinusoidal':
        return SinusoidalEncoding(config)
    elif encoding_type == 'learned':
        return LearnedPositionalEncoding(config)
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(config)
    elif encoding_type == 't5_relative':
        return T5RelativeEncoding(config)
    elif encoding_type == 'rope':
        return RoPEEncoding(config)
    else:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}")


def compare_encodings(
    config: ModelConfig,
    encoding_types: List[str],
    seq_len: int = 64,
    metrics: List[str] = ['similarity', 'distance', 'frequency']
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compare different positional encoding methods.
    
    Args:
        config: Model configuration
        encoding_types: List of encoding types to compare
        seq_len: Sequence length for comparison
        metrics: List of metrics to compute
        
    Returns:
        Comparison results dictionary
    """
    results = {}
    encodings = {}
    
    # Create encoding instances
    for enc_type in encoding_types:
        temp_config = ModelConfig(**config.__dict__)
        temp_config.encoding_type = enc_type
        encoding = get_positional_encoding(temp_config)
        encodings[enc_type] = encoding
    
    # Extract encoding representations
    encoding_matrices = {}
    for enc_type, encoding in encodings.items():
        if enc_type in ['sinusoidal', 'learned']:
            enc_output = encoding.forward(seq_len, config.d_model)
            encoding_matrices[enc_type] = enc_output.squeeze(0)  # (seq_len, d_model)
        elif enc_type in ['relative', 't5_relative']:
            # For relative encodings, we'll use a simplified representation
            if hasattr(encoding, 'relative_positions_keys'):
                # Standard relative encoding
                rel_data = encoding.forward(seq_len, config.d_model)
                # Use relative keys as representation
                encoding_matrices[enc_type] = rel_data['relative_keys'].mean(dim=1)  # Average over positions
            else:
                # T5 relative encoding - create a representative matrix
                bias = encoding.forward(seq_len, config.d_model)  # (1, n_heads, seq_len, seq_len)
                # Convert to position representation by averaging
                encoding_matrices[enc_type] = bias.mean(dim=1).squeeze(0).mean(dim=0, keepdim=True).repeat(seq_len, 1)
        elif enc_type == 'rope':
            rope_data = encoding.forward(seq_len, config.d_model)
            # Combine cos and sin as representation
            cos_sin = torch.cat([rope_data['cos'], rope_data['sin']], dim=-1)
            encoding_matrices[enc_type] = cos_sin.squeeze(0)  # (seq_len, 2*d_model)
    
    # Compute comparison metrics
    for metric in metrics:
        results[metric] = {}
        
        if metric == 'similarity':
            results[metric] = _compute_similarity_metrics(encoding_matrices)
        elif metric == 'distance':
            results[metric] = _compute_distance_metrics(encoding_matrices)
        elif metric == 'frequency':
            results[metric] = _compute_frequency_metrics(encodings, seq_len)
        elif metric == 'extrapolation':
            results[metric] = _compute_extrapolation_metrics(encodings, seq_len)
    
    return results


def _compute_similarity_metrics(encoding_matrices: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute similarity metrics between encoding methods."""
    similarities = {}
    
    encoding_names = list(encoding_matrices.keys())
    
    for i, enc1 in enumerate(encoding_names):
        for j, enc2 in enumerate(encoding_names[i:], i):
            if i == j:
                continue
            
            matrix1 = encoding_matrices[enc1]
            matrix2 = encoding_matrices[enc2]
            
            # Align dimensions if different
            min_dim = min(matrix1.size(-1), matrix2.size(-1))
            matrix1_aligned = matrix1[:, :min_dim]
            matrix2_aligned = matrix2[:, :min_dim]
            
            # Compute position-wise cosine similarity
            position_similarities = F.cosine_similarity(matrix1_aligned, matrix2_aligned, dim=-1)
            
            # Compute overall similarity matrices
            sim1 = torch.mm(matrix1_aligned, matrix1_aligned.t()) / min_dim
            sim2 = torch.mm(matrix2_aligned, matrix2_aligned.t()) / min_dim
            
            pair_key = f"{enc1}_vs_{enc2}"
            similarities[pair_key] = {
                'position_wise_similarity': position_similarities,
                'mean_position_similarity': position_similarities.mean(),
                'similarity_matrix_correlation': torch.corrcoef(torch.stack([
                    sim1.flatten(), sim2.flatten()
                ]))[0, 1],
                'pattern_difference': (sim1 - sim2).abs().mean()
            }
    
    return similarities


def _compute_distance_metrics(encoding_matrices: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute distance-based metrics between encoding methods."""
    distances = {}
    
    for enc_name, matrix in encoding_matrices.items():
        # Compute pairwise distances within each encoding
        pairwise_distances = torch.cdist(matrix, matrix, p=2)
        
        # Analyze distance patterns
        seq_len = matrix.size(0)
        position_differences = torch.abs(
            torch.arange(seq_len).float().unsqueeze(0) - 
            torch.arange(seq_len).float().unsqueeze(1)
        )
        
        # Flatten for correlation analysis
        flat_distances = pairwise_distances.flatten()
        flat_pos_diffs = position_differences.flatten()
        
        # Remove diagonal (distance = 0, pos_diff = 0)
        mask = flat_pos_diffs != 0
        
        if mask.sum() > 1:
            correlation = torch.corrcoef(torch.stack([
                flat_distances[mask], flat_pos_diffs[mask]
            ]))[0, 1]
        else:
            correlation = torch.tensor(0.0)
        
        distances[enc_name] = {
            'pairwise_distances': pairwise_distances,
            'distance_position_correlation': correlation,
            'mean_distance': pairwise_distances.mean(),
            'distance_std': pairwise_distances.std()
        }
    
    return distances


def _compute_frequency_metrics(encodings: Dict[str, nn.Module], seq_len: int) -> Dict[str, torch.Tensor]:
    """Compute frequency-based metrics for different encodings."""
    frequencies = {}
    
    for enc_name, encoding in encodings.items():
        if enc_name == 'sinusoidal':
            freq_analysis = encoding.analyze_frequency_components()
            frequencies[enc_name] = freq_analysis
        elif enc_name == 'rope':
            freq_analysis = encoding.analyze_rotation_frequencies()
            frequencies[enc_name] = freq_analysis
        elif enc_name in ['learned']:
            # For learned encodings, analyze spectral properties
            embedding_matrix = encoding.get_embedding_matrix()[:seq_len]
            
            # Compute FFT for each dimension
            fft_results = {}
            for dim in range(min(16, embedding_matrix.size(1))):  # Sample dimensions
                signal = embedding_matrix[:, dim].cpu().numpy()
                fft = np.fft.fft(signal)
                fft_results[dim] = {
                    'magnitude': torch.tensor(np.abs(fft)),
                    'dominant_frequency': torch.tensor(np.argmax(np.abs(fft[1:seq_len//2])) + 1)
                }
            
            frequencies[enc_name] = {
                'fft_analysis': fft_results,
                'spectral_centroid': _compute_spectral_centroid(embedding_matrix)
            }
        else:
            # For relative encodings, analyze bias patterns
            frequencies[enc_name] = {'type': 'relative', 'analysis': 'bias_based'}
    
    return frequencies


def _compute_spectral_centroid(matrix: torch.Tensor) -> torch.Tensor:
    """Compute spectral centroid for learned embeddings."""
    centroids = []
    
    for dim in range(min(16, matrix.size(1))):
        signal = matrix[:, dim].cpu().numpy()
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft)
        
        freqs = np.fft.fftfreq(len(signal))
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        centroids.append(abs(centroid))
    
    return torch.tensor(centroids)


def _compute_extrapolation_metrics(encodings: Dict[str, nn.Module], base_seq_len: int) -> Dict[str, Dict]:
    """Compute extrapolation capabilities of different encodings."""
    extrapolation = {}
    test_lengths = [base_seq_len * 2, base_seq_len * 4]
    
    for enc_name, encoding in encodings.items():
        if enc_name in ['sinusoidal', 'rope']:
            # These can extrapolate naturally
            extrap_results = {}
            
            for test_len in test_lengths:
                if enc_name == 'sinusoidal':
                    base_encoding = encoding.forward(base_seq_len, encoding.d_model)
                    extended_encoding = encoding.forward(test_len, encoding.d_model)
                    
                    # Compare overlapping region
                    overlap_similarity = F.cosine_similarity(
                        base_encoding.squeeze(0),
                        extended_encoding.squeeze(0)[:base_seq_len],
                        dim=-1
                    ).mean()
                    
                elif enc_name == 'rope':
                    base_data = encoding.forward(base_seq_len, encoding.d_model)
                    extended_data = encoding.forward(test_len, encoding.d_model)
                    
                    # Compare rotation patterns in overlapping region
                    base_cos = base_data['cos'].squeeze(0)
                    extended_cos = extended_data['cos'].squeeze(0)[:base_seq_len]
                    
                    overlap_similarity = F.cosine_similarity(
                        base_cos, extended_cos, dim=-1
                    ).mean()
                
                extrap_results[f'length_{test_len}'] = {
                    'overlap_similarity': overlap_similarity,
                    'extrapolation_factor': test_len / base_seq_len
                }
            
            extrapolation[enc_name] = {
                'can_extrapolate': True,
                'results': extrap_results
            }
        else:
            extrapolation[enc_name] = {
                'can_extrapolate': False,
                'reason': 'Limited by training sequence length'
            }
    
    return extrapolation


def visualize_encoding_patterns(
    encodings: Dict[str, nn.Module],
    seq_len: int = 64,
    save_path: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """Create visualizations for different positional encodings.
    
    Args:
        encodings: Dictionary of encoding name to module
        seq_len: Sequence length to visualize
        save_path: Optional path to save figures
        
    Returns:
        Dictionary of figures
    """
    figures = {}
    
    # 1. Encoding pattern heatmaps
    fig, axes = plt.subplots(2, len(encodings), figsize=(4*len(encodings), 8))
    if len(encodings) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (enc_name, encoding) in enumerate(encodings.items()):
        # Get encoding matrix
        if enc_name in ['sinusoidal', 'learned']:
            enc_output = encoding.forward(seq_len, encoding.d_model)
            matrix = enc_output.squeeze(0)  # (seq_len, d_model)
        elif enc_name == 'rope':
            rope_data = encoding.forward(seq_len, encoding.d_model)
            matrix = rope_data['cos'].squeeze(0)  # Use cos component
        else:
            continue  # Skip relative encodings for heatmap
        
        # Plot encoding heatmap
        im1 = axes[0, idx].imshow(matrix.T.cpu().numpy(), aspect='auto', cmap='viridis')
        axes[0, idx].set_title(f'{enc_name.title()} Encodings')
        axes[0, idx].set_xlabel('Position')
        axes[0, idx].set_ylabel('Dimension')
        plt.colorbar(im1, ax=axes[0, idx])
        
        # Plot similarity matrix
        similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
        im2 = axes[1, idx].imshow(similarities.cpu().numpy(), cmap='coolwarm')
        axes[1, idx].set_title(f'{enc_name.title()} Position Similarities')
        axes[1, idx].set_xlabel('Position')
        axes[1, idx].set_ylabel('Position')
        plt.colorbar(im2, ax=axes[1, idx])
    
    plt.tight_layout()
    figures['encoding_patterns'] = fig
    
    if save_path:
        fig.savefig(f"{save_path}/encoding_patterns.png", dpi=300, bbox_inches='tight')
    
    # 2. Frequency analysis (for applicable encodings)
    freq_fig, freq_axes = plt.subplots(1, 1, figsize=(10, 6))
    
    for enc_name, encoding in encodings.items():
        if enc_name == 'sinusoidal':
            freq_analysis = encoding.analyze_frequency_components()
            frequencies = freq_analysis['frequencies'].cpu().numpy()
            freq_axes.semilogy(frequencies, 'o-', label=f'{enc_name} frequencies')
        elif enc_name == 'rope':
            freq_analysis = encoding.analyze_rotation_frequencies()
            frequencies = freq_analysis['frequencies'].cpu().numpy()
            freq_axes.semilogy(frequencies, 's-', label=f'{enc_name} frequencies')
    
    freq_axes.set_xlabel('Dimension Pair')
    freq_axes.set_ylabel('Frequency')
    freq_axes.set_title('Frequency Analysis of Positional Encodings')
    freq_axes.legend()
    freq_axes.grid(True, alpha=0.3)
    
    figures['frequency_analysis'] = freq_fig
    
    if save_path:
        freq_fig.savefig(f"{save_path}/frequency_analysis.png", dpi=300, bbox_inches='tight')
    
    return figures


def analyze_sequence_length_effects(
    encodings: Dict[str, nn.Module],
    sequence_lengths: List[int],
    metrics: List[str] = ['similarity_decay', 'extrapolation_quality']
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Analyze how different encodings handle varying sequence lengths.
    
    Args:
        encodings: Dictionary of encoding modules
        sequence_lengths: List of sequence lengths to test
        metrics: List of metrics to compute
        
    Returns:
        Analysis results dictionary
    """
    results = {}
    
    for enc_name, encoding in encodings.items():
        enc_results = {}
        
        for metric in metrics:
            if metric == 'similarity_decay':
                enc_results[metric] = _analyze_similarity_decay(encoding, sequence_lengths)
            elif metric == 'extrapolation_quality':
                enc_results[metric] = _analyze_extrapolation_quality(encoding, sequence_lengths)
            elif metric == 'computational_cost':
                enc_results[metric] = _analyze_computational_cost(encoding, sequence_lengths)
        
        results[enc_name] = enc_results
    
    return results


def _analyze_similarity_decay(encoding: nn.Module, sequence_lengths: List[int]) -> Dict[str, torch.Tensor]:
    """Analyze how position similarities decay with distance across sequence lengths."""
    decay_analysis = {}
    
    for seq_len in sequence_lengths:
        if hasattr(encoding, 'forward'):
            if hasattr(encoding, 'encoding_type') and encoding.config.encoding_type in ['sinusoidal', 'learned']:
                enc_output = encoding.forward(seq_len, encoding.d_model)
                matrix = enc_output.squeeze(0)
            elif hasattr(encoding, 'config') and encoding.config.encoding_type == 'rope':
                rope_data = encoding.forward(seq_len, encoding.d_model)
                matrix = torch.cat([rope_data['cos'], rope_data['sin']], dim=-1).squeeze(0)
            else:
                continue
            
            # Compute similarities
            similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
            
            # Analyze decay pattern
            distances = torch.arange(seq_len).float()
            decay_pattern = []
            
            for dist in range(1, min(seq_len, 32)):  # Analyze up to distance 32
                mask = torch.abs(
                    torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
                ) == dist
                
                if mask.sum() > 0:
                    avg_similarity = similarities[mask].mean()
                    decay_pattern.append(avg_similarity)
                else:
                    decay_pattern.append(torch.tensor(0.0))
            
            decay_analysis[f'seq_len_{seq_len}'] = {
                'decay_pattern': torch.stack(decay_pattern),
                'distances': torch.arange(1, len(decay_pattern) + 1),
                'max_similarity': similarities.max(),
                'min_similarity': similarities.min()
            }
    
    return decay_analysis


def _analyze_extrapolation_quality(encoding: nn.Module, sequence_lengths: List[int]) -> Dict[str, torch.Tensor]:
    """Analyze extrapolation quality for different sequence lengths."""
    if not hasattr(encoding, 'config'):
        return {'error': 'Encoding module missing config attribute'}
    
    enc_type = encoding.config.encoding_type
    
    if enc_type not in ['sinusoidal', 'rope']:
        return {'can_extrapolate': False}
    
    base_length = min(sequence_lengths)
    extrapolation_results = {}
    
    for seq_len in sequence_lengths:
        if seq_len <= base_length:
            continue
        
        # Compare patterns in overlapping region
        if enc_type == 'sinusoidal':
            base_enc = encoding.forward(base_length, encoding.d_model).squeeze(0)
            extended_enc = encoding.forward(seq_len, encoding.d_model).squeeze(0)
            
            overlap_similarity = F.cosine_similarity(
                base_enc, extended_enc[:base_length], dim=-1
            ).mean()
        
        elif enc_type == 'rope':
            base_data = encoding.forward(base_length, encoding.d_model)
            extended_data = encoding.forward(seq_len, encoding.d_model)
            
            base_cos = base_data['cos'].squeeze(0)
            extended_cos = extended_data['cos'].squeeze(0)[:base_length]
            
            overlap_similarity = F.cosine_similarity(
                base_cos, extended_cos, dim=-1
            ).mean()
        
        extrapolation_results[f'seq_len_{seq_len}'] = {
            'overlap_similarity': overlap_similarity,
            'extrapolation_factor': seq_len / base_length
        }
    
    return extrapolation_results


def _analyze_computational_cost(encoding: nn.Module, sequence_lengths: List[int]) -> Dict[str, float]:
    """Analyze computational cost scaling with sequence length."""
    import time
    
    cost_analysis = {}
    
    for seq_len in sequence_lengths:
        # Measure encoding computation time
        start_time = time.time()
        
        try:
            if hasattr(encoding, 'config'):
                encoding.forward(seq_len, encoding.d_model)
            else:
                # Fallback for simple encodings
                encoding.forward(seq_len, 512)  # Assume reasonable dimension
            
            end_time = time.time()
            computation_time = end_time - start_time
            
        except Exception as e:
            computation_time = float('inf')  # Mark as failed
        
        cost_analysis[f'seq_len_{seq_len}'] = computation_time
    
    return cost_analysis


def create_encoding_comparison_report(
    config: ModelConfig,
    encoding_types: List[str],
    sequence_lengths: List[int] = [16, 32, 64, 128],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a comprehensive comparison report of positional encodings.
    
    Args:
        config: Model configuration
        encoding_types: List of encoding types to compare
        sequence_lengths: List of sequence lengths to analyze
        output_path: Optional path to save the report
        
    Returns:
        Comprehensive comparison report
    """
    report = {
        'config': config.__dict__,
        'encoding_types': encoding_types,
        'sequence_lengths': sequence_lengths,
        'comparisons': {},
        'analysis': {},
        'recommendations': {}
    }
    
    # Run comprehensive comparison
    comparison_results = compare_encodings(
        config, encoding_types, max(sequence_lengths),
        metrics=['similarity', 'distance', 'frequency', 'extrapolation']
    )
    report['comparisons'] = comparison_results
    
    # Analyze sequence length effects
    encodings = {enc_type: get_positional_encoding(
        ModelConfig(**{**config.__dict__, 'encoding_type': enc_type})
    ) for enc_type in encoding_types}
    
    length_analysis = analyze_sequence_length_effects(
        encodings, sequence_lengths,
        metrics=['similarity_decay', 'extrapolation_quality', 'computational_cost']
    )
    report['analysis'] = length_analysis
    
    # Generate recommendations
    recommendations = _generate_encoding_recommendations(
        comparison_results, length_analysis, config
    )
    report['recommendations'] = recommendations
    
    # Save report if path provided
    if output_path:
        torch.save(report, f"{output_path}/encoding_comparison_report.pt")
    
    return report


def _generate_encoding_recommendations(
    comparison_results: Dict,
    length_analysis: Dict,
    config: ModelConfig
) -> Dict[str, str]:
    """Generate recommendations based on analysis results."""
    recommendations = {}
    
    # Analyze results and provide recommendations
    if 'extrapolation' in comparison_results:
        extrap_results = comparison_results['extrapolation']
        
        extrapolating_encodings = [
            enc for enc, results in extrap_results.items()
            if results.get('can_extrapolate', False)
        ]
        
        if extrapolating_encodings:
            recommendations['extrapolation'] = (
                f"For tasks requiring sequence length extrapolation, consider: "
                f"{', '.join(extrapolating_encodings)}"
            )
    
    # Computational efficiency recommendations
    if any('computational_cost' in results for results in length_analysis.values()):
        recommendations['efficiency'] = (
            "Sinusoidal and RoPE encodings generally have better computational efficiency "
            "compared to learned encodings, especially for longer sequences."
        )
    
    # Task-specific recommendations
    if config.max_seq_len > 512:
        recommendations['long_sequences'] = (
            "For long sequences, RoPE or sinusoidal encodings are recommended "
            "due to their extrapolation capabilities and computational efficiency."
        )
    
    if config.max_seq_len <= 128:
        recommendations['short_sequences'] = (
            "For short sequences, learned positional encodings may provide "
            "better task-specific adaptation."
        )
    
    return recommendations
