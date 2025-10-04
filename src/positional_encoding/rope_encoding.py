
"""Rotary Position Embedding (RoPE) implementation with visualization support."""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional, List, Union

from config import ModelConfig


class RoPEEncoding(nn.Module):
    """Rotary Position Embedding with comprehensive analysis capabilities."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.base = getattr(config, 'rope_theta', 10000.0)
        
        # Pre-compute rotation frequencies
        self.register_buffer('inv_freq', self._compute_inv_freq())
        
        # Visualization storage
        self.rotation_analysis = {}
        self.position_analysis = {}
        
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for rotary embeddings.
        
        Returns:
            Inverse frequencies tensor
        """
        # For RoPE, we use half the dimensions (pairs of dimensions)
        dim = self.d_model // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, dtype=torch.float32) / dim))
        return inv_freq
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine values for rotary embeddings.
        
        Args:
            seq_len: Sequence length
            device: Device for computation
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute frequencies for each position
        freqs = torch.outer(t, self.inv_freq.to(device))  # (seq_len, d_model//2)
        
        # Compute cos and sin
        cos = torch.cos(freqs)  # (seq_len, d_model//2)
        sin = torch.sin(freqs)  # (seq_len, d_model//2)
        
        return cos, sin
    
    def forward(self, seq_len: int, d_model: int) -> Dict[str, torch.Tensor]:
        """Get rotary position embeddings.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            Dictionary containing cos and sin rotation matrices
        """
        device = self.inv_freq.device
        cos, sin = self._compute_cos_sin(seq_len, device)
        
        # Expand to full dimension by duplicating
        cos = torch.cat([cos, cos], dim=-1)  # (seq_len, d_model)
        sin = torch.cat([sin, sin], dim=-1)  # (seq_len, d_model)
        
        return {
            'cos': cos.unsqueeze(0),  # (1, seq_len, d_model)
            'sin': sin.unsqueeze(0),  # (1, seq_len, d_model)
            'freqs': torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), self.inv_freq)
        }
    
    def apply_rope(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        store_analysis: bool = False
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            cos: Cosine rotation matrix
            sin: Sine rotation matrix
            store_analysis: Whether to store rotation analysis
            
        Returns:
            Rotated tensor
        """
        # Split x into pairs for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]  # Even and odd dimensions
        cos_half, sin_half = cos[..., ::2], sin[..., ::2]  # Use only half (since we duplicated)
        
        if store_analysis:
            self.rotation_analysis['input'] = x.clone()
            self.rotation_analysis['x1'] = x1.clone()
            self.rotation_analysis['x2'] = x2.clone()
            self.rotation_analysis['cos'] = cos_half.clone()
            self.rotation_analysis['sin'] = sin_half.clone()
        
        # Apply 2D rotation
        rotated_x1 = x1 * cos_half - x2 * sin_half
        rotated_x2 = x1 * sin_half + x2 * cos_half
        
        if store_analysis:
            self.rotation_analysis['rotated_x1'] = rotated_x1.clone()
            self.rotation_analysis['rotated_x2'] = rotated_x2.clone()
        
        # Interleave back to original shape
        output = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        
        if store_analysis:
            self.rotation_analysis['output'] = output.clone()
        
        return output
    
    def analyze_rotation_frequencies(self) -> Dict[str, torch.Tensor]:
        """Analyze rotation frequencies across dimensions.
        
        Returns:
            Frequency analysis results
        """
        analysis = {}
        
        # Frequency analysis
        frequencies = 1.0 / self.inv_freq  # Convert back to frequencies
        analysis['frequencies'] = frequencies
        analysis['inv_frequencies'] = self.inv_freq
        
        # Wavelength analysis (2π / frequency)
        wavelengths = 2 * math.pi / frequencies
        analysis['wavelengths'] = wavelengths
        
        # Frequency range
        analysis['min_frequency'] = frequencies.min()
        analysis['max_frequency'] = frequencies.max()
        analysis['frequency_ratio'] = frequencies.max() / frequencies.min()
        
        # Wavelength range
        analysis['min_wavelength'] = wavelengths.min()
        analysis['max_wavelength'] = wavelengths.max()
        analysis['wavelength_ratio'] = wavelengths.max() / wavelengths.min()
        
        # Periodicity analysis
        analysis['max_unambiguous_distance'] = wavelengths.min() / 2
        analysis['max_representable_distance'] = wavelengths.max() / 2
        
        return analysis
    
    def visualize_rotation_patterns(
        self, 
        seq_len: int,
        dimensions: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Create visualization data for rotation patterns.
        
        Args:
            seq_len: Sequence length
            dimensions: Specific dimensions to visualize
            
        Returns:
            Visualization data
        """
        if dimensions is None:
            dimensions = list(range(0, min(self.d_model, 16), 2))  # Sample even dimensions
        
        device = self.inv_freq.device
        cos, sin = self._compute_cos_sin(seq_len, device)
        
        viz_data = {
            'seq_len': seq_len,
            'positions': torch.arange(seq_len),
            'dimensions': dimensions
        }
        
        # Cosine and sine patterns for specific dimensions
        for dim in dimensions:
            if dim // 2 < cos.shape[1]:
                dim_idx = dim // 2
                viz_data[f'cos_dim_{dim}'] = cos[:, dim_idx]
                viz_data[f'sin_dim_{dim}'] = sin[:, dim_idx]
                
                # Frequency for this dimension
                freq = 1.0 / self.inv_freq[dim_idx]
                viz_data[f'frequency_dim_{dim}'] = freq
                viz_data[f'wavelength_dim_{dim}'] = 2 * math.pi / freq
        
        # 2D rotation visualization (complex plane representation)
        rotation_matrices = []
        for pos in range(seq_len):
            rotation_2d = []
            for dim_pair in range(0, min(len(dimensions), self.d_model), 2):
                if dim_pair // 2 < cos.shape[1]:
                    dim_idx = dim_pair // 2
                    c = cos[pos, dim_idx].item()
                    s = sin[pos, dim_idx].item()
                    
                    # 2D rotation matrix
                    rot_matrix = torch.tensor([[c, -s], [s, c]])
                    rotation_2d.append(rot_matrix)
            
            if rotation_2d:
                rotation_matrices.append(torch.stack(rotation_2d))
        
        if rotation_matrices:
            viz_data['rotation_matrices'] = torch.stack(rotation_matrices)  # (seq_len, num_pairs, 2, 2)
        
        return viz_data
    
    def analyze_extrapolation_capability(
        self, 
        train_length: int,
        test_lengths: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Analyze RoPE's extrapolation capability to longer sequences.
        
        Args:
            train_length: Training sequence length
            test_lengths: List of test sequence lengths
            
        Returns:
            Extrapolation analysis results
        """
        analysis = {}
        device = self.inv_freq.device
        
        # Baseline: training length
        train_cos, train_sin = self._compute_cos_sin(train_length, device)
        analysis['train_length'] = train_length
        analysis['train_cos'] = train_cos
        analysis['train_sin'] = train_sin
        
        # Test different lengths
        for test_len in test_lengths:
            test_cos, test_sin = self._compute_cos_sin(test_len, device)
            
            # Compare patterns in overlapping region
            overlap_len = min(train_length, test_len)
            
            cos_similarity = F.cosine_similarity(
                train_cos[:overlap_len].flatten(),
                test_cos[:overlap_len].flatten(),
                dim=0
            )
            
            sin_similarity = F.cosine_similarity(
                train_sin[:overlap_len].flatten(),
                test_sin[:overlap_len].flatten(),
                dim=0
            )
            
            analysis[f'test_length_{test_len}'] = {
                'test_cos': test_cos,
                'test_sin': test_sin,
                'cos_similarity': cos_similarity,
                'sin_similarity': sin_similarity,
                'extrapolation_factor': test_len / train_length
            }
        
        return analysis
    
    def compute_position_similarities(
        self, 
        seq_len: int,
        similarity_metric: str = 'cosine'
    ) -> torch.Tensor:
        """Compute similarities between positions using RoPE.
        
        Args:
            seq_len: Sequence length
            similarity_metric: 'cosine' or 'rotation_angle'
            
        Returns:
            Position similarity matrix
        """
        device = self.inv_freq.device
        cos, sin = self._compute_cos_sin(seq_len, device)
        
        if similarity_metric == 'cosine':
            # Treat each position as a vector in rotation space
            position_vectors = torch.cat([cos, sin], dim=1)  # (seq_len, d_model)
            similarities = torch.mm(position_vectors, position_vectors.t())
            # Normalize by dimension
            similarities = similarities / position_vectors.shape[1]
        
        elif similarity_metric == 'rotation_angle':
            # Compute relative rotation angles between positions
            similarities = torch.zeros(seq_len, seq_len)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    # Compute relative angle for each frequency
                    relative_angles = abs(i - j) * self.inv_freq
                    # Average cosine of relative angles (similarity measure)
                    avg_similarity = torch.cos(relative_angles).mean()
                    similarities[i, j] = avg_similarity
        
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        return similarities
    
    def compare_with_sinusoidal(
        self, 
        sinusoidal_encoding: torch.Tensor,
        seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Compare RoPE with sinusoidal positional encoding.
        
        Args:
            sinusoidal_encoding: Sinusoidal positional encodings
            seq_len: Sequence length
            
        Returns:
            Comparison results
        """
        rope_data = self.forward(seq_len, self.d_model)
        
        comparison = {}
        
        # Create RoPE "position vectors" for comparison
        rope_cos = rope_data['cos'].squeeze(0)  # (seq_len, d_model)
        rope_sin = rope_data['sin'].squeeze(0)  # (seq_len, d_model)
        
        # RoPE doesn't directly provide position vectors, but we can use the rotation parameters
        rope_vectors = torch.cat([rope_cos, rope_sin], dim=1)  # (seq_len, 2*d_model)
        
        # Truncate sinusoidal to match sequence length
        sin_vectors = sinusoidal_encoding[:seq_len]  # (seq_len, d_model)
        
        # Compare position similarity patterns
        rope_similarities = torch.mm(rope_vectors, rope_vectors.t()) / (2 * self.d_model)
        sin_similarities = torch.mm(sin_vectors, sin_vectors.t()) / self.d_model
        
        comparison['rope_similarities'] = rope_similarities
        comparison['sinusoidal_similarities'] = sin_similarities
        comparison['similarity_difference'] = (rope_similarities - sin_similarities[:seq_len, :seq_len]).abs()
        comparison['mean_similarity_difference'] = comparison['similarity_difference'].mean()
        
        # Frequency comparison
        freq_analysis = self.analyze_rotation_frequencies()
        comparison['rope_frequencies'] = freq_analysis['frequencies']
        
        # Sinusoidal frequencies (approximation)
        sin_freqs = []
        for dim in range(0, self.d_model, 2):
            freq = 1.0 / (10000.0 ** (dim / self.d_model))
            sin_freqs.append(freq)
        
        comparison['sinusoidal_frequencies'] = torch.tensor(sin_freqs)
        comparison['frequency_correlation'] = torch.corrcoef(torch.stack([
            comparison['rope_frequencies'],
            comparison['sinusoidal_frequencies'][:len(comparison['rope_frequencies'])]
        ]))[0, 1]
        
        return comparison
    
    def get_rotation_analysis(self) -> Dict[str, torch.Tensor]:
        """Get stored rotation analysis data.
        
        Returns:
            Rotation analysis data
        """
        if not self.rotation_analysis:
            raise ValueError("No rotation analysis stored. Run apply_rope with store_analysis=True")
        
        return self.rotation_analysis
    
    def visualize_2d_rotations(
        self, 
        seq_len: int,
        sample_positions: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Visualize 2D rotations in complex plane.
        
        Args:
            seq_len: Sequence length
            sample_positions: Positions to visualize (None for all)
            
        Returns:
            2D rotation visualization data
        """
        if sample_positions is None:
            sample_positions = list(range(0, seq_len, max(1, seq_len // 16)))
        
        device = self.inv_freq.device
        cos, sin = self._compute_cos_sin(seq_len, device)
        
        viz_data = {
            'sample_positions': sample_positions,
            'rotations_2d': {}
        }
        
        # For each sampled position, show how unit vectors are rotated
        unit_vectors = torch.eye(self.d_model // 2)  # Unit vectors for each dimension pair
        
        for pos in sample_positions:
            rotations = []
            
            for dim_pair in range(self.d_model // 2):
                c = cos[pos, dim_pair]
                s = sin[pos, dim_pair]
                
                # Rotation matrix
                rot_matrix = torch.tensor([[c, -s], [s, c]])
                
                # Apply to unit vector
                unit_vec = torch.tensor([1.0, 0.0])
                rotated_vec = torch.mv(rot_matrix, unit_vec)
                
                rotations.append({
                    'rotation_matrix': rot_matrix,
                    'original_vector': unit_vec,
                    'rotated_vector': rotated_vec,
                    'angle': math.atan2(s.item(), c.item()),
                    'frequency': 1.0 / self.inv_freq[dim_pair].item()
                })
            
            viz_data['rotations_2d'][pos] = rotations
        
        return viz_data


class RotaryEmbedding(nn.Module):
    """Simplified Rotary Embedding interface for direct use in attention."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos and sin values
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos and sin values."""
        if seq_len > self._cached_seq_len:
            # Compute new cache
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            
            self._cached_cos = cos
            self._cached_sin = sin
            self._cached_seq_len = seq_len
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, n_heads, seq_len, head_dim)
            seq_len: Sequence length (inferred if None)
            
        Returns:
            Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # Update cache if needed
        self._update_cache(seq_len, q.device, q.dtype)
        
        # Get cos and sin values
        cos = self._cached_cos[:seq_len]  # (seq_len, dim//2)
        sin = self._cached_sin[:seq_len]  # (seq_len, dim//2)
        
        # Apply rotation
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotation(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotation to input tensor."""
        # Split into even and odd dimensions
        x1 = x[..., ::2]   # Even dimensions
        x2 = x[..., 1::2]  # Odd dimensions
        
        # Expand cos and sin to match tensor dimensions
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        
        return rotated
    
    def get_rotation_info(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """Get rotation information for analysis.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Rotation information dictionary
        """
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        
        return {
            'frequencies': 1.0 / self.inv_freq,
            'positions': t,
            'rotation_angles': freqs,
            'cos_values': torch.cos(freqs),
            'sin_values': torch.sin(freqs)
        }
