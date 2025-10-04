"""Absolute positional encoding implementations with visualization support."""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional, List
import numpy as np

from config import ModelConfig


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding with comprehensive analysis capabilities."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        
        # Pre-compute positional encodings
        self.register_buffer('pe', self._create_sinusoidal_encodings())
        
        # Visualization storage
        self.encoding_analysis = {}
        
    def _create_sinusoidal_encodings(self) -> torch.Tensor:
        """Create sinusoidal positional encodings.
        
        Returns:
            Positional encodings of shape (max_seq_len, d_model)
        """
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        
        # Create division term for frequencies
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * 
            -(math.log(10000.0) / self.d_model)
        )
        
        # Apply sin to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Get positional encodings for given sequence length.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            Positional encodings of shape (1, seq_len, d_model)
        """
        return self.pe[:seq_len, :d_model].unsqueeze(0)
    
    def get_encoding_at_position(self, position: int) -> torch.Tensor:
        """Get encoding vector at specific position.
        
        Args:
            position: Position index
            
        Returns:
            Encoding vector of shape (d_model,)
        """
        return self.pe[position]
    
    def analyze_frequency_components(self) -> Dict[str, torch.Tensor]:
        """Analyze frequency components of sinusoidal encodings.
        
        Returns:
            Dictionary with frequency analysis results
        """
        analysis = {}
        
        # Compute frequencies for each dimension
        frequencies = []
        for dim in range(0, self.d_model, 2):
            freq = 1.0 / (10000.0 ** (dim / self.d_model))
            frequencies.append(freq)
            if dim + 1 < self.d_model:
                frequencies.append(freq)  # Same frequency for cos component
        
        analysis['frequencies'] = torch.tensor(frequencies[:self.d_model])
        analysis['wavelengths'] = 2 * math.pi / analysis['frequencies']
        
        # Analyze frequency spectrum
        analysis['min_frequency'] = analysis['frequencies'].min()
        analysis['max_frequency'] = analysis['frequencies'].max()
        analysis['frequency_ratio'] = analysis['max_frequency'] / analysis['min_frequency']
        
        # Periodicity analysis
        max_period = analysis['wavelengths'].max().item()
        analysis['max_period'] = max_period
        analysis['periods_in_max_seq'] = self.max_seq_len / max_period
        
        return analysis
    
    def visualize_encoding_patterns(
        self, 
        positions: Optional[List[int]] = None,
        dimensions: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Create visualization data for encoding patterns.
        
        Args:
            positions: Specific positions to visualize (None for all)
            dimensions: Specific dimensions to visualize (None for all)
            
        Returns:
            Visualization data dictionary
        """
        if positions is None:
            positions = list(range(min(100, self.max_seq_len)))
        if dimensions is None:
            dimensions = list(range(min(64, self.d_model)))
        
        viz_data = {}
        
        # Extract encoding subset
        encoding_subset = self.pe[positions][:, dimensions]
        viz_data['encoding_matrix'] = encoding_subset
        viz_data['positions'] = torch.tensor(positions)
        viz_data['dimensions'] = torch.tensor(dimensions)
        
        # Compute position similarities
        position_similarities = torch.mm(
            encoding_subset, encoding_subset.t()
        ) / self.d_model
        viz_data['position_similarities'] = position_similarities
        
        # Dimension-wise patterns
        viz_data['dimension_patterns'] = {}
        for i, dim in enumerate(dimensions):
            pattern = self.pe[positions, dim]
            viz_data['dimension_patterns'][dim] = pattern
        
        # Frequency domain analysis
        if len(positions) > 1:
            fft_results = {}
            for i, dim in enumerate(dimensions[:16]):  # Limit for visualization
                signal = self.pe[positions, dim].numpy()
                fft = np.fft.fft(signal)
                fft_results[dim] = {
                    'magnitude': torch.tensor(np.abs(fft)),
                    'phase': torch.tensor(np.angle(fft)),
                    'power': torch.tensor(np.abs(fft) ** 2)
                }
            viz_data['fft_analysis'] = fft_results
        
        return viz_data
    
    def compute_position_distances(
        self, 
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """Compute distances between positional encodings.
        
        Args:
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Distance matrix of shape (seq_len, seq_len)
        """
        encodings = self.pe[:self.max_seq_len]  # (seq_len, d_model)
        
        if metric == 'cosine':
            # Normalize encodings
            normalized = torch.nn.functional.normalize(encodings, p=2, dim=1)
            distances = 1 - torch.mm(normalized, normalized.t())
        elif metric == 'euclidean':
            distances = torch.cdist(encodings, encodings, p=2)
        elif metric == 'manhattan':
            distances = torch.cdist(encodings, encodings, p=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distances
    
    def analyze_interpolation_capability(
        self, 
        test_lengths: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Analyze how well encodings interpolate to different sequence lengths.
        
        Args:
            test_lengths: List of sequence lengths to test
            
        Returns:
            Interpolation analysis results
        """
        results = {}
        
        for length in test_lengths:
            if length <= self.max_seq_len:
                # Direct encoding
                encoding = self.pe[:length]
                results[f'length_{length}'] = {
                    'encoding': encoding,
                    'type': 'direct'
                }
            else:
                # Need to extrapolate
                extrapolated = self._extrapolate_encoding(length)
                results[f'length_{length}'] = {
                    'encoding': extrapolated,
                    'type': 'extrapolated'
                }
        
        return results
    
    def _extrapolate_encoding(self, target_length: int) -> torch.Tensor:
        """Extrapolate sinusoidal encoding to longer sequences.
        
        Args:
            target_length: Target sequence length
            
        Returns:
            Extrapolated positional encodings
        """
        pe = torch.zeros(target_length, self.d_model)
        position = torch.arange(0, target_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * 
            -(math.log(10000.0) / self.d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def compare_with_learned(
        self, 
        learned_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compare sinusoidal encoding with learned positional embeddings.
        
        Args:
            learned_embeddings: Learned positional embeddings
            
        Returns:
            Comparison results
        """
        seq_len = min(learned_embeddings.size(0), self.max_seq_len)
        sinusoidal = self.pe[:seq_len]
        learned = learned_embeddings[:seq_len]
        
        comparison = {}
        
        # Cosine similarity
        sin_norm = torch.nn.functional.normalize(sinusoidal, p=2, dim=1)
        learned_norm = torch.nn.functional.normalize(learned, p=2, dim=1)
        cosine_sim = torch.sum(sin_norm * learned_norm, dim=1)
        
        comparison['cosine_similarities'] = cosine_sim
        comparison['mean_cosine_similarity'] = cosine_sim.mean()
        
        # L2 distance
        l2_distances = torch.norm(sinusoidal - learned, dim=1)
        comparison['l2_distances'] = l2_distances
        comparison['mean_l2_distance'] = l2_distances.mean()
        
        # Dimension-wise correlation
        correlations = []
        for dim in range(self.d_model):
            corr = torch.corrcoef(torch.stack([
                sinusoidal[:, dim], 
                learned[:, dim]
            ]))[0, 1]
            correlations.append(corr)
        
        comparison['dimension_correlations'] = torch.stack(correlations)
        comparison['mean_dimension_correlation'] = torch.stack(correlations).mean()
        
        return comparison


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding with analysis capabilities."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Visualization storage
        self.learning_history = []
        
    def _init_embeddings(self):
        """Initialize positional embeddings."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Get learned positional encodings.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            Positional encodings of shape (1, seq_len, d_model)
        """
        positions = torch.arange(seq_len, device=self.position_embeddings.weight.device)
        return self.position_embeddings(positions).unsqueeze(0)
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the full learned embedding matrix."""
        return self.position_embeddings.weight.data
    
    def analyze_learned_patterns(self) -> Dict[str, torch.Tensor]:
        """Analyze patterns in learned positional embeddings.
        
        Returns:
            Analysis of learned patterns
        """
        embeddings = self.position_embeddings.weight  # (max_seq_len, d_model)
        
        analysis = {}
        
        # Position similarity analysis
        similarities = torch.mm(embeddings, embeddings.t()) / self.d_model
        analysis['position_similarities'] = similarities
        
        # Distance-based patterns
        distances = torch.cdist(embeddings, embeddings, p=2)
        analysis['position_distances'] = distances
        
        # Analyze if distance correlates with position difference
        pos_diffs = torch.abs(
            torch.arange(self.max_seq_len).float().unsqueeze(0) - 
            torch.arange(self.max_seq_len).float().unsqueeze(1)
        )
        
        # Flatten for correlation analysis
        flat_distances = distances.flatten()
        flat_pos_diffs = pos_diffs.flatten()
        
        # Remove diagonal (distance = 0, pos_diff = 0)
        mask = flat_pos_diffs != 0
        correlation = torch.corrcoef(torch.stack([
            flat_distances[mask], 
            flat_pos_diffs[mask]
        ]))[0, 1]
        
        analysis['distance_position_correlation'] = correlation
        
        # Dimension importance analysis
        dim_variance = embeddings.var(dim=0)
        analysis['dimension_importance'] = dim_variance
        analysis['effective_dimensions'] = (dim_variance > 0.01 * dim_variance.max()).sum()
        
        # Periodicity detection
        analysis['periodicity'] = self._detect_periodicity(embeddings)
        
        return analysis
    
    def _detect_periodicity(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect periodic patterns in learned embeddings.
        
        Args:
            embeddings: Position embeddings matrix
            
        Returns:
            Periodicity analysis results
        """
        periodicity = {}
        
        # Autocorrelation analysis for each dimension
        autocorrelations = []
        for dim in range(min(16, self.d_model)):  # Limit for efficiency
            signal = embeddings[:, dim].cpu().numpy()
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorrelations.append(torch.tensor(autocorr))
        
        periodicity['autocorrelations'] = autocorrelations
        
        # Find potential periods
        periods = []
        for autocorr in autocorrelations:
            # Find peaks in autocorrelation (excluding the first peak at lag=0)
            peaks = []
            for i in range(2, min(len(autocorr), self.max_seq_len // 4)):
                if autocorr[i] > 0.5 * autocorr[0] and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            periods.extend(peaks)
        
        if periods:
            unique_periods = torch.tensor(list(set(periods)))
            periodicity['detected_periods'] = unique_periods
        else:
            periodicity['detected_periods'] = torch.tensor([])
        
        return periodicity
    
    def track_learning_progress(self, epoch: int):
        """Track how embeddings evolve during training.
        
        Args:
            epoch: Current training epoch
        """
        current_state = {
            'epoch': epoch,
            'embeddings': self.position_embeddings.weight.data.clone(),
            'embedding_norms': self.position_embeddings.weight.data.norm(dim=1),
            'mean_similarity': self._compute_mean_similarity()
        }
        
        self.learning_history.append(current_state)
    
    def _compute_mean_similarity(self) -> float:
        """Compute mean pairwise similarity of position embeddings."""
        embeddings = self.position_embeddings.weight
        similarities = torch.mm(embeddings, embeddings.t()) / self.d_model
        
        # Exclude diagonal and compute mean
        mask = torch.eye(similarities.size(0), dtype=torch.bool)
        mean_sim = similarities[~mask].mean().item()
        
        return mean_sim
    
    def get_learning_dynamics(self) -> Dict[str, torch.Tensor]:
        """Analyze learning dynamics over training.
        
        Returns:
            Learning dynamics analysis
        """
        if not self.learning_history:
            raise ValueError("No learning history available. Use track_learning_progress() during training.")
        
        dynamics = {}
        
        # Extract data over time
        epochs = [state['epoch'] for state in self.learning_history]
        mean_similarities = [state['mean_similarity'] for state in self.learning_history]
        
        dynamics['epochs'] = torch.tensor(epochs)
        dynamics['mean_similarities'] = torch.tensor(mean_similarities)
        
        # Norm evolution
        norm_evolution = []
        for state in self.learning_history:
            norm_evolution.append(state['embedding_norms'])
        
        dynamics['norm_evolution'] = torch.stack(norm_evolution)  # (num_epochs, max_seq_len)
        
        # Stability analysis
        if len(self.learning_history) > 1:
            embedding_changes = []
            for i in range(1, len(self.learning_history)):
                prev_emb = self.learning_history[i-1]['embeddings']
                curr_emb = self.learning_history[i]['embeddings']
                change = torch.norm(curr_emb - prev_emb, dim=1).mean()
                embedding_changes.append(change)
            
            dynamics['embedding_changes'] = torch.tensor(embedding_changes)
        
        return dynamics
    
    def initialize_with_sinusoidal(self):
        """Initialize learned embeddings with sinusoidal patterns."""
        sinusoidal_enc = SinusoidalEncoding(self.config)
        sinusoidal_weights = sinusoidal_enc.pe[:self.max_seq_len]
        
        with torch.no_grad():
            self.position_embeddings.weight.copy_(sinusoidal_weights)
    
    def interpolate_for_longer_sequences(self, target_length: int) -> torch.Tensor:
        """Interpolate learned embeddings for longer sequences.
        
        Args:
            target_length: Target sequence length
            
        Returns:
            Interpolated position embeddings
        """
        if target_length <= self.max_seq_len:
            return self.position_embeddings.weight[:target_length]
        
        # Linear interpolation for positions beyond max_seq_len
        existing_embeddings = self.position_embeddings.weight  # (max_seq_len, d_model)
        
        # Create interpolation grid
        original_positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        target_positions = torch.linspace(0, self.max_seq_len - 1, target_length)
        
        # Interpolate each dimension
        interpolated = torch.zeros(target_length, self.d_model)
        for dim in range(self.d_model):
            interpolated[:, dim] = torch.nn.functional.interpolate(
                existing_embeddings[:, dim].unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return interpolated
