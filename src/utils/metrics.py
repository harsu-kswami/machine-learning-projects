"""Metrics and evaluation utilities for positional encoding analysis."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math
from scipy import stats
from sklearn.metrics import mutual_info_score
from collections import defaultdict
import warnings


class AttentionMetrics:
    """Metrics for analyzing attention patterns."""
    
    def __init__(self):
        self.epsilon = 1e-8
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute attention entropy metrics.
        
        Args:
            attention_weights: Attention weights of shape (..., seq_len, seq_len)
            
        Returns:
            Dictionary with entropy metrics
        """
        # Add small epsilon to avoid log(0)
        attn_probs = attention_weights + self.epsilon
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1)
        
        results = {
            'entropy': entropy,
            'mean_entropy': entropy.mean(),
            'std_entropy': entropy.std(),
            'min_entropy': entropy.min(),
            'max_entropy': entropy.max()
        }
        
        return results
    
    def compute_attention_distance(self, attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute average attention distance for each query position.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Dictionary with distance metrics
        """
        seq_len = attention_weights.shape[-1]
        
        # Create position matrix
        positions = torch.arange(seq_len, dtype=torch.float32, device=attention_weights.device)
        position_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
        position_distances = torch.abs(position_matrix)
        
        # Compute weighted average distance for each query
        weighted_distances = (attention_weights * position_distances.unsqueeze(0)).sum(dim=-1)
        
        results = {
            'attention_distances': weighted_distances,
            'mean_distance': weighted_distances.mean(),
            'std_distance': weighted_distances.std(),
            'distance_by_position': weighted_distances.mean(dim=0) if len(weighted_distances.shape) > 1 else weighted_distances
        }
        
        return results
    
    def compute_attention_sparsity(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute attention sparsity metrics.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Dictionary with sparsity metrics
        """
        # Gini coefficient as sparsity measure
        def gini_coefficient(x):
            # Sort values
            sorted_x = torch.sort(x.flatten()).values
            n = len(sorted_x)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
            gini = (2 * (index * sorted_x).sum()) / (n * sorted_x.sum()) - (n + 1) / n
            return gini
        
        # L1/L2 ratio (another sparsity measure)
        l1_norm = attention_weights.abs().sum(dim=-1)
        l2_norm = (attention_weights ** 2).sum(dim=-1).sqrt()
        l1_l2_ratio = l1_norm / (l2_norm + self.epsilon)
        
        # Effective attention span (number of positions with > threshold attention)
        threshold = 0.1
        effective_span = (attention_weights > threshold).sum(dim=-1).float()
        
        results = {
            'gini_coefficient': gini_coefficient(attention_weights).item(),
            'l1_l2_ratio': l1_l2_ratio.mean().item(),
            'effective_span': effective_span.mean().item(),
            'max_attention': attention_weights.max().item(),
            'attention_concentration': (attention_weights > 0.5).sum().float().item() / attention_weights.numel()
        }
        
        return results
    
    def compute_head_similarity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute similarity between attention heads.
        
        Args:
            attention_weights: Multi-head attention weights (batch, heads, seq_len, seq_len)
            
        Returns:
            Head similarity matrix
        """
        if attention_weights.dim() != 4:
            raise ValueError("Expected 4D tensor (batch, heads, seq_len, seq_len)")
        
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Flatten attention matrices for correlation computation
        flattened_attention = attention_weights.view(batch_size, n_heads, -1)
        
        # Compute correlation between heads
        head_similarities = torch.zeros(batch_size, n_heads, n_heads)
        
        for b in range(batch_size):
            for i in range(n_heads):
                for j in range(n_heads):
                    if i != j:
                        attn_i = flattened_attention[b, i]
                        attn_j = flattened_attention[b, j]
                        
                        # Compute correlation
                        correlation = torch.corrcoef(torch.stack([attn_i, attn_j]))[0, 1]
                        head_similarities[b, i, j] = correlation
        
        return head_similarities
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Comprehensive attention pattern analysis.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {}
        
        # Basic entropy analysis
        results['entropy'] = self.compute_attention_entropy(attention_weights)
        
        # Distance analysis
        results['distance'] = self.compute_attention_distance(attention_weights)
        
        # Sparsity analysis
        results['sparsity'] = self.compute_attention_sparsity(attention_weights)
        
        # Head similarity (if multi-head)
        if attention_weights.dim() == 4:
            results['head_similarity'] = self.compute_head_similarity(attention_weights)
        
        # Pattern classification
        results['pattern_type'] = self._classify_attention_pattern(attention_weights)
        
        return results
    
    def _classify_attention_pattern(self, attention_weights: torch.Tensor) -> str:
        """Classify the type of attention pattern.
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Pattern type string
        """
        # Average across batch dimension if present
        if attention_weights.dim() > 2:
            attn_matrix = attention_weights.mean(dim=0)
            if attn_matrix.dim() > 2:  # Multi-head
                attn_matrix = attn_matrix.mean(dim=0)
        else:
            attn_matrix = attention_weights
        
        seq_len = attn_matrix.shape[0]
        
        # Analyze diagonal dominance (self-attention)
        diagonal_sum = torch.diag(attn_matrix).sum().item()
        total_sum = attn_matrix.sum().item()
        diagonal_ratio = diagonal_sum / total_sum
        
        # Analyze local vs global attention
        local_mask = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)) <= 2
        local_attention = attn_matrix[local_mask].sum().item() / total_sum
        
        # Classify pattern
        if diagonal_ratio > 0.4:
            return "self-focused"
        elif local_attention > 0.6:
            return "local"
        elif diagonal_ratio < 0.1 and local_attention < 0.3:
            return "global"
        else:
            return "mixed"


class EncodingMetrics:
    """Metrics for evaluating positional encodings."""
    
    def __init__(self):
        self.epsilon = 1e-8
    
    def compute_position_similarity(
        self, 
        encoding_matrix: torch.Tensor,
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """Compute similarity matrix between positions.
        
        Args:
            encoding_matrix: Encoding matrix of shape (seq_len, d_model)
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Position similarity matrix
        """
        if metric == 'cosine':
            # Normalize encodings
            normalized = F.normalize(encoding_matrix, p=2, dim=1)
            similarities = torch.mm(normalized, normalized.t())
        elif metric == 'euclidean':
            # Compute pairwise Euclidean distances
            distances = torch.cdist(encoding_matrix, encoding_matrix, p=2)
            # Convert to similarities (higher is more similar)
            similarities = 1.0 / (1.0 + distances)
        elif metric == 'dot':
            similarities = torch.mm(encoding_matrix, encoding_matrix.t())
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities
    
    def compute_encoding_quality(self, encoding_matrix: torch.Tensor) -> Dict[str, float]:
        """Compute overall quality metrics for positional encoding.
        
        Args:
            encoding_matrix: Encoding matrix
            
        Returns:
            Dictionary with quality metrics
        """
        seq_len, d_model = encoding_matrix.shape
        
        # Position distinguishability: how well can we distinguish different positions
        similarities = self.compute_position_similarity(encoding_matrix, 'cosine')
        
        # Remove diagonal (self-similarity)
        off_diagonal_mask = ~torch.eye(seq_len, dtype=torch.bool)
        off_diagonal_similarities = similarities[off_diagonal_mask]
        
        # Lower off-diagonal similarities are better (positions are more distinguishable)
        distinguishability = 1.0 - off_diagonal_similarities.mean().item()
        
        # Encoding variance: higher variance means more information
        encoding_variance = encoding_matrix.var().item()
        
        # Dimension utilization: how evenly are dimensions used
        dim_variances = encoding_matrix.var(dim=0)
        dim_utilization = 1.0 - (dim_variances.std() / (dim_variances.mean() + self.epsilon)).item()
        
        # Periodicity detection (for sinusoidal encodings)
        periodicity_score = self._detect_periodicity(encoding_matrix)
        
        results = {
            'distinguishability': distinguishability,
            'encoding_variance': encoding_variance,
            'dimension_utilization': dim_utilization,
            'periodicity_score': periodicity_score,
            'mean_similarity': off_diagonal_similarities.mean().item(),
            'similarity_std': off_diagonal_similarities.std().item()
        }
        
        return results
    
    def _detect_periodicity(self, encoding_matrix: torch.Tensor) -> float:
        """Detect periodicity in encoding patterns.
        
        Args:
            encoding_matrix: Encoding matrix
            
        Returns:
            Periodicity score (0-1, higher means more periodic)
        """
        # Compute autocorrelation for first few dimensions
        periodicity_scores = []
        
        for dim in range(min(8, encoding_matrix.shape[1])):
            signal = encoding_matrix[:, dim].cpu().numpy()
            
            # Compute autocorrelation
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks (excluding the first one at lag 0)
            if len(autocorr) > 2:
                peak_score = np.max(autocorr[1:len(autocorr)//2]) / autocorr[0]
                periodicity_scores.append(peak_score)
        
        return np.mean(periodicity_scores) if periodicity_scores else 0.0
    
    def compare_encodings(
        self,
        encoding_matrices: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple encoding methods.
        
        Args:
            encoding_matrices: Dictionary of encoding name to matrix
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Compute individual quality metrics
        for name, matrix in encoding_matrices.items():
            results[name] = self.compute_encoding_quality(matrix)
        
        # Compute pairwise similarities between encoding methods
        encoding_names = list(encoding_matrices.keys())
        pairwise_similarities = {}
        
        for i, name1 in enumerate(encoding_names):
            for j, name2 in enumerate(encoding_names[i+1:], i+1):
                matrix1 = encoding_matrices[name1]
                matrix2 = encoding_matrices[name2]
                
                # Align dimensions if different
                min_dim = min(matrix1.size(-1), matrix2.size(-1))
                min_seq = min(matrix1.size(0), matrix2.size(0))
                
                aligned1 = matrix1[:min_seq, :min_dim]
                aligned2 = matrix2[:min_seq, :min_dim]
                
                # Compute similarity
                similarity = F.cosine_similarity(
                    aligned1.flatten(), aligned2.flatten(), dim=0
                ).item()
                
                pairwise_similarities[f"{name1}_vs_{name2}"] = similarity
        
        results['pairwise_similarities'] = pairwise_similarities
        
        return results
    
    def evaluate_extrapolation(
        self,
        encoding_generator,
        base_length: int,
        test_lengths: List[int]
    ) -> Dict[str, float]:
        """Evaluate encoding extrapolation capability.
        
        Args:
            encoding_generator: Function to generate encoding for given length
            base_length: Base sequence length for comparison
            test_lengths: Test sequence lengths
            
        Returns:
            Extrapolation quality metrics
        """
        base_encoding = encoding_generator(base_length)
        base_similarities = self.compute_position_similarity(base_encoding)
        
        extrapolation_scores = {}
        
        for test_length in test_lengths:
            if test_length <= base_length:
                continue
            
            test_encoding = encoding_generator(test_length)
            test_similarities = self.compute_position_similarity(test_encoding)
            
            # Compare similarity patterns in overlapping region
            overlap_base = base_similarities
            overlap_test = test_similarities[:base_length, :base_length]
            
            # Compute similarity between similarity matrices
            similarity_correlation = F.cosine_similarity(
                overlap_base.flatten(), overlap_test.flatten(), dim=0
            ).item()
            
            extrapolation_scores[test_length] = similarity_correlation
        
        return extrapolation_scores


class PerformanceMetrics:
    """Performance and efficiency metrics."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
    
    def measure_encoding_time(
        self,
        encoding_fn,
        sequence_lengths: List[int],
        num_runs: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """Measure encoding computation time.
        
        Args:
            encoding_fn: Function to compute encoding
            sequence_lengths: List of sequence lengths to test
            num_runs: Number of runs for averaging
            
        Returns:
            Timing results for each sequence length
        """
        import time
        
        timing_results = {}
        
        for seq_len in sequence_lengths:
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                _ = encoding_fn(seq_len)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            timing_results[seq_len] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        return timing_results
    
    def measure_memory_usage(
        self,
        model,
        input_sizes: List[Tuple[int, int]],
        device: str = 'cpu'
    ) -> Dict[Tuple[int, int], float]:
        """Measure memory usage for different input sizes.
        
        Args:
            model: Model to test
            input_sizes: List of (batch_size, seq_len) tuples
            device: Device to test on
            
        Returns:
            Memory usage in MB for each input size
        """
        memory_usage = {}
        
        for batch_size, seq_len in input_sizes:
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create dummy input
                dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                memory_usage[(batch_size, seq_len)] = memory_mb
                
            else:
                # CPU memory measurement is more complex and less reliable
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / (1024 * 1024)
                
                dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_usage[(batch_size, seq_len)] = memory_after - memory_before
        
        return memory_usage
    
    def compute_flops(
        self,
        d_model: int,
        seq_len: int,
        n_heads: int,
        encoding_type: str
    ) -> Dict[str, int]:
        """Estimate FLOPs for different operations.
        
        Args:
            d_model: Model dimension
            seq_len: Sequence length
            n_heads: Number of attention heads
            encoding_type: Type of positional encoding
            
        Returns:
            FLOP estimates for different operations
        """
        d_k = d_model // n_heads
        
        flops = {}
        
        # Attention FLOPs
        # Q @ K^T: (seq_len, d_k) @ (d_k, seq_len) = seq_len^2 * d_k
        qk_flops = seq_len * seq_len * d_k * n_heads
        
        # Softmax: approximately 3 * seq_len^2 operations per head
        softmax_flops = 3 * seq_len * seq_len * n_heads
        
        # Attention @ V: (seq_len, seq_len) @ (seq_len, d_k) = seq_len^2 * d_k
        av_flops = seq_len * seq_len * d_k * n_heads
        
        flops['attention'] = qk_flops + softmax_flops + av_flops
        
        # Positional encoding FLOPs
        if encoding_type == 'sinusoidal':
            # Pre-computed, so minimal FLOPs
            flops['positional_encoding'] = seq_len * d_model  # Just addition
        elif encoding_type == 'learned':
            # Embedding lookup
            flops['positional_encoding'] = seq_len * d_model
        elif encoding_type == 'rope':
            # Rotation operations
            flops['positional_encoding'] = seq_len * d_model * 4  # cos, sin, multiply operations
        else:
            flops['positional_encoding'] = 0
        
        # Total FLOPs
        flops['total'] = flops['attention'] + flops['positional_encoding']
        
        return flops


# Convenience functions
def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute attention entropy.
    
    Args:
        attention_weights: Attention weights tensor
        
    Returns:
        Entropy values
    """
    metrics = AttentionMetrics()
    return metrics.compute_attention_entropy(attention_weights)['entropy']


def compute_position_similarity(encoding_matrix: torch.Tensor, metric: str = 'cosine') -> torch.Tensor:
    """Compute position similarity matrix.
    
    Args:
        encoding_matrix: Positional encoding matrix
        metric: Similarity metric
        
    Returns:
        Similarity matrix
    """
    metrics = EncodingMetrics()
    return metrics.compute_position_similarity(encoding_matrix, metric)


def evaluate_encoding_quality(encoding_matrix: torch.Tensor) -> Dict[str, float]:
    """Evaluate encoding quality.
    
    Args:
        encoding_matrix: Positional encoding matrix
        
    Returns:
        Quality metrics
    """
    metrics = EncodingMetrics()
    return metrics.compute_encoding_quality(encoding_matrix)
