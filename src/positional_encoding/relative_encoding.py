"""Relative positional encoding implementations with visualization support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List

from config import ModelConfig


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding with comprehensive analysis capabilities."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_relative_position = getattr(config, 'max_relative_position', 128)
        
        # Relative position embeddings for keys and values
        self.relative_positions_keys = nn.Embedding(
            2 * self.max_relative_position - 1, config.d_model // config.n_heads
        )
        self.relative_positions_values = nn.Embedding(
            2 * self.max_relative_position - 1, config.d_model // config.n_heads
        )
        
        self._init_embeddings()
        
        # Visualization storage
        self.relative_bias_matrices = {}
        self.position_analysis = {}
    
    def _init_embeddings(self):
        """Initialize relative position embeddings."""
        nn.init.xavier_uniform_(self.relative_positions_keys.weight)
        nn.init.xavier_uniform_(self.relative_positions_values.weight)
    
    def _get_relative_positions_matrix(self, seq_len: int) -> torch.Tensor:
        """Generate relative position matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position matrix of shape (seq_len, seq_len)
        """
        range_vec = torch.arange(seq_len)
        relative_matrix = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # Clip to maximum relative position
        clipped_relative_matrix = torch.clamp(
            relative_matrix,
            -self.max_relative_position + 1,
            self.max_relative_position - 1
        )
        
        # Shift to make all values positive for embedding lookup
        relative_positions = clipped_relative_matrix + self.max_relative_position - 1
        
        return relative_positions
    
    def forward(self, seq_len: int, d_model: int) -> Dict[str, torch.Tensor]:
        """Get relative positional encodings and bias matrices.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            Dictionary containing relative encodings and bias matrices
        """
        # Get relative position matrix
        relative_positions = self._get_relative_positions_matrix(seq_len)
        
        # Get relative position embeddings
        relative_keys = self.relative_positions_keys(relative_positions)
        relative_values = self.relative_positions_values(relative_positions)
        
        result = {
            'relative_positions_matrix': relative_positions,
            'relative_keys': relative_keys,
            'relative_values': relative_values,
            'seq_len': seq_len
        }
        
        return result
    
    def compute_relative_attention_bias(
        self, 
        query: torch.Tensor,
        seq_len: int,
        store_analysis: bool = False
    ) -> torch.Tensor:
        """Compute relative attention bias.
        
        Args:
            query: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            seq_len: Sequence length
            store_analysis: Whether to store bias analysis
            
        Returns:
            Relative attention bias of shape (batch_size, n_heads, seq_len, seq_len)
        """
        relative_positions = self._get_relative_positions_matrix(seq_len)
        relative_keys = self.relative_positions_keys(relative_positions)
        
        # Compute relative attention scores
        # query: (batch_size, n_heads, seq_len, d_k)
        # relative_keys: (seq_len, seq_len, d_k)
        
        relative_scores = torch.einsum('bhid,ijd->bhij', query, relative_keys)
        
        if store_analysis:
            self.relative_bias_matrices[seq_len] = {
                'positions_matrix': relative_positions,
                'relative_keys': relative_keys,
                'bias_scores': relative_scores,
                'query_used': query.clone()
            }
        
        return relative_scores
    
    def analyze_relative_patterns(self, seq_lengths: List[int]) -> Dict[str, torch.Tensor]:
        """Analyze relative encoding patterns across different sequence lengths.
        
        Args:
            seq_lengths: List of sequence lengths to analyze
            
        Returns:
            Analysis results dictionary
        """
        analysis = {}
        
        for seq_len in seq_lengths:
            # Generate relative position matrix
            rel_pos_matrix = self._get_relative_positions_matrix(seq_len)
            
            # Get relative embeddings
            relative_keys = self.relative_positions_keys(rel_pos_matrix)
            relative_values = self.relative_positions_values(rel_pos_matrix)
            
            # Analyze distance decay
            distance_analysis = self._analyze_distance_decay(rel_pos_matrix, relative_keys)
            
            analysis[f'seq_len_{seq_len}'] = {
                'relative_positions': rel_pos_matrix,
                'relative_keys': relative_keys,
                'relative_values': relative_values,
                'distance_decay': distance_analysis,
                'embedding_statistics': self._compute_embedding_stats(relative_keys)
            }
        
        return analysis
    
    def _analyze_distance_decay(
        self, 
        positions: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze how relative embeddings change with distance.
        
        Args:
            positions: Relative position matrix
            embeddings: Relative embeddings
            
        Returns:
            Distance decay analysis
        """
        seq_len = positions.size(0)
        analysis = {}
        
        # Group embeddings by distance
        distances = torch.abs(positions - (self.max_relative_position - 1))
        unique_distances = torch.unique(distances)
        
        distance_embeddings = {}
        distance_norms = {}
        
        for dist in unique_distances:
            mask = distances == dist
            if mask.sum() > 0:
                dist_embeddings = embeddings[mask]  # (num_positions_at_distance, d_k)
                distance_embeddings[dist.item()] = dist_embeddings
                distance_norms[dist.item()] = dist_embeddings.norm(dim=-1).mean()
        
        analysis['distance_embeddings'] = distance_embeddings
        analysis['distance_norms'] = distance_norms
        
        # Analyze decay pattern
        distances_list = sorted(distance_norms.keys())
        norms_list = [distance_norms[d] for d in distances_list]
        
        analysis['distances'] = torch.tensor(distances_list)
        analysis['norm_decay'] = torch.stack(norms_list)
        
        return analysis
    
    def _compute_embedding_stats(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute statistics for relative embeddings.
        
        Args:
            embeddings: Relative embeddings tensor
            
        Returns:
            Embedding statistics
        """
        stats = {}
        
        # Basic statistics
        stats['mean'] = embeddings.mean()
        stats['std'] = embeddings.std()
        stats['min'] = embeddings.min()
        stats['max'] = embeddings.max()
        
        # Norm statistics
        norms = embeddings.norm(dim=-1)
        stats['norm_mean'] = norms.mean()
        stats['norm_std'] = norms.std()
        stats['norm_min'] = norms.min()
        stats['norm_max'] = norms.max()
        
        # Dimension-wise analysis
        stats['dim_variance'] = embeddings.var(dim=(0, 1))
        stats['dim_mean'] = embeddings.mean(dim=(0, 1))
        
        return stats
    
    def visualize_relative_bias_matrix(
        self, 
        seq_len: int,
        head_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Create visualization data for relative bias matrix.
        
        Args:
            seq_len: Sequence length
            head_idx: Attention head index
            
        Returns:
            Visualization data
        """
        if seq_len not in self.relative_bias_matrices:
            raise ValueError(f"No bias matrix stored for seq_len={seq_len}. Run compute_relative_attention_bias first.")
        
        bias_data = self.relative_bias_matrices[seq_len]
        
        viz_data = {
            'positions_matrix': bias_data['positions_matrix'],
            'bias_matrix': bias_data['bias_scores'][0, head_idx] if len(bias_data['bias_scores'].shape) > 2 else bias_data['bias_scores'],
            'relative_keys': bias_data['relative_keys'],
            'seq_len': seq_len
        }
        
        # Add distance-based analysis
        distances = torch.abs(bias_data['positions_matrix'].float() - (self.max_relative_position - 1))
        viz_data['distance_matrix'] = distances
        
        # Compute average bias by distance
        unique_distances = torch.unique(distances)
        avg_bias_by_distance = {}
        
        for dist in unique_distances:
            mask = distances == dist
            if mask.sum() > 0:
                avg_bias = viz_data['bias_matrix'][mask].mean()
                avg_bias_by_distance[dist.item()] = avg_bias.item()
        
        viz_data['average_bias_by_distance'] = avg_bias_by_distance
        
        return viz_data
    
    def compare_with_absolute(
        self, 
        absolute_encodings: torch.Tensor,
        seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Compare relative encoding effects with absolute encodings.
        
        Args:
            absolute_encodings: Absolute positional encodings
            seq_len: Sequence length
            
        Returns:
            Comparison results
        """
        # Get relative encodings
        relative_data = self.forward(seq_len, self.d_model)
        relative_keys = relative_data['relative_keys']
        
        comparison = {}
        
        # Compare position representation capacity
        abs_similarities = F.cosine_similarity(
            absolute_encodings.unsqueeze(1),
            absolute_encodings.unsqueeze(0),
            dim=-1
        )
        
        # For relative encodings, compute average similarity at each distance
        distances = torch.abs(relative_data['relative_positions_matrix'].float() - (self.max_relative_position - 1))
        unique_distances = torch.unique(distances)
        
        rel_similarities = torch.zeros_like(abs_similarities)
        for dist in unique_distances:
            mask = distances == dist
            if mask.sum() > 0:
                # Average relative key embedding for this distance
                avg_rel_emb = relative_keys[mask].mean(dim=0)
                rel_similarities[mask] = F.cosine_similarity(
                    avg_rel_emb.unsqueeze(0),
                    avg_rel_emb.unsqueeze(0),
                    dim=-1
                ).item()
        
        comparison['absolute_similarities'] = abs_similarities
        comparison['relative_similarities'] = rel_similarities
        comparison['similarity_difference'] = (abs_similarities - rel_similarities).abs()
        comparison['mean_similarity_difference'] = comparison['similarity_difference'].mean()
        
        return comparison


class T5RelativeEncoding(nn.Module):
    """T5-style relative positional encoding with bucketed positions."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.num_buckets = getattr(config, 'relative_attention_num_buckets', 32)
        self.max_distance = getattr(config, 'relative_attention_max_distance', 128)
        
        # Relative attention bias
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)
        
        self._init_bias()
        
        # Visualization storage
        self.bucket_analysis = {}
    
    def _init_bias(self):
        """Initialize relative attention bias."""
        nn.init.xavier_uniform_(self.relative_attention_bias.weight)
    
    def _relative_position_bucket(
        self, 
        relative_position: torch.Tensor,
        bidirectional: bool = True
    ) -> torch.Tensor:
        """Map relative positions to buckets.
        
        Args:
            relative_position: Relative position matrix
            bidirectional: Whether to use bidirectional attention
            
        Returns:
            Bucket indices
        """
        relative_buckets = 0
        num_buckets = self.num_buckets
        
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # Small distances get their own bucket
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # Large distances get logarithmically spaced buckets
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_distance / max_exact) * 
            (num_buckets - max_exact)
        ).long()
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small,
            relative_position,
            relative_position_if_large
        )
        
        return relative_buckets
    
    def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Compute T5 relative attention bias.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension (unused but kept for consistency)
            
        Returns:
            Relative attention bias of shape (1, n_heads, seq_len, seq_len)
        """
        # Create relative position matrix
        context_position = torch.arange(seq_len)[:, None]
        memory_position = torch.arange(seq_len)[None, :]
        relative_position = memory_position - context_position
        
        # Map to buckets
        relative_buckets = self._relative_position_bucket(relative_position)
        
        # Get bias values
        bias = self.relative_attention_bias(relative_buckets)  # (seq_len, seq_len, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, seq_len, seq_len)
        
        return bias
    
    def analyze_bucketing_strategy(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """Analyze the bucketing strategy for relative positions.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Bucketing analysis results
        """
        # Create relative position matrix
        context_position = torch.arange(seq_len)[:, None]
        memory_position = torch.arange(seq_len)[None, :]
        relative_position = memory_position - context_position
        
        # Map to buckets
        relative_buckets = self._relative_position_bucket(relative_position)
        
        analysis = {
            'relative_positions': relative_position,
            'bucket_assignments': relative_buckets,
            'unique_buckets': torch.unique(relative_buckets),
            'num_unique_buckets': len(torch.unique(relative_buckets))
        }
        
        # Analyze bucket distribution
        bucket_counts = {}
        bucket_positions = {}
        
        for bucket_id in analysis['unique_buckets']:
            mask = relative_buckets == bucket_id
            count = mask.sum().item()
            positions = relative_position[mask].unique()
            
            bucket_counts[bucket_id.item()] = count
            bucket_positions[bucket_id.item()] = positions
        
        analysis['bucket_counts'] = bucket_counts
        analysis['bucket_positions'] = bucket_positions
        
        # Analyze bias values per bucket
        bias_values = self.relative_attention_bias.weight  # (num_buckets, n_heads)
        analysis['bias_per_bucket'] = bias_values
        
        # Compute statistics per head
        head_analysis = {}
        for head_idx in range(self.n_heads):
            head_bias = bias_values[:, head_idx]
            head_analysis[head_idx] = {
                'bias_mean': head_bias.mean(),
                'bias_std': head_bias.std(),
                'bias_range': head_bias.max() - head_bias.min(),
                'bias_values': head_bias
            }
        
        analysis['head_analysis'] = head_analysis
        
        return analysis
    
    def visualize_bucket_mapping(self, max_seq_len: int = 64) -> Dict[str, torch.Tensor]:
        """Create visualization data for bucket mapping.
        
        Args:
            max_seq_len: Maximum sequence length to visualize
            
        Returns:
            Visualization data
        """
        viz_data = {}
        
        # Create distance range
        distances = torch.arange(-max_seq_len, max_seq_len + 1)
        
        # Map distances to buckets
        buckets = self._relative_position_bucket(distances)
        
        viz_data['distances'] = distances
        viz_data['bucket_assignments'] = buckets
        
        # Create bucket-to-distance mapping
        bucket_to_distances = {}
        for i, bucket in enumerate(buckets):
            bucket_id = bucket.item()
            if bucket_id not in bucket_to_distances:
                bucket_to_distances[bucket_id] = []
            bucket_to_distances[bucket_id].append(distances[i].item())
        
        viz_data['bucket_to_distances'] = bucket_to_distances
        
        # Add bias visualization
        bias_matrix = self.forward(max_seq_len, self.config.d_model)
        viz_data['bias_matrix'] = bias_matrix.squeeze(0)  # (n_heads, seq_len, seq_len)
        
        return viz_data
    
    def compare_bucketing_strategies(
        self, 
        alternative_num_buckets: List[int],
        seq_len: int = 32
    ) -> Dict[str, Dict]:
        """Compare different bucketing strategies.
        
        Args:
            alternative_num_buckets: List of alternative bucket numbers
            seq_len: Sequence length for comparison
            
        Returns:
            Comparison results
        """
        original_num_buckets = self.num_buckets
        results = {}
        
        # Analyze original strategy
        results['original'] = self.analyze_bucketing_strategy(seq_len)
        results['original']['num_buckets'] = original_num_buckets
        
        # Analyze alternatives
        for num_buckets in alternative_num_buckets:
            self.num_buckets = num_buckets
            # Reinitialize bias with new number of buckets
            self.relative_attention_bias = nn.Embedding(num_buckets, self.n_heads)
            self._init_bias()
            
            analysis = self.analyze_bucketing_strategy(seq_len)
            analysis['num_buckets'] = num_buckets
            results[f'buckets_{num_buckets}'] = analysis
        
        # Restore original configuration
        self.num_buckets = original_num_buckets
        self.relative_attention_bias = nn.Embedding(original_num_buckets, self.n_heads)
        self._init_bias()
        
        return results
    
    def get_effective_bias_range(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """Get the effective range of bias values used for a sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Effective bias range analysis
        """
        bias_matrix = self.forward(seq_len, self.config.d_model)  # (1, n_heads, seq_len, seq_len)
        
        analysis = {}
        
        for head_idx in range(self.n_heads):
            head_bias = bias_matrix[0, head_idx]  # (seq_len, seq_len)
            
            analysis[f'head_{head_idx}'] = {
                'min_bias': head_bias.min(),
                'max_bias': head_bias.max(),
                'mean_bias': head_bias.mean(),
                'std_bias': head_bias.std(),
                'unique_values': torch.unique(head_bias),
                'num_unique_values': len(torch.unique(head_bias))
            }
        
        # Overall statistics
        analysis['overall'] = {
            'min_bias': bias_matrix.min(),
            'max_bias': bias_matrix.max(),
            'mean_bias': bias_matrix.mean(),
            'std_bias': bias_matrix.std()
        }
        
        return analysis

