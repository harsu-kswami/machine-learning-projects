"""Token embedding implementation with visualization capabilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from config import ModelConfig


class TokenEmbedding(nn.Module):
    """Token embedding layer with comprehensive visualization support."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Optional scaling (as in original Transformer paper)
        self.scale_embeddings = True
        self.embedding_scale = math.sqrt(config.d_model)
        
        # Initialize embeddings
        self._init_weights()
        
        # Visualization storage
        self.embedding_stats = {}
        self.token_similarities = None
        
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding layer.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embedded tokens of shape (batch_size, seq_len, d_model)
        """
        embeddings = self.embedding(input_ids)
        
        if self.scale_embeddings:
            embeddings = embeddings * self.embedding_scale
            
        return embeddings
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the full embedding matrix."""
        return self.embedding.weight.data
    
    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a specific token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token embedding vector
        """
        return self.embedding.weight[token_id]
    
    def compute_token_similarities(
        self, 
        token_ids: Optional[List[int]] = None,
        similarity_metric: str = 'cosine'
    ) -> torch.Tensor:
        """Compute similarities between token embeddings.
        
        Args:
            token_ids: List of token IDs to compare (None for all tokens)
            similarity_metric: 'cosine' or 'euclidean'
            
        Returns:
            Similarity matrix
        """
        if token_ids is None:
            embeddings = self.embedding.weight  # (vocab_size, d_model)
        else:
            embeddings = self.embedding.weight[token_ids]  # (num_tokens, d_model)
        
        if similarity_metric == 'cosine':
            # Normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
            similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
        elif similarity_metric == 'euclidean':
            # Compute pairwise Euclidean distances
            distances = torch.cdist(embeddings, embeddings, p=2)
            # Convert to similarities (higher is more similar)
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        self.token_similarities = similarities
        return similarities
    
    def find_similar_tokens(
        self, 
        token_id: int, 
        k: int = 10,
        exclude_self: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k most similar tokens to a given token.
        
        Args:
            token_id: Target token ID
            k: Number of similar tokens to return
            exclude_self: Whether to exclude the token itself
            
        Returns:
            Tuple of (similar_token_ids, similarity_scores)
        """
        # Compute similarities if not cached
        if self.token_similarities is None:
            self.compute_token_similarities()
        
        token_similarities = self.token_similarities[token_id]
        
        if exclude_self:
            # Set self-similarity to -inf to exclude it
            token_similarities[token_id] = float('-inf')
        
        # Get top-k similar tokens
        top_k_similarities, top_k_indices = torch.topk(token_similarities, k)
        
        return top_k_indices, top_k_similarities
    
    def analyze_embedding_distribution(self) -> Dict[str, torch.Tensor]:
        """Analyze the distribution of embedding vectors.
        
        Returns:
            Dictionary with embedding statistics
        """
        embeddings = self.embedding.weight  # (vocab_size, d_model)
        
        analysis = {}
        
        # Basic statistics
        analysis['mean'] = embeddings.mean(dim=0)  # (d_model,)
        analysis['std'] = embeddings.std(dim=0)   # (d_model,)
        analysis['min'] = embeddings.min(dim=0)[0]  # (d_model,)
        analysis['max'] = embeddings.max(dim=0)[0]  # (d_model,)
        
        # Global statistics
        analysis['global_mean'] = embeddings.mean()
        analysis['global_std'] = embeddings.std()
        analysis['global_norm'] = embeddings.norm(dim=1).mean()
        
        # Dimension-wise analysis
        analysis['dimension_variance'] = embeddings.var(dim=0)
        analysis['dimension_range'] = analysis['max'] - analysis['min']
        
        # Sparsity analysis
        analysis['zero_fraction'] = (embeddings == 0).float().mean()
        analysis['near_zero_fraction'] = (embeddings.abs() < 0.01).float().mean()
        
        # Norm distribution
        embedding_norms = embeddings.norm(dim=1)
        analysis['norm_mean'] = embedding_norms.mean()
        analysis['norm_std'] = embedding_norms.std()
        analysis['norm_min'] = embedding_norms.min()
        analysis['norm_max'] = embedding_norms.max()
        
        self.embedding_stats = analysis
        return analysis
    
    def visualize_embedding_space(
        self, 
        token_ids: List[int],
        method: str = 'pca'
    ) -> torch.Tensor:
        """Create 2D visualization of embedding space.
        
        Args:
            token_ids: Token IDs to visualize
            method: Dimensionality reduction method ('pca', 'tsne')
            
        Returns:
            2D coordinates for visualization
        """
        embeddings = self.embedding.weight[token_ids]  # (num_tokens, d_model)
        
        if method == 'pca':
            # Simple PCA implementation
            centered = embeddings - embeddings.mean(dim=0)
            U, S, V = torch.svd(centered)
            coords_2d = torch.mm(centered, V[:, :2])
        elif method == 'tsne':
            # Note: This is a placeholder - in practice, you'd use sklearn's t-SNE
            # For now, just return first two dimensions
            coords_2d = embeddings[:, :2]
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        return coords_2d
    
    def compare_with_pretrained(
        self, 
        pretrained_embeddings: torch.Tensor,
        token_ids: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compare current embeddings with pretrained ones.
        
        Args:
            pretrained_embeddings: Pretrained embedding matrix
            token_ids: Specific tokens to compare (None for all)
            
        Returns:
            Comparison statistics
        """
        if token_ids is None:
            current_emb = self.embedding.weight
            pretrained_emb = pretrained_embeddings
        else:
            current_emb = self.embedding.weight[token_ids]
            pretrained_emb = pretrained_embeddings[token_ids]
        
        comparison = {}
        
        # Cosine similarity
        current_norm = F.normalize(current_emb, p=2, dim=-1)
        pretrained_norm = F.normalize(pretrained_emb, p=2, dim=-1)
        cosine_sim = (current_norm * pretrained_norm).sum(dim=-1)
        
        comparison['cosine_similarities'] = cosine_sim
        comparison['mean_cosine_similarity'] = cosine_sim.mean()
        
        # L2 distance
        l2_distances = torch.norm(current_emb - pretrained_emb, dim=-1)
        comparison['l2_distances'] = l2_distances
        comparison['mean_l2_distance'] = l2_distances.mean()
        
        # Dimension-wise correlation
        dim_correlations = []
        for dim in range(current_emb.size(-1)):
            corr = torch.corrcoef(torch.stack([
                current_emb[:, dim], 
                pretrained_emb[:, dim]
            ]))[0, 1]
            dim_correlations.append(corr)
        
        comparison['dimension_correlations'] = torch.stack(dim_correlations)
        comparison['mean_dimension_correlation'] = torch.stack(dim_correlations).mean()
        
        return comparison


class LearnedEmbedding(TokenEmbedding):
    """Token embedding with learned positional information."""
    
    def __init__(self, config: ModelConfig, learn_position: bool = True):
        super().__init__(config)
        
        self.learn_position = learn_position
        if learn_position:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
            self._init_position_weights()
    
    def _init_position_weights(self):
        """Initialize positional embedding weights."""
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional learned positional embeddings."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeddings = super().forward(input_ids)
        
        if self.learn_position:
            # Add learned positional embeddings
            positions = torch.arange(seq_len, device=input_ids.device)
            position_embeddings = self.position_embedding(positions)
            
            # Combine token and position embeddings
            embeddings = token_embeddings + position_embeddings.unsqueeze(0)
        else:
            embeddings = token_embeddings
        
        return embeddings
    
    def get_position_embeddings(self) -> torch.Tensor:
        """Get the learned positional embeddings."""
        if not self.learn_position:
            raise ValueError("Position learning is disabled")
        return self.position_embedding.weight.data
    
    def analyze_position_patterns(self) -> Dict[str, torch.Tensor]:
        """Analyze patterns in learned positional embeddings."""
        if not self.learn_position:
            raise ValueError("Position learning is disabled")
        
        pos_embeddings = self.position_embedding.weight  # (max_seq_len, d_model)
        
        analysis = {}
        
        # Position similarity matrix
        pos_similarities = F.cosine_similarity(
            pos_embeddings.unsqueeze(1), 
            pos_embeddings.unsqueeze(0), 
            dim=-1
        )
        analysis['position_similarities'] = pos_similarities
        
        # Distance decay pattern
        max_seq_len = pos_embeddings.size(0)
        distances = torch.arange(max_seq_len).float()
        
        position_distances = []
        for i in range(max_seq_len):
            pos_dist = torch.norm(
                pos_embeddings - pos_embeddings[i].unsqueeze(0), 
                dim=-1
            )
            position_distances.append(pos_dist)
        
        analysis['position_distances'] = torch.stack(position_distances)
        
        return analysis


class FactorizedEmbedding(nn.Module):
    """Factorized embedding for large vocabularies with visualization."""
    
    def __init__(self, config: ModelConfig, factorization_dim: int = 128):
        super().__init__()
        self.config = config
        self.factorization_dim = factorization_dim
        
        # Two-stage embedding: vocab -> factorization_dim -> d_model
        self.token_embedding = nn.Embedding(config.vocab_size, factorization_dim)
        self.projection = nn.Linear(factorization_dim, config.d_model, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize factorized embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through factorized embedding."""
        # First stage: token -> factorization dimension
        factorized = self.token_embedding(input_ids)
        
        # Second stage: factorization dimension -> model dimension
        embeddings = self.projection(factorized)
        
        return embeddings
    
    def get_factorized_embeddings(self) -> torch.Tensor:
        """Get the factorized token embeddings."""
        return self.token_embedding.weight.data
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Get the projection matrix."""
        return self.projection.weight.data
    
    def analyze_factorization_efficiency(self) -> Dict[str, torch.Tensor]:
        """Analyze the efficiency of the factorization."""
        factorized_emb = self.token_embedding.weight  # (vocab_size, factorization_dim)
        projection_matrix = self.projection.weight     # (d_model, factorization_dim)
        
        analysis = {}
        
        # Effective rank of factorized embeddings
        U, S, V = torch.svd(factorized_emb)
        analysis['factorized_singular_values'] = S
        analysis['factorized_effective_rank'] = (S > 0.01 * S[0]).sum().float()
        
        # Projection matrix analysis
        U_proj, S_proj, V_proj = torch.svd(projection_matrix)
        analysis['projection_singular_values'] = S_proj
        analysis['projection_effective_rank'] = (S_proj > 0.01 * S_proj[0]).sum().float()
        
        # Reconstruction quality
        full_embeddings = torch.mm(factorized_emb, projection_matrix.t())
        analysis['embedding_norms'] = full_embeddings.norm(dim=1)
        analysis['factorization_compression_ratio'] = (
            self.config.vocab_size * self.config.d_model
        ) / (
            self.config.vocab_size * self.factorization_dim + 
            self.factorization_dim * self.config.d_model
        )
        
        return analysis
