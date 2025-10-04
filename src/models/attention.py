"""Attention mechanisms for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from config import ModelConfig


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, n_heads, seq_len, d_v)
            mask: Optional attention mask
            store_weights: Whether to store attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        # Store weights if requested
        stored_weights = attention_weights if store_weights else None
        
        return output, stored_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.d_v = config.d_model // config.n_heads
        
        # Ensure d_model is divisible by n_heads
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=config.dropout)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-head attention forward pass.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            store_weights: Whether to store attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)    # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.attention(
            Q, K, V, mask, store_weights
        )
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights
    
    def get_attention_patterns(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention patterns for visualization.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention weights tensor
        """
        with torch.no_grad():
            _, attention_weights = self.forward(
                query, key, value, mask, store_weights=True
            )
            return attention_weights
    
    def analyze_head_importance(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Analyze the importance of different attention heads.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Dictionary with head importance metrics
        """
        with torch.no_grad():
            batch_size, seq_len, d_model = query.size()
            
            # Get attention weights
            _, attention_weights = self.forward(
                query, key, value, mask, store_weights=True
            )
            
            if attention_weights is None:
                return {}
            
            # Compute head importance metrics
            head_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-9), dim=-1
            ).mean(dim=(0, 2))  # Average over batch and sequence
            
            head_max_attention = attention_weights.max(dim=-1)[0].mean(dim=(0, 2))
            head_attention_variance = attention_weights.var(dim=-1).mean(dim=(0, 2))
            
            return {
                'entropy': head_entropy,
                'max_attention': head_max_attention,
                'variance': head_attention_variance,
                'attention_weights': attention_weights
            }


class RelativePositionalAttention(nn.Module):
    """Relative positional attention mechanism."""
    
    def __init__(self, config: ModelConfig, max_relative_position: int = 32):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )
        
        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Get relative position indices."""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip to max relative position
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Shift to positive indices
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Relative positional attention forward pass."""
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Get relative positions
        relative_positions = self._get_relative_positions(seq_len).to(query.device)
        relative_positions = relative_positions.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.n_heads, -1, -1
        )
        
        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings(
            relative_positions.view(-1)
        ).view(batch_size, self.n_heads, seq_len, seq_len, self.d_k)
        
        # Compute content-based attention
        content_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Compute position-based attention
        position_scores = torch.matmul(
            Q.unsqueeze(3), relative_embeddings.transpose(-2, -1)
        ).squeeze(3)
        
        # Combine scores
        scores = content_scores + position_scores
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        stored_weights = attention_weights if store_weights else None
        
        return output, stored_weights


class SparseAttention(nn.Module):
    """Sparse attention mechanism with configurable patterns."""
    
    def __init__(self, config: ModelConfig, attention_pattern: str = "strided"):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.attention_pattern = attention_pattern
        
        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        
    def _create_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Create attention mask based on pattern."""
        mask = torch.zeros(seq_len, seq_len)
        
        if self.attention_pattern == "strided":
            # Strided pattern: attend to every k-th position
            stride = 2
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    if abs(i - j) <= stride:
                        mask[i, j] = 1
                        
        elif self.attention_pattern == "local":
            # Local pattern: attend to nearby positions
            window_size = 4
            for i in range(seq_len):
                start = max(0, i - window_size)
                end = min(seq_len, i + window_size + 1)
                mask[i, start:end] = 1
                
        elif self.attention_pattern == "random":
            # Random pattern: attend to random positions
            import random
            random.seed(42)  # For reproducibility
            for i in range(seq_len):
                num_attended = min(8, seq_len)
                attended_positions = random.sample(range(seq_len), num_attended)
                mask[i, attended_positions] = 1
                
        return mask.bool()
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sparse attention forward pass."""
        batch_size, seq_len, d_model = query.size()
        
        # Create sparse attention mask
        sparse_mask = self._create_attention_mask(seq_len).to(query.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.n_heads, -1, -1
        )
        
        # Combine with provided mask
        if mask is not None:
            combined_mask = sparse_mask & mask.unsqueeze(1)
        else:
            combined_mask = sparse_mask
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply sparse attention
        attn_output, attention_weights = self.attention(
            Q, K, V, combined_mask, store_weights
        )
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


def create_attention_mechanism(
    config: ModelConfig,
    attention_type: str = "multi_head"
) -> nn.Module:
    """Create attention mechanism based on type.
    
    Args:
        config: Model configuration
        attention_type: Type of attention mechanism
        
    Returns:
        Attention mechanism module
    """
    if attention_type == "multi_head":
        return MultiHeadAttention(config)
    elif attention_type == "relative_positional":
        return RelativePositionalAttention(config)
    elif attention_type == "sparse":
        return SparseAttention(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def analyze_attention_patterns(
    attention_weights: torch.Tensor,
    method: str = "entropy"
) -> Dict[str, torch.Tensor]:
    """Analyze attention patterns for visualization.
    
    Args:
        attention_weights: Attention weights tensor
        method: Analysis method
        
    Returns:
        Dictionary with analysis results
    """
    if attention_weights is None:
        return {}
    
    results = {}
    
    if method == "entropy":
        # Compute attention entropy
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-9), dim=-1
        )
        results['entropy'] = entropy
        
    elif method == "max_attention":
        # Find maximum attention positions
        max_attention, max_indices = attention_weights.max(dim=-1)
        results['max_attention'] = max_attention
        results['max_indices'] = max_indices
        
    elif method == "attention_distance":
        # Compute attention distance (how far attention spreads)
        seq_len = attention_weights.size(-1)
        positions = torch.arange(seq_len, device=attention_weights.device)
        expected_position = torch.sum(
            attention_weights * positions.unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1
        )
        results['expected_position'] = expected_position
        
    return results