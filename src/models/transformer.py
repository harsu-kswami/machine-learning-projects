"""Transformer encoder implementation with visualization hooks."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layer_norm import LayerNorm
from .embedding import TokenEmbedding
from src.positional_encoding import get_positional_encoding
from config import ModelConfig


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with visualization capabilities."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config)
        
        # Layer normalization
        self.norm1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Visualization storage
        self.attention_weights = None
        self.intermediate_outputs = {}
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        store_attention: bool = False
    ) -> torch.Tensor:
        """Forward pass through transformer encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            store_attention: Whether to store attention weights for visualization
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Store input for visualization
        if store_attention:
            self.intermediate_outputs['input'] = x.clone()
        
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(
            query=x, key=x, value=x, mask=mask, store_weights=store_attention
        )
        
        if store_attention:
            self.attention_weights = attention_weights
            self.intermediate_outputs['attention_output'] = attn_output.clone()
        
        # First residual connection and layer norm
        x_attn = self.norm1(x + self.dropout(attn_output))
        
        if store_attention:
            self.intermediate_outputs['after_norm1'] = x_attn.clone()
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(x_attn, store_intermediate=store_attention)
        output = self.norm2(x_attn + self.dropout(ff_output))
        
        if store_attention:
            self.intermediate_outputs['feed_forward_output'] = ff_output.clone()
            self.intermediate_outputs['final_output'] = output.clone()
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get stored attention weights."""
        return self.attention_weights
    
    def get_intermediate_outputs(self) -> Dict[str, torch.Tensor]:
        """Get stored intermediate outputs for visualization."""
        return self.intermediate_outputs
    
    def reset_visualization_cache(self):
        """Clear stored visualization data."""
        self.attention_weights = None
        self.intermediate_outputs = {}


class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder with comprehensive visualization support."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        
        # Token embedding
        self.embedding = TokenEmbedding(config)
        
        # Positional encoding
        self.positional_encoding = get_positional_encoding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Visualization storage
        self.layer_outputs = []
        self.all_attention_weights = []
        self.positional_encodings = None
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        store_visualizations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer encoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            store_visualizations: Whether to store data for visualization
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final layer output
                - hidden_states: All layer outputs (if storing visualizations)
                - attention_weights: All attention weights (if storing visualizations)
        """
        batch_size, seq_len = input_ids.shape
        
        # Reset visualization cache
        if store_visualizations:
            self.reset_visualization_cache()
        
        # Token embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Add positional encodings
        pos_encodings = self.positional_encoding(seq_len, self.d_model)
        if store_visualizations:
            self.positional_encodings = pos_encodings.clone()
        
        # Combine embeddings and positional encodings
        x = embeddings + pos_encodings
        x = self.dropout(x)
        
        # Store initial embeddings for visualization
        if store_visualizations:
            self.layer_outputs.append({
                'embeddings': embeddings.clone(),
                'positional_encodings': pos_encodings.clone(),
                'input': x.clone()
            })
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=attention_mask, store_attention=store_visualizations)
            
            if store_visualizations:
                # Store layer outputs and attention weights
                layer_info = {
                    'output': x.clone(),
                    'attention_weights': layer.get_attention_weights(),
                    'intermediate_outputs': layer.get_intermediate_outputs()
                }
                self.layer_outputs.append(layer_info)
                
                if layer.get_attention_weights() is not None:
                    self.all_attention_weights.append(layer.get_attention_weights())
        
        # Final layer normalization
        output = self.final_norm(x)
        
        # Prepare return dictionary
        result = {
            'last_hidden_state': output,
        }
        
        if store_visualizations:
            result.update({
                'hidden_states': [layer_info['output'] for layer_info in self.layer_outputs[1:]],
                'attention_weights': self.all_attention_weights,
                'positional_encodings': self.positional_encodings,
                'embeddings': self.layer_outputs[0]['embeddings'],
                'layer_outputs': self.layer_outputs
            })
        
        return result
    
    def get_attention_patterns(
        self,
        input_ids: torch.Tensor,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Get attention patterns for visualization.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Specific layer to analyze (None for all layers)
            head_idx: Specific head to analyze (None for all heads)
            
        Returns:
            Attention weights tensor
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, store_visualizations=True)
            attention_weights = outputs['attention_weights']
            
            if layer_idx is not None:
                attention_weights = attention_weights[layer_idx]
                
            if head_idx is not None:
                attention_weights = attention_weights[:, head_idx:head_idx+1]
                
            return attention_weights
    
    def compare_positional_encodings(
        self,
        input_ids: torch.Tensor,
        encoding_types: List[str]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compare different positional encoding methods.
        
        Args:
            input_ids: Input token IDs
            encoding_types: List of encoding types to compare
            
        Returns:
            Dictionary with results for each encoding type
        """
        results = {}
        original_encoding = self.config.encoding_type
        
        for encoding_type in encoding_types:
            # Temporarily change encoding type
            self.config.encoding_type = encoding_type
            self.positional_encoding = get_positional_encoding(self.config)
            
            # Run forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids, store_visualizations=True)
                
            results[encoding_type] = {
                'attention_weights': outputs['attention_weights'],
                'positional_encodings': outputs['positional_encodings'],
                'final_output': outputs['last_hidden_state']
            }
        
        # Restore original encoding
        self.config.encoding_type = original_encoding
        self.positional_encoding = get_positional_encoding(self.config)
        
        return results
    
    def analyze_sequence_length_effects(
        self,
        base_sequence: torch.Tensor,
        lengths: List[int]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Analyze how sequence length affects attention patterns.
        
        Args:
            base_sequence: Base sequence to extend/truncate
            lengths: List of sequence lengths to test
            
        Returns:
            Dictionary with results for each sequence length
        """
        results = {}
        
        for length in lengths:
            # Adjust sequence length
            if length <= base_sequence.size(1):
                # Truncate
                sequence = base_sequence[:, :length]
            else:
                # Extend with padding
                pad_length = length - base_sequence.size(1)
                padding = torch.zeros(
                    base_sequence.size(0), pad_length,
                    dtype=base_sequence.dtype, device=base_sequence.device
                )
                sequence = torch.cat([base_sequence, padding], dim=1)
            
            # Run analysis
            with torch.no_grad():
                outputs = self.forward(sequence, store_visualizations=True)
                
            results[length] = {
                'attention_weights': outputs['attention_weights'],
                'sequence_length': length,
                'effective_length': (sequence != 0).sum(dim=1).float().mean().item()
            }
        
        return results
    
    def reset_visualization_cache(self):
        """Clear all stored visualization data."""
        self.layer_outputs = []
        self.all_attention_weights = []
        self.positional_encodings = None
        
        # Reset layer caches
        for layer in self.layers:
            layer.reset_visualization_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for visualization."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'd_ff': self.config.d_ff,
                'vocab_size': self.config.vocab_size,
                'max_seq_len': self.config.max_seq_len
            },
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            },
            'config': {
                'encoding_type': self.config.encoding_type,
                'dropout': self.config.dropout,
                'activation': self.config.activation,
                'device': str(next(self.parameters()).device)
            }
        }
    
    @torch.no_grad()
    def generate_attention_rollout(
        self,
        input_ids: torch.Tensor,
        start_layer: int = 0
    ) -> torch.Tensor:
        """Generate attention rollout for analyzing information flow.
        
        Args:
            input_ids: Input token IDs
            start_layer: Layer to start rollout from
            
        Returns:
            Attention rollout matrix
        """
        outputs = self.forward(input_ids, store_visualizations=True)
        attention_weights = outputs['attention_weights']
        
        # Initialize rollout with identity matrix
        batch_size, seq_len = input
