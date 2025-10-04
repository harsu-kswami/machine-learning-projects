"""Feed-forward network implementation with visualization support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

from config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network with step-by-step visualization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        
        # Linear layers
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        
        # Activation function
        self.activation = self._get_activation(config.activation)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Visualization storage
        self.intermediate_outputs = {}
        
    def _get_activation(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'swish': lambda x: x * torch.sigmoid(x),
            'silu': F.silu
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        
        return activations[activation_name]
    
    def forward(
        self, 
        x: torch.Tensor, 
        store_intermediate: bool = False
    ) -> torch.Tensor:
        """Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            store_intermediate: Whether to store intermediate outputs
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if store_intermediate:
            self.intermediate_outputs = {}
            self.intermediate_outputs['input'] = x.clone()
        
        # First linear transformation
        x1 = self.linear1(x)  # (batch_size, seq_len, d_ff)
        
        if store_intermediate:
            self.intermediate_outputs['after_linear1'] = x1.clone()
        
        # Apply activation function
        x_activated = self.activation(x1)
        
        if store_intermediate:
            self.intermediate_outputs['after_activation'] = x_activated.clone()
        
        # Apply dropout
        x_dropout = self.dropout(x_activated)
        
        if store_intermediate:
            self.intermediate_outputs['after_dropout'] = x_dropout.clone()
        
        # Second linear transformation
        output = self.linear2(x_dropout)  # (batch_size, seq_len, d_model)
        
        if store_intermediate:
            self.intermediate_outputs['final_output'] = output.clone()
        
        return output
    
    def get_intermediate_outputs(self) -> Dict[str, torch.Tensor]:
        """Get stored intermediate outputs for visualization."""
        return self.intermediate_outputs
    
    def analyze_activation_patterns(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze activation patterns in the feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with activation analysis results
        """
        with torch.no_grad():
            # Forward pass with intermediate storage
            output = self.forward(x, store_intermediate=True)
            
            analysis = {}
            
            # Pre-activation statistics
            pre_activation = self.intermediate_outputs['after_linear1']
            analysis['pre_activation_mean'] = pre_activation.mean(dim=(0, 1))
            analysis['pre_activation_std'] = pre_activation.std(dim=(0, 1))
            analysis['pre_activation_min'] = pre_activation.min(dim=0)[0].min(dim=0)[0]
            analysis['pre_activation_max'] = pre_activation.max(dim=0)[0].max(dim=0)[0]
            
            # Post-activation statistics
            post_activation = self.intermediate_outputs['after_activation']
            analysis['post_activation_mean'] = post_activation.mean(dim=(0, 1))
            analysis['post_activation_std'] = post_activation.std(dim=(0, 1))
            analysis['activation_sparsity'] = (post_activation == 0).float().mean()
            
            # Activation function specific analysis
            if self.config.activation == 'relu':
                analysis['relu_dead_neurons'] = (post_activation.max(dim=0)[0].max(dim=0)[0] == 0).float().mean()
            
            # Weight statistics
            analysis['linear1_weight_norm'] = self.linear1.weight.norm(dim=1)
            analysis['linear2_weight_norm'] = self.linear2.weight.norm(dim=0)
            
            return analysis
    
    def visualize_transformation_effect(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Visualize how the feed-forward network transforms representations.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with transformation visualization data
        """
        with torch.no_grad():
            self.forward(x, store_intermediate=True)
            
            visualization_data = {}
            
            # Input-output comparison
            input_tensor = self.intermediate_outputs['input']
            output_tensor = self.intermediate_outputs['final_output']
            
            # Magnitude changes
            input_norms = torch.norm(input_tensor, dim=-1)  # (batch_size, seq_len)
            output_norms = torch.norm(output_tensor, dim=-1)  # (batch_size, seq_len)
            
            visualization_data['magnitude_change'] = output_norms / (input_norms + 1e-8)
            visualization_data['input_norms'] = input_norms
            visualization_data['output_norms'] = output_norms
            
            # Direction changes (cosine similarity)
            input_flat = input_tensor.flatten(0, 1)  # (batch_size * seq_len, d_model)
            output_flat = output_tensor.flatten(0, 1)  # (batch_size * seq_len, d_model)
            
            cosine_sim = F.cosine_similarity(input_flat, output_flat, dim=-1)
            visualization_data['direction_similarity'] = cosine_sim.view(input_tensor.shape[:2])
            
            # Dimensional analysis
            visualization_data['dimension_importance'] = {
                'input_variance_per_dim': input_tensor.var(dim=(0, 1)),
                'output_variance_per_dim': output_tensor.var(dim=(0, 1)),
                'linear1_weight_importance': self.linear1.weight.abs().mean(dim=0),
                'linear2_weight_importance': self.linear2.weight.abs().mean(dim=1)
            }
            
            return visualization_data
    
    def reset_visualization_cache(self):
        """Clear stored intermediate outputs."""
        self.intermediate_outputs = {}


class PositionwiseFeedForward(FeedForward):
    """Alternative implementation with different initialization and optional GLU."""
    
    def __init__(self, config: ModelConfig, use_glu: bool = False):
        super().__init__(config)
        self.use_glu = use_glu
        
        if use_glu:
            # GLU (Gated Linear Unit) variant
            self.gate_linear = nn.Linear(config.d_model, config.d_ff)
            
    def forward(self, x: torch.Tensor, store_intermediate: bool = False) -> torch.Tensor:
        """Forward pass with optional GLU mechanism."""
        if store_intermediate:
            self.intermediate_outputs = {}
            self.intermediate_outputs['input'] = x.clone()
        
        if self.use_glu:
            # GLU implementation: (linear1(x) * activation(gate(x))) * linear2
            linear_out = self.linear1(x)
            gate_out = self.gate_linear(x)
            
            if store_intermediate:
                self.intermediate_outputs['linear_output'] = linear_out.clone()
                self.intermediate_outputs['gate_output'] = gate_out.clone()
            
            gated_output = linear_out * self.activation(gate_out)
            
            if store_intermediate:
                self.intermediate_outputs['gated_output'] = gated_output.clone()
            
            output = self.linear2(self.dropout(gated_output))
        else:
            # Standard implementation
            output = super().forward(x, store_intermediate)
        
        return output


class ExpertFeedForward(nn.Module):
    """Mixture of Experts style feed-forward network for advanced visualization."""
    
    def __init__(self, config: ModelConfig, num_experts: int = 4):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(config.d_model, num_experts)
        
        # Visualization storage
        self.expert_weights = None
        self.expert_outputs = []
        
    def forward(
        self, 
        x: torch.Tensor, 
        store_expert_info: bool = False
    ) -> torch.Tensor:
        """Forward pass through mixture of experts.
        
        Args:
            x: Input tensor
            store_expert_info: Whether to store expert information
            
        Returns:
            Weighted combination of expert outputs
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute expert weights
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        expert_weights = F.softmax(gate_logits, dim=-1)
        
        if store_expert_info:
            self.expert_weights = expert_weights.clone()
            self.expert_outputs = []
        
        # Compute expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x, store_intermediate=store_expert_info)
            expert_outputs.append(expert_out)
            
            if store_expert_info:
                self.expert_outputs.append(expert_out.clone())
        
        # Combine expert outputs
        expert_stack = torch.stack(expert_outputs, dim=-1)  # (batch, seq, d_model, num_experts)
        expert_weights_expanded = expert_weights.unsqueeze(-2)  # (batch, seq, 1, num_experts)
        
        output = (expert_stack * expert_weights_expanded).sum(dim=-1)
        
        return output
    
    def analyze_expert_utilization(self) -> Dict[str, torch.Tensor]:
        """Analyze how different experts are utilized."""
        if self.expert_weights is None:
            raise ValueError("No expert weights stored. Run forward pass with store_expert_info=True")
        
        analysis = {}
        
        # Average expert utilization
        analysis['expert_utilization'] = self.expert_weights.mean(dim=(0, 1))
        
        # Expert specialization (entropy of weights)
        epsilon = 1e-8
        entropy = -(self.expert_weights * torch.log(self.expert_weights + epsilon)).sum(dim=-1)
        analysis['routing_entropy'] = entropy.mean(dim=(0, 1))
        
        # Expert diversity
        expert_outputs_stack = torch.stack(self.expert_outputs, dim=0)  # (num_experts, batch, seq, d_model)
        similarities = torch.zeros(self.num_experts, self.num_experts)
        
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                cosine_sim = F.cosine_similarity(
                    expert_outputs_stack[i].flatten(),
                    expert_outputs_stack[j].flatten(),
                    dim=0
                )
                similarities[i, j] = cosine_sim
                similarities[j, i] = cosine_sim
        
        analysis['expert_similarities'] = similarities
        
        return analysis
