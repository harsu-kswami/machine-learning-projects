"""Layer normalization implementations with visualization support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

from config import ModelConfig


class LayerNorm(nn.Module):
    """Layer normalization with comprehensive visualization capabilities."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Visualization storage
        self.normalization_stats = {}
        self.pre_norm_stats = {}
        self.post_norm_stats = {}
        
    def forward(self, input: torch.Tensor, store_stats: bool = False) -> torch.Tensor:
        """Apply layer normalization.
        
        Args:
            input: Input tensor of shape (..., normalized_shape)
            store_stats: Whether to store normalization statistics
            
        Returns:
            Layer normalized tensor
        """
        if store_stats:
            self._store_pre_norm_stats(input)
        
        # Compute mean and variance
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input - mean) / torch.sqrt(var + self.eps)
        
        if store_stats:
            self._store_normalization_stats(mean, var, normalized)
        
        # Apply learnable parameters
        if self.elementwise_affine:
            output = self.weight * normalized + self.bias
        else:
            output = normalized
        
        if store_stats:
            self._store_post_norm_stats(output)
        
        return output
    
    def _store_pre_norm_stats(self, input: torch.Tensor):
        """Store pre-normalization statistics."""
        self.pre_norm_stats = {
            'mean': input.mean(dim=-1).detach().clone(),
            'var': input.var(dim=-1, unbiased=False).detach().clone(),
            'std': input.std(dim=-1, unbiased=False).detach().clone(),
            'min': input.min(dim=-1)[0].detach().clone(),
            'max': input.max(dim=-1)[0].detach().clone(),
            'norm': input.norm(dim=-1).detach().clone()
        }
    
    def _store_normalization_stats(self, mean: torch.Tensor, var: torch.Tensor, normalized: torch.Tensor):
        """Store normalization process statistics."""
        self.normalization_stats = {
            'computed_mean': mean.squeeze(-1).detach().clone(),
            'computed_var': var.squeeze(-1).detach().clone(),
            'computed_std': torch.sqrt(var + self.eps).squeeze(-1).detach().clone(),
            'normalized_mean': normalized.mean(dim=-1).detach().clone(),
            'normalized_var': normalized.var(dim=-1, unbiased=False).detach().clone(),
            'normalized_norm': normalized.norm(dim=-1).detach().clone()
        }
    
    def _store_post_norm_stats(self, output: torch.Tensor):
        """Store post-normalization statistics."""
        self.post_norm_stats = {
            'mean': output.mean(dim=-1).detach().clone(),
            'var': output.var(dim=-1, unbiased=False).detach().clone(),
            'std': output.std(dim=-1, unbiased=False).detach().clone(),
            'min': output.min(dim=-1)[0].detach().clone(),
            'max': output.max(dim=-1)[0].detach().clone(),
            'norm': output.norm(dim=-1).detach().clone()
        }
    
    def get_normalization_effects(self) -> Dict[str, torch.Tensor]:
        """Get comprehensive normalization effect analysis.
        
        Returns:
            Dictionary with normalization effect statistics
        """
        if not self.normalization_stats:
            raise ValueError("No normalization stats stored. Run forward pass with store_stats=True")
        
        effects = {}
        
        # Mean centering effect
        effects['mean_centering'] = {
            'pre_norm_mean': self.pre_norm_stats['mean'],
            'post_normalize_mean': self.normalization_stats['normalized_mean'],
            'final_mean': self.post_norm_stats['mean']
        }
        
        # Variance scaling effect
        effects['variance_scaling'] = {
            'pre_norm_var': self.pre_norm_stats['var'],
            'post_normalize_var': self.normalization_stats['normalized_var'],
            'final_var': self.post_norm_stats['var']
        }
        
        # Distribution shape changes
        effects['distribution_changes'] = {
            'pre_norm_range': self.pre_norm_stats['max'] - self.pre_norm_stats['min'],
            'post_norm_range': self.post_norm_stats['max'] - self.post_norm_stats['min'],
            'norm_ratio': self.post_norm_stats['norm'] / (self.pre_norm_stats['norm'] + 1e-8)
        }
        
        # Parameter effects (if applicable)
        if self.elementwise_affine:
            effects['parameter_effects'] = {
                'weight_mean': self.weight.mean(),
                'weight_std': self.weight.std(), 
                'weight_min': self.weight.min(),
                'weight_max': self.weight.max(),
                'bias_mean': self.bias.mean(),
                'bias_std': self.bias.std(),
                'bias_min': self.bias.min(),
                'bias_max': self.bias.max()
            }
        
        return effects
    
    def analyze_parameter_impact(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze the impact of weight and bias parameters.
        
        Args:
            input: Input tensor to analyze
            
        Returns:
            Dictionary with parameter impact analysis
        """
        if not self.elementwise_affine:
            raise ValueError("Parameter analysis requires elementwise_affine=True")
        
        with torch.no_grad():
            # Forward pass without parameters
            mean = input.mean(dim=-1, keepdim=True)
            var = input.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (input - mean) / torch.sqrt(var + self.eps)
            
            # Analyze weight impact
            weight_contribution = self.weight * normalized
            bias_contribution = self.bias.unsqueeze(0).expand_as(normalized)
            
            analysis = {
                'normalized_without_params': normalized,
                'weight_contribution': weight_contribution,
                'bias_contribution': bias_contribution,
                'weight_magnitude_per_dim': self.weight.abs(),
                'bias_magnitude_per_dim': self.bias.abs(),
                'effective_scaling_per_dim': self.weight,
                'effective_shift_per_dim': self.bias,
                'weight_dominance': weight_contribution.abs().mean(dim=(0, 1)),
                'bias_dominance': bias_contribution.abs().mean(dim=(0, 1))
            }
        
        return analysis
    
    def visualize_normalization_process(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create visualization data for the normalization process.
        
        Args:
            input: Input tensor
            
        Returns:
            Visualization data dictionary
        """
        with torch.no_grad():
            # Store intermediate steps
            mean = input.mean(dim=-1, keepdim=True)
            var = input.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + self.eps)
            
            # Step-by-step transformation
            centered = input - mean
            normalized = centered / std
            
            if self.elementwise_affine:
                scaled = self.weight * normalized
                final = scaled + self.bias
            else:
                scaled = normalized
                final = normalized
            
            visualization_data = {
                'step1_original': input,
                'step2_mean': mean.expand_as(input),
                'step3_centered': centered,
                'step4_std': std.expand_as(input),
                'step5_normalized': normalized,
                'step6_scaled': scaled,
                'step7_final': final,
                'statistics': {
                    'input_mean_per_sample': mean.squeeze(-1),
                    'input_std_per_sample': std.squeeze(-1),
                    'dimension_means': input.mean(dim=(0, 1)),
                    'dimension_stds': input.std(dim=(0, 1))
                }
            }
        
        return visualization_data
    
    def reset_stats(self):
        """Clear stored statistics."""
        self.normalization_stats = {}
        self.pre_norm_stats = {}
        self.post_norm_stats = {}


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with visualization support."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter('weight', None)
        
        # Visualization storage
        self.rms_stats = {}
    
    def forward(self, input: torch.Tensor, store_stats: bool = False) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            input: Input tensor
            store_stats: Whether to store RMS statistics
            
        Returns:
            RMS normalized tensor
        """
        # Compute RMS
        mean_square = input.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        
        # Normalize by RMS
        normalized = input / rms
        
        if store_stats:
            self.rms_stats = {
                'input_mean_square': mean_square.squeeze(-1).detach().clone(),
                'rms': rms.squeeze(-1).detach().clone(),
                'normalized_rms': normalized.pow(2).mean(dim=-1).sqrt().detach().clone()
            }
        
        # Apply learnable parameter
        if self.elementwise_affine:
            output = self.weight * normalized
        else:
            output = normalized
        
        return output
    
    def compare_with_layernorm(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compare RMSNorm with standard LayerNorm.
        
        Args:
            input: Input tensor
            
        Returns:
            Comparison statistics
        """
        with torch.no_grad():
            # RMSNorm computation
            mean_square = input.pow(2).mean(dim=-1, keepdim=True)
            rms = torch.sqrt(mean_square + self.eps)
            rms_normalized = input / rms
            
            # LayerNorm computation
            mean = input.mean(dim=-1, keepdim=True)
            var = input.var(dim=-1, keepdim=True, unbiased=False)
            ln_normalized = (input - mean) / torch.sqrt(var + self.eps)
            
            comparison = {
                'rms_values': rms.squeeze(-1),
                'ln_std_values': torch.sqrt(var + self.eps).squeeze(-1),
                'rms_normalized': rms_normalized,
                'ln_normalized': ln_normalized,
                'normalization_difference': (rms_normalized - ln_normalized).abs().mean(dim=-1),
                'correlation': torch.stack([
                    rms_normalized.flatten(), 
                    ln_normalized.flatten()
                ]).corrcoef()[0, 1]
            }
        
        return comparison


class AdaptiveLayerNorm(LayerNorm):
    """Adaptive layer normalization that can switch between different normalization strategies."""
    
    def __init__(
        self, 
        normalized_shape: int, 
        eps: float = 1e-6, 
        adaptation_dim: int = 64
    ):
        super().__init__(normalized_shape, eps)
        
        # Adaptation network to predict normalization parameters
        self.adaptation_dim = adaptation_dim
        self.adaptation_net = nn.Sequential(
            nn.Linear(normalized_shape, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, 2)  # Predict scale and shift factors
        )
        
    def forward(self, input: torch.Tensor, store_stats: bool = False) -> torch.Tensor:
        """Adaptive layer normalization forward pass."""
        # Standard layer normalization
        normalized = super().forward(input, store_stats=False)
        
        # Predict adaptive parameters
        adaptation_params = self.adaptation_net(input.mean(dim=-2))  # (batch_size, 2)
        scale_factor = adaptation_params[:, 0:1]  # (batch_size, 1)
        shift_factor = adaptation_params[:, 1:2]  # (batch_size, 1)
        
        # Apply adaptive transformation
        output = normalized * scale_factor.unsqueeze(-1) + shift_factor.unsqueeze(-1)
        
        if store_stats:
            self.adaptive_stats = {
                'scale_factors': scale_factor.detach().clone(),
                'shift_factors': shift_factor.detach().clone(),
                'scale_mean': scale_factor.mean().item(),
                'scale_std': scale_factor.std().item(),
                'shift_mean': shift_factor.mean().item(),
                'shift_std': shift_factor.std().item()
            }
        
        return output
    
    def get_adaptive_stats(self) -> Dict[str, torch.Tensor]:
        """Get adaptive normalization statistics."""
        if not hasattr(self, 'adaptive_stats'):
            raise ValueError("No adaptive stats stored. Run forward pass with store_stats=True")
        return self.adaptive_stats
