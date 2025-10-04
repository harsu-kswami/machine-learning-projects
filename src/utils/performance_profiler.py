"""Performance profiling utilities for transformer models and encodings."""

import torch
import torch.nn as nn
import time
import psutil
import os
import gc
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
import numpy as np
from contextlib import contextmanager
import threading
import functools
import warnings
import json
from pathlib import Path
from dataclasses import dataclass

from config import ModelConfig


@dataclass
class ProfilingResult:
    """Result of profiling operation."""
    operation_name: str
    execution_time: float
    memory_usage: Dict[str, float]
    additional_metrics: Dict[str, Any]
    timestamp: float


class PerformanceProfiler:
    """General-purpose performance profiler for timing and memory usage."""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._determine_device(device)
        self.results = []
        self.memory_baseline = self._get_memory_baseline()
        
    def _determine_device(self, device: str) -> str:
        """Determine the device to use for profiling."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _get_memory_baseline(self) -> Dict[str, float]:
        """Get baseline memory usage."""
        baseline = {}
        
        # CPU memory
        process = psutil.Process()
        baseline['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        if self.device == 'cuda' and torch.cuda.is_available():
            baseline['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            baseline['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return baseline
    
    @contextmanager
    def profile(self, operation_name: str, **kwargs):
        """Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
            **kwargs: Additional metadata
        """
        # Clear GPU cache if using CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear Python garbage collector
        gc.collect()
        
        # Record start state
        start_time = time.perf_counter()
        start_memory = self._get_current_memory()
        
        try:
            yield
        finally:
            # Synchronize if using CUDA
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Record end state
            end_time = time.perf_counter()
            end_memory = self._get_current_memory()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = {
                key: end_memory[key] - start_memory[key]
                for key in start_memory.keys()
            }
            
            # Store result
            result = ProfilingResult(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                additional_metrics=kwargs,
                timestamp=start_time
            )
            
            self.results.append(result)
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory = {}
        
        # CPU memory
        process = psutil.Process()
        memory['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory
        if self.device == 'cuda' and torch.cuda.is_available():
            memory['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return memory
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, ProfilingResult]:
        """Profile a function call.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function_result, profiling_result)
        """
        operation_name = getattr(func, '__name__', 'anonymous_function')
        
        with self.profile(operation_name):
            result = func(*args, **kwargs)
        
        return result, self.results[-1]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling results.
        
        Returns:
            Summary statistics
        """
        if not self.results:
            return {"message": "No profiling results available"}
        
        # Group results by operation name
        grouped_results = defaultdict(list)
        for result in self.results:
            grouped_results[result.operation_name].append(result)
        
        summary = {}
        
        for operation_name, results in grouped_results.items():
            execution_times = [r.execution_time for r in results]
            
            operation_summary = {
                'count': len(results),
                'total_time': sum(execution_times),
                'mean_time': np.mean(execution_times),
                'std_time': np.std(execution_times),
                'min_time': min(execution_times),
                'max_time': max(execution_times),
                'memory_usage': {}
            }
            
            # Memory statistics
            if results:
                memory_keys = results[0].memory_usage.keys()
                for key in memory_keys:
                    memory_values = [r.memory_usage[key] for r in results]
                    operation_summary['memory_usage'][key] = {
                        'mean': np.mean(memory_values),
                        'std': np.std(memory_values),
                        'min': min(memory_values),
                        'max': max(memory_values)
                    }
            
            summary[operation_name] = operation_summary
        
        return summary
    
    def clear_results(self):
        """Clear all profiling results."""
        self.results = []
    
    def save_results(self, filepath: str):
        """Save profiling results to file.
        
        Args:
            filepath: Path to save results
        """
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'operation_name': result.operation_name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'additional_metrics': result.additional_metrics,
                'timestamp': result.timestamp
            }
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump({
                'device': self.device,
                'memory_baseline': self.memory_baseline,
                'results': serializable_results,
                'summary': self.get_summary()
            }, f, indent=2)


class ModelProfiler(PerformanceProfiler):
    """Specialized profiler for transformer models."""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        super().__init__(device)
        self.model = model
        self.model.to(self.device)
        
    def profile_forward_pass(
        self, 
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch_sizes: Optional[List[int]] = None,
        sequence_lengths: Optional[List[int]] = None
    ) -> Dict[str, ProfilingResult]:
        """Profile forward pass with different input sizes.
        
        Args:
            input_data: Input data or data generator function
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary of profiling results
        """
        results = {}
        
        # Default test sizes
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
        if sequence_lengths is None:
            sequence_lengths = [32, 64, 128, 256]
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                for seq_len in sequence_lengths:
                    # Generate input data
                    if callable(input_data):
                        inputs = input_data(batch_size, seq_len)
                    elif isinstance(input_data, torch.Tensor):
                        # Expand/truncate input data
                        inputs = self._resize_input_tensor(input_data, batch_size, seq_len)
                    else:
                        # Dictionary of tensors
                        inputs = {
                            key: self._resize_input_tensor(tensor, batch_size, seq_len)
                            for key, tensor in input_data.items()
                        }
                    
                    # Move to device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    elif isinstance(inputs, dict):
                        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
                    
                    # Profile forward pass
                    operation_name = f"forward_batch{batch_size}_seq{seq_len}"
                    
                    with self.profile(operation_name, batch_size=batch_size, seq_len=seq_len):
                        if isinstance(inputs, torch.Tensor):
                            _ = self.model(inputs)
                        else:
                            _ = self.model(**inputs)
                    
                    results[operation_name] = self.results[-1]
        
        return results
    
    def _resize_input_tensor(
        self, 
        tensor: torch.Tensor, 
        batch_size: int, 
        seq_len: int
    ) -> torch.Tensor:
        """Resize input tensor to specified dimensions."""
        current_batch, current_seq = tensor.shape[:2]
        
        # Adjust batch size
        if batch_size != current_batch:
            if batch_size > current_batch:
                # Repeat tensor
                repeat_factor = (batch_size + current_batch - 1) // current_batch
                tensor = tensor.repeat(repeat_factor, *([1] * (tensor.dim() - 1)))
            tensor = tensor[:batch_size]
        
        # Adjust sequence length
        if seq_len != current_seq:
            if seq_len > current_seq:
                # Pad with zeros
                pad_length = seq_len - current_seq
                if tensor.dim() == 2:
                    padding = torch.zeros(tensor.shape[0], pad_length, dtype=tensor.dtype)
                else:
                    padding = torch.zeros(tensor.shape[0], pad_length, *tensor.shape[2:], dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
            else:
                tensor = tensor[:, :seq_len]
        
        return tensor
    
    def profile_attention_computation(
        self, 
        sequence_lengths: List[int],
        d_model: int = 512,
        n_heads: int = 8
    ) -> Dict[str, ProfilingResult]:
        """Profile attention computation separately.
        
        Args:
            sequence_lengths: List of sequence lengths to test
            d_model: Model dimension
            n_heads: Number of attention heads
            
        Returns:
            Profiling results for attention computation
        """
        results = {}
        d_k = d_model // n_heads
        
        # Create attention layer
        attention_layer = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        attention_layer.to(self.device)
        attention_layer.eval()
        
        with torch.no_grad():
            for seq_len in sequence_lengths:
                # Create dummy input
                x = torch.randn(1, seq_len, d_model, device=self.device)
                
                operation_name = f"attention_seq{seq_len}"
                
                with self.profile(operation_name, seq_len=seq_len, d_model=d_model, n_heads=n_heads):
                    _ = attention_layer(x, x, x)
                
                results[operation_name] = self.results[-1]
        
        return results
    
    def profile_positional_encoding(
        self,
        encodings: Dict[str, Callable],
        sequence_lengths: List[int],
        d_model: int = 512
    ) -> Dict[str, ProfilingResult]:
        """Profile different positional encodings.
        
        Args:
            encodings: Dictionary of encoding name to encoding function
            sequence_lengths: List of sequence lengths to test
            d_model: Model dimension
            
        Returns:
            Profiling results for encodings
        """
        results = {}
        
        for enc_name, encoding_fn in encodings.items():
            for seq_len in sequence_lengths:
                operation_name = f"{enc_name}_seq{seq_len}"
                
                with self.profile(operation_name, encoding_type=enc_name, seq_len=seq_len):
                    _ = encoding_fn(seq_len, d_model)
                
                results[operation_name] = self.results[-1]
        
        return results
    
    def analyze_computational_complexity(self) -> Dict[str, Any]:
        """Analyze computational complexity from profiling results.
        
        Returns:
            Complexity analysis
        """
        summary = self.get_summary()
        complexity_analysis = {}
        
        # Group results by operation type
        forward_results = {k: v for k, v in summary.items() if 'forward' in k}
        attention_results = {k: v for k, v in summary.items() if 'attention' in k}
        
        # Analyze forward pass complexity
        if forward_results:
            complexity_analysis['forward_pass'] = self._analyze_scaling_behavior(forward_results)
        
        # Analyze attention complexity
        if attention_results:
            complexity_analysis['attention'] = self._analyze_scaling_behavior(attention_results)
        
        return complexity_analysis
    
    def _analyze_scaling_behavior(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scaling behavior from timing results."""
        # Extract sequence lengths and corresponding times
        seq_lens = []
        times = []
        
        for operation_name, result_data in results.items():
            # Parse sequence length from operation name
            if 'seq' in operation_name:
                seq_len_str = operation_name.split('seq')[1].split('_')[0]
                try:
                    seq_len = int(seq_len_str)
                    seq_lens.append(seq_len)
                    times.append(result_data['mean_time'])
                except ValueError:
                    continue
        
        if len(seq_lens) < 2:
            return {"error": "Insufficient data for scaling analysis"}
        
        # Sort by sequence length
        sorted_data = sorted(zip(seq_lens, times))
        seq_lens, times = zip(*sorted_data)
        
        # Fit different complexity models
        seq_lens_np = np.array(seq_lens)
        times_np = np.array(times)
        
        analysis = {}
        
        # Linear complexity O(n)
        try:
            linear_coef = np.polyfit(seq_lens_np, times_np, 1)
            linear_pred = np.polyval(linear_coef, seq_lens_np)
            linear_r2 = 1 - np.sum((times_np - linear_pred)**2) / np.sum((times_np - np.mean(times_np))**2)
            analysis['linear_fit'] = {'r2': linear_r2, 'coefficients': linear_coef.tolist()}
        except:
            analysis['linear_fit'] = {'error': 'Failed to fit linear model'}
        
        # Quadratic complexity O(n^2)
        try:
            quad_coef = np.polyfit(seq_lens_np, times_np, 2)
            quad_pred = np.polyval(quad_coef, seq_lens_np)
            quad_r2 = 1 - np.sum((times_np - quad_pred)**2) / np.sum((times_np - np.mean(times_np))**2)
            analysis['quadratic_fit'] = {'r2': quad_r2, 'coefficients': quad_coef.tolist()}
        except:
            analysis['quadratic_fit'] = {'error': 'Failed to fit quadratic model'}
        
        # Determine best fit
        if 'linear_fit' in analysis and 'quadratic_fit' in analysis:
            if (analysis['quadratic_fit'].get('r2', 0) > analysis['linear_fit'].get('r2', 0) and
                analysis['quadratic_fit'].get('r2', 0) > 0.9):
                analysis['likely_complexity'] = 'O(n^2)'
            elif analysis['linear_fit'].get('r2', 0) > 0.9:
                analysis['likely_complexity'] = 'O(n)'
            else:
                analysis['likely_complexity'] = 'Unknown'
        
        analysis['data_points'] = {'sequence_lengths': seq_lens, 'times': times}
        
        return analysis


class MemoryProfiler:
    """Specialized memory profiler for detailed memory analysis."""
    
    def __init__(self, device: str = 'auto'):
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
        self.memory_snapshots = []
        
    @contextmanager
    def memory_snapshot(self, operation_name: str):
        """Context manager for taking memory snapshots.
        
        Args:
            operation_name: Name of the operation
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Take before snapshot
        before = self._get_detailed_memory_info()
        
        try:
            yield
        finally:
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Take after snapshot
            after = self._get_detailed_memory_info()
            
            snapshot = {
                'operation_name': operation_name,
                'before': before,
                'after': after,
                'difference': {
                    key: after[key] - before[key]
                    for key in before.keys()
                    if isinstance(before[key], (int, float))
                },
                'timestamp': time.time()
            }
            
            self.memory_snapshots.append(snapshot)
    
    def _get_detailed_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information."""
        info = {}
        
        # CPU memory
        process = psutil.Process()
        cpu_info = process.memory_info()
        info['cpu_rss_mb'] = cpu_info.rss / (1024 * 1024)
        info['cpu_vms_mb'] = cpu_info.vms / (1024 * 1024)
        
        # System memory
        system_memory = psutil.virtual_memory()
        info['system_total_mb'] = system_memory.total / (1024 * 1024)
        info['system_available_mb'] = system_memory.available / (1024 * 1024)
        info['system_used_percent'] = system_memory.percent
        
        # GPU memory
        if self.device == 'cuda' and torch.cuda.is_available():
            info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
            info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # GPU device info
            device_props = torch.cuda.get_device_properties(0)
            info['gpu_total_mb'] = device_props.total_memory / (1024 * 1024)
            info['gpu_name'] = device_props.name
        
        return info
    
    def analyze_memory_leaks(self) -> Dict[str, Any]:
        """Analyze potential memory leaks from snapshots.
        
        Returns:
            Memory leak analysis
        """
        if len(self.memory_snapshots) < 2:
            return {"error": "Need at least 2 snapshots for leak analysis"}
        
        analysis = {}
        
        # Check for continuous memory growth
        gpu_allocated = [s['after']['gpu_allocated_mb'] for s in self.memory_snapshots 
                        if 'gpu_allocated_mb' in s['after']]
        cpu_rss = [s['after']['cpu_rss_mb'] for s in self.memory_snapshots]
        
        if gpu_allocated:
            gpu_growth = gpu_allocated[-1] - gpu_allocated[0]
            analysis['gpu_memory_growth_mb'] = gpu_growth
            analysis['potential_gpu_leak'] = gpu_growth > 100  # More than 100MB growth
        
        cpu_growth = cpu_rss[-1] - cpu_rss[0]
        analysis['cpu_memory_growth_mb'] = cpu_growth
        analysis['potential_cpu_leak'] = cpu_growth > 50  # More than 50MB growth
        
        # Identify operations with high memory usage
        high_memory_ops = []
        for snapshot in self.memory_snapshots:
            if 'gpu_allocated_mb' in snapshot['difference']:
                if snapshot['difference']['gpu_allocated_mb'] > 100:
                    high_memory_ops.append({
                        'operation': snapshot['operation_name'],
                        'gpu_memory_mb': snapshot['difference']['gpu_allocated_mb']
                    })
        
        analysis['high_memory_operations'] = high_memory_ops
        
        return analysis
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_snapshots:
            return {"message": "No memory snapshots available"}
        
        summary = {}
        
        # Overall memory statistics
        if self.memory_snapshots:
            latest = self.memory_snapshots[-1]['after']
            summary['current_memory'] = latest
            
            # Peak memory usage
            if 'gpu_allocated_mb' in latest:
                peak_gpu = max(s['after']['gpu_allocated_mb'] for s in self.memory_snapshots 
                              if 'gpu_allocated_mb' in s['after'])
                summary['peak_gpu_memory_mb'] = peak_gpu
            
            peak_cpu = max(s['after']['cpu_rss_mb'] for s in self.memory_snapshots)
            summary['peak_cpu_memory_mb'] = peak_cpu
        
        # Per-operation memory usage
        operation_memory = defaultdict(list)
        for snapshot in self.memory_snapshots:
            op_name = snapshot['operation_name']
            if 'gpu_allocated_mb' in snapshot['difference']:
                operation_memory[op_name].append(snapshot['difference']['gpu_allocated_mb'])
        
        summary['memory_by_operation'] = {}
        for op_name, memory_values in operation_memory.items():
            summary['memory_by_operation'][op_name] = {
                'mean_mb': np.mean(memory_values),
                'max_mb': max(memory_values),
                'total_mb': sum(memory_values)
            }
        
        return summary


# Convenience functions
def profile_model(
    model: nn.Module,
    input_generator: Callable,
    batch_sizes: List[int] = [1, 4, 8],
    sequence_lengths: List[int] = [32, 64, 128],
    device: str = 'auto'
) -> Dict[str, Any]:
    """Profile model performance.
    
    Args:
        model: Model to profile
        input_generator: Function to generate inputs (batch_size, seq_len) -> inputs
        batch_sizes: List of batch sizes to test
        sequence_lengths: List of sequence lengths to test
        device: Device to use
        
    Returns:
        Profiling results
    """
    profiler = ModelProfiler(model, device)
    
    # Profile forward passes
    forward_results = profiler.profile_forward_pass(
        input_generator, batch_sizes, sequence_lengths
    )
    
    # Get summary and complexity analysis
    summary = profiler.get_summary()
    complexity = profiler.analyze_computational_complexity()
    
    return {
        'forward_results': forward_results,
        'summary': summary,
        'complexity_analysis': complexity,
        'device': device
    }


def measure_inference_time(
    model: nn.Module,
    inputs: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'auto'
) -> Dict[str, float]:
    """Measure model inference time.
    
    Args:
        model: Model to measure
        inputs: Input data
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
        device: Device to use
        
    Returns:
        Timing statistics
    """
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
    
    model.to(device)
    inputs = inputs.to(device)
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(inputs)
    
    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(inputs)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': min(times),
        'max_time': max(times),
        'median_time': np.median(times),
        'num_runs': num_runs
    }


def analyze_memory_usage(
    model: nn.Module,
    input_sizes: List[Tuple[int, int]],
    device: str = 'auto'
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Analyze memory usage for different input sizes.
    
    Args:
        model: Model to analyze
        input_sizes: List of (batch_size, seq_len) tuples
        device: Device to use
        
    Returns:
        Memory usage for each input size
    """
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
    model.to(device)
    model.eval()
    
    memory_profiler = MemoryProfiler(device)
    memory_usage = {}
    
    for batch_size, seq_len in input_sizes:
        # Generate dummy input
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        with memory_profiler.memory_snapshot(f'batch{batch_size}_seq{seq_len}'):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Get memory usage for this input size
        if memory_profiler.memory_snapshots:
            snapshot = memory_profiler.memory_snapshots[-1]
            memory_usage[(batch_size, seq_len)] = snapshot['difference']
    
    return memory_usage
