#!/usr/bin/env python3
"""
Encoding Benchmark - Comprehensive benchmarking of positional encoding methods
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Any, Tuple
import warnings

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass

from config import ModelConfig
from models import TransformerEncoder
from positional_encoding import get_positional_encoding
from utils.metrics import AttentionMetrics, EncodingMetrics
from utils.performance_profiler import PerformanceProfiler, ModelProfiler
from utils.tokenizer import SimpleTokenizer

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    sequence_lengths: List[int] = None
    model_dimensions: List[int] = None
    encoding_types: List[str] = None
    batch_sizes: List[int] = None
    num_runs: int = 10
    warmup_runs: int = 3
    device: str = 'auto'
    seed: int = 42
    
    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [16, 32, 64, 128, 256]
        if self.model_dimensions is None:
            self.model_dimensions = [128, 256, 512]
        if self.encoding_types is None:
            self.encoding_types = ['sinusoidal', 'learned', 'rope']
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 16]

class EncodingBenchmark:
    """Comprehensive benchmarking suite for positional encodings"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._setup_device()
        self.results = {}
        self.profiler = PerformanceProfiler(device=self.device)
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        print(f"🎯 Benchmark initialized on {self.device}")
    
    def _setup_device(self) -> str:
        """Setup computation device"""
        if self.config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.config.device
        
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def benchmark_encoding_generation(self) -> Dict[str, Any]:
        """Benchmark encoding generation speed and memory usage"""
        
        print("⚡ Benchmarking Encoding Generation")
        
        results = {}
        
        for encoding_type in self.config.encoding_types:
            print(f"  Testing {encoding_type}")
            
            encoding_results = {}
            
            for seq_len in self.config.sequence_lengths:
                for d_model in self.config.model_dimensions:
                    
                    config_key = f"seq{seq_len}_dim{d_model}"
                    
                    try:
                        model_config = ModelConfig(
                            d_model=d_model,
                            encoding_type=encoding_type,
                            max_seq_len=seq_len
                        )
                        
                        encoding = get_positional_encoding(model_config)
                        
                        # Warmup runs
                        for _ in range(self.config.warmup_runs):
                            if hasattr(encoding, 'forward'):
                                _ = encoding.forward(seq_len, d_model)
                        
                        # Benchmark runs
                        times = []
                        
                        for run in range(self.config.num_runs):
                            start_time = time.perf_counter()
                            
                            if hasattr(encoding, 'forward'):
                                output = encoding.forward(seq_len, d_model)
                            
                            if self.device == 'cuda':
                                torch.cuda.synchronize()
                            
                            end_time = time.perf_counter()
                            times.append(end_time - start_time)
                        
                        # Memory usage estimation
                        if hasattr(encoding, 'forward'):
                            output = encoding.forward(seq_len, d_model)
                            if isinstance(output, dict):
                                memory_usage = sum(v.numel() * v.element_size() for v in output.values())
                            else:
                                memory_usage = output.numel() * output.element_size()
                        else:
                            memory_usage = 0
                        
                        encoding_results[config_key] = {
                            'mean_time': np.mean(times),
                            'std_time': np.std(times),
                            'min_time': np.min(times),
                            'max_time': np.max(times),
                            'memory_bytes': int(memory_usage),
                            'memory_mb': memory_usage / (1024 * 1024),
                            'throughput_seq_per_sec': 1.0 / np.mean(times)
                        }
                        
                    except Exception as e:
                        encoding_results[config_key] = {'error': str(e)}
                        print(f"    ❌ {config_key}: {e}")
            
            results[encoding_type] = encoding_results
        
        self.results['encoding_generation'] = results
        return results
    
    def benchmark_model_performance(self) -> Dict[str, Any]:
        """Benchmark full model performance with different encodings"""
        
        print("🤖 Benchmarking Model Performance")
        
        results = {}
        
        for encoding_type in self.config.encoding_types:
            print(f"  Testing {encoding_type}")
            
            model_results = {}
            
            for seq_len in self.config.sequence_lengths[:3]:  # Limit for full model tests
                for d_model in [128, 256]:  # Smaller models for full benchmarks
                    for batch_size in self.config.batch_sizes:
                        
                        config_key = f"seq{seq_len}_dim{d_model}_batch{batch_size}"
                        
                        try:
                            model_config = ModelConfig(
                                d_model=d_model,
                                n_heads=max(1, d_model // 64),
                                n_layers=4,
                                encoding_type=encoding_type,
                                max_seq_len=seq_len
                            )
                            
                            model = TransformerEncoder(model_config).to(self.device)
                            model.eval()
                            
                            # Create input
                            input_ids = torch.randint(
                                0, 1000, 
                                (batch_size, seq_len),
                                device=self.device
                            )
                            
                            # Warmup
                            with torch.no_grad():
                                for _ in range(self.config.warmup_runs):
                                    _ = model(input_ids)
                            
                            # Benchmark forward pass
                            forward_times = []
                            
                            with torch.no_grad():
                                for run in range(self.config.num_runs):
                                    start_time = time.perf_counter()
                                    outputs = model(input_ids)
                                    
                                    if self.device == 'cuda':
                                        torch.cuda.synchronize()
                                    
                                    end_time = time.perf_counter()
                                    forward_times.append(end_time - start_time)
                            
                            # Memory usage
                            if self.device == 'cuda':
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                                
                                with torch.no_grad():
                                    _ = model(input_ids)
                                
                                peak_memory = torch.cuda.max_memory_allocated()
                                memory_mb = peak_memory / (1024 * 1024)
                            else:
                                memory_mb = 0  # CPU memory harder to measure accurately
                            
                            # Parameter count
                            total_params = sum(p.numel() for p in model.parameters())
                            
                            model_results[config_key] = {
                                'forward_time_mean': np.mean(forward_times),
                                'forward_time_std': np.std(forward_times),
                                'throughput_samples_per_sec': batch_size / np.mean(forward_times),
                                'peak_memory_mb': memory_mb,
                                'total_parameters': total_params,
                                'parameters_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
                            }
                            
                        except Exception as e:
                            model_results[config_key] = {'error': str(e)}
                            print(f"    ❌ {config_key}: {e}")
                        
                        finally:
                            # Clean up
                            if 'model' in locals():
                                del model
                            if self.device == 'cuda':
                                torch.cuda.empty_cache()
            
            results[encoding_type] = model_results
        
        self.results['model_performance'] = results
        return results
    
    def benchmark_quality_metrics(self) -> Dict[str, Any]:
        """Benchmark encoding quality across different configurations"""
        
        print("📊 Benchmarking Quality Metrics")
        
        results = {}
        metrics_computer = EncodingMetrics()
        
        for encoding_type in self.config.encoding_types:
            print(f"  Analyzing {encoding_type}")
            
            quality_results = {}
            
            for seq_len in self.config.sequence_lengths:
                for d_model in self.config.model_dimensions:
                    
                    config_key = f"seq{seq_len}_dim{d_model}"
                    
                    try:
                        model_config = ModelConfig(
                            d_model=d_model,
                            encoding_type=encoding_type,
                            max_seq_len=seq_len
                        )
                        
                        encoding = get_positional_encoding(model_config)
                        
                        if hasattr(encoding, 'forward'):
                            encoding_output = encoding.forward(seq_len, d_model)
                            
                            if isinstance(encoding_output, dict):  # RoPE case
                                encoding_matrix = torch.cat([
                                    encoding_output['cos'].squeeze(0),
                                    encoding_output['sin'].squeeze(0)
                                ], dim=-1)
                            else:
                                encoding_matrix = encoding_output.squeeze(0)
                            
                            # Compute quality metrics
                            quality_metrics = metrics_computer.compute_encoding_quality(encoding_matrix)
                            
                            # Position similarity analysis
                            similarities = metrics_computer.compute_position_similarity(encoding_matrix)
                            
                            quality_results[config_key] = {
                                **quality_metrics,
                                'mean_similarity': float(similarities.mean()),
                                'similarity_std': float(similarities.std()),
                                'max_similarity': float(similarities.max()),
                                'min_similarity': float(similarities.min())
                            }
                            
                    except Exception as e:
                        quality_results[config_key] = {'error': str(e)}
                        print(f"    ❌ {config_key}: {e}")
            
            results[encoding_type] = quality_results
        
        self.results['quality_metrics'] = results
        return results
    
    def benchmark_extrapolation(self) -> Dict[str, Any]:
        """Benchmark extrapolation capabilities"""
        
        print("🔍 Benchmarking Extrapolation")
        
        results = {}
        metrics_computer = EncodingMetrics()
        
        train_lengths = [32, 64]
        test_multipliers = [2, 4, 8]  # Test at 2x, 4x, 8x training length
        
        for encoding_type in self.config.encoding_types:
            print(f"  Testing {encoding_type}")
            
            extrapolation_results = {}
            
            for train_len in train_lengths:
                for d_model in [128, 256]:  # Limit dimensions
                    
                    baseline_key = f"train{train_len}_dim{d_model}"
                    
                    try:
                        model_config = ModelConfig(
                            d_model=d_model,
                            encoding_type=encoding_type,
                            max_seq_len=train_len * max(test_multipliers)
                        )
                        
                        encoding = get_positional_encoding(model_config)
                        
                        # Generate baseline encoding
                        if hasattr(encoding, 'forward'):
                            baseline_output = encoding.forward(train_len, d_model)
                            
                            if isinstance(baseline_output, dict):
                                baseline_matrix = torch.cat([
                                    baseline_output['cos'].squeeze(0),
                                    baseline_output['sin'].squeeze(0)
                                ], dim=-1)
                            else:
                                baseline_matrix = baseline_output.squeeze(0)
                            
                            baseline_similarities = metrics_computer.compute_position_similarity(baseline_matrix)
                            
                            test_results = {}
                            
                            for multiplier in test_multipliers:
                                test_len = train_len * multiplier
                                
                                if encoding_type == 'learned' and test_len > train_len:
                                    # Learned encodings can't extrapolate
                                    test_results[f'{multiplier}x'] = {
                                        'extrapolation_possible': False,
                                        'error': 'Learned encodings cannot extrapolate beyond training length'
                                    }
                                    continue
                                
                                try:
                                    test_output = encoding.forward(test_len, d_model)
                                    
                                    if isinstance(test_output, dict):
                                        test_matrix = torch.cat([
                                            test_output['cos'].squeeze(0),
                                            test_output['sin'].squeeze(0)
                                        ], dim=-1)
                                    else:
                                        test_matrix = test_output.squeeze(0)
                                    
                                    # Compare patterns in overlapping region
                                    overlap_test_similarities = metrics_computer.compute_position_similarity(
                                        test_matrix[:train_len]
                                    )
                                    
                                    # Correlation between baseline and test similarities
                                    correlation = torch.corrcoef(torch.stack([
                                        baseline_similarities.flatten(),
                                        overlap_test_similarities.flatten()
                                    ]))[0, 1].item()
                                    
                                    # Full sequence quality
                                    full_quality = metrics_computer.compute_encoding_quality(test_matrix)
                                    
                                    test_results[f'{multiplier}x'] = {
                                        'extrapolation_possible': True,
                                        'similarity_correlation': correlation,
                                        'quality_degradation': full_quality.get('distinguishability', 0),
                                        'test_length': test_len
                                    }
                                    
                                except Exception as e:
                                    test_results[f'{multiplier}x'] = {
                                        'extrapolation_possible': False,
                                        'error': str(e)
                                    }
                            
                            extrapolation_results[baseline_key] = test_results
                            
                    except Exception as e:
                        extrapolation_results[baseline_key] = {'error': str(e)}
                        print(f"    ❌ {baseline_key}: {e}")
            
            results[encoding_type] = extrapolation_results
        
        self.results['extrapolation'] = results
        return results
    
    def benchmark_attention_quality(self) -> Dict[str, Any]:
        """Benchmark attention quality with different encodings"""
        
        print("👁️ Benchmarking Attention Quality")
        
        results = {}
        attention_metrics = AttentionMetrics()
        tokenizer = SimpleTokenizer()
        
        # Test sentences
        test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require extensive training data.",
            "Attention mechanisms enable transformers to focus selectively."
        ]
        
        for encoding_type in self.config.encoding_types:
            print(f"  Testing {encoding_type}")
            
            attention_results = {}
            
            for sentence_idx, sentence in enumerate(test_sentences):
                tokens = tokenizer.tokenize(sentence)
                
                if len(tokens) > 32:  # Limit token length
                    tokens = tokens[:32]
                
                try:
                    model_config = ModelConfig(
                        d_model=256,
                        n_heads=8,
                        n_layers=4,
                        encoding_type=encoding_type,
                        max_seq_len=len(tokens)
                    )
                    
                    model = TransformerEncoder(model_config).to(self.device)
                    model.eval()
                    
                    input_ids = torch.tensor([list(range(len(tokens)))], device=self.device)
                    
                    with torch.no_grad():
                        outputs = model(input_ids, store_visualizations=True)
                    
                    attention_weights = outputs['attention_weights']
                    
                    if attention_weights:
                        # Analyze attention patterns for each layer
                        layer_analyses = []
                        
                        for layer_idx, layer_attention in enumerate(attention_weights):
                            analysis = attention_metrics.analyze_attention_patterns(layer_attention)
                            
                            layer_analyses.append({
                                'layer': layer_idx,
                                'entropy': float(analysis['entropy']['mean_entropy']),
                                'sparsity': analysis['sparsity']['gini_coefficient'],
                                'pattern_type': analysis.get('pattern_type', 'unknown')
                            })
                        
                        attention_results[f'sentence_{sentence_idx}'] = {
                            'sentence': sentence,
                            'num_tokens': len(tokens),
                            'layer_analyses': layer_analyses,
                            'mean_entropy': np.mean([la['entropy'] for la in layer_analyses]),
                            'mean_sparsity': np.mean([la['sparsity'] for la in layer_analyses])
                        }
                    
                except Exception as e:
                    attention_results[f'sentence_{sentence_idx}'] = {'error': str(e)}
                    print(f"    ❌ sentence_{sentence_idx}: {e}")
                
                finally:
                    if 'model' in locals():
                        del model
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            results[encoding_type] = attention_results
        
        self.results['attention_quality'] = results
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark categories"""
        
        print("🚀 Running Comprehensive Benchmark")
        print("=" * 50)
        
        benchmark_start = time.time()
        
        try:
            # Run all benchmarks
            self.benchmark_encoding_generation()
            self.benchmark_quality_metrics()
            self.benchmark_extrapolation()
            self.benchmark_model_performance()
            self.benchmark_attention_quality()
            
            benchmark_time = time.time() - benchmark_start
            
            # Add metadata
            self.results['metadata'] = {
                'benchmark_time_seconds': benchmark_time,
                'device': self.device,
                'config': self.config.__dict__,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                self.results['metadata'][f'cuda_device_name'] = torch.cuda.get_device_name()
                self.results['metadata']['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            
            print(f"\n✅ Comprehensive benchmark completed in {benchmark_time:.1f}s")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, output_path: str):
        """Save benchmark results to file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_path}")
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Create summary report of benchmark results"""
        
        if not self.results:
            return {}
        
        summary = {
            'overview': {
                'encodings_tested': self.config.encoding_types,
                'sequence_lengths': self.config.sequence_lengths,
                'model_dimensions': self.config.model_dimensions,
                'total_configurations': len(self.config.encoding_types) * len(self.config.sequence_lengths) * len(self.config.model_dimensions)
            },
            'performance_winners': {},
            'quality_winners': {},
            'recommendations': []
        }
        
        # Analyze performance winners
        if 'encoding_generation' in self.results:
            fastest_encoding = None
            fastest_time = float('inf')
            
            for encoding_type, configs in self.results['encoding_generation'].items():
                avg_times = []
                for config_name, config_results in configs.items():
                    if 'mean_time' in config_results:
                        avg_times.append(config_results['mean_time'])
                
                if avg_times:
                    mean_time = np.mean(avg_times)
                    if mean_time < fastest_time:
                        fastest_time = mean_time
                        fastest_encoding = encoding_type
            
            summary['performance_winners']['fastest_encoding'] = fastest_encoding
        
        # Analyze quality winners
        if 'quality_metrics' in self.results:
            best_quality_encoding = None
            best_quality_score = 0
            
            for encoding_type, configs in self.results['quality_metrics'].items():
                quality_scores = []
                for config_name, config_results in configs.items():
                    if 'distinguishability' in config_results:
                        quality_scores.append(config_results['distinguishability'])
                
                if quality_scores:
                    mean_quality = np.mean(quality_scores)
                    if mean_quality > best_quality_score:
                        best_quality_score = mean_quality
                        best_quality_encoding = encoding_type
            
            summary['quality_winners']['best_quality'] = best_quality_encoding
        
        # Generate recommendations
        recommendations = []
        
        if 'extrapolation' in self.results:
            # Check extrapolation capabilities
            rope_extrapolates = any(
                'rope' in self.results['extrapolation'] and 
                any('extrapolation_possible' in test_result and test_result['extrapolation_possible'] 
                    for test_config in self.results['extrapolation']['rope'].values()
                    for test_result in test_config.values() if isinstance(test_result, dict))
            )
            
            if rope_extrapolates:
                recommendations.append("RoPE shows good extrapolation capabilities for longer sequences")
        
        if fastest_encoding:
            recommendations.append(f"{fastest_encoding} encoding offers the best generation performance")
        
        if best_quality_encoding:
            recommendations.append(f"{best_quality_encoding} encoding provides the highest quality scores")
        
        summary['recommendations'] = recommendations
        
        return summary

def main():
    """Main benchmark runner"""
    
    parser = argparse.ArgumentParser(description='Benchmark positional encoding methods')
    
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run benchmarks on'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with reduced configurations'
    )
    
    parser.add_argument(
        '--encodings',
        nargs='+',
        default=['sinusoidal', 'learned', 'rope'],
        help='Encoding types to benchmark'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of runs for each benchmark'
    )
    
    args = parser.parse_args()
    
    print("🏁 Positional Encoding Benchmark Suite")
    print("=" * 50)
    
    # Configure benchmark
    if args.quick:
        config = BenchmarkConfig(
            sequence_lengths=[16, 64],
            model_dimensions=[128, 256],
            encoding_types=args.encodings,
            batch_sizes=[1, 4],
            num_runs=5,
            device=args.device
        )
    else:
        config = BenchmarkConfig(
            encoding_types=args.encodings,
            num_runs=args.num_runs,
            device=args.device
        )
    
    # Run benchmark
    benchmark = EncodingBenchmark(config)
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        
        if results:
            # Save results
            benchmark.save_results(args.output)
            
            # Create and save summary
            summary = benchmark.create_summary_report()
            summary_path = args.output.replace('.json', '_summary.json')
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📊 Summary saved to {summary_path}")
            
            # Print key findings
            print("\n🎯 Key Findings:")
            if summary.get('performance_winners', {}).get('fastest_encoding'):
                print(f"  Fastest: {summary['performance_winners']['fastest_encoding']}")
            if summary.get('quality_winners', {}).get('best_quality'):
                print(f"  Best Quality: {summary['quality_winners']['best_quality']}")
            
            for rec in summary.get('recommendations', []):
                print(f"  💡 {rec}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark stopped by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    main()
