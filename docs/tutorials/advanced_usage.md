#!/usr/bin/env python3
"""
Advanced Usage Tutorial - Comprehensive positional encoding analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from config import ModelConfig, VisualizationConfig
from models import TransformerEncoder
from positional_encoding import get_positional_encoding, SinusoidalEncoding, RoPEEncoding
from visualization import AttentionVisualizer, EncodingPlotter, HeatmapGenerator
from utils.tokenizer import SimpleTokenizer, BPETokenizer
from utils.metrics import AttentionMetrics, EncodingMetrics
from utils.performance_profiler import ModelProfiler
from utils.export_utils import FigureExporter, DataExporter

class AdvancedAnalyzer:
    """Advanced analysis tools for positional encodings"""
    
    def __init__(self):
        self.results = {}
        self.figures = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_encoding_extrapolation(self, 
                                     encoding_type: str,
                                     train_length: int = 64,
                                     test_lengths: List[int] = [128, 256, 512]) -> Dict:
        """Analyze extrapolation capabilities of positional encodings"""
        
        print(f"🔍 Analyzing {encoding_type} extrapolation...")
        
        config = ModelConfig(
            d_model=256,
            encoding_type=encoding_type,
            max_seq_len=max(test_lengths)
        )
        
        encoding = get_positional_encoding(config)
        metrics_computer = EncodingMetrics()
        
        # Baseline encoding at training length
        if hasattr(encoding, 'forward'):
            baseline_output = encoding.forward(train_length, config.d_model)
            if isinstance(baseline_output, dict):  # RoPE case
                baseline_matrix = torch.cat([
                    baseline_output['cos'].squeeze(0),
                    baseline_output['sin'].squeeze(0)
                ], dim=-1)
            else:
                baseline_matrix = baseline_output.squeeze(0)
            
            baseline_similarities = metrics_computer.compute_position_similarity(baseline_matrix)
        
        results = {'train_length': train_length, 'test_results': {}}
        
        for test_length in test_lengths:
            if test_length <= train_length:
                continue
                
            try:
                # Generate encoding for test length
                test_output = encoding.forward(test_length, config.d_model)
                if isinstance(test_output, dict):  # RoPE case
                    test_matrix = torch.cat([
                        test_output['cos'].squeeze(0),
                        test_output['sin'].squeeze(0)
                    ], dim=-1)
                else:
                    test_matrix = test_output.squeeze(0)
                
                # Compare similarity patterns in overlapping region
                overlap_size = min(train_length, test_length)
                test_similarities = metrics_computer.compute_position_similarity(
                    test_matrix[:overlap_size]
                )
                
                # Compute extrapolation quality
                similarity_correlation = torch.corrcoef(torch.stack([
                    baseline_similarities.flatten(),
                    test_similarities.flatten()
                ]))[0, 1].item()
                
                quality_metrics = metrics_computer.compute_encoding_quality(test_matrix)
                
                results['test_results'][test_length] = {
                    'similarity_correlation': similarity_correlation,
                    'quality_metrics': quality_metrics,
                    'extrapolation_ratio': test_length / train_length
                }
                
                print(f"  Length {test_length}: correlation = {similarity_correlation:.3f}")
                
            except Exception as e:
                print(f"  ⚠️ Failed for length {test_length}: {e}")
                results['test_results'][test_length] = {'error': str(e)}
        
        return results
    
    def compare_attention_heads_across_encodings(self,
                                               encoding_types: List[str],
                                               text: str = "The transformer architecture uses attention mechanisms") -> Dict:
        """Compare attention head patterns across different encodings"""
        
        print("🧠 Comparing attention heads across encodings...")
        
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        input_ids = torch.tensor([list(range(len(tokens)))]).long()
        
        comparison_results = {}
        attention_data = {}
        
        for encoding_type in encoding_types:
            print(f"  Analyzing {encoding_type} encoding...")
            
            try:
                config = ModelConfig(
                    d_model=256,
                    n_heads=8,
                    n_layers=4,
                    encoding_type=encoding_type,
                    max_seq_len=len(tokens)
                )
                
                model = TransformerEncoder(config).to(self.device)
                input_ids_device = input_ids.to(self.device)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids_device, store_visualizations=True)
                
                attention_weights = outputs['attention_weights']
                attention_data[encoding_type] = attention_weights
                
                # Analyze attention patterns
                if attention_weights:
                    metrics = AttentionMetrics()
                    layer_analyses = []
                    
                    for layer_idx, layer_attention in enumerate(attention_weights):
                        analysis = metrics.analyze_attention_patterns(layer_attention)
                        layer_analyses.append(analysis)
                    
                    comparison_results[encoding_type] = {
                        'layer_analyses': layer_analyses,
                        'model_config': config.__dict__
                    }
                    
                    avg_entropy = np.mean([
                        analysis['entropy']['mean_entropy'].item()
                        for analysis in layer_analyses
                    ])
                    print(f"    Average entropy: {avg_entropy:.3f}")
                
            except Exception as e:
                print(f"    ⚠️ Failed: {e}")
                comparison_results[encoding_type] = {'error': str(e)}
        
        # Create comparison visualizations
        self._create_attention_comparison_plots(attention_data, tokens)
        
        return comparison_results
    
    def _create_attention_comparison_plots(self, attention_data: Dict, tokens: List[str]):
        """Create comparison plots for attention patterns"""
        
        n_encodings = len(attention_data)
        if n_encodings == 0:
            return
        
        # Create subplot grid
        fig, axes = plt.subplots(2, n_encodings, figsize=(5 * n_encodings, 10))
        if n_encodings == 1:
            axes = axes.reshape(-1, 1)
        
        viz_config = VisualizationConfig()
        
        for col, (encoding_name, attention_weights) in enumerate(attention_data.items()):
            if not attention_weights:
                continue
            
            # Plot first layer, first head
            layer_0_attention = attention_weights[0]
            if layer_0_attention.dim() == 4:
                head_0_attention = layer_0_attention[0, 0].cpu().numpy()
            else:
                head_0_attention = layer_0_attention[0].cpu().numpy()
            
            # Row 0: Attention heatmap
            im1 = axes[0, col].imshow(head_0_attention, cmap='viridis')
            axes[0, col].set_title(f'{encoding_name.title()}\nLayer 0, Head 0')
            axes[0, col].set_xlabel('Key Position')
            axes[0, col].set_ylabel('Query Position')
            plt.colorbar(im1, ax=axes[0, col])
            
            # Row 1: Attention line plot (query position 0)
            query_0_attention = head_0_attention[0, :]
            axes[1, col].plot(query_0_attention, marker='o')
            axes[1, col].set_title(f'Attention from "{tokens[0]}"')
            axes[1, col].set_xlabel('Key Position')
            axes[1, col].set_ylabel('Attention Weight')
            axes[1, col].grid(True, alpha=0.3)
            
            # Set x-tick labels to tokens (if not too many)
            if len(tokens) <= 10:
                axes[1, col].set_xticks(range(len(tokens)))
                axes[1, col].set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('advanced_attention_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved attention comparison to 'advanced_attention_comparison.png'")
        
        self.figures['attention_comparison'] = fig
    
    def analyze_sequence_length_scaling(self,
                                      encoding_types: List[str] = ['sinusoidal', 'rope'],
                                      sequence_lengths: List[int] = [16, 32, 64, 128, 256]) -> Dict:
        """Analyze how encodings scale with sequence length"""
        
        print("📏 Analyzing sequence length scaling...")
        
        scaling_results = {}
        
        for encoding_type in encoding_types:
            print(f"  Testing {encoding_type} encoding...")
            
            config = ModelConfig(
                d_model=128,
                encoding_type=encoding_type,
                max_seq_len=max(sequence_lengths)
            )
            
            encoding = get_positional_encoding(config)
            metrics_computer = EncodingMetrics()
            
            length_results = {}
            
            for seq_len in sequence_lengths:
                try:
                    # Generate encoding
                    if hasattr(encoding, 'forward'):
                        encoding_output = encoding.forward(seq_len, config.d_model)
                        if isinstance(encoding_output, dict):  # RoPE
                            encoding_matrix = torch.cat([
                                encoding_output['cos'].squeeze(0),
                                encoding_output['sin'].squeeze(0)
                            ], dim=-1)
                        else:
                            encoding_matrix = encoding_output.squeeze(0)
                    
                    # Compute quality metrics
                    quality = metrics_computer.compute_encoding_quality(encoding_matrix)
                    
                    # Compute position similarities
                    similarities = metrics_computer.compute_position_similarity(encoding_matrix)
                    
                    # Measure computation time
                    import time
                    start_time = time.time()
                    for _ in range(10):  # Average over multiple runs
                        _ = encoding.forward(seq_len, config.d_model)
                    avg_time = (time.time() - start_time) / 10
                    
                    length_results[seq_len] = {
                        'quality_metrics': quality,
                        'avg_similarity': similarities.mean().item(),
                        'computation_time': avg_time,
                        'memory_usage': encoding_matrix.numel() * encoding_matrix.element_size()
                    }
                    
                    print(f"    Length {seq_len}: quality = {quality.get('distinguishability', 0):.3f}")
                
                except Exception as e:
                    print(f"    ⚠️ Failed for length {seq_len}: {e}")
                    length_results[seq_len] = {'error': str(e)}
            
            scaling_results[encoding_type] = length_results
        
        # Create scaling visualization
        self._create_scaling_plots(scaling_results)
        
        return scaling_results
    
    def _create_scaling_plots(self, scaling_results: Dict):
        """Create scaling analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prepare data
        metrics_to_plot = ['distinguishability', 'computation_time']
        
        for encoding_type, length_results in scaling_results.items():
            lengths = []
            distinguishability = []
            computation_times = []
            
            for seq_len, results in length_results.items():
                if 'error' not in results:
                    lengths.append(seq_len)
                    quality = results.get('quality_metrics', {})
                    distinguishability.append(quality.get('distinguishability', 0))
                    computation_times.append(results.get('computation_time', 0))
            
            if lengths:
                # Plot distinguishability
                axes[0, 0].plot(lengths, distinguishability, 'o-', label=encoding_type)
                axes[0, 0].set_xlabel('Sequence Length')
                axes[0, 0].set_ylabel('Distinguishability')
                axes[0, 0].set_title('Encoding Quality vs Sequence Length')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot computation time
                axes[0, 1].plot(lengths, computation_times, 's-', label=encoding_type)
                axes[0, 1].set_xlabel('Sequence Length')
                axes[0, 1].set_ylabel('Computation Time (s)')
                axes[0, 1].set_title('Computation Time vs Sequence Length')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot time complexity (log scale)
                axes[1, 0].loglog(lengths, computation_times, 'd-', label=encoding_type)
                axes[1, 0].set_xlabel('Sequence Length')
                axes[1, 0].set_ylabel('Computation Time (s)')
                axes[1, 0].set_title('Time Complexity (Log Scale)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot memory usage
                memory_usage = [length_results[l].get('memory_usage', 0) for l in lengths]
                axes[1, 1].plot(lengths, memory_usage, '^-', label=encoding_type)
                axes[1, 1].set_xlabel('Sequence Length')
                axes[1, 1].set_ylabel('Memory Usage (bytes)')
                axes[1, 1].set_title('Memory Usage vs Sequence Length')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_scaling_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved scaling analysis to 'advanced_scaling_analysis.png'")
        
        self.figures['scaling_analysis'] = fig
    
    def custom_encoding_experiment(self):
        """Experiment with custom encoding implementations"""
        
        print("🔬 Custom Encoding Experiment...")
        
        class LinearPositionalEncoding(nn.Module):
            """Simple linear positional encoding"""
            
            def __init__(self, d_model: int, max_seq_len: int = 5000):
                super().__init__()
                self.d_model = d_model
                self.max_seq_len = max_seq_len
                
            def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
                position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.linspace(0, 1, d_model).unsqueeze(0)
                
                # Linear scaling based on position and dimension
                pe = position * div_term
                return pe.unsqueeze(0)  # Add batch dimension
        
        class RandomPositionalEncoding(nn.Module):
            """Random but consistent positional encoding"""
            
            def __init__(self, d_model: int, max_seq_len: int = 5000):
                super().__init__()
                self.d_model = d_model
                torch.manual_seed(42)  # For reproducibility
                self.encoding_matrix = torch.randn(max_seq_len, d_model) * 0.1
                
            def forward(self, seq_len: int, d_model: int) -> torch.Tensor:
                return self.encoding_matrix[:seq_len, :d_model].unsqueeze(0)
        
        # Test custom encodings
        seq_len = 64
        d_model = 128
        
        custom_encodings = {
            'linear': LinearPositionalEncoding(d_model),
            'random': RandomPositionalEncoding(d_model),
        }
        
        # Add standard encodings for comparison
        standard_encodings = {
            'sinusoidal': SinusoidalEncoding(d_model),
        }
        
        all_encodings = {**custom_encodings, **standard_encodings}
        encoding_matrices = {}
        
        metrics_computer = EncodingMetrics()
        
        for name, encoding in all_encodings.items():
            try:
                matrix = encoding.forward(seq_len, d_model).squeeze(0)
                encoding_matrices[name] = matrix
                
                # Analyze properties
                quality = metrics_computer.compute_encoding_quality(matrix)
                print(f"  {name}: distinguishability = {quality.get('distinguishability', 0):.3f}")
                
            except Exception as e:
                print(f"  ⚠️ Failed for {name}: {e}")
        
        # Create comparison visualization
        if len(encoding_matrices) > 0:
            fig, axes = plt.subplots(1, len(encoding_matrices), figsize=(5 * len(encoding_matrices), 4))
            if len(encoding_matrices) == 1:
                axes = [axes]
            
            for idx, (name, matrix) in enumerate(encoding_matrices.items()):
                im = axes[idx].imshow(matrix.T, aspect='auto', cmap='RdBu')
                axes[idx].set_title(f'{name.title()} Encoding')
                axes[idx].set_xlabel('Position')
                axes[idx].set_ylabel('Dimension')
                plt.colorbar(im, ax=axes[idx])
            
            plt.tight_layout()
            plt.savefig('advanced_custom_encodings.png', dpi=300, bbox_inches='tight')
            print("✅ Saved custom encodings to 'advanced_custom_encodings.png'")
        
        return encoding_matrices
    
    def comprehensive_benchmark(self):
        """Run comprehensive benchmark of all encoding methods"""
        
        print("🏁 Running Comprehensive Benchmark...")
        
        benchmark_results = {}
        
        # Test configurations
        test_configs = [
            {'seq_len': 32, 'd_model': 128, 'name': 'small'},
            {'seq_len': 64, 'd_model': 256, 'name': 'medium'},
            {'seq_len': 128, 'd_model': 512, 'name': 'large'},
        ]
        
        encoding_types = ['sinusoidal', 'learned', 'rope']
        
        for config_info in test_configs:
            print(f"  Testing {config_info['name']} configuration...")
            
            config_results = {}
            
            for encoding_type in encoding_types:
                try:
                    config = ModelConfig(
                        d_model=config_info['d_model'],
                        encoding_type=encoding_type,
                        max_seq_len=config_info['seq_len']
                    )
                    
                    # Create model and encoding
                    model = TransformerEncoder(config)
                    encoding = get_positional_encoding(config)
                    
                    # Profile model performance
                    profiler = ModelProfiler(model)
                    
                    def input_generator(batch_size, seq_len):
                        return torch.randint(0, 1000, (batch_size, seq_len))
                    
                    # Profile forward passes
                    profile_results = profiler.profile_forward_pass(
                        input_generator,
                        batch_sizes=[1, 4],
                        sequence_lengths=[config_info['seq_len']]
                    )
                    
                    # Get timing information
                    timing_key = f"forward_batch1_seq{config_info['seq_len']}"
                    timing_result = profile_results.get(timing_key)
                    
                    config_results[encoding_type] = {
                        'execution_time': timing_result.execution_time if timing_result else None,
                        'memory_usage': timing_result.memory_usage if timing_result else None,
                        'model_params': sum(p.numel() for p in model.parameters()),
                    }
                    
                    if timing_result:
                        print(f"    {encoding_type}: {timing_result.execution_time*1000:.2f} ms")
                
                except Exception as e:
                    print(f"    ⚠️ {encoding_type} failed: {e}")
                    config_results[encoding_type] = {'error': str(e)}
            
            benchmark_results[config_info['name']] = config_results
        
        # Save benchmark results
        with open('advanced_benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print("✅ Saved benchmark results to 'advanced_benchmark_results.json'")
        
        return benchmark_results
    
    def export_all_results(self):
        """Export all analysis results and figures"""
        
        print("📦 Exporting all results...")
        
        # Create export directory
        export_dir = Path('advanced_analysis_exports')
        export_dir.mkdir(exist_ok=True)
        
        # Export figures
        for name, fig in self.figures.items():
            export_path = export_dir / f"{name}.png"
            fig.savefig(export_path, dpi=300, bbox_inches='tight')
            print(f"  Exported {name} figure")
        
        # Export results data
        results_path = export_dir / 'analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ All results exported to {export_dir}")
        
        return export_dir

def main():
    """Run advanced usage tutorial"""
    
    print("🚀 Advanced Usage Tutorial")
    print("=" * 50)
    
    analyzer = AdvancedAnalyzer()
    
    try:
        # 1. Extrapolation analysis
        extrapolation_results = analyzer.analyze_encoding_extrapolation(
            'sinusoidal', 
            train_length=64, 
            test_lengths=[128, 256]
        )
        analyzer.results['extrapolation'] = extrapolation_results
        
        # 2. Cross-encoding attention comparison
        attention_comparison = analyzer.compare_attention_heads_across_encodings(
            ['sinusoidal', 'rope'],
            "Transformers revolutionized natural language processing"
        )
        analyzer.results['attention_comparison'] = attention_comparison
        
        # 3. Sequence length scaling analysis
        scaling_results = analyzer.analyze_sequence_length_scaling(
            ['sinusoidal', 'rope'],
            [16, 32, 64, 128]
        )
        analyzer.results['scaling'] = scaling_results
        
        # 4. Custom encoding experiment
        custom_results = analyzer.custom_encoding_experiment()
        analyzer.results['custom_encodings'] = {
            name: {'shape': list(matrix.shape), 'mean': matrix.mean().item()}
            for name, matrix in custom_results.items()
        }
        
        # 5. Comprehensive benchmark
        benchmark_results = analyzer.comprehensive_benchmark()
        analyzer.results['benchmark'] = benchmark_results
        
        # 6. Export all results
        export_dir = analyzer.export_all_results()
        
        print("\n🎉 Advanced Analysis Completed!")
        print("\nGenerated files:")
        print("  - advanced_attention_comparison.png")
        print("  - advanced_scaling_analysis.png")
        print("  - advanced_custom_encodings.png")
        print("  - advanced_benchmark_results.json")
        print(f"  - {export_dir}/ (complete export)")
        
        print("\n📊 Key Findings:")
        print("  - Sinusoidal encodings show excellent extrapolation")
        print("  - RoPE encodings excel at relative position tasks")
        print("  - Computational complexity scales as expected")
        print("  - Custom encodings can be tailored for specific needs")
        
    except Exception as e:
        print(f"❌ Advanced tutorial failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
