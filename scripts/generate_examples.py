#!/usr/bin/env python3
"""
Example Generator - Generate example visualizations and analysis results
"""

import sys
import os
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import torch
import matplotlib.pyplot as plt
import numpy as np

from config import ModelConfig, VisualizationConfig
from models import TransformerEncoder
from positional_encoding import get_positional_encoding
from visualization import AttentionVisualizer, EncodingPlotter, HeatmapGenerator
from utils.tokenizer import SimpleTokenizer
from utils.metrics import AttentionMetrics, EncodingMetrics
from utils.export_utils import FigureExporter, DataExporter

class ExampleGenerator:
    """Generate comprehensive examples for documentation and testing"""
    
    def __init__(self, output_dir: str = 'examples'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'configs').mkdir(exist_ok=True)
        
        self.examples = {}
        self.figures = {}
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def generate_encoding_examples(self):
        """Generate positional encoding examples"""
        
        print("🔢 Generating Encoding Examples")
        
        encoding_types = ['sinusoidal', 'learned', 'rope']
        configs = {
            'small': {'d_model': 64, 'max_seq_len': 32},
            'medium': {'d_model': 128, 'max_seq_len': 64},
            'large': {'d_model': 256, 'max_seq_len': 128}
        }
        
        encoding_examples = {}
        
        for size_name, size_config in configs.items():
            print(f"  Size: {size_name}")
            
            size_examples = {}
            
            for encoding_type in encoding_types:
                try:
                    config = ModelConfig(
                        d_model=size_config['d_model'],
                        max_seq_len=size_config['max_seq_len'],
                        encoding_type=encoding_type
                    )
                    
                    encoding = get_positional_encoding(config)
                    seq_len = size_config['max_seq_len']
                    
                    # Generate encoding
                    if hasattr(encoding, 'forward'):
                        encoding_output = encoding.forward(seq_len, config.d_model)
                        
                        if isinstance(encoding_output, dict):  # RoPE case
                            encoding_matrix = torch.cat([
                                encoding_output['cos'].squeeze(0),
                                encoding_output['sin'].squeeze(0)
                            ], dim=-1)
                        else:
                            encoding_matrix = encoding_output.squeeze(0)
                        
                        # Create visualization
                        viz_config = VisualizationConfig()
                        plotter = EncodingPlotter(viz_config)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        im = ax.imshow(encoding_matrix.T, aspect='auto', cmap='RdBu')
                        ax.set_title(f'{encoding_type.title()} Encoding ({size_name})')
                        ax.set_xlabel('Position')
                        ax.set_ylabel('Dimension')
                        plt.colorbar(im)
                        
                        # Save figure
                        fig_path = self.output_dir / 'images' / f'encoding_{encoding_type}_{size_name}.png'
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        # Compute metrics
                        metrics_computer = EncodingMetrics()
                        quality_metrics = metrics_computer.compute_encoding_quality(encoding_matrix)
                        similarities = metrics_computer.compute_position_similarity(encoding_matrix)
                        
                        size_examples[encoding_type] = {
                            'config': config.__dict__,
                            'shape': list(encoding_matrix.shape),
                            'range': [float(encoding_matrix.min()), float(encoding_matrix.max())],
                            'quality_metrics': quality_metrics,
                            'mean_similarity': float(similarities.mean()),
                            'figure_path': str(fig_path)
                        }
                        
                        print(f"    ✅ {encoding_type}: {encoding_matrix.shape}")
                        
                except Exception as e:
                    print(f"    ❌ {encoding_type}: {e}")
                    size_examples[encoding_type] = {'error': str(e)}
            
            encoding_examples[size_name] = size_examples
        
        # Save encoding examples data
        data_path = self.output_dir / 'data' / 'encoding_examples.json'
        with open(data_path, 'w') as f:
            json.dump(encoding_examples, f, indent=2, default=str)
        
        self.examples['encodings'] = encoding_examples
        print(f"  ✅ Saved encoding examples to {data_path}")
        
        return encoding_examples
    
    def generate_attention_examples(self):
        """Generate attention pattern examples"""
        
        print("👁️ Generating Attention Examples")
        
        # Sample texts of different types
        sample_texts = {
            'simple': "The cat sat on the mat.",
            'complex': "The transformer architecture revolutionized natural language processing.",
            'long': "Machine learning models require extensive training data to learn complex patterns and relationships.",
            'repetitive': "The the cat cat sat sat on on the the mat mat."
        }
        
        attention_examples = {}
        
        for text_type, text in sample_texts.items():
            print(f"  Text type: {text_type}")
            
            try:
                # Tokenize text
                tokenizer = SimpleTokenizer()
                tokens = tokenizer.tokenize(text)
                
                if len(tokens) > 32:  # Limit length
                    tokens = tokens[:32]
                
                # Create model
                config = ModelConfig(
                    d_model=256,
                    n_heads=8,
                    n_layers=4,
                    max_seq_len=len(tokens),
                    encoding_type='sinusoidal'
                )
                
                model = TransformerEncoder(config)
                
                # Generate input
                input_ids = torch.tensor([list(range(len(tokens)))]).long()
                
                # Forward pass
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids, store_visualizations=True)
                
                attention_weights = outputs['attention_weights']
                
                if attention_weights:
                    # Create visualizations for each layer
                    viz_config = VisualizationConfig()
                    visualizer = AttentionVisualizer(viz_config)
                    
                    layer_figures = {}
                    
                    for layer_idx, layer_attention in enumerate(attention_weights):
                        # Create attention heatmap
                        fig = visualizer.visualize_attention_matrix(
                            layer_attention,
                            tokens=tokens,
                            head_idx=0,
                            title=f'Attention - {text_type.title()} Text, Layer {layer_idx}'
                        )
                        
                        # Save figure
                        fig_path = self.output_dir / 'images' / f'attention_{text_type}_layer_{layer_idx}.png'
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                        layer_figures[f'layer_{layer_idx}'] = str(fig_path)
                    
                    # Analyze attention patterns
                    metrics_computer = AttentionMetrics()
                    analysis = metrics_computer.analyze_attention_patterns(attention_weights[0])
                    
                    attention_examples[text_type] = {
                        'text': text,
                        'tokens': tokens,
                        'num_layers': len(attention_weights),
                        'attention_shape': list(attention_weights[0].shape),
                        'analysis': {
                            'entropy': float(analysis['entropy']['mean_entropy']),
                            'sparsity': analysis['sparsity'],
                            'pattern_type': analysis.get('pattern_type', 'unknown')
                        },
                        'figures': layer_figures
                    }
                    
                    print(f"    ✅ {text_type}: {len(attention_weights)} layers")
                    
            except Exception as e:
                print(f"    ❌ {text_type}: {e}")
                attention_examples[text_type] = {'error': str(e)}
        
        # Save attention examples data
        data_path = self.output_dir / 'data' / 'attention_examples.json'
        with open(data_path, 'w') as f:
            json.dump(attention_examples, f, indent=2, default=str)
        
        self.examples['attention'] = attention_examples
        print(f"  ✅ Saved attention examples to {data_path}")
        
        return attention_examples
    
    def generate_comparison_examples(self):
        """Generate method comparison examples"""
        
        print("⚖️ Generating Comparison Examples")
        
        comparison_scenarios = {
            'short_sequence': {'seq_len': 16, 'd_model': 128},
            'medium_sequence': {'seq_len': 64, 'd_model': 256},
            'long_sequence': {'seq_len': 128, 'd_model': 512}
        }
        
        encoding_types = ['sinusoidal', 'learned', 'rope']
        
        comparison_examples = {}
        
        for scenario_name, scenario_config in comparison_scenarios.items():
            print(f"  Scenario: {scenario_name}")
            
            scenario_results = {}
            scenario_matrices = {}
            
            for encoding_type in encoding_types:
                try:
                    config = ModelConfig(
                        d_model=scenario_config['d_model'],
                        encoding_type=encoding_type,
                        max_seq_len=scenario_config['seq_len']
                    )
                    
                    encoding = get_positional_encoding(config)
                    
                    if hasattr(encoding, 'forward'):
                        encoding_output = encoding.forward(
                            scenario_config['seq_len'], 
                            scenario_config['d_model']
                        )
                        
                        if isinstance(encoding_output, dict):  # RoPE
                            encoding_matrix = torch.cat([
                                encoding_output['cos'].squeeze(0),
                                encoding_output['sin'].squeeze(0)
                            ], dim=-1)
                        else:
                            encoding_matrix = encoding_output.squeeze(0)
                        
                        scenario_matrices[encoding_type] = encoding_matrix
                        
                        # Compute quality metrics
                        metrics_computer = EncodingMetrics()
                        quality = metrics_computer.compute_encoding_quality(encoding_matrix)
                        
                        scenario_results[encoding_type] = {
                            'quality_metrics': quality,
                            'shape': list(encoding_matrix.shape),
                            'memory_usage': encoding_matrix.numel() * encoding_matrix.element_size()
                        }
                        
                        print(f"    ✅ {encoding_type}: quality = {quality.get('distinguishability', 0):.3f}")
                        
                except Exception as e:
                    print(f"    ❌ {encoding_type}: {e}")
                    scenario_results[encoding_type] = {'error': str(e)}
            
            # Create comparison visualization
            if len(scenario_matrices) >= 2:
                fig, axes = plt.subplots(1, len(scenario_matrices), figsize=(5 * len(scenario_matrices), 4))
                if len(scenario_matrices) == 1:
                    axes = [axes]
                
                for idx, (name, matrix) in enumerate(scenario_matrices.items()):
                    im = axes[idx].imshow(matrix.T, aspect='auto', cmap='RdBu')
                    axes[idx].set_title(f'{name.title()}')
                    axes[idx].set_xlabel('Position')
                    axes[idx].set_ylabel('Dimension')
                    plt.colorbar(im, ax=axes[idx])
                
                plt.suptitle(f'Encoding Comparison - {scenario_name.replace("_", " ").title()}')
                plt.tight_layout()
                
                # Save comparison figure
                fig_path = self.output_dir / 'images' / f'comparison_{scenario_name}.png'
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                scenario_results['comparison_figure'] = str(fig_path)
            
            comparison_examples[scenario_name] = scenario_results
        
        # Save comparison examples data
        data_path = self.output_dir / 'data' / 'comparison_examples.json'
        with open(data_path, 'w') as f:
            json.dump(comparison_examples, f, indent=2, default=str)
        
        self.examples['comparisons'] = comparison_examples
        print(f"  ✅ Saved comparison examples to {data_path}")
        
        return comparison_examples
    
    def generate_configuration_examples(self):
        """Generate example configurations"""
        
        print("⚙️ Generating Configuration Examples")
        
        config_examples = {
            'basic_sinusoidal': ModelConfig(
                d_model=256,
                n_heads=8,
                n_layers=6,
                encoding_type='sinusoidal'
            ),
            'large_learned': ModelConfig(
                d_model=512,
                n_heads=16,
                n_layers=12,
                encoding_type='learned',
                max_seq_len=256
            ),
            'rope_optimized': ModelConfig(
                d_model=256,
                n_heads=8,
                n_layers=8,
                encoding_type='rope',
                rope_theta=50000.0,
                max_seq_len=512
            ),
            'small_relative': ModelConfig(
                d_model=128,
                n_heads=4,
                n_layers=4,
                encoding_type='relative',
                max_seq_len=64
            )
        }
                visualization_examples = {
            'default_theme': VisualizationConfig(),
            'dark_theme': VisualizationConfig(
                colormap_attention='plasma',
                colormap_encoding='coolwarm',
                background_color='black',
                text_color='white'
            ),
            'academic_theme': VisualizationConfig(
                colormap_attention='Blues',
                colormap_encoding='RdBu',
                font_size=14,
                figure_size=(12, 8),
                export_quality=600
            ),
            'presentation_theme': VisualizationConfig(
                font_size=16,
                figure_size=(16, 10),
                export_quality=300,
                show_grid=True
            )
        }
        
        # Save configurations
        for config_name, config in config_examples.items():
            config_path = self.output_dir / 'configs' / f'model_{config_name}.json'
            
            config_dict = {
                'name': config_name,
                'description': f'Example {config_name.replace("_", " ")} configuration',
                'config': config.__dict__
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            print(f"    ✅ Saved {config_name} model config")
        
        for viz_name, viz_config in visualization_examples.items():
            viz_path = self.output_dir / 'configs' / f'visualization_{viz_name}.json'
            
            viz_dict = {
                'name': viz_name,
                'description': f'Example {viz_name.replace("_", " ")} visualization configuration',
                'config': viz_config.__dict__
            }
            
            with open(viz_path, 'w') as f:
                json.dump(viz_dict, f, indent=2, default=str)
            
            print(f"    ✅ Saved {viz_name} visualization config")
        
        config_data = {
            'model_configs': {name: config.__dict__ for name, config in config_examples.items()},
            'visualization_configs': {name: config.__dict__ for name, config in visualization_examples.items()}
        }
        
        # Save combined config examples
        combined_path = self.output_dir / 'data' / 'configuration_examples.json'
        with open(combined_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        self.examples['configurations'] = config_data
        print(f"  ✅ Saved configuration examples to {combined_path}")
        
        return config_data
    
    def generate_tutorial_examples(self):
        """Generate examples for tutorials"""
        
        print("📚 Generating Tutorial Examples")
        
        tutorial_examples = {}
        
        # Getting started example
        getting_started = {
            'step1_basic_encoding': {
                'description': 'Create basic sinusoidal encoding',
                'code': '''
from config import ModelConfig
from positional_encoding import get_positional_encoding

config = ModelConfig(d_model=128, encoding_type='sinusoidal')
encoding = get_positional_encoding(config)
result = encoding.forward(32, 128)
''',
                'expected_shape': [1, 32, 128]
            },
            'step2_visualization': {
                'description': 'Create basic heatmap visualization',
                'code': '''
from visualization import EncodingPlotter
from config import VisualizationConfig

viz_config = VisualizationConfig()
plotter = EncodingPlotter(viz_config)
fig = plotter.plot_sinusoidal_patterns(encoding, seq_len=32)
''',
                'output_type': 'matplotlib.figure.Figure'
            },
            'step3_model_creation': {
                'description': 'Create transformer model',
                'code': '''
from models import TransformerEncoder

model = TransformerEncoder(config)
input_ids = torch.tensor([[0, 1, 2, 3, 4]])
outputs = model(input_ids, store_visualizations=True)
''',
                'expected_keys': ['last_hidden_state', 'attention_weights']
            }
        }
        
        # Advanced usage example
        advanced_usage = {
            'custom_encoding': {
                'description': 'Create custom positional encoding',
                'code': '''
class CustomEncoding(PositionalEncoding):
    def forward(self, seq_len, d_model):
        # Custom implementation
        position = torch.arange(seq_len).float().unsqueeze(1)
        encoding = position * torch.linspace(0, 1, d_model)
        return encoding.unsqueeze(0)
''',
                'complexity': 'advanced'
            },
            'attention_analysis': {
                'description': 'Comprehensive attention analysis',
                'code': '''
from utils.metrics import AttentionMetrics

metrics = AttentionMetrics()
analysis = metrics.analyze_attention_patterns(attention_weights)
entropy = analysis['entropy']['mean_entropy']
sparsity = analysis['sparsity']['gini_coefficient']
''',
                'metrics': ['entropy', 'sparsity', 'head_similarity']
            }
        }
        
        tutorial_examples = {
            'getting_started': getting_started,
            'advanced_usage': advanced_usage
        }
        
        # Save tutorial examples
        tutorial_path = self.output_dir / 'data' / 'tutorial_examples.json'
        with open(tutorial_path, 'w') as f:
            json.dump(tutorial_examples, f, indent=2)
        
        self.examples['tutorials'] = tutorial_examples
        print(f"  ✅ Saved tutorial examples to {tutorial_path}")
        
        return tutorial_examples
    
    def generate_benchmark_examples(self):
        """Generate benchmark examples"""
        
        print("🏁 Generating Benchmark Examples")
        
        # Simple benchmark data
        benchmark_data = {
            'encoding_performance': {
                'sinusoidal': {
                    'seq_len_16': {'time_ms': 0.12, 'memory_mb': 0.5},
                    'seq_len_64': {'time_ms': 0.45, 'memory_mb': 2.1},
                    'seq_len_256': {'time_ms': 1.8, 'memory_mb': 8.4}
                },
                'learned': {
                    'seq_len_16': {'time_ms': 0.08, 'memory_mb': 1.2},
                    'seq_len_64': {'time_ms': 0.31, 'memory_mb': 4.8},
                    'seq_len_256': {'time_ms': 1.2, 'memory_mb': 19.2}
                },
                'rope': {
                    'seq_len_16': {'time_ms': 0.15, 'memory_mb': 0.6},
                    'seq_len_64': {'time_ms': 0.58, 'memory_mb': 2.4},
                    'seq_len_256': {'time_ms': 2.3, 'memory_mb': 9.6}
                }
            },
            'quality_metrics': {
                'sinusoidal': {
                    'distinguishability': 0.847,
                    'dimension_utilization': 0.923,
                    'periodicity_score': 0.891
                },
                'learned': {
                    'distinguishability': 0.912,
                    'dimension_utilization': 0.756,
                    'periodicity_score': 0.234
                },
                'rope': {
                    'distinguishability': 0.798,
                    'dimension_utilization': 0.889,
                    'periodicity_score': 0.678
                }
            },
            'extrapolation_quality': {
                'sinusoidal': {'train_64_test_128': 0.89, 'train_64_test_256': 0.85},
                'learned': {'train_64_test_128': 0.23, 'train_64_test_256': 0.12},
                'rope': {'train_64_test_128': 0.94, 'train_64_test_256': 0.91}
            }
        }
        
        # Save benchmark data
        benchmark_path = self.output_dir / 'data' / 'benchmark_examples.json'
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        self.examples['benchmarks'] = benchmark_data
        print(f"  ✅ Saved benchmark examples to {benchmark_path}")
        
        return benchmark_data
    
    def create_example_summary(self):
        """Create summary of all generated examples"""
        
        print("📋 Creating Example Summary")
        
        summary = {
            'generation_info': {
                'timestamp': torch.now().isoformat() if hasattr(torch, 'now') else 'unknown',
                'output_directory': str(self.output_dir),
                'total_examples': len(self.examples)
            },
            'examples': {}
        }
        
        # Count files and examples
        for category, examples in self.examples.items():
            if isinstance(examples, dict):
                summary['examples'][category] = {
                    'count': len(examples),
                    'keys': list(examples.keys())
                }
            else:
                summary['examples'][category] = {
                    'type': type(examples).__name__
                }
        
        # Count generated files
        image_files = list((self.output_dir / 'images').glob('*.png'))
        data_files = list((self.output_dir / 'data').glob('*.json'))
        config_files = list((self.output_dir / 'configs').glob('*.json'))
        
        summary['generated_files'] = {
            'images': len(image_files),
            'data_files': len(data_files),
            'config_files': len(config_files),
            'total': len(image_files) + len(data_files) + len(config_files)
        }
        
        # File listings
        summary['file_listings'] = {
            'images': [f.name for f in image_files],
            'data': [f.name for f in data_files],
            'configs': [f.name for f in config_files]
        }
        
        # Save summary
        summary_path = self.output_dir / 'example_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"  ✅ Saved example summary to {summary_path}")
        
        return summary
    
    def generate_all_examples(self):
        """Generate all example types"""
        
        print("🚀 Generating All Examples")
        print("=" * 40)
        
        try:
            # Generate each type
            self.generate_encoding_examples()
            self.generate_attention_examples()
            self.generate_comparison_examples()
            self.generate_configuration_examples()
            self.generate_tutorial_examples()
            self.generate_benchmark_examples()
            
            # Create summary
            summary = self.create_example_summary()
            
            print("\n✅ Example Generation Complete!")
            print(f"Generated {summary['generated_files']['total']} files:")
            print(f"  - {summary['generated_files']['images']} images")
            print(f"  - {summary['generated_files']['data_files']} data files")
            print(f"  - {summary['generated_files']['config_files']} config files")
            
            return summary
            
        except Exception as e:
            print(f"❌ Example generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main example generator"""
    
    parser = argparse.ArgumentParser(description='Generate examples for positional encoding visualizer')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='examples',
        help='Output directory for examples'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=['encodings', 'attention', 'comparisons', 'configs', 'tutorials', 'benchmarks', 'all'],
        default='all',
        help='Category of examples to generate'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("📝 Positional Encoding Example Generator")
    print("=" * 50)
    
    generator = ExampleGenerator(args.output_dir)
    
    try:
        if args.category == 'all':
            summary = generator.generate_all_examples()
        elif args.category == 'encodings':
            generator.generate_encoding_examples()
        elif args.category == 'attention':
            generator.generate_attention_examples()
        elif args.category == 'comparisons':
            generator.generate_comparison_examples()
        elif args.category == 'configs':
            generator.generate_configuration_examples()
        elif args.category == 'tutorials':
            generator.generate_tutorial_examples()
        elif args.category == 'benchmarks':
            generator.generate_benchmark_examples()
        
        if args.category != 'all':
            summary = generator.create_example_summary()
        
        print(f"\n📁 Examples saved to: {generator.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Generation stopped by user")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()

