#!/usr/bin/env python3
"""
Visualization Exporter - Export visualizations in various formats and create reports
"""

import sys
import os
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Any
import warnings

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from config import ModelConfig, VisualizationConfig
from models import TransformerEncoder
from positional_encoding import get_positional_encoding
from visualization import AttentionVisualizer, EncodingPlotter, HeatmapGenerator, ThreeDVisualizer
from utils.tokenizer import SimpleTokenizer
from utils.export_utils import FigureExporter, DataExporter, ReportGenerator
from utils.metrics import AttentionMetrics, EncodingMetrics

class VisualizationExporter:
    """Export comprehensive visualizations and reports"""
    
    def __init__(self, output_dir: str = 'exports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        (self.output_dir / 'interactive').mkdir(exist_ok=True)
        
        self.viz_config = VisualizationConfig()
        self.figure_exporter = FigureExporter(self.viz_config)
        self.data_exporter = DataExporter()
        self.report_generator = ReportGenerator(self.viz_config)
        
        self.exported_items = {}
        self.analysis_results = {}
        
        print(f"📁 Export directory: {self.output_dir}")
    
    def export_encoding_visualizations(self, 
                                     encoding_types: List[str] = None,
                                     formats: List[str] = None) -> Dict[str, List[str]]:
        """Export positional encoding visualizations"""
        
        print("🔢 Exporting Encoding Visualizations")
        
        if encoding_types is None:
            encoding_types = ['sinusoidal', 'learned', 'rope', 'relative']
        
        if formats is None:
            formats = ['png', 'pdf', 'svg']
        
        exported_files = {}
        
        # Test configurations
        configs = [
            {'d_model': 128, 'seq_len': 32, 'name': 'small'},
            {'d_model': 256, 'seq_len': 64, 'name': 'medium'},
            {'d_model': 512, 'seq_len': 128, 'name': 'large'}
        ]
        
        plotter = EncodingPlotter(self.viz_config)
        heatmap_generator = HeatmapGenerator(self.viz_config)
        viz_3d = ThreeDVisualizer(self.viz_config)
        
        for config_info in configs:
            print(f"  Configuration: {config_info['name']}")
            
            config_files = []
            encoding_matrices = {}
            
            # Generate encodings
            for encoding_type in encoding_types:
                try:
                    model_config = ModelConfig(
                        d_model=config_info['d_model'],
                        encoding_type=encoding_type,
                        max_seq_len=config_info['seq_len']
                    )
                    
                    encoding = get_positional_encoding(model_config)
                    
                    if hasattr(encoding, 'forward'):
                        encoding_output = encoding.forward(
                            config_info['seq_len'], 
                            config_info['d_model']
                        )
                        
                        if isinstance(encoding_output, dict):  # RoPE
                            encoding_matrix = torch.cat([
                                encoding_output['cos'].squeeze(0),
                                encoding_output['sin'].squeeze(0)
                            ], dim=-1)
                        else:
                            encoding_matrix = encoding_output.squeeze(0)
                        
                        encoding_matrices[encoding_type] = encoding_matrix
                        
                        # 2D Heatmap
                        fig_2d = heatmap_generator.create_encoding_heatmap(
                            encoding_matrix,
                            encoding_name=f"{encoding_type.title()} ({config_info['name']})"
                        )
                        
                        # Export in multiple formats
                        for fmt in formats:
                            filename = f"encoding_{encoding_type}_{config_info['name']}_2d.{fmt}"
                            filepath = self.output_dir / 'images' / filename
                            
                            self.figure_exporter.export_matplotlib_figure(
                                fig_2d, str(filepath), format=fmt
                            )
                            config_files.append(str(filepath))
                        
                        plt.close(fig_2d)
                        
                        # 3D Surface (for PNG only to avoid large files)
                        if 'png' in formats:
                            fig_3d = viz_3d.create_encoding_3d_surface(
                                encoding_matrix,
                                encoding_name=f"{encoding_type.title()} ({config_info['name']})"
                            )
                            
                            filename_3d = f"encoding_{encoding_type}_{config_info['name']}_3d.html"
                            filepath_3d = self.output_dir / 'interactive' / filename_3d
                            
                            self.figure_exporter.export_plotly_figure(
                                fig_3d, str(filepath_3d), format='html'
                            )
                            config_files.append(str(filepath_3d))
                        
                        print(f"    ✅ {encoding_type}")
                        
                except Exception as e:
                    print(f"    ❌ {encoding_type}: {e}")
            
            # Create comparison visualization
            if len(encoding_matrices) >= 2:
                comparison_fig = heatmap_generator.create_comparison_heatmap(
                    encoding_matrices,
                    title=f"Encoding Comparison ({config_info['name']})"
                )
                
                for fmt in formats:
                    filename = f"encoding_comparison_{config_info['name']}.{fmt}"
                    filepath = self.output_dir / 'images' / filename
                    
                    self.figure_exporter.export_matplotlib_figure(
                        comparison_fig, str(filepath), format=fmt
                    )
                    config_files.append(str(filepath))
                
                plt.close(comparison_fig)
            
            exported_files[config_info['name']] = config_files
        
        self.exported_items['encodings'] = exported_files
        return exported_files
    
    def export_attention_visualizations(self,
                                       sample_texts: List[str] = None,
                                       formats: List[str] = None) -> Dict[str, List[str]]:
        """Export attention pattern visualizations"""
        
        print("👁️ Exporting Attention Visualizations")
        
        if sample_texts is None:
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Transformers use attention mechanisms for sequence processing.",
                "Machine learning models require extensive training data."
            ]
        
        if formats is None:
            formats = ['png', 'pdf']
        
        exported_files = {}
        tokenizer = SimpleTokenizer()
        
        visualizer = AttentionVisualizer(self.viz_config)
        heatmap_generator = HeatmapGenerator(self.viz_config)
        
        encoding_types = ['sinusoidal', 'rope']  # Focus on most interesting ones
        
        for text_idx, text in enumerate(sample_texts):
            print(f"  Text {text_idx + 1}: {text[:50]}...")
            
            text_files = []
            tokens = tokenizer.tokenize(text)
            
            if len(tokens) > 32:  # Limit length
                tokens = tokens[:32]
            
            for encoding_type in encoding_types:
                try:
                    # Create model
                    model_config = ModelConfig(
                        d_model=256,
                        n_heads=8,
                        n_layers=4,
                        encoding_type=encoding_type,
                        max_seq_len=len(tokens)
                    )
                    
                    model = TransformerEncoder(model_config)
                    model.eval()
                    
                    # Generate attention
                    input_ids = torch.tensor([list(range(len(tokens)))])
                    
                    with torch.no_grad():
                        outputs = model(input_ids, store_visualizations=True)
                    
                    attention_weights = outputs['attention_weights']
                    
                    if attention_weights:
                        # Export attention for each layer
                        for layer_idx, layer_attention in enumerate(attention_weights):
                            
                            # Single head attention heatmap
                            fig_single = visualizer.visualize_attention_matrix(
                                layer_attention,
                                tokens=tokens,
                                head_idx=0,
                                title=f"Attention - {encoding_type.title()}, Layer {layer_idx}"
                            )
                            
                            for fmt in formats:
                                filename = f"attention_text{text_idx}_layer{layer_idx}_{encoding_type}.{fmt}"
                                filepath = self.output_dir / 'images' / filename
                                
                                self.figure_exporter.export_matplotlib_figure(
                                    fig_single, str(filepath), format=fmt
                                )
                                text_files.append(str(filepath))
                            
                            plt.close(fig_single)
                            
                            # Multi-head visualization
                            fig_multi = visualizer.visualize_multi_head_attention(
                                layer_attention,
                                tokens=tokens
                            )
                            
                            filename_multi = f"attention_multihead_text{text_idx}_layer{layer_idx}_{encoding_type}.png"
                            filepath_multi = self.output_dir / 'images' / filename_multi
                            
                            self.figure_exporter.export_matplotlib_figure(
                                fig_multi, str(filepath_multi)
                            )
                            text_files.append(str(filepath_multi))
                            
                            plt.close(fig_multi)
                        
                        # Layer evolution
                        fig_evolution = heatmap_generator.create_multi_layer_heatmap(
                            attention_weights,
                            tokens=tokens,
                            head_idx=0,
                            title=f"Attention Evolution - {encoding_type.title()}"
                        )
                        
                        filename_evolution = f"attention_evolution_text{text_idx}_{encoding_type}.png"
                        filepath_evolution = self.output_dir / 'images' / filename_evolution
                        
                        self.figure_exporter.export_matplotlib_figure(
                            fig_evolution, str(filepath_evolution)
                        )
                        text_files.append(str(filepath_evolution))
                        
                        plt.close(fig_evolution)
                        
                        print(f"    ✅ {encoding_type} - {len(attention_weights)} layers")
                        
                except Exception as e:
                    print(f"    ❌ {encoding_type}: {e}")
                
                finally:
                    if 'model' in locals():
                        del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            exported_files[f'text_{text_idx}'] = text_files
        
        self.exported_items['attention'] = exported_files
        return exported_files
    
    def export_comparison_analysis(self, formats: List[str] = None) -> Dict[str, List[str]]:
        """Export comprehensive comparison analysis"""
        
        print("⚖️ Exporting Comparison Analysis")
        
        if formats is None:
            formats = ['png', 'pdf']
        
        exported_files = []
        
        # Compare encoding methods
        encoding_types = ['sinusoidal', 'learned', 'rope']
        metrics_computer = EncodingMetrics()
        
        # Different scenarios
        scenarios = [
            {'seq_len': 32, 'd_model': 128, 'name': 'short'},
            {'seq_len': 64, 'd_model': 256, 'name': 'medium'},
            {'seq_len': 128, 'd_model': 512, 'name': 'long'}
        ]
        
        comparison_data = {}
        
        for scenario in scenarios:
            print(f"  Scenario: {scenario['name']}")
            
            scenario_results = {}
            encoding_matrices = {}
            
            for encoding_type in encoding_types:
                try:
                    model_config = ModelConfig(
                        d_model=scenario['d_model'],
                        encoding_type=encoding_type,
                        max_seq_len=scenario['seq_len']
                    )
                    
                    encoding = get_positional_encoding(model_config)
                    
                    if hasattr(encoding, 'forward'):
                        encoding_output = encoding.forward(scenario['seq_len'], scenario['d_model'])
                        
                        if isinstance(encoding_output, dict):
                            encoding_matrix = torch.cat([
                                encoding_output['cos'].squeeze(0),
                                encoding_output['sin'].squeeze(0)
                            ], dim=-1)
                        else:
                            encoding_matrix = encoding_output.squeeze(0)
                        
                        encoding_matrices[encoding_type] = encoding_matrix
                        
                        # Compute metrics
                        quality_metrics = metrics_computer.compute_encoding_quality(encoding_matrix)
                        similarities = metrics_computer.compute_position_similarity(encoding_matrix)
                        
                        scenario_results[encoding_type] = {
                            **quality_metrics,
                            'mean_similarity': float(similarities.mean()),
                            'similarity_std': float(similarities.std())
                        }
                        
                except Exception as e:
                    print(f"    ❌ {encoding_type}: {e}")
                    scenario_results[encoding_type] = {'error': str(e)}
            
            comparison_data[scenario['name']] = scenario_results
            
            # Create visual comparisons
            if len(encoding_matrices) >= 2:
                heatmap_generator = HeatmapGenerator(self.viz_config)
                
                # Side-by-side heatmaps
                comparison_fig = heatmap_generator.create_comparison_heatmap(
                    encoding_matrices,
                    title=f"Method Comparison - {scenario['name'].title()} Sequences"
                )
                
                for fmt in formats:
                    filename = f"method_comparison_{scenario['name']}.{fmt}"
                    filepath = self.output_dir / 'images' / filename
                    
                    self.figure_exporter.export_matplotlib_figure(
                        comparison_fig, str(filepath), format=fmt
                    )
                    exported_files.append(str(filepath))
                
                plt.close(comparison_fig)
        
        # Create metrics comparison chart
        self._create_metrics_comparison_chart(comparison_data, exported_files, formats)
        
        # Save comparison data
        data_path = self.output_dir / 'data' / 'comparison_analysis.json'
        with open(data_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        exported_files.append(str(data_path))
        
        self.exported_items['comparisons'] = exported_files
        self.analysis_results['comparisons'] = comparison_data
        
        return {'comparison_analysis': exported_files}
    
    def _create_metrics_comparison_chart(self, comparison_data: Dict, exported_files: List, formats: List):
        """Create metrics comparison chart"""
        
        # Prepare data for plotting
        scenarios = list(comparison_data.keys())
        encoding_types = ['sinusoidal', 'learned', 'rope']
        metrics = ['distinguishability', 'encoding_variance', 'dimension_utilization']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            x_pos = np.arange(len(scenarios))
            width = 0.25
            
            for enc_idx, encoding_type in enumerate(encoding_types):
                values = []
                for scenario in scenarios:
                    scenario_data = comparison_data[scenario]
                    if encoding_type in scenario_data and metric in scenario_data[encoding_type]:
                        values.append(scenario_data[encoding_type][metric])
                    else:
                        values.append(0)
                
                ax.bar(x_pos + enc_idx * width, values, width, 
                      label=encoding_type.title(), alpha=0.8)
            
            ax.set_xlabel('Scenario')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels([s.title() for s in scenarios])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        for fmt in formats:
            filename = f"metrics_comparison.{fmt}"
            filepath = self.output_dir / 'images' / filename
            
            self.figure_exporter.export_matplotlib_figure(fig, str(filepath), format=fmt)
            exported_files.append(str(filepath))
        
        plt.close(fig)
    
    def export_interactive_visualizations(self) -> Dict[str, List[str]]:
        """Export interactive HTML visualizations"""
        
        print("🌐 Exporting Interactive Visualizations")
        
        exported_files = []
        
        try:
            # Interactive encoding explorer
            config = ModelConfig(d_model=256, encoding_type='sinusoidal')
            encoding = get_positional_encoding(config)
            
            if hasattr(encoding, 'forward'):
                encoding_output = encoding.forward(64, 256)
                encoding_matrix = encoding_output.squeeze(0)
                
                # Interactive 3D surface
                viz_3d = ThreeDVisualizer(self.viz_config)
                fig_3d = viz_3d.create_encoding_3d_surface(
                    encoding_matrix,
                    encoding_name="Interactive Sinusoidal Encoding"
                )
                
                interactive_path = self.output_dir / 'interactive' / 'encoding_3d_explorer.html'
                self.figure_exporter.export_plotly_figure(fig_3d, str(interactive_path))
                exported_files.append(str(interactive_path))
                
                print("  ✅ Interactive 3D encoding explorer")
                
                # Interactive heatmap with hover data
                fig_interactive = go.Figure(data=go.Heatmap(
                    z=encoding_matrix.T.numpy(),
                    colorscale='Viridis',
                    hovertemplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
                ))
                
                fig_interactive.update_layout(
                    title="Interactive Encoding Heatmap",
                    xaxis_title="Position",
                    yaxis_title="Dimension",
                    height=600
                )
                
                heatmap_path = self.output_dir / 'interactive' / 'encoding_heatmap.html'
                self.figure_exporter.export_plotly_figure(fig_interactive, str(heatmap_path))
                exported_files.append(str(heatmap_path))
                
                print("  ✅ Interactive encoding heatmap")
        
        except Exception as e:
            print(f"  ❌ Interactive encodings: {e}")
        
        # Interactive attention visualization
        try:
            tokenizer = SimpleTokenizer()
            text = "The transformer architecture uses self-attention mechanisms."
            tokens = tokenizer.tokenize(text)[:16]  # Limit for interactivity
            
            model_config = ModelConfig(
                d_model=256, n_heads=8, n_layers=4,
                encoding_type='sinusoidal', max_seq_len=len(tokens)
            )
            
            model = TransformerEncoder(model_config)
            model.eval()
            
            input_ids = torch.tensor([list(range(len(tokens)))])
            
            with torch.no_grad():
                outputs = model(input_ids, store_visualizations=True)
            
            attention_weights = outputs['attention_weights']
            
            if attention_weights:
                # Interactive multi-layer attention
                layer_attention = attention_weights[0][0, 0].detach().cpu().numpy()  # First head
                
                fig_attention = go.Figure(data=go.Heatmap(
                    z=layer_attention,
                    x=tokens,
                    y=tokens,
                    colorscale='Viridis',
                    hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
                ))
                
                fig_attention.update_layout(
                    title="Interactive Attention Pattern",
                    xaxis_title="Key Tokens",
                    yaxis_title="Query Tokens",
                    height=600
                )
                
                attention_path = self.output_dir / 'interactive' / 'attention_pattern.html'
                self.figure_exporter.export_plotly_figure(fig_attention, str(attention_path))
                exported_files.append(str(attention_path))
                
                print("  ✅ Interactive attention pattern")
        
        except Exception as e:
            print(f"  ❌ Interactive attention: {e}")
        
        self.exported_items['interactive'] = exported_files
        return {'interactive': exported_files}
    
    def create_comprehensive_report(self) -> str:
        """Create comprehensive HTML report"""
        
        print("📄 Creating Comprehensive Report")
        
        # Collect all analysis results
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'export_directory': str(self.output_dir),
                'total_files_exported': sum(len(files) for files in self.exported_items.values() if isinstance(files, list))
            },
            'analysis_results': self.analysis_results,
            'exported_items': self.exported_items
        }
        
        # Generate HTML report
        report_path = self.output_dir / 'reports' / 'comprehensive_report.html'
        
        try:
            report_html_path = self.report_generator.generate_html_report(
                report_data,
                {},  # Empty figures dict since we're referencing file paths
                str(report_path),
                title="Positional Encoding Analysis Report"
            )
            
            print(f"  ✅ Report saved to {report_html_path}")
            
            # Also save data as JSON
            json_path = self.output_dir / 'data' / 'comprehensive_analysis.json'
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            return report_html_path
            
        except Exception as e:
            print(f"  ❌ Report generation failed: {e}")
            
            # Fallback: create simple report
            simple_report = self._create_simple_html_report(report_data)
            with open(report_path, 'w') as f:
                f.write(simple_report)
            
            return str(report_path)
    
    def _create_simple_html_report(self, report_data: Dict) -> str:
        """Create simple HTML report as fallback"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Positional Encoding Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                .file-list {{ list-style-type: none; padding: 0; }}
                .file-list li {{ padding: 5px; margin: 2px 0; background: white; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 Positional Encoding Analysis Report</h1>
                <p>Generated on: {report_data['metadata']['generated_at']}</p>
            </div>
            
            <div class="section">
                <h2>📊 Export Summary</h2>
                <p>Total files exported: {report_data['metadata']['total_files_exported']}</p>
                <p>Export directory: {report_data['metadata']['export_directory']}</p>
            </div>
        """
        
        # Add exported items
        for category, items in report_data['exported_items'].items():
            if isinstance(items, list) and items:
                html += f"""
                <div class="section">
                    <h3>📁 {category.title()}</h3>
                    <ul class="file-list">
                """
                for item in items[:10]:  # Limit display
                    filename = Path(item).name
                    html += f"<li>{filename}</li>"
                
                if len(items) > 10:
                    html += f"<li>... and {len(items) - 10} more files</li>"
                
                html += "</ul></div>"
        
        html += """
            <div class="section">
                <h3>🔍 Analysis Results</h3>
                <p>Detailed analysis results are available in the exported JSON files.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def export_all(self, 
                   formats: List[str] = None,
                   include_interactive: bool = True) -> Dict[str, Any]:
        """Export all visualizations and create comprehensive report"""
        
        print("🚀 Exporting All Visualizations")
        print("=" * 50)
        
        if formats is None:
            formats = ['png', 'pdf']
        
        export_summary = {}
        
        try:
            # Export encoding visualizations
            encoding_exports = self.export_encoding_visualizations(formats=formats)
            export_summary['encodings'] = encoding_exports
            
            # Export attention visualizations
            attention_exports = self.export_attention_visualizations(formats=formats)
            export_summary['attention'] = attention_exports
            
            # Export comparison analysis
            comparison_exports = self.export_comparison_analysis(formats=formats)
            export_summary['comparisons'] = comparison_exports
            
            # Export interactive visualizations
            if include_interactive:
                interactive_exports = self.export_interactive_visualizations()
                export_summary['interactive'] = interactive_exports
            
            # Create comprehensive report
            report_path = self.create_comprehensive_report()
            export_summary['report'] = report_path
            
            # Summary statistics
            total_files = sum(
                len(files) for category_files in self.exported_items.values()
                for files in (category_files if isinstance(category_files, list) else category_files.values())
                if isinstance(files, list)
            )
            
            print(f"\n✅ Export completed successfully!")
            print(f"Total files exported: {total_files}")
            print(f"Export directory: {self.output_dir}")
            print(f"Report: {report_path}")
            
            return export_summary
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    """Main export script"""
    
    parser = argparse.ArgumentParser(description='Export positional encoding visualizations')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exports',
        help='Output directory for exports'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['png', 'pdf'],
        choices=['png', 'pdf', 'svg', 'html'],
        help='Export formats'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=['encodings', 'attention', 'comparisons', 'interactive', 'all'],
        default='all',
        help='Category to export'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive visualizations'
    )
    
    args = parser.parse_args()
    
    print("📊 Positional Encoding Visualization Exporter")
    print("=" * 50)
    
    exporter = VisualizationExporter(args.output_dir)
    
    try:
        if args.category == 'all':
            export_summary = exporter.export_all(
                formats=args.formats,
                include_interactive=not args.no_interactive
            )
        elif args.category == 'encodings':
            export_summary = exporter.export_encoding_visualizations(formats=args.formats)
        elif args.category == 'attention':
            export_summary = exporter.export_attention_visualizations(formats=args.formats)
        elif args.category == 'comparisons':
            export_summary = exporter.export_comparison_analysis(formats=args.formats)
        elif args.category == 'interactive':
            export_summary = exporter.export_interactive_visualizations()
        
        if export_summary:
            print("\n📁 Exported files available in:")
            print(f"  Images: {exporter.output_dir / 'images'}")
            print(f"  Data: {exporter.output_dir / 'data'}")
            print(f"  Reports: {exporter.output_dir / 'reports'}")
            if not args.no_interactive:
                print(f"  Interactive: {exporter.output_dir / 'interactive'}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Export stopped by user")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()
