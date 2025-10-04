"""Gradio interface for positional encoding exploration."""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import json
import io
import base64
from PIL import Image
import warnings

from config import ModelConfig, VisualizationConfig
from src.models import TransformerEncoder
from src.positional_encoding import get_positional_encoding
from src.utils.tokenizer import SimpleTokenizer
from src.utils.metrics import AttentionMetrics, EncodingMetrics
from src.utils.export_utils import FigureExporter
from src.visualization import AttentionVisualizer, EncodingPlotter


class GradioInterface:
    """Gradio-based interface for positional encoding exploration."""
    
    def __init__(self):
        self.model_cache = {}
        self.analysis_cache = {}
        self.tokenizer = SimpleTokenizer()
        
        # Default configurations
        self.default_config = ModelConfig()
        self.viz_config = VisualizationConfig()
        
        # Create interface
        self.interface = self.create_interface()
    
    def create_interface(self):
        """Create the main Gradio interface."""
        with gr.Blocks(
            title="🧠 Positional Encoding Visualizer",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🧠 Positional Encoding Visualizer
            
            **Explore transformer positional encodings interactively!**
            
            This tool helps you understand how different positional encoding methods work 
            and how they affect attention patterns in transformer models.
            """)
            
            with gr.Tabs():
                with gr.TabItem("🏠 Home"):
                    self.create_home_tab()
                
                with gr.TabItem("📊 Encoding Analysis"):
                    self.create_encoding_analysis_tab()
                
                with gr.TabItem("🔍 Attention Patterns"):
                    self.create_attention_patterns_tab()
                
                with gr.TabItem("⚖️ Method Comparison"):
                    self.create_comparison_tab()
                
                with gr.TabItem("🎯 Interactive Lab"):
                    self.create_interactive_lab_tab()
                
                with gr.TabItem("📄 Export & Reports"):
                    self.create_export_tab()
        
        return interface
    
    def _get_custom_css(self):
        """Get custom CSS for styling."""
        return """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin: 1rem 0;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .info-panel {
            background-color: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .warning-panel {
            background-color: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        """
    
    def create_home_tab(self):
        """Create the home tab."""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ## Welcome! 🎉
                
                This interactive tool helps you understand positional encodings in transformers:
                
                ### 🎯 Key Features:
                - **📊 Encoding Analysis**: Visualize positional encoding patterns
                - **🔍 Attention Patterns**: See how encodings affect attention
                - **⚖️ Method Comparison**: Compare different encoding methods
                - **🎯 Interactive Lab**: Real-time parameter exploration
                - **📄 Export & Reports**: Generate comprehensive reports
                
                ### 🚀 Quick Start:
                1. Go to **Encoding Analysis** to explore basic patterns
                2. Try **Attention Patterns** to see real effects
                3. Use **Interactive Lab** for live experimentation
                4. Compare methods in **Method Comparison**
                
                ### 💡 Tips:
                - Start with sinusoidal encoding to understand basics
                - Try different sequence lengths to see scaling effects
                - Use the Interactive Lab for intuitive exploration
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Quick Configuration")
                
                with gr.Group():
                    quick_encoding = gr.Dropdown(
                        choices=["sinusoidal", "learned", "rope", "relative"],
                        value="sinusoidal",
                        label="Encoding Type"
                    )
                    
                    quick_seq_len = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        label="Sequence Length"
                    )
                    
                    quick_d_model = gr.Dropdown(
                        choices=[64, 128, 256, 512],
                        value=256,
                        label="Model Dimension"
                    )
                    
                    quick_demo_btn = gr.Button("🚀 Quick Demo", variant="primary")
                
                quick_demo_output = gr.Plot(label="Quick Demo Plot")
                
                # Quick demo functionality
                quick_demo_btn.click(
                    fn=self.generate_quick_demo,
                    inputs=[quick_encoding, quick_seq_len, quick_d_model],
                    outputs=[quick_demo_output]
                )
    
    def create_encoding_analysis_tab(self):
        """Create the encoding analysis tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Configuration")
                
                # Model settings
                with gr.Group():
                    gr.Markdown("#### Model Settings")
                    
                    d_model = gr.Dropdown(
                        choices=[64, 128, 256, 512, 1024],
                        value=256,
                        label="Model Dimension"
                    )
                    
                    encoding_type = gr.Dropdown(
                        choices=["sinusoidal", "learned", "rope", "relative"],
                        value="sinusoidal",
                        label="Encoding Type"
                    )
                    
                    seq_len = gr.Slider(
                        minimum=8,
                        maximum=256,
                        value=64,
                        step=8,
                        label="Sequence Length"
                    )
                
                # Visualization settings
                with gr.Group():
                    gr.Markdown("#### Visualization")
                    
                    viz_type = gr.Dropdown(
                        choices=["Heatmap", "Line Plot", "3D Surface", "Frequency Analysis"],
                        value="Heatmap",
                        label="Visualization Type"
                    )
                    
                    show_similarities = gr.Checkbox(
                        label="Show Position Similarities",
                        value=True
                    )
                    
                    max_dims_display = gr.Slider(
                        minimum=8,
                        maximum=64,
                        value=32,
                        step=8,
                        label="Max Dimensions to Display"
                    )
                
                analyze_btn = gr.Button("🔍 Analyze Encoding", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Encoding Visualization")
                
                with gr.Tabs():
                    with gr.TabItem("Main Plot"):
                        main_plot = gr.Plot(label="Encoding Pattern")
                    
                    with gr.TabItem("Similarities"):
                        similarity_plot = gr.Plot(label="Position Similarities")
                    
                    with gr.TabItem("Statistics"):
                        with gr.Row():
                            metrics_json = gr.JSON(label="Encoding Metrics")
                        
                        with gr.Row():
                            quality_score = gr.Number(label="Quality Score", precision=4)
                            distinguishability = gr.Number(label="Distinguishability", precision=4)
                            complexity = gr.Number(label="Complexity", precision=4)
        
        # Connect analysis functionality
        analyze_btn.click(
            fn=self.analyze_encoding,
            inputs=[encoding_type, d_model, seq_len, viz_type, show_similarities, max_dims_display],
            outputs=[main_plot, similarity_plot, metrics_json, quality_score, distinguishability, complexity]
        )
        
        # Auto-update on parameter change
        for component in [encoding_type, d_model, seq_len, viz_type]:
            component.change(
                fn=self.analyze_encoding,
                inputs=[encoding_type, d_model, seq_len, viz_type, show_similarities, max_dims_display],
                outputs=[main_plot, similarity_plot, metrics_json, quality_score, distinguishability, complexity]
            )
    
    def create_attention_patterns_tab(self):
        """Create the attention patterns tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Model Configuration")
                
                # Model parameters
                with gr.Group():
                    gr.Markdown("#### Architecture")
                    
                    attn_d_model = gr.Dropdown(
                        choices=[128, 256, 512],
                        value=256,
                        label="Model Dimension"
                    )
                    
                    attn_n_heads = gr.Dropdown(
                        choices=[2, 4, 8, 12, 16],
                        value=8,
                        label="Attention Heads"
                    )
                    
                    attn_n_layers = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=4,
                        step=1,
                        label="Number of Layers"
                    )
                    
                    attn_encoding = gr.Dropdown(
                        choices=["sinusoidal", "learned", "rope"],
                        value="sinusoidal",
                        label="Positional Encoding"
                    )
                
                # Input text
                with gr.Group():
                    gr.Markdown("#### Input Text")
                    
                    input_text = gr.Textbox(
                        value="The quick brown fox jumps over the lazy dog.",
                        label="Text to Analyze",
                        lines=3
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=8,
                        maximum=64,
                        value=32,
                        step=4,
                        label="Max Tokens"
                    )
                
                # Analysis controls
                with gr.Group():
                    gr.Markdown("#### Analysis")
                    
                    layer_idx = gr.Slider(
                        minimum=0,
                        maximum=11,
                        value=0,
                        step=1,
                        label="Layer to Analyze"
                    )
                    
                    head_idx = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=0,
                        step=1,
                        label="Attention Head"
                    )
                    
                    attention_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.05,
                        label="Attention Threshold"
                    )
                
                compute_attention_btn = gr.Button("🔍 Compute Attention", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Attention Analysis")
                
                with gr.Tabs():
                    with gr.TabItem("Attention Heatmap"):
                        attention_heatmap = gr.Plot(label="Attention Pattern")
                    
                    with gr.TabItem("Multi-Head View"):
                        multihead_plot = gr.Plot(label="Multi-Head Comparison")
                    
                    with gr.TabItem("Layer Evolution"):
                        layer_evolution_plot = gr.Plot(label="Attention Evolution")
                    
                    with gr.TabItem("Head Analysis"):
                        with gr.Row():
                            head_entropy_plot = gr.Plot(label="Head Entropy", scale=1)
                            head_similarity_plot = gr.Plot(label="Head Similarity", scale=1)
        
        # Connect attention analysis
        compute_attention_btn.click(
            fn=self.compute_attention_analysis,
            inputs=[
                input_text, attn_d_model, attn_n_heads, attn_n_layers,
                attn_encoding, max_tokens, layer_idx, head_idx, attention_threshold
            ],
            outputs=[
                attention_heatmap, multihead_plot, layer_evolution_plot,
                head_entropy_plot, head_similarity_plot
            ]
        )
        
        # Update layer/head sliders based on model config
        attn_n_layers.change(
            fn=lambda n_layers: gr.Slider.update(maximum=n_layers-1),
            inputs=[attn_n_layers],
            outputs=[layer_idx]
        )
        
        attn_n_heads.change(
            fn=lambda n_heads: gr.Slider.update(maximum=n_heads-1),
            inputs=[attn_n_heads],
            outputs=[head_idx]
        )
    
    def create_comparison_tab(self):
        """Create the method comparison tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Comparison Settings")
                
                # Method selection
                with gr.Group():
                    gr.Markdown("#### Methods to Compare")
                    
                    compare_methods = gr.CheckboxGroup(
                        choices=["sinusoidal", "learned", "rope", "relative"],
                        value=["sinusoidal", "rope"],
                        label="Select Methods"
                    )
                
                # Comparison parameters
                with gr.Group():
                    gr.Markdown("#### Parameters")
                    
                    comp_seq_len = gr.Slider(
                        minimum=16,
                        maximum=128,
                        value=64,
                        step=16,
                        label="Sequence Length"
                    )
                    
                    comp_d_model = gr.Dropdown(
                        choices=[128, 256, 512],
                        value=256,
                        label="Model Dimension"
                    )
                    
                    comparison_metric = gr.Dropdown(
                        choices=[
                            "Encoding Patterns",
                            "Position Similarities", 
                            "Quality Metrics",
                            "Frequency Analysis"
                        ],
                        value="Encoding Patterns",
                        label="Comparison Type"
                    )
                
                run_comparison_btn = gr.Button("⚖️ Run Comparison", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Comparison Results")
                
                with gr.Tabs():
                    with gr.TabItem("Visual Comparison"):
                        comparison_plot = gr.Plot(label="Method Comparison")
                    
                    with gr.TabItem("Metrics Table"):
                        comparison_table = gr.Dataframe(
                            headers=["Method", "Quality", "Complexity", "Distinguishability"],
                            datatype=["str", "number", "number", "number"],
                            label="Comparison Metrics"
                        )
                    
                    with gr.TabItem("Performance"):
                        performance_plot = gr.Plot(label="Performance Comparison")
        
        # Connect comparison functionality
        run_comparison_btn.click(
            fn=self.run_method_comparison,
            inputs=[compare_methods, comp_seq_len, comp_d_model, comparison_metric],
            outputs=[comparison_plot, comparison_table, performance_plot]
        )
    
    def create_interactive_lab_tab(self):
        """Create the interactive lab tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Live Controls")
                
                with gr.Group():
                    gr.Markdown("#### Real-time Parameters")
                    
                    live_encoding = gr.Dropdown(
                        choices=["sinusoidal", "rope"],
                        value="sinusoidal",
                        label="Encoding Type"
                    )
                    
                    live_seq_len = gr.Slider(
                        minimum=8,
                        maximum=64,
                        value=32,
                        step=4,
                        label="Sequence Length"
                    )
                    
                    live_d_model = gr.Dropdown(
                        choices=[64, 128, 256],
                        value=128,
                        label="Model Dimension"
                    )
                
                # Encoding-specific parameters
                with gr.Group():
                    gr.Markdown("#### Advanced Parameters")
                    
                    rope_theta = gr.Slider(
                        minimum=1000,
                        maximum=50000,
                        value=10000,
                        step=1000,
                        label="RoPE Base (θ)",
                        visible=False
                    )
                    
                    frequency_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Frequency Scale",
                        visible=True
                    )
                
                # Visualization options
                with gr.Group():
                    gr.Markdown("#### Visualization")
                    
                    live_viz_type = gr.Dropdown(
                        choices=["Heatmap", "3D Surface", "Animation"],
                        value="Heatmap",
                        label="Visualization Type"
                    )
                    
                    auto_update = gr.Checkbox(
                        label="Auto Update",
                        value=True
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Live Visualization")
                
                live_plot = gr.Plot(label="Live Encoding Pattern")
                
                # Real-time metrics
                with gr.Row():
                    live_quality = gr.Number(label="Quality", precision=4)
                    live_complexity = gr.Number(label="Complexity", precision=4)
                    live_periodicity = gr.Number(label="Periodicity", precision=4)
        
        # Show/hide parameters based on encoding type
        live_encoding.change(
            fn=self.update_parameter_visibility,
            inputs=[live_encoding],
            outputs=[rope_theta, frequency_scale]
        )
        
        # Live update functionality
        for component in [live_encoding, live_seq_len, live_d_model, rope_theta, frequency_scale, live_viz_type]:
            component.change(
                fn=self.update_live_visualization,
                inputs=[live_encoding, live_seq_len, live_d_model, rope_theta, frequency_scale, live_viz_type, auto_update],
                outputs=[live_plot, live_quality, live_complexity, live_periodicity]
            )
    
    def create_export_tab(self):
        """Create the export and reports tab."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Export Options")
                
                with gr.Group():
                    gr.Markdown("#### Report Settings")
                    
                    report_title = gr.Textbox(
                        value="Positional Encoding Analysis Report",
                        label="Report Title"
                    )
                    
                    report_sections = gr.CheckboxGroup(
                        choices=[
                            "Executive Summary",
                            "Encoding Analysis", 
                            "Attention Patterns",
                            "Method Comparison",
                            "Performance Metrics",
                            "Technical Details"
                        ],
                        value=["Executive Summary", "Encoding Analysis"],
                        label="Include Sections"
                    )
                    
                    export_format = gr.Dropdown(
                        choices=["HTML", "PDF", "JSON", "CSV"],
                        value="HTML",
                        label="Export Format"
                    )
                
                with gr.Group():
                    gr.Markdown("#### Configuration Export")
                    
                    include_config = gr.Checkbox(
                        label="Include Configuration",
                        value=True
                    )
                    
                    include_data = gr.Checkbox(
                        label="Include Raw Data",
                        value=False
                    )
                
                generate_report_btn = gr.Button("📄 Generate Report", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📊 Export Results")
                
                report_preview = gr.HTML(label="Report Preview")
                
                download_file = gr.File(
                    label="Download Report",
                    visible=False
                )
                
                export_status = gr.Markdown("")
        
        # Connect export functionality
        generate_report_btn.click(
            fn=self.generate_report,
            inputs=[report_title, report_sections, export_format, include_config, include_data],
            outputs=[report_preview, download_file, export_status]
        )
    
    # Analysis methods
    def generate_quick_demo(self, encoding_type, seq_len, d_model):
        """Generate quick demo visualization."""
        try:
            config = ModelConfig(d_model=d_model, encoding_type=encoding_type)
            encoding = get_positional_encoding(config)
            
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, d_model)
                matrix = enc_output.squeeze(0).detach().cpu().numpy()
            else:
                return go.Figure()
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix.T,
                colorscale='Viridis',
                hovertemplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{encoding_type.title()} Encoding Pattern",
                xaxis_title="Position",
                yaxis_title="Dimension",
                height=400
            )
            
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error: {str(e)}")
    
    def analyze_encoding(self, encoding_type, d_model, seq_len, viz_type, show_similarities, max_dims_display):
        """Analyze positional encoding."""
        try:
            config = ModelConfig(d_model=d_model, encoding_type=encoding_type)
            encoding = get_positional_encoding(config)
            
            # Get encoding matrix
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, d_model)
                matrix = enc_output.squeeze(0)
            else:
                empty_fig = go.Figure()
                return empty_fig, empty_fig, {}, 0.0, 0.0, 0.0
            
            # Create main visualization
            main_fig = self._create_encoding_plot(matrix, viz_type, max_dims_display, encoding_type)
            
            # Create similarity plot
            similarity_fig = go.Figure()
            if show_similarities:
                similarities = self._compute_similarities(matrix)
                similarity_fig = go.Figure(data=go.Heatmap(
                    z=similarities.cpu().numpy(),
                    colorscale='RdBu',
                    zmid=0
                ))
                similarity_fig.update_layout(
                    title="Position Similarities",
                    xaxis_title="Position",
                    yaxis_title="Position"
                )
            
            # Compute metrics
            metrics = self._compute_encoding_metrics_gradio(matrix)
            
            return (
                main_fig,
                similarity_fig, 
                metrics,
                metrics.get('distinguishability', 0.0),
                metrics.get('distinguishability', 0.0),
                metrics.get('encoding_variance', 0.0)
            )
            
        except Exception as e:
            empty_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
            return empty_fig, empty_fig, {"error": str(e)}, 0.0, 0.0, 0.0
    
    def compute_attention_analysis(self, input_text, d_model, n_heads, n_layers, 
                                 encoding_type, max_tokens, layer_idx, head_idx, threshold):
        """Compute attention analysis."""
        try:
            # Tokenize input
            tokens = self.tokenizer.tokenize(input_text)[:max_tokens]
            
            if len(tokens) < 2:
                empty_fig = go.Figure().add_annotation(text="Need at least 2 tokens")
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
            # Create model
            config = ModelConfig(
                d_model=d_model,
                n_heads=n_heads, 
                n_layers=n_layers,
                encoding_type=encoding_type,
                max_seq_len=len(tokens)
            )
            
            model = self._get_or_create_model_gradio(config)
            
            # Generate attention
            input_ids = torch.tensor([list(range(len(tokens)))])
            
            with torch.no_grad():
                outputs = model.forward(input_ids, store_visualizations=True)
                attention_weights = outputs['attention_weights']
            
            if not attention_weights or layer_idx >= len(attention_weights):
                empty_fig = go.Figure().add_annotation(text="No attention data available")
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
            # Create visualizations
            attention_fig = self._create_attention_heatmap_gradio(
                attention_weights[layer_idx], tokens, head_idx, threshold
            )
            
            multihead_fig = self._create_multihead_plot_gradio(
                attention_weights[layer_idx], tokens
            )
            
            evolution_fig = self._create_layer_evolution_gradio(attention_weights, tokens, head_idx)
            
            entropy_fig, similarity_fig = self._create_head_analysis_gradio(
                attention_weights[layer_idx]
            )
            
            return attention_fig, multihead_fig, evolution_fig, entropy_fig, similarity_fig
            
        except Exception as e:
            error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
            return error_fig, error_fig, error_fig, error_fig, error_fig
    
    def run_method_comparison(self, methods, seq_len, d_model, comparison_type):
        """Run method comparison analysis."""
        try:
            if len(methods) < 2:
                empty_fig = go.Figure().add_annotation(text="Select at least 2 methods")
                empty_df = pd.DataFrame()
                return empty_fig, empty_df, empty_fig
            
            # Generate encodings for each method
            results = {}
            metrics_data = []
            
            for method in methods:
                config = ModelConfig(d_model=d_model, encoding_type=method)
                encoding = get_positional_encoding(config)
                
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, d_model)
                    matrix = enc_output.squeeze(0)
                    results[method] = matrix
                    
                    # Compute metrics
                    metrics = self._compute_encoding_metrics_gradio(matrix)
                    metrics_data.append({
                        'Method': method.title(),
                        'Quality': metrics.get('distinguishability', 0.0),
                        'Complexity': metrics.get('encoding_variance', 0.0),
                        'Distinguishability': metrics.get('distinguishability', 0.0)
                    })
            
            # Create comparison plot
            comparison_fig = self._create_method_comparison_plot(results, comparison_type)
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(metrics_data)
            
            # Create performance plot
            performance_fig = self._create_performance_plot(metrics_data)
            
            return comparison_fig, metrics_df, performance_fig
            
        except Exception as e:
            error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
            empty_df = pd.DataFrame()
            return error_fig, empty_df, error_fig
    
    def update_parameter_visibility(self, encoding_type):
        """Update parameter visibility based on encoding type."""
        if encoding_type == "rope":
            return gr.Slider.update(visible=True), gr.Slider.update(visible=False)
        else:
            return gr.Slider.update(visible=False), gr.Slider.update(visible=True)
    
    def update_live_visualization(self, encoding_type, seq_len, d_model, rope_theta, 
                                freq_scale, viz_type, auto_update):
        """Update live visualization."""
        if not auto_update:
            return gr.Plot.update(), 0.0, 0.0, 0.0
        
        try:
            config = ModelConfig(d_model=d_model, encoding_type=encoding_type)
            if encoding_type == "rope":
                config.rope_theta = rope_theta
                
            encoding = get_positional_encoding(config)
            
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, d_model)
                matrix = enc_output.squeeze(0)
            else:
                return go.Figure(), 0.0, 0.0, 0.0
            
            # Create visualization
            fig = self._create_encoding_plot(matrix, viz_type, 32, encoding_type)
            
            # Compute live metrics
            metrics = self._compute_encoding_metrics_gradio(matrix)
            
            return (
                fig,
                metrics.get('distinguishability', 0.0),
                metrics.get('encoding_variance', 0.0),
                metrics.get('periodicity_score', 0.0)
            )
            
        except Exception as e:
            error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
            return error_fig, 0.0, 0.0, 0.0
    
    def generate_report(self, title, sections, format_type, include_config, include_data):
        """Generate comprehensive report."""
        try:
            # Generate report content
            report_html = f"""
            <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
                <h1 style="color: #2c3e50; text-align: center;">{title}</h1>
                <p style="text-align: center; color: #7f8c8d;">
                    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            """
            
            if "Executive Summary" in sections:
                report_html += """
                <h2 style="color: #34495e;">Executive Summary</h2>
                <p>This report provides a comprehensive analysis of positional encoding methods
                in transformer models, including their patterns, effectiveness, and comparative performance.</p>
                """
            
            if "Encoding Analysis" in sections:
                report_html += """
                <h2 style="color: #34495e;">Encoding Analysis</h2>
                <p>Detailed analysis of positional encoding patterns and their mathematical properties.</p>
                """
            
            report_html += "</div>"
            
            # Create downloadable file
            if format_type == "HTML":
                file_content = report_html
                filename = f"{title.replace(' ', '_')}.html"
            elif format_type == "JSON":
                report_data = {
                    "title": title,
                    "sections": sections,
                    "generated_at": pd.Timestamp.now().isoformat(),
                    "config": include_config,
                    "data": include_data
                }
                file_content = json.dumps(report_data, indent=2)
                filename = f"{title.replace(' ', '_')}.json"
            else:
                file_content = f"Report: {title}\nGenerated: {pd.Timestamp.now()}\nSections: {', '.join(sections)}"
                filename = f"{title.replace(' ', '_')}.txt"
            
            # Save file
            with open(filename, 'w') as f:
                f.write(file_content)
            
            status_msg = f"✅ Report generated successfully! Format: {format_type}"
            
            return report_html, filename, status_msg
            
        except Exception as e:
            error_html = f"<div style='color: red;'>Error generating report: {str(e)}</div>"
            return error_html, None, f"❌ Error: {str(e)}"
    
    # Helper methods
    def _get_or_create_model_gradio(self, config):
        """Get or create model for Gradio interface."""
        config_key = f"{config.d_model}_{config.n_heads}_{config.n_layers}_{config.encoding_type}"
        
        if config_key not in self.model_cache:
            model = TransformerEncoder(config)
            self.model_cache[config_key] = model
        
        return self.model_cache[config_key]
    
    def _create_encoding_plot(self, matrix, viz_type, max_dims, encoding_type):
        """Create encoding visualization plot."""
        matrix_np = matrix.detach().cpu().numpy()
        
        if viz_type == "Heatmap":
            fig = go.Figure(data=go.Heatmap(
                z=matrix_np.T,
                colorscale='Viridis'
            ))
            fig.update_layout(
                title=f"{encoding_type.title()} Encoding Heatmap",
                xaxis_title="Position",
                yaxis_title="Dimension"
            )
        
        elif viz_type == "Line Plot":
            fig = go.Figure()
            for dim in range(min(8, matrix_np.shape[1])):
                fig.add_trace(go.Scatter(
                    x=list(range(matrix_np.shape[0])),
                    y=matrix_np[:, dim],
                    mode='lines+markers',
                    name=f'Dim {dim}'
                ))
            fig.update_layout(
                title=f"{encoding_type.title()} Line Plot",
                xaxis_title="Position",
                yaxis_title="Value"
            )
        
        elif viz_type == "3D Surface":
            # Limit dimensions for 3D
            matrix_subset = matrix_np[:, :min(max_dims, matrix_np.shape[1])]
            x = np.arange(matrix_np.shape[0])
            y = np.arange(matrix_subset.shape[1])
            X, Y = np.meshgrid(x, y)
            
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=matrix_subset.T,
                colorscale='Viridis'
            )])
            fig.update_layout(
                title=f"{encoding_type.title()} 3D Surface",
                scene=dict(
                    xaxis_title="Position",
                    yaxis_title="Dimension",
                    zaxis_title="Value"
                )
            )
        
        else:
            fig = go.Figure()
        
        return fig
    
    def _compute_similarities(self, matrix):
        """Compute position similarity matrix."""
        similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
        return similarities
    
    def _compute_encoding_metrics_gradio(self, matrix):
        """Compute encoding metrics for Gradio interface."""
        try:
            from ..utils.metrics import EncodingMetrics
            
            metrics_computer = EncodingMetrics()
            return metrics_computer.compute_encoding_quality(matrix)
        except Exception:
            return {
                'distinguishability': 0.0,
                'encoding_variance': 0.0,
                'periodicity_score': 0.0
            }
    
    def _create_attention_heatmap_gradio(self, attention_weights, tokens, head_idx, threshold):
        """Create attention heatmap for Gradio."""
        if attention_weights.dim() == 4:
            attn_matrix = attention_weights[0, head_idx].detach().cpu().numpy()
        else:
            attn_matrix = attention_weights[head_idx].detach().cpu().numpy()
        
        # Apply threshold
        display_matrix = np.where(attn_matrix >= threshold, attn_matrix, 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=display_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f"Attention Pattern - Head {head_idx}",
            xaxis_title="Key Position",
            yaxis_title="Query Position"
        )
        
        return fig
    
    def _create_multihead_plot_gradio(self, attention_weights, tokens):
        """Create multi-head comparison plot."""
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dim
        
        n_heads = attention_weights.shape[0]
        query_pos = 0  # Use first position
        
        fig = go.Figure()
        
        for head_idx in range(min(n_heads, 8)):  # Limit to 8 heads
            attn_values = attention_weights[head_idx, query_pos].detach().cpu().numpy()
            
            fig.add_trace(go.Bar(
                x=tokens,
                y=attn_values,
                name=f'Head {head_idx}',
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f'Multi-Head Attention for "{tokens[0]}"',
            xaxis_title='Key Position',
            yaxis_title='Attention Weight',
            barmode='group'
        )
        
        return fig
    
    def _create_layer_evolution_gradio(self, attention_weights, tokens, head_idx):
        """Create layer evolution plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Layer {i}' for i in range(min(4, len(attention_weights)))]
        )
        
        for i, layer_attn in enumerate(attention_weights[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if layer_attn.dim() == 4:
                attn_matrix = layer_attn[0, head_idx].detach().cpu().numpy()
            else:
                attn_matrix = layer_attn[head_idx].detach().cpu().numpy()
            
            fig.add_trace(
                go.Heatmap(z=attn_matrix, showscale=False),
                row=row, col=col
            )
        
        fig.update_layout(title="Attention Evolution Across Layers")
        return fig
    
    def _create_head_analysis_gradio(self, attention_weights):
        """Create head analysis plots."""
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dim
        
        n_heads = attention_weights.shape[0]
        
        # Compute entropy for each head
        entropies = []
        for head_idx in range(n_heads):
            head_attn = attention_weights[head_idx]
            entropy = -(head_attn * torch.log(head_attn + 1e-8)).sum(dim=-1).mean()
            entropies.append(entropy.item())
        
        # Entropy plot
        entropy_fig = go.Figure(data=go.Bar(
            x=[f'Head {i}' for i in range(n_heads)],
            y=entropies
        ))
        entropy_fig.update_layout(
            title="Attention Entropy by Head",
            xaxis_title="Head",
            yaxis_title="Entropy"
        )
        
        # Head similarity matrix
        similarities = torch.zeros(n_heads, n_heads)
        for i in range(n_heads):
            for j in range(n_heads):
                if i != j:
                    attn_i = attention_weights[i].flatten()
                    attn_j = attention_weights[j].flatten()
                    correlation = torch.corrcoef(torch.stack([attn_i, attn_j]))[0, 1]
                    similarities[i, j] = correlation
        
        similarity_fig = go.Figure(data=go.Heatmap(
            z=similarities.numpy(),
            colorscale='RdBu',
            zmid=0
        ))
        similarity_fig.update_layout(
            title="Head Similarity Matrix",
            xaxis_title="Head",
            yaxis_title="Head"
        )
        
        return entropy_fig, similarity_fig
    
    def _create_method_comparison_plot(self, results, comparison_type):
        """Create method comparison visualization."""
        if comparison_type == "Encoding Patterns":
            fig = make_subplots(
                rows=1, cols=len(results),
                subplot_titles=list(results.keys())
            )
            
            for i, (method, matrix) in enumerate(results.items()):
                matrix_np = matrix.detach().cpu().numpy()
                
                fig.add_trace(
                    go.Heatmap(z=matrix_np.T, showscale=(i==0)),
                    row=1, col=i+1
                )
            
            fig.update_layout(title="Encoding Pattern Comparison")
            
        elif comparison_type == "Position Similarities":
            fig = make_subplots(
                rows=1, cols=len(results),
                subplot_titles=list(results.keys())
            )
            
            for i, (method, matrix) in enumerate(results.items()):
                similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
                
                fig.add_trace(
                    go.Heatmap(z=similarities.numpy(), showscale=(i==0)),
                    row=1, col=i+1
                )
            
            fig.update_layout(title="Position Similarity Comparison")
        
        else:
            fig = go.Figure()
        
        return fig
    
    def _create_performance_plot(self, metrics_data):
        """Create performance comparison plot."""
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        
        metrics = ['Quality', 'Complexity', 'Distinguishability']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=df['Method'],
                y=df[metric],
                name=metric,
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Performance Metrics Comparison",
            xaxis_title="Method",
            yaxis_title="Score",
            barmode='group'
        )
        
        return fig
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        return self.interface.launch(**kwargs)


def create_gradio_interface():
    """Create and return a Gradio interface instance."""
    interface = GradioInterface()
    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
