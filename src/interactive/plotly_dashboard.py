"""Plotly Dash dashboard for positional encoding exploration."""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
import json
from typing import Dict, List, Optional, Any
import base64
import io

from config import ModelConfig, VisualizationConfig
from src.models import TransformerEncoder
from src.positional_encoding import get_positional_encoding
from src.utils.tokenizer import SimpleTokenizer
from src.utils.metrics import AttentionMetrics, EncodingMetrics
from src.visualization import AttentionVisualizer, EncodingPlotter


class PlotlyDashboard:
    """Plotly Dash dashboard for interactive positional encoding exploration."""
    
    def __init__(self, debug=False):
        self.app = dash.Dash(__name__)
        self.debug = debug
        
        # Initialize components
        self.model_cache = {}
        self.tokenizer = SimpleTokenizer()
        self.default_config = ModelConfig()
        self.viz_config = VisualizationConfig()
        
        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🧠 Positional Encoding Visualizer", 
                       className="main-title"),
                html.P("Interactive exploration of transformer positional encodings",
                      className="subtitle")
            ], className="header"),
            
            # Navigation tabs
            dcc.Tabs(id="main-tabs", value="home", children=[
                dcc.Tab(label="🏠 Home", value="home"),
                dcc.Tab(label="📊 Encoding Analysis", value="encoding"),
                dcc.Tab(label="🔍 Attention Patterns", value="attention"),
                dcc.Tab(label="⚖️ Method Comparison", value="comparison"),
                dcc.Tab(label="🎯 Interactive Lab", value="lab"),
            ]),
            
            # Tab content
            html.Div(id="tab-content"),
            
            # Store components for state management
            dcc.Store(id="model-cache"),
            dcc.Store(id="analysis-results"),
            dcc.Store(id="current-config"),
            
        ], className="dashboard-container")
        
        # Add CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Positional Encoding Visualizer</title>
                {%favicon%}
                {%css%}
                <style>
                    .dashboard-container {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    
                    .header {
                        text-align: center;
                        margin-bottom: 30px;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-radius: 10px;
                    }
                    
                    .main-title {
                        margin: 0;
                        font-size: 2.5rem;
                        font-weight: 700;
                    }
                    
                    .subtitle {
                        margin: 10px 0 0 0;
                        font-size: 1.1rem;
                        opacity: 0.9;
                    }
                    
                    .control-panel {
                        background-color: #f8f9fa;
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                    }
                    
                    .metric-card {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px;
                        border-radius: 8px;
                        text-align: center;
                        margin: 10px;
                    }
                    
                    .metric-value {
                        font-size: 1.5rem;
                        font-weight: bold;
                        margin-bottom: 5px;
                    }
                    
                    .metric-label {
                        font-size: 0.9rem;
                        opacity: 0.9;
                    }
                    
                    .info-box {
                        background-color: #e3f2fd;
                        border-left: 4px solid #2196f3;
                        padding: 15px;
                        margin: 15px 0;
                        border-radius: 4px;
                    }
                    
                    .warning-box {
                        background-color: #fff3e0;
                        border-left: 4px solid #ff9800;
                        padding: 15px;
                        margin: 15px 0;
                        border-radius: 4px;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "value")
        )
        def render_tab_content(active_tab):
            """Render content based on active tab."""
            if active_tab == "home":
                return self.create_home_content()
            elif active_tab == "encoding":
                return self.create_encoding_content()
            elif active_tab == "attention":
                return self.create_attention_content()
            elif active_tab == "comparison":
                return self.create_comparison_content()
            elif active_tab == "lab":
                return self.create_lab_content()
            
            return html.Div("Select a tab to get started!")
        
        # Encoding analysis callbacks
        @self.app.callback(
            [Output("encoding-plot", "figure"),
             Output("similarity-plot", "figure"),
             Output("metrics-display", "children")],
            [Input("analyze-encoding-btn", "n_clicks")],
            [State("encoding-type", "value"),
             State("d-model", "value"),
             State("sequence-length", "value"),
             State("viz-type", "value")]
        )
        def update_encoding_analysis(n_clicks, encoding_type, d_model, seq_len, viz_type):
            """Update encoding analysis plots."""
            if not n_clicks:
                return {}, {}, []
            
            try:
                # Create encoding
                config = ModelConfig(d_model=d_model, encoding_type=encoding_type)
                encoding = get_positional_encoding(config)
                
                # Get encoding matrix
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, d_model)
                    matrix = enc_output.squeeze(0).detach().cpu().numpy()
                else:
                    return {}, {}, [html.Div("Error: Could not generate encoding")]
                
                # Create main plot
                main_fig = self._create_encoding_plot_dash(matrix, viz_type, encoding_type)
                
                # Create similarity plot
                similarities = np.dot(matrix, matrix.T) / matrix.shape[1]
                similarity_fig = go.Figure(data=go.Heatmap(
                    z=similarities,
                    colorscale='RdBu',
                    zmid=0
                ))
                similarity_fig.update_layout(
                    title="Position Similarities",
                    xaxis_title="Position",
                    yaxis_title="Position"
                )
                
                # Compute metrics
                metrics = self._compute_metrics_dash(matrix)
                metrics_display = self._create_metrics_display(metrics)
                
                return main_fig, similarity_fig, metrics_display
                
            except Exception as e:
                error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
                return error_fig, error_fig, [html.Div(f"Error: {str(e)}", className="warning-box")]
        
        # Attention analysis callbacks
        @self.app.callback(
            [Output("attention-heatmap", "figure"),
             Output("multihead-plot", "figure"),
             Output("head-analysis", "children")],
            [Input("compute-attention-btn", "n_clicks")],
            [State("input-text", "value"),
             State("attn-d-model", "value"),
             State("attn-n-heads", "value"),
             State("attn-n-layers", "value"),
             State("attn-encoding", "value"),
             State("layer-idx", "value"),
             State("head-idx", "value")]
        )
        def update_attention_analysis(n_clicks, input_text, d_model, n_heads, n_layers, 
                                    encoding_type, layer_idx, head_idx):
            """Update attention analysis."""
            if not n_clicks:
                return {}, {}, []
            
            try:
                # Tokenize input
                tokens = self.tokenizer.tokenize(input_text)[:32]  # Limit tokens
                
                if len(tokens) < 2:
                    error_msg = "Need at least 2 tokens for analysis"
                    error_fig = go.Figure().add_annotation(text=error_msg)
                    return error_fig, error_fig, [html.Div(error_msg, className="warning-box")]
                
                # Create model
                config = ModelConfig(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    encoding_type=encoding_type,
                    max_seq_len=len(tokens)
                )
                
                model = self._get_or_create_model_dash(config)
                
                # Compute attention
                input_ids = torch.tensor([list(range(len(tokens)))])
                
                with torch.no_grad():
                    outputs = model.forward(input_ids, store_visualizations=True)
                    attention_weights = outputs['attention_weights']
                
                if not attention_weights or layer_idx >= len(attention_weights):
                    error_msg = "No attention data available for selected layer"
                    error_fig = go.Figure().add_annotation(text=error_msg)
                    return error_fig, error_fig, [html.Div(error_msg, className="warning-box")]
                
                # Create attention heatmap
                layer_attention = attention_weights[layer_idx]
                if layer_attention.dim() == 4:
                    attn_matrix = layer_attention[0, head_idx].detach().cpu().numpy()
                else:
                    attn_matrix = layer_attention[head_idx].detach().cpu().numpy()
                
                attention_fig = go.Figure(data=go.Heatmap(
                    z=attn_matrix,
                    x=tokens,
                    y=tokens,
                    colorscale='Viridis'
                ))
                attention_fig.update_layout(
                    title=f"Attention Pattern - Layer {layer_idx}, Head {head_idx}",
                    xaxis_title="Key Position",
                    yaxis_title="Query Position"
                )
                
                # Create multi-head comparison
                multihead_fig = self._create_multihead_comparison_dash(layer_attention, tokens)
                
                # Create head analysis
                head_analysis = self._create_head_analysis_dash(layer_attention)
                
                return attention_fig, multihead_fig, head_analysis
                
            except Exception as e:
                error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
                return error_fig, error_fig, [html.Div(f"Error: {str(e)}", className="warning-box")]
        
        # Method comparison callbacks
        @self.app.callback(
            [Output("comparison-plot", "figure"),
             Output("comparison-table", "data")],
            [Input("run-comparison-btn", "n_clicks")],
            [State("comparison-methods", "value"),
             State("comp-seq-len", "value"),
             State("comp-d-model", "value")]
        )
        def update_method_comparison(n_clicks, methods, seq_len, d_model):
            """Update method comparison."""
            if not n_clicks or not methods or len(methods) < 2:
                return {}, []
            
            try:
                results = {}
                table_data = []
                
                # Generate encodings for each method
                for method in methods:
                    config = ModelConfig(d_model=d_model, encoding_type=method)
                    encoding = get_positional_encoding(config)
                    
                    if hasattr(encoding, 'forward'):
                        enc_output = encoding.forward(seq_len, d_model)
                        matrix = enc_output.squeeze(0).detach().cpu().numpy()
                        results[method] = matrix
                        
                        # Compute metrics for table
                        metrics = self._compute_metrics_dash(matrix)
                        table_data.append({
                            'Method': method.title(),
                            'Quality': f"{metrics.get('quality', 0):.4f}",
                            'Variance': f"{metrics.get('variance', 0):.4f}",
                            'Complexity': f"{metrics.get('complexity', 0):.4f}"
                        })
                
                # Create comparison plot
                comparison_fig = self._create_comparison_plot_dash(results)
                
                return comparison_fig, table_data
                
            except Exception as e:
                error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
                return error_fig, []
        
        # Interactive lab callbacks
        @self.app.callback(
            [Output("live-plot", "figure"),
             Output("live-metrics", "children")],
            [Input("live-encoding", "value"),
             Input("live-seq-len", "value"),
             Input("live-d-model", "value"),
             Input("live-viz-type", "value")]
        )
        def update_live_visualization(encoding_type, seq_len, d_model, viz_type):
            """Update live visualization in interactive lab."""
            try:
                config = ModelConfig(d_model=d_model, encoding_type=encoding_type)
                encoding = get_positional_encoding(config)
                
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, d_model)
                    matrix = enc_output.squeeze(0).detach().cpu().numpy()
                else:
                    return {}, []
                
                # Create live plot
                live_fig = self._create_encoding_plot_dash(matrix, viz_type, encoding_type)
                
                # Create live metrics
                metrics = self._compute_metrics_dash(matrix)
                live_metrics = self._create_live_metrics_display(metrics)
                
                return live_fig, live_metrics
                
            except Exception as e:
                error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}")
                return error_fig, [html.Div(f"Error: {str(e)}")]
    
    def create_home_content(self):
        """Create home tab content."""
        return html.Div([
            html.Div([
                html.Div([
                    html.H2("Welcome to the Positional Encoding Visualizer! 🎉"),
                    html.P([
                        "This interactive tool helps you understand how different positional ",
                        "encoding methods work in transformer models and how they affect attention patterns."
                    ]),
                    html.H3("🎯 Key Features:"),
                    html.Ul([
                        html.Li("📊 Encoding Analysis: Visualize positional encoding patterns"),
                        html.Li("🔍 Attention Patterns: See how encodings affect attention"),
                        html.Li("⚖️ Method Comparison: Compare different encoding methods"),
                        html.Li("🎯 Interactive Lab: Real-time parameter exploration"),
                    ]),
                    html.H3("🚀 Getting Started:"),
                    html.Ol([
                        html.Li("Start with 'Encoding Analysis' to explore basic patterns"),
                        html.Li("Try 'Attention Patterns' to see real effects on attention"),
                        html.Li("Use 'Interactive Lab' for live experimentation"),
                        html.Li("Compare methods in 'Method Comparison'"),
                    ])
                ], className="col-8"),
                
                html.Div([
                    html.H3("🎛️ Quick Demo"),
                    html.Div([
                        html.Label("Encoding Type:"),
                        dcc.Dropdown(
                            id="demo-encoding",
                            options=[
                                {"label": "Sinusoidal", "value": "sinusoidal"},
                                {"label": "RoPE", "value": "rope"},
                            ],
                            value="sinusoidal"
                        ),
                    ], className="control-group"),
                    html.Div([
                        html.Label("Sequence Length:"),
                        dcc.Slider(
                            id="demo-seq-len",
                            min=8, max=64, step=8, value=32,
                            marks={i: str(i) for i in range(8, 65, 16)}
                        ),
                    ], className="control-group"),
                    html.Button("🚀 Generate Demo", id="demo-btn", className="btn-primary"),
                    dcc.Graph(id="demo-plot")
                ], className="col-4 control-panel")
            ], className="row"),
            
            # Educational content
            html.Div([
                html.H3("🧠 Understanding Positional Encoding"),
                html.Div([
                    html.Div([
                        html.H4("Why Do We Need It?"),
                        html.P([
                            "Transformers process all tokens in parallel, but they need to understand ",
                            "the order of words. Positional encoding adds position information to embeddings."
                        ])
                    ], className="info-box"),
                    
                    html.Div([
                        html.H4("Different Methods:"),
                        html.Ul([
                            html.Li("🌊 Sinusoidal: Mathematical patterns, good extrapolation"),
                            html.Li("🎓 Learned: Adaptive to data, limited by training length"),
                            html.Li("🔄 RoPE: Relative positions, excellent extrapolation"),
                            html.Li("📏 Relative: Direct relative position modeling"),
                        ])
                    ], className="info-box")
                ])
            ])
        ])
        
        # Demo callback
        @self.app.callback(
            Output("demo-plot", "figure"),
            [Input("demo-btn", "n_clicks")],
            [State("demo-encoding", "value"),
             State("demo-seq-len", "value")]
        )
        def update_demo(n_clicks, encoding_type, seq_len):
            if not n_clicks:
                return {}
            
            try:
                config = ModelConfig(d_model=128, encoding_type=encoding_type)
                encoding = get_positional_encoding(config)
                
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, 128)
                    matrix = enc_output.squeeze(0).detach().cpu().numpy()
                else:
                    return {}
                
                fig = go.Figure(data=go.Heatmap(
                    z=matrix.T,
                    colorscale='Viridis'
                ))
                fig.update_layout(
                    title=f"{encoding_type.title()} Encoding Pattern",
                    xaxis_title="Position",
                    yaxis_title="Dimension"
                )
                return fig
            except:
                return {}
    
    def create_encoding_content(self):
        """Create encoding analysis tab content."""
        return html.Div([
            html.H2("📊 Encoding Analysis"),
            
            html.Div([
                # Controls
                html.Div([
                    html.H3("🎛️ Configuration"),
                    
                    html.Div([
                        html.Label("Encoding Type:"),
                        dcc.Dropdown(
                            id="encoding-type",
                            options=[
                                {"label": "Sinusoidal", "value": "sinusoidal"},
                                {"label": "Learned", "value": "learned"},
                                {"label": "RoPE", "value": "rope"},
                                {"label": "Relative", "value": "relative"},
                            ],
                            value="sinusoidal"
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Model Dimension:"),
                        dcc.Dropdown(
                            id="d-model",
                            options=[{"label": str(d), "value": d} for d in [64, 128, 256, 512]],
                            value=256
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Sequence Length:"),
                        dcc.Slider(
                            id="sequence-length",
                            min=8, max=128, step=8, value=64,
                            marks={i: str(i) for i in range(8, 129, 32)}
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Visualization Type:"),
                        dcc.Dropdown(
                            id="viz-type",
                            options=[
                                {"label": "Heatmap", "value": "heatmap"},
                                {"label": "Line Plot", "value": "line"},
                                {"label": "3D Surface", "value": "3d"},
                            ],
                            value="heatmap"
                        ),
                    ], className="control-group"),
                    
                    html.Button("🔍 Analyze Encoding", id="analyze-encoding-btn", className="btn-primary"),
                    
                ], className="col-3 control-panel"),
                
                # Visualizations
                html.Div([
                    html.H3("📈 Encoding Visualization"),
                    dcc.Graph(id="encoding-plot"),
                    
                    html.H3("🔗 Position Similarities"),
                    dcc.Graph(id="similarity-plot"),
                ], className="col-6"),
                
                # Metrics
                html.Div([
                    html.H3("📊 Metrics"),
                    html.Div(id="metrics-display"),
                ], className="col-3")
                
            ], className="row")
        ])
    
    def create_attention_content(self):
        """Create attention patterns tab content."""
        return html.Div([
            html.H2("🔍 Attention Patterns"),
            
            html.Div([
                # Controls
                html.Div([
                    html.H3("🎛️ Model Configuration"),
                    
                    html.Div([
                        html.Label("Input Text:"),
                        dcc.Textarea(
                            id="input-text",
                            value="The quick brown fox jumps over the lazy dog.",
                            style={'width': '100%', 'height': 80}
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Model Dimension:"),
                        dcc.Dropdown(
                            id="attn-d-model",
                            options=[{"label": str(d), "value": d} for d in [128, 256, 512]],
                            value=256
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Attention Heads:"),
                        dcc.Dropdown(
                            id="attn-n-heads",
                            options=[{"label": str(h), "value": h} for h in [2, 4, 8, 12, 16]],
                            value=8
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Number of Layers:"),
                        dcc.Slider(
                            id="attn-n-layers",
                            min=1, max=8, step=1, value=4,
                            marks={i: str(i) for i in range(1, 9)}
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Encoding Type:"),
                        dcc.Dropdown(
                            id="attn-encoding",
                            options=[
                                {"label": "Sinusoidal", "value": "sinusoidal"},
                                {"label": "Learned", "value": "learned"},
                                {"label": "RoPE", "value": "rope"},
                            ],
                            value="sinusoidal"
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Layer to Analyze:"),
                        dcc.Slider(
                            id="layer-idx",
                            min=0, max=7, step=1, value=0,
                            marks={i: f"L{i}" for i in range(8)}
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Attention Head:"),
                        dcc.Slider(
                            id="head-idx",
                            min=0, max=15, step=1, value=0,
                            marks={i: f"H{i}" for i in range(0, 16, 4)}
                        ),
                    ], className="control-group"),
                    
                    html.Button("🔍 Compute Attention", id="compute-attention-btn", className="btn-primary"),
                    
                ], className="col-3 control-panel"),
                
                # Visualizations
                html.Div([
                    html.H3("🎯 Attention Heatmap"),
                    dcc.Graph(id="attention-heatmap"),
                    
                    html.H3("👥 Multi-Head Comparison"),
                    dcc.Graph(id="multihead-plot"),
                ], className="col-6"),
                
                # Analysis
                html.Div([
                    html.H3("📊 Head Analysis"),
                    html.Div(id="head-analysis"),
                ], className="col-3")
                
            ], className="row")
        ])
    
    def create_comparison_content(self):
        """Create method comparison tab content."""
        return html.Div([
            html.H2("⚖️ Method Comparison"),
            
            html.Div([
                # Controls
                html.Div([
                    html.H3("🎛️ Comparison Settings"),
                    
                    html.Div([
                        html.Label("Methods to Compare:"),
                        dcc.Checklist(
                            id="comparison-methods",
                            options=[
                                {"label": " Sinusoidal", "value": "sinusoidal"},
                                {"label": " Learned", "value": "learned"},
                                {"label": " RoPE", "value": "rope"},
                                {"label": " Relative", "value": "relative"},
                            ],
                            value=["sinusoidal", "rope"],
                            inline=False
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Sequence Length:"),
                        dcc.Slider(
                            id="comp-seq-len",
                            min=16, max=128, step=16, value=64,
                            marks={i: str(i) for i in range(16, 129, 32)}
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Model Dimension:"),
                        dcc.Dropdown(
                            id="comp-d-model",
                            options=[{"label": str(d), "value": d} for d in [128, 256, 512]],
                            value=256
                        ),
                    ], className="control-group"),
                    
                    html.Button("⚖️ Run Comparison", id="run-comparison-btn", className="btn-primary"),
                    
                ], className="col-3 control-panel"),
                
                # Results
                html.Div([
                    html.H3("📊 Comparison Results"),
                    dcc.Graph(id="comparison-plot"),
                    
                    html.H3("📋 Metrics Table"),
                    dash_table.DataTable(
                        id="comparison-table",
                        columns=[
                            {"name": "Method", "id": "Method"},
                            {"name": "Quality", "id": "Quality"},
                            {"name": "Variance", "id": "Variance"},
                            {"name": "Complexity", "id": "Complexity"},
                        ],
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': 'bold'},
                        style_data={'backgroundColor': '#f8f9fa'}
                    )
                ], className="col-9")
            ], className="row")
        ])
    
    def create_lab_content(self):
        """Create interactive lab tab content."""
        return html.Div([
            html.H2("🎯 Interactive Lab"),
            html.P("Experiment with parameters in real-time!"),
            
            html.Div([
                # Live controls
                html.Div([
                    html.H3("🎛️ Live Controls"),
                    
                    html.Div([
                        html.Label("Encoding Type:"),
                        dcc.Dropdown(
                            id="live-encoding",
                            options=[
                                {"label": "Sinusoidal", "value": "sinusoidal"},
                                {"label": "RoPE", "value": "rope"},
                            ],
                            value="sinusoidal"
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Sequence Length:"),
                        dcc.Slider(
                            id="live-seq-len",
                            min=8, max=64, step=4, value=32,
                            marks={i: str(i) for i in range(8, 65, 16)}
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Model Dimension:"),
                        dcc.Dropdown(
                            id="live-d-model",
                            options=[{"label": str(d), "value": d} for d in [64, 128, 256]],
                            value=128
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Visualization:"),
                        dcc.Dropdown(
                            id="live-viz-type",
                            options=[
                                {"label": "Heatmap", "value": "heatmap"},
                                {"label": "3D Surface", "value": "3d"},
                                {"label": "Line Plot", "value": "line"},
                            ],
                            value="heatmap"
                        ),
                    ], className="control-group"),
                    
                ], className="col-3 control-panel"),
                
                # Live visualization
                html.Div([
                    html.H3("📊 Live Visualization"),
                    dcc.Graph(id="live-plot"),
                ], className="col-6"),
                
                # Live metrics
                html.Div([
                    html.H3("📈 Live Metrics"),
                    html.Div(id="live-metrics"),
                ], className="col-3")
                
            ], className="row")
        ])
    
    # Helper methods
    def _get_or_create_model_dash(self, config):
        """Get or create model for Dash interface."""
        config_key = f"{config.d_model}_{config.n_heads}_{config.n_layers}_{config.encoding_type}"
        
        if config_key not in self.model_cache:
            model = TransformerEncoder(config)
            self.model_cache[config_key] = model
        
        return self.model_cache[config_key]
    
    def _create_encoding_plot_dash(self, matrix, viz_type, encoding_type):
        """Create encoding plot for Dash."""
        if viz_type == "heatmap":
            fig = go.Figure(data=go.Heatmap(
                z=matrix.T,
                colorscale='Viridis'
            ))
            fig.update_layout(
                title=f"{encoding_type.title()} Encoding Heatmap",
                xaxis_title="Position",
                yaxis_title="Dimension"
            )
        
        elif viz_type == "line":
            fig = go.Figure()
            for dim in range(min(8, matrix.shape[1])):
                fig.add_trace(go.Scatter(
                    x=list(range(matrix.shape[0])),
                    y=matrix[:, dim],
                    mode='lines+markers',
                    name=f'Dim {dim}'
                ))
            fig.update_layout(
                title=f"{encoding_type.title()} Line Plot",
                xaxis_title="Position",
                yaxis_title="Value"
            )
        
        elif viz_type == "3d":
            # Create 3D surface
            matrix_subset = matrix[:, :min(32, matrix.shape[1])]
            x = np.arange(matrix.shape[0])
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
    
    def _compute_metrics_dash(self, matrix):
        """Compute metrics for Dash interface."""
        # Basic statistics
        variance = np.var(matrix)
        mean = np.mean(matrix)
        
        # Quality metrics (simplified)
        quality = variance / (1 + np.abs(mean))
        complexity = np.std(matrix, axis=0).mean()
        
        return {
            'variance': variance,
            'quality': quality,
            'complexity': complexity,
            'mean': mean,
            'std': np.std(matrix)
        }
    
    def _create_metrics_display(self, metrics):
        """Create metrics display components."""
        return [
            html.Div([
                html.Div(f"{metrics['quality']:.4f}", className="metric-value"),
                html.Div("Quality Score", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{metrics['variance']:.4f}", className="metric-value"),
                html.Div("Variance", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{metrics['complexity']:.4f}", className="metric-value"),
                html.Div("Complexity", className="metric-label")
            ], className="metric-card"),
        ]
    
    def _create_live_metrics_display(self, metrics):
        """Create live metrics display."""
        return [
            html.Div([
                html.H4("Real-time Metrics"),
                html.P(f"Quality: {metrics['quality']:.4f}"),
                html.P(f"Variance: {metrics['variance']:.4f}"),
                html.P(f"Complexity: {metrics['complexity']:.4f}"),
                html.P(f"Mean: {metrics['mean']:.4f}"),
                html.P(f"Std: {metrics['std']:.4f}"),
            ], className="info-box")
        ]
    
    def _create_multihead_comparison_dash(self, attention_weights, tokens):
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
            title=f'Multi-Head Attention for "{tokens[0] if tokens else "Token 0"}"',
            xaxis_title='Key Position',
            yaxis_title='Attention Weight',
            barmode='group'
        )
        
        return fig
    
    def _create_head_analysis_dash(self, attention_weights):
        """Create head analysis components."""
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Remove batch dim
        
        n_heads = attention_weights.shape[0]
        
        # Compute entropy for each head
        entropies = []
        for head_idx in range(n_heads):
            head_attn = attention_weights[head_idx]
            entropy = -(head_attn * torch.log(head_attn + 1e-8)).sum(dim=-1).mean()
            entropies.append(entropy.item())
        
        return [
            html.Div([
                html.H4("Head Statistics"),
                html.P(f"Number of heads: {n_heads}"),
                html.P(f"Average entropy: {np.mean(entropies):.4f}"),
                html.P(f"Entropy range: {np.min(entropies):.4f} - {np.max(entropies):.4f}"),
            ], className="info-box")
        ]
    
    def _create_comparison_plot_dash(self, results):
        """Create method comparison plot."""
        fig = make_subplots(
            rows=1, cols=len(results),
            subplot_titles=list(results.keys())
        )
        
        for i, (method, matrix) in enumerate(results.items()):
            fig.add_trace(
                go.Heatmap(z=matrix.T, showscale=(i==0)),
                row=1, col=i+1
            )
        
        fig.update_layout(title="Method Comparison")
        return fig
    
    def run_server(self, **kwargs):
        """Run the Dash server."""
        return self.app.run_server(debug=self.debug, **kwargs)


def create_plotly_dashboard(debug=False):
    """Create and return a Plotly Dash dashboard instance."""
    dashboard = PlotlyDashboard(debug=debug)
    return dashboard


if __name__ == "__main__":
    dashboard = create_plotly_dashboard(debug=True)
    dashboard.run_server(host="0.0.0.0", port=8050)
