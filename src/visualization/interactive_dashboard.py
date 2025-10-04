"""Interactive dashboard for comprehensive positional encoding exploration."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import io
import base64

from config import ModelConfig, VisualizationConfig
from src.models import TransformerEncoder
from src.positional_encoding import get_positional_encoding
from src.utils.tokenizer import SimpleTokenizer
from .attention_visualizer import AttentionVisualizer
from .encoding_plots import EncodingPlotter


class InteractiveDashboard:
    """Main interactive dashboard for positional encoding visualization."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Positional Encoding Visualizer",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #34495E;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #F8F9FA;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3498DB;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_state(self):
        """Initialize session state variables."""
        if 'model_config' not in st.session_state:
            st.session_state.model_config = ModelConfig()
        
        if 'viz_config' not in st.session_state:
            st.session_state.viz_config = VisualizationConfig()
        
        if 'current_text' not in st.session_state:
            st.session_state.current_text = "The quick brown fox jumps over the lazy dog."
        
        if 'model_cache' not in st.session_state:
            st.session_state.model_cache = {}
    
    def run(self):
        """Main dashboard execution."""
        # Header
        st.markdown('<h1 class="main-header">🧠 Positional Encoding Visualizer</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Explore transformer positional encodings through interactive visualizations**
        
        This tool helps you understand how different positional encoding methods work and 
        how they affect attention patterns in transformer models.
        """)
        
        # Sidebar configuration
        self.create_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Encoding Patterns", 
            "🔍 Attention Analysis", 
            "📈 Comparison", 
            "🎯 Interactive Exploration",
            "📚 Educational Mode"
        ])
        
        with tab1:
            self.encoding_patterns_tab()
        
        with tab2:
            self.attention_analysis_tab()
        
        with tab3:
            self.comparison_tab()
        
        with tab4:
            self.interactive_exploration_tab()
        
        with tab5:
            self.educational_mode_tab()
    
    def create_sidebar(self):
        """Create sidebar with configuration options."""
        st.sidebar.markdown("## ⚙️ Configuration")
        
        # Model Configuration
        with st.sidebar.expander("🤖 Model Settings", expanded=True):
            st.session_state.model_config.d_model = st.selectbox(
                "Model Dimension",
                [64, 128, 256, 512],
                index=2,
                help="Dimension of the model embeddings"
            )
            
            st.session_state.model_config.n_heads = st.selectbox(
                "Number of Attention Heads",
                [1, 2, 4, 8, 16],
                index=2,
                help="Number of attention heads in multi-head attention"
            )
            
            st.session_state.model_config.n_layers = st.slider(
                "Number of Layers",
                1, 8, 3,
                help="Number of transformer encoder layers"
            )
            
            st.session_state.model_config.encoding_type = st.selectbox(
                "Positional Encoding Type",
                ["sinusoidal", "learned", "rope", "relative"],
                help="Type of positional encoding to use"
            )
        
        # Sequence Configuration
        with st.sidebar.expander("📝 Sequence Settings"):
            max_seq_len = st.slider(
                "Maximum Sequence Length",
                16, 512, 128,
                help="Maximum sequence length for the model"
            )
            st.session_state.model_config.max_seq_len = max_seq_len
            
            current_seq_len = st.slider(
                "Current Sequence Length",
                5, min(max_seq_len, 100), 
                min(32, max_seq_len),
                help="Sequence length for current visualization"
            )
            
            # Text input
            st.session_state.current_text = st.text_area(
                "Input Text",
                st.session_state.current_text,
                height=100,
                help="Text to analyze (will be tokenized)"
            )
        
        # Visualization Configuration
        with st.sidebar.expander("🎨 Visualization Settings"):
            theme = st.selectbox(
                "Color Theme",
                ["default", "dark", "academic", "colorblind_friendly"],
                help="Visual theme for plots"
            )
            
            show_values = st.checkbox(
                "Show Attention Values",
                value=True,
                help="Display numerical values in attention heatmaps"
            )
            
            max_tokens_display = st.slider(
                "Max Tokens to Display",
                5, 50, 20,
                help="Maximum number of tokens to show in visualizations"
            )
        
        return current_seq_len
    
    def encoding_patterns_tab(self):
        """Encoding patterns visualization tab."""
        st.markdown('<h2 class="section-header">📊 Encoding Patterns</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            # Controls
            st.markdown("### Controls")
            
            seq_len = st.slider("Sequence Length", 5, 100, 32, key="enc_seq_len")
            
            encoding_type = st.session_state.model_config.encoding_type
            
            # Create encoding instance
            encoding = get_positional_encoding(st.session_state.model_config)
            
            # Show encoding info
            st.markdown("### Encoding Information")
            info_placeholder = st.empty()
            
        with col1:
            # Main visualization
            if encoding_type in ['sinusoidal', 'learned']:
                enc_output = encoding.forward(seq_len, st.session_state.model_config.d_model)
                matrix = enc_output.squeeze(0).detach().cpu().numpy()
                
                # Encoding heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=matrix.T,
                    colorscale='Viridis',
                    hovertemplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{encoding_type.title()} Positional Encoding",
                    xaxis_title="Position",
                    yaxis_title="Dimension",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Position similarity matrix
                similarities = torch.mm(
                    torch.from_numpy(matrix), 
                    torch.from_numpy(matrix).t()
                ) / matrix.shape[1]
                
                fig2 = go.Figure(data=go.Heatmap(
                    z=similarities.numpy(),
                    colorscale='RdBu',
                    zmid=0,
                    hovertemplate='Pos1: %{x}<br>Pos2: %{y}<br>Similarity: %{z:.3f}<extra></extra>'
                ))
                
                fig2.update_layout(
                    title="Position Similarity Matrix",
                    xaxis_title="Position",
                    yaxis_title="Position",
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
            elif encoding_type == 'rope':
                rope_data = encoding.forward(seq_len, st.session_state.model_config.d_model)
                
                # RoPE visualization
                cos_values = rope_data['cos'].squeeze(0).detach().cpu().numpy()
                sin_values = rope_data['sin'].squeeze(0).detach().cpu().numpy()
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Cosine Values', 'Sine Values']
                )
                
                fig.add_trace(
                    go.Heatmap(z=cos_values.T, colorscale='Viridis', showscale=False),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Heatmap(z=sin_values.T, colorscale='Plasma'),
                    row=1, col=2
                )
                
                fig.update_layout(title="RoPE Encoding Patterns", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Update encoding info
        with info_placeholder.container():
            if encoding_type == 'sinusoidal':
                if hasattr(encoding, 'analyze_frequency_components'):
                    freq_analysis = encoding.analyze_frequency_components()
                    st.markdown(f"""
                    <div class="metric-card">
                    <strong>Frequency Range:</strong><br>
                    Min: {freq_analysis['min_frequency']:.6f}<br>
                    Max: {freq_analysis['max_frequency']:.6f}<br>
                    Ratio: {freq_analysis['frequency_ratio']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                    
            elif encoding_type == 'rope':
                if hasattr(encoding, 'analyze_rotation_frequencies'):
                    freq_analysis = encoding.analyze_rotation_frequencies()
                    st.markdown(f"""
                    <div class="metric-card">
                    <strong>Rotation Frequencies:</strong><br>
                    Min: {freq_analysis['min_frequency']:.6f}<br>
                    Max: {freq_analysis['max_frequency']:.6f}<br>
                    Max Period: {freq_analysis['max_wavelength']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
    
    def attention_analysis_tab(self):
        """Attention analysis tab."""
        st.markdown('<h2 class="section-header">🔍 Attention Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Tokenize input text
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(st.session_state.current_text)
        
        if len(tokens) > st.session_state.model_config.max_seq_len:
            tokens = tokens[:st.session_state.model_config.max_seq_len]
            st.warning(f"Text truncated to {st.session_state.model_config.max_seq_len} tokens")
        
        # Create model
        model = self.get_or_create_model()
        
        # Get attention weights
        with torch.no_grad():
            input_ids = torch.tensor([range(len(tokens))])  # Simple tokenization
            outputs = model.forward(input_ids, store_visualizations=True)
            attention_weights = outputs['attention_weights']
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### Controls")
            
            layer_idx = st.selectbox(
                "Layer", 
                range(st.session_state.model_config.n_layers),
                format_func=lambda x: f"Layer {x}"
            )
            
            head_idx = st.selectbox(
                "Attention Head",
                range(st.session_state.model_config.n_heads),
                format_func=lambda x: f"Head {x}"
            )
            
            show_token_labels = st.checkbox("Show Token Labels", value=True)
            
        with col1:
            # Attention heatmap
            if len(attention_weights) > layer_idx:
                attn_matrix = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
                
                # Create labels
                labels = tokens if show_token_labels else [f"Pos {i}" for i in range(len(tokens))]
                
                fig = go.Figure(data=go.Heatmap(
                    z=attn_matrix,
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Attention Pattern - Layer {layer_idx}, Head {head_idx}",
                    xaxis_title="Key Position",
                    yaxis_title="Query Position",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Multi-head comparison
        st.markdown("### Multi-Head Comparison")
        
        if len(attention_weights) > layer_idx:
            query_pos = st.selectbox(
                "Query Position",
                range(len(tokens)),
                format_func=lambda x: f"{tokens[x]} (pos {x})"
            )
            
            # Create comparison plot
            n_heads = st.session_state.model_config.n_heads
            attention_data = []
            
            for head in range(n_heads):
                attn_values = attention_weights[layer_idx][0, head, query_pos].detach().cpu().numpy()
                for pos, (token, attn) in enumerate(zip(tokens, attn_values)):
                    attention_data.append({
                        'Head': f'Head {head}',
                        'Position': pos,
                        'Token': token,
                        'Attention': attn
                    })
            
            df = pd.DataFrame(attention_data)
            
            fig = px.bar(
                df, x='Token', y='Attention', color='Head',
                title=f'Attention Patterns for Query: "{tokens[query_pos]}"',
                barmode='group'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def comparison_tab(self):
        """Encoding comparison tab."""
        st.markdown('<h2 class="section-header">📈 Comparison</h2>', 
                   unsafe_allow_html=True)
        
        # Select encodings to compare
        st.markdown("### Select Encodings to Compare")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            compare_sinusoidal = st.checkbox("Sinusoidal", value=True)
        with col2:
            compare_learned = st.checkbox("Learned", value=True)
        with col3:
            compare_rope = st.checkbox("RoPE", value=True)
        with col4:
            compare_relative = st.checkbox("Relative", value=False)
        
        selected_encodings = []
        if compare_sinusoidal:
            selected_encodings.append("sinusoidal")
        if compare_learned:
            selected_encodings.append("learned")
        if compare_rope:
            selected_encodings.append("rope")
        if compare_relative:
            selected_encodings.append("relative")
        
        if len(selected_encodings) < 2:
            st.warning("Please select at least 2 encodings to compare.")
            return
        
        seq_len = st.slider("Sequence Length for Comparison", 5, 100, 32)
        
        # Create comparison visualizations
        encodings = {}
        encoding_matrices = {}
        
        for enc_type in selected_encodings:
            config = ModelConfig(**st.session_state.model_config.__dict__)
            config.encoding_type = enc_type
            encoding = get_positional_encoding(config)
            encodings[enc_type] = encoding
            
            # Get encoding representation
            if enc_type in ['sinusoidal', 'learned']:
                enc_output = encoding.forward(seq_len, config.d_model)
                matrix = enc_output.squeeze(0)
            elif enc_type == 'rope':
                rope_data = encoding.forward(seq_len, config.d_model)
                matrix = rope_data['cos'].squeeze(0)
            else:
                continue  # Skip unsupported types for now
            
            encoding_matrices[enc_type] = matrix
        
        # Create comparison plots
        if len(encoding_matrices) >= 2:
            # Encoding patterns
            fig = make_subplots(
                rows=2, cols=len(encoding_matrices),
                subplot_titles=[f"{name.title()}" for name in encoding_matrices.keys()] +
                              [f"{name.title()} Similarities" for name in encoding_matrices.keys()],
                vertical_spacing=0.1
            )
            
            for col, (enc_name, matrix) in enumerate(encoding_matrices.items(), 1):
                matrix_np = matrix.detach().cpu().numpy()
                
                # Encoding heatmap
                fig.add_trace(
                    go.Heatmap(z=matrix_np.T, colorscale='Viridis', showscale=(col==1)),
                    row=1, col=col
                )
                
                # Similarity matrix
                similarities = torch.mm(matrix, matrix.t()) / matrix.size(-1)
                fig.add_trace(
                    go.Heatmap(z=similarities.detach().cpu().numpy(), 
                              colorscale='RdBu', zmid=0, showscale=(col==1)),
                    row=2, col=col
                )
            
            fig.update_layout(
                title="Encoding Comparison",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quantitative comparison
            st.markdown("### Quantitative Comparison")
            
            comparison_data = []
            encoding_names = list(encoding_matrices.keys())
            
            for i, enc1 in enumerate(encoding_names):
                for j, enc2 in enumerate(encoding_names[i+1:], i+1):
                    matrix1 = encoding_matrices[enc1]
                    matrix2 = encoding_matrices[enc2]
                    
                    # Align dimensions
                    min_dim = min(matrix1.size(-1), matrix2.size(-1))
                    matrix1_aligned = matrix1[:, :min_dim]
                    matrix2_aligned = matrix2[:, :min_dim]
                    
                    # Compute similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        matrix1_aligned.flatten(), 
                        matrix2_aligned.flatten(), 
                        dim=0
                    ).item()
                    
                    comparison_data.append({
                        'Encoding 1': enc1.title(),
                        'Encoding 2': enc2.title(),
                        'Cosine Similarity': similarity,
                        'Difference': 1 - similarity
                    })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
    
    def interactive_exploration_tab(self):
        """Interactive exploration tab with real-time parameter adjustment."""
        st.markdown('<h2 class="section-header">🎯 Interactive Exploration</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("### Real-time Parameter Exploration")
        st.markdown("Adjust parameters below to see real-time changes in encoding patterns.")
        
        # Parameter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            live_seq_len = st.slider("Sequence Length", 5, 64, 32, key="live_seq_len")
            live_d_model = st.selectbox("Model Dimension", [64, 128, 256], index=1, key="live_d_model")
        
        with col2:
            live_encoding_type = st.selectbox(
                "Encoding Type", 
                ["sinusoidal", "rope"], 
                key="live_encoding_type"
            )
            
            if live_encoding_type == "rope":
                rope_base = st.slider("RoPE Base (θ)", 1000.0, 50000.0, 10000.0, step=1000.0)
        
        with col3:
            visualization_type = st.selectbox(
                "Visualization Type",
                ["Heatmap", "Line Plot", "3D Surface"],
                key="live_viz_type"
            )
        
        # Create encoding with current parameters
        live_config = ModelConfig(
            d_model=live_d_model,
            encoding_type=live_encoding_type,
            max_seq_len=live_seq_len * 2
        )
        
        if live_encoding_type == "rope":
            live_config.rope_theta = rope_base
        
        encoding = get_positional_encoding(live_config)
        
        # Generate visualization
        if live_encoding_type in ['sinusoidal']:
            enc_output = encoding.forward(live_seq_len, live_d_model)
            matrix = enc_output.squeeze(0).detach().cpu().numpy()
        elif live_encoding_type == 'rope':
            rope_data = encoding.forward(live_seq_len, live_d_model)
            matrix = rope_data['cos'].squeeze(0).detach().cpu().numpy()
        
        # Create visualization based on type
        if visualization_type == "Heatmap":
            fig = go.Figure(data=go.Heatmap(
                z=matrix.T,
                colorscale='Viridis',
                hoveremplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title=f"Live {live_encoding_type.title()} Encoding",
                xaxis_title="Position",
                yaxis_title="Dimension",
                height=500
            )
            
        elif visualization_type == "Line Plot":
            fig = go.Figure()
            
            # Plot selected dimensions
            for dim in range(0, min(matrix.shape[1], 8), 2):
                fig.add_trace(go.Scatter(
                    x=list(range(live_seq_len)),
                    y=matrix[:, dim],
                    mode='lines+markers',
                    name=f'Dim {dim}',
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f"Live {live_encoding_type.title()} Patterns",
                xaxis_title="Position",
                yaxis_title="Encoding Value",
                height=500
            )
            
        elif visualization_type == "3D Surface":
            # Create meshgrid for 3D surface
            x = np.arange(live_seq_len)
            y = np.arange(min(matrix.shape[1], 32))  # Limit dimensions for 3D
            X, Y = np.meshgrid(x, y)
            Z = matrix[:, :min(matrix.shape[1], 32)].T
            
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                hoveremplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f"3D {live_encoding_type.title()} Surface",
                scene=dict(
                    xaxis_title="Position",
                    yaxis_title="Dimension", 
                    zaxis_title="Encoding Value"
                ),
                height=600
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Real-time statistics
        st.markdown("### Real-time Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Value", f"{matrix.mean():.4f}")
        
        with col2:
            st.metric("Std Deviation", f"{matrix.std():.4f}")
        
        with col3:
            st.metric("Min Value", f"{matrix.min():.4f}")
        
        with col4:
            st.metric("Max Value", f"{matrix.max():.4f}")
    
    def educational_mode_tab(self):
        """Educational mode with guided tutorials."""
        st.markdown('<h2 class="section-header">📚 Educational Mode</h2>', 
                   unsafe_allow_html=True)
        
        # Tutorial selection
        tutorial = st.selectbox(
            "Select Tutorial",
            [
                "Introduction to Positional Encoding",
                "Understanding Sinusoidal Patterns", 
                "Rotary Position Embedding (RoPE)",
                "Attention Patterns Analysis",
                "Comparing Encoding Methods"
            ]
        )
        
        if tutorial == "Introduction to Positional Encoding":
            self.tutorial_introduction()
        elif tutorial == "Understanding Sinusoidal Patterns":
            self.tutorial_sinusoidal()
        elif tutorial == "Rotary Position Embedding (RoPE)":
            self.tutorial_rope()
        elif tutorial == "Attention Patterns Analysis":
            self.tutorial_attention()
        elif tutorial == "Comparing Encoding Methods":
            self.tutorial_comparison()
    
    def tutorial_introduction(self):
        """Introduction tutorial."""
        st.markdown("""
        ## Why Do We Need Positional Encoding?
        
        Transformer models process all tokens in parallel, but they need to understand the 
        **order** and **position** of words in a sequence. Positional encoding solves this problem.
        
        ### Key Concepts:
        
        1. **Position Information**: Each position gets a unique encoding
        2. **Additive Property**: Position encodings are added to token embeddings
        3. **Learnable vs Fixed**: Some encodings are learned, others use mathematical functions
        """)
        
        # Interactive example
        st.markdown("### Interactive Example")
        
        example_text = st.text_input("Enter text:", "Hello world")
        tokens = example_text.split()
        
        if tokens:
            # Simple visualization
            positions = list(range(len(tokens)))
            
            # Create simple encoding visualization
            fig = go.Figure()
            
            # Show token positions
            fig.add_trace(go.Scatter(
                x=positions,
                y=[0] * len(tokens),
                mode='markers+text',
                text=tokens,
                textposition="top center",
                marker=dict(size=20, color='blue'),
                name="Tokens"
            ))
            
            fig.update_layout(
                title="Token Positions in Sequence",
                xaxis_title="Position",
                yaxis=dict(visible=False),
                height=200,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **What you see:**
            - {len(tokens)} tokens at positions {positions}
            - Each position needs a unique encoding
            - Position encoding will be added to token embeddings
            """)
    
    def tutorial_sinusoidal(self):
        """Sinusoidal encoding tutorial."""
        st.markdown("""
        ## Sinusoidal Positional Encoding
        
        The original Transformer paper uses sinusoidal functions to create position encodings:
        
        - **Even dimensions**: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        - **Odd dimensions**: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        ### Key Properties:
        1. **Deterministic**: Same encoding for same position
        2. **Extrapolation**: Can handle longer sequences than training
        3. **Relative Position**: Model can learn relative positions
        """)
        
        # Interactive sinusoidal demo
        st.markdown("### Interactive Demo")
        
        demo_seq_len = st.slider("Sequence Length", 5, 50, 20, key="sin_demo_len")
        demo_d_model = st.selectbox("Model Dimension", [16, 32, 64], index=1, key="sin_demo_dim")
        
        # Create sinusoidal encoding
        config = ModelConfig(d_model=demo_d_model, encoding_type="sinusoidal")
        encoding = get_positional_encoding(config)
        enc_output = encoding.forward(demo_seq_len, demo_d_model)
        matrix = enc_output.squeeze(0).detach().cpu().numpy()
        
        # Show pattern for specific dimensions
        selected_dims = st.multiselect(
            "Select dimensions to visualize",
            list(range(0, demo_d_model, 2)),
            default=list(range(0, min(demo_d_model, 8), 2))
        )
        
        if selected_dims:
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            for i, dim in enumerate(selected_dims):
                fig.add_trace(go.Scatter(
                    x=list(range(demo_seq_len)),
                    y=matrix[:, dim],
                    mode='lines+markers',
                    name=f'Dim {dim}',
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
            
            fig.update_layout(
                title="Sinusoidal Patterns by Dimension",
                xaxis_title="Position",
                yaxis_title="Encoding Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Frequency analysis
        if hasattr(encoding, 'analyze_frequency_components'):
            freq_analysis = encoding.analyze_frequency_components()
            
            st.markdown("### Frequency Analysis")
            st.markdown(f"""
            - **Frequency Range**: {freq_analysis['min_frequency']:.6f} to {freq_analysis['max_frequency']:.6f}
            - **Wavelength Range**: {freq_analysis['wavelengths'].min():.2f} to {freq_analysis['wavelengths'].max():.2f}
            - **Frequency Ratio**: {freq_analysis['frequency_ratio']:.2f}
            """)
    
    def tutorial_rope(self):
        """RoPE tutorial."""
        st.markdown("""
        ## Rotary Position Embedding (RoPE)
        
        RoPE applies rotations to query and key vectors based on their positions:
        
        ### How it works:
        1. **2D Rotations**: Pairs of dimensions are rotated in 2D space
        2. **Position-dependent**: Rotation angle depends on position
        3. **Relative**: Attention between positions depends on their relative distance
        
        ### Mathematical Foundation:
        - Rotation matrix: [[cos θ, -sin θ], [sin θ, cos θ]]
        - θ = pos / 10000^(2i/d_model)
        """)
        
        # Interactive RoPE demo
        st.markdown("### Interactive RoPE Visualization")
        
        rope_pos = st.slider("Position", 0, 31, 0, key="rope_pos")
        rope_dim_pair = st.selectbox("Dimension Pair", [0, 1, 2, 3], key="rope_dim")
        
        # Create RoPE encoding
        config = ModelConfig(d_model=64, encoding_type="rope")
        encoding = get_positional_encoding(config)
        
        # Get rotation data for visualization
        rope_data = encoding.forward(32, 64)
        cos_values = rope_data['cos'].squeeze(0)
        sin_values = rope_data['sin'].squeeze(0)
        
        # 2D rotation visualization
        fig = go.Figure()
        
        # Unit circle
        theta_circle = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta_circle),
            y=np.sin(theta_circle),
            mode='lines',
            name='Unit Circle',
            line=dict(color='gray', dash='dash')
        ))
        
        # Original vector
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 0],
            mode='lines+markers',
            name='Original Vector',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Rotated vector
        cos_val = cos_values[rope_pos, rope_dim_pair].item()
        sin_val = sin_values[rope_pos, rope_dim_pair].item()
        
        fig.add_trace(go.Scatter(
            x=[0, cos_val], y=[0, sin_val],
            mode='lines+markers',
            name=f'Rotated (pos={rope_pos})',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"RoPE Rotation - Position {rope_pos}, Dimension Pair {rope_dim_pair}",
            xaxis=dict(range=[-1.5, 1.5], scaleanchor="y"),
            yaxis=dict(range=[-1.5, 1.5]),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show rotation angle
        angle = math.atan2(sin_val, cos_val)
        st.markdown(f"""
        **Rotation Details:**
        - **Angle**: {angle:.4f} radians ({math.degrees(angle):.2f}°)
        - **Cos**: {cos_val:.4f}
        - **Sin**: {sin_val:.4f}
        """)
    
    def tutorial_attention(self):
        """Attention analysis tutorial."""
        st.markdown("""
        ## Understanding Attention Patterns
        
        Attention weights show how much each position "pays attention" to other positions:
        
        ### Key Concepts:
        1. **Query-Key Interaction**: Attention = softmax(Q·K^T / √d_k)
        2. **Position Influence**: Positional encoding affects Q and K
        3. **Pattern Types**: Local, global, sparse, etc.
        """)
        
        # Simple attention demo
        demo_text = st.text_input("Demo text:", "The cat sat on the mat", key="attn_demo")
        tokens = demo_text.split()[:8]  # Limit for demo
        
        if len(tokens) >= 2:
            # Create simple model for demo
            config = ModelConfig(d_model=64, n_heads=2, encoding_type="sinusoidal")
            model = self.get_or_create_model(config)
            
            with torch.no_grad():
                input_ids = torch.tensor([list(range(len(tokens)))])
                outputs = model.forward(input_ids, store_visualizations=True)
                attention_weights = outputs['attention_weights']
            
            if attention_weights:
                attn_matrix = attention_weights[0][0, 0].detach().cpu().numpy()
                
                fig = go.Figure(data=go.Heatmap(
                    z=attn_matrix,
                    x=tokens,
                    y=tokens,
                    colorscale='Viridis',
                    hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Attention Pattern (Layer 0, Head 0)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analysis
                st.markdown("### Pattern Analysis")
                
                # Find highest attention for each query
                for i, query_token in enumerate(tokens):
                    attention_row = attn_matrix[i]
                    max_idx = np.argmax(attention_row)
                    max_attention = attention_row[max_idx]
                    
                    st.markdown(f"**{query_token}** pays most attention to **{tokens[max_idx]}** ({max_attention:.3f})")
    
    def tutorial_comparison(self):
        """Encoding comparison tutorial."""
        st.markdown("""
        ## Comparing Encoding Methods
        
        Different positional encodings have different properties:
        
        | Method | Extrapolation | Learnable | Relative | Efficiency |
        |--------|---------------|-----------|----------|------------|
        | Sinusoidal | ✅ | ❌ | ⚠️ | ✅ |
        | Learned | ❌ | ✅ | ❌ | ✅ |
        | RoPE | ✅ | ❌ | ✅ | ✅ |
        | Relative | ❌ | ✅ | ✅ | ⚠️ |
        """)
        
        # Interactive comparison
        st.markdown("### Interactive Comparison")
        
        compare_methods = st.multiselect(
            "Select methods to compare",
            ["sinusoidal", "rope"],
            default=["sinusoidal", "rope"]
        )
        
        comparison_seq_len = st.slider("Sequence Length", 5, 32, 16, key="comp_seq_len")
        
        if len(compare_methods) >= 2:
            fig = make_subplots(
                rows=1, cols=len(compare_methods),
                subplot_titles=[method.title() for method in compare_methods]
            )
            
            for col, method in enumerate(compare_methods, 1):
                config = ModelConfig(d_model=64, encoding_type=method)
                encoding = get_positional_encoding(config)
                
                if method in ['sinusoidal']:
                    enc_output = encoding.forward(comparison_seq_len, 64)
                    matrix = enc_output.squeeze(0).detach().cpu().numpy()
                elif method == 'rope':
                    rope_data = encoding.forward(comparison_seq_len, 64)
                    matrix = rope_data['cos'].squeeze(0).detach().cpu().numpy()
                
                fig.add_trace(
                    go.Heatmap(z=matrix.T, colorscale='Viridis', showscale=(col==1)),
                    row=1, col=col
                )
            
            fig.update_layout(title="Encoding Comparison", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def get_or_create_model(self, config=None):
        """Get or create model from cache."""
        if config is None:
            config = st.session_state.model_config
        
        config_key = f"{config.d_model}_{config.n_heads}_{config.n_layers}_{config.encoding_type}"
        
        if config_key not in st.session_state.model_cache:
            st.session_state.model_cache[config_key] = TransformerEncoder(config)
        
        return st.session_state.model_cache[config_key]


class DashboardManager:
    """Manager class for running the dashboard."""
    
    @staticmethod
    def run():
        """Run the interactive dashboard."""
        dashboard = InteractiveDashboard()
        dashboard.run()


# Entry point for Streamlit
if __name__ == "__main__":
    DashboardManager.run()
