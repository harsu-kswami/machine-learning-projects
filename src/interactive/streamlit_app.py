"""Streamlit application for interactive positional encoding exploration."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import io
import base64

from config import ModelConfig, VisualizationConfig
from src.models import TransformerEncoder
from src.positional_encoding import get_positional_encoding
from src.utils.tokenizer import SimpleTokenizer
from src.utils.metrics import AttentionMetrics, EncodingMetrics
from src.utils.export_utils import FigureExporter, generate_analysis_report
from src.visualization import AttentionVisualizer, EncodingPlotter


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_css()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="🧠 Positional Encoding Visualizer",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/positional-encoding-visualizer',
                'Report a bug': "https://github.com/your-repo/positional-encoding-visualizer/issues",
                'About': "Interactive tool for understanding transformer positional encodings"
            }
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Model configuration
        if 'model_config' not in st.session_state:
            st.session_state.model_config = ModelConfig()
        
        # Visualization configuration
        if 'viz_config' not in st.session_state:
            st.session_state.viz_config = VisualizationConfig()
        
        # Current input text
        if 'input_text' not in st.session_state:
            st.session_state.input_text = "The quick brown fox jumps over the lazy dog."
        
        # Model cache
        if 'model_cache' not in st.session_state:
            st.session_state.model_cache = {}
        
        # Analysis results cache
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        
        # Export settings
        if 'export_settings' not in st.session_state:
            st.session_state.export_settings = {
                'format': 'png',
                'dpi': 300,
                'transparent': False
            }
    
    def setup_css(self):
        """Setup custom CSS styling."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 700;
        }
        
        .section-header {
            font-size: 1.8rem;
            color: #1f2937;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem 0;
        }
        
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .info-box {
            background-color: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .warning-box {
            background-color: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .success-box {
            background-color: #dcfce7;
            border: 1px solid #16a34a;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .sidebar .stSelectbox > div > div {
            background-color: #f8fafc;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application execution."""
        # Header
        st.markdown('<h1 class="main-header">🧠 Positional Encoding Visualizer</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Home",
            "📊 Encoding Analysis", 
            "🔍 Attention Patterns",
            "⚖️ Method Comparison",
            "🎯 Interactive Lab",
            "📄 Export & Reports"
        ])
        
        with tab1:
            self.home_tab()
        
        with tab2:
            self.encoding_analysis_tab()
        
        with tab3:
            self.attention_patterns_tab()
        
        with tab4:
            self.method_comparison_tab()
        
        with tab5:
            self.interactive_lab_tab()
        
        with tab6:
            self.export_reports_tab()
    
    def create_sidebar(self):
        """Create application sidebar with controls."""
        st.sidebar.markdown("## ⚙️ Configuration")
        
        # Model Configuration
        with st.sidebar.expander("🤖 Model Settings", expanded=True):
            st.session_state.model_config.d_model = st.selectbox(
                "Model Dimension",
                [64, 128, 256, 512, 1024],
                index=2,
                help="Dimension of the model embeddings"
            )
            
            st.session_state.model_config.n_heads = st.selectbox(
                "Number of Attention Heads",
                [1, 2, 4, 8, 12, 16],
                index=2,
                help="Number of attention heads in multi-head attention"
            )
            
            st.session_state.model_config.n_layers = st.slider(
                "Number of Layers",
                1, 12, 4,
                help="Number of transformer encoder layers"
            )
            
            st.session_state.model_config.encoding_type = st.selectbox(
                "Positional Encoding Type",
                ["sinusoidal", "learned", "rope", "relative"],
                help="Type of positional encoding to use"
            )
        
        # Sequence Configuration
        with st.sidebar.expander("📝 Sequence Settings", expanded=True):
            max_seq_len = st.slider(
                "Maximum Sequence Length",
                16, 1024, 256,
                step=16,
                help="Maximum sequence length for the model"
            )
            st.session_state.model_config.max_seq_len = max_seq_len
            
            st.session_state.input_text = st.text_area(
                "Input Text",
                st.session_state.input_text,
                height=100,
                help="Text to analyze (will be tokenized)"
            )
        
        # Visualization Settings
        with st.sidebar.expander("🎨 Visualization Settings"):
            theme = st.selectbox(
                "Color Theme",
                ["default", "dark", "academic", "colorblind_friendly"],
                help="Visual theme for plots"
            )
            
            if theme != "default":
                from config import get_viz_config
                st.session_state.viz_config = get_viz_config(theme)
            
            show_values = st.checkbox(
                "Show Numerical Values",
                value=True,
                help="Display numerical values in visualizations"
            )
            
            max_display_tokens = st.slider(
                "Max Tokens to Display",
                5, 100, 32,
                help="Maximum number of tokens to show in visualizations"
            )
        
        # Advanced Settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            if st.session_state.model_config.encoding_type == "rope":
                st.session_state.model_config.rope_theta = st.number_input(
                    "RoPE Base (θ)",
                    min_value=1000.0,
                    max_value=100000.0,
                    value=10000.0,
                    step=1000.0,
                    help="Base frequency for RoPE encoding"
                )
            
            st.session_state.model_config.dropout = st.slider(
                "Dropout Rate",
                0.0, 0.5, 0.1,
                step=0.05,
                help="Dropout rate for training (affects model behavior)"
            )
            
            device = st.selectbox(
                "Device",
                ["auto", "cpu", "cuda"],
                help="Computation device"
            )
            
            if device != "auto":
                st.session_state.model_config.device = device
        
        # Quick Actions
        st.sidebar.markdown("## 🚀 Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Reset Config"):
                st.session_state.model_config = ModelConfig()
                st.session_state.viz_config = VisualizationConfig()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache"):
                st.session_state.model_cache = {}
                st.session_state.analysis_cache = {}
                st.success("Cache cleared!")
        
        # Model Info
        st.sidebar.markdown("## ℹ️ Model Info")
        model_info = self._get_model_info()
        for key, value in model_info.items():
            st.sidebar.metric(key.replace('_', ' ').title(), value)
    
    def home_tab(self):
        """Home tab with introduction and overview."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Welcome to the Positional Encoding Visualizer! 🎉
            
            This interactive tool helps you understand how different positional encoding methods work 
            in transformer models and how they affect attention patterns.
            
            ### 🎯 What You Can Explore:
            
            1. **📊 Encoding Analysis**: Visualize how different positional encodings represent position information
            2. **🔍 Attention Patterns**: See how positional encodings affect attention weights
            3. **⚖️ Method Comparison**: Compare multiple encoding methods side-by-side  
            4. **🎯 Interactive Lab**: Real-time parameter exploration
            5. **📄 Export & Reports**: Generate comprehensive analysis reports
            
            ### 🚀 Getting Started:
            
            1. Choose your model settings in the sidebar
            2. Select a positional encoding method
            3. Explore the different tabs to understand how encodings work
            4. Try the Interactive Lab for real-time experimentation!
            """)
            
            # Quick tutorial
            with st.expander("📚 Quick Tutorial"):
                st.markdown("""
                **Step 1**: Configure your model in the sidebar
                - Choose model dimension (start with 256)
                - Select number of attention heads (try 8)
                - Pick an encoding type (start with 'sinusoidal')
                
                **Step 2**: Go to "Encoding Analysis" tab
                - See how your chosen encoding represents positions
                - Understand the mathematical patterns
                
                **Step 3**: Check "Attention Patterns" tab  
                - See how attention weights change with your encoding
                - Analyze attention head behaviors
                
                **Step 4**: Use "Method Comparison" tab
                - Compare different encoding methods
                - Understand trade-offs between approaches
                
                **Step 5**: Experiment in "Interactive Lab"
                - Change parameters in real-time
                - See immediate effects on patterns
                """)
        
        with col2:
            st.markdown("### 📈 Current Configuration")
            
            config_display = {
                "Model Dimension": st.session_state.model_config.d_model,
                "Attention Heads": st.session_state.model_config.n_heads,
                "Layers": st.session_state.model_config.n_layers,
                "Encoding": st.session_state.model_config.encoding_type.title(),
                "Max Sequence Length": st.session_state.model_config.max_seq_len
            }
            
            for key, value in config_display.items():
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{key}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sample text preview
            st.markdown("### 📝 Sample Text")
            tokenizer = SimpleTokenizer()
            tokens = tokenizer.tokenize(st.session_state.input_text)
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Tokens ({len(tokens)}):</strong><br>
                {' • '.join(tokens[:10])}
                {' • ...' if len(tokens) > 10 else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Educational content
            st.markdown("### 🧠 Why Positional Encoding?")
            st.markdown("""
            Transformers process all tokens in parallel, but they need to understand 
            the **order** of words. Positional encoding solves this by adding position 
            information to word embeddings.
            
            Different methods have different properties:
            - 🌊 **Sinusoidal**: Mathematical patterns, good extrapolation
            - 🎓 **Learned**: Adaptive to data, limited length  
            - 🔄 **RoPE**: Relative positions, excellent extrapolation
            - 📏 **Relative**: Direct relative position modeling
            """)
    
    def encoding_analysis_tab(self):
        """Encoding analysis tab."""
        st.markdown('<h2 class="section-header">📊 Encoding Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Get current encoding
        encoding = get_positional_encoding(st.session_state.model_config)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 🎛️ Analysis Controls")
            
            analysis_seq_len = st.slider(
                "Sequence Length for Analysis",
                8, 128, 64,
                key="analysis_seq_len"
            )
            
            visualization_type = st.selectbox(
                "Visualization Type",
                ["Heatmap", "Line Plot", "3D Surface", "Frequency Analysis"],
                key="analysis_viz_type"
            )
            
            if st.session_state.model_config.encoding_type in ["sinusoidal", "rope"]:
                show_frequency_analysis = st.checkbox(
                    "Show Frequency Analysis",
                    value=True,
                    key="show_freq_analysis"
                )
            
            # Analysis metrics
            st.markdown("### 📊 Encoding Metrics")
            
            if st.button("🔍 Analyze Encoding"):
                with st.spinner("Analyzing encoding..."):
                    metrics = self._compute_encoding_metrics(encoding, analysis_seq_len)
                    
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                            )
        
        with col1:
            st.markdown("### 📈 Encoding Visualization")
            
            # Generate visualization based on selected type
            fig = self._create_encoding_visualization(
                encoding, analysis_seq_len, visualization_type
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional analysis
            if st.session_state.model_config.encoding_type == "sinusoidal" and hasattr(encoding, 'analyze_frequency_components'):
                st.markdown("### 🌊 Frequency Components")
                freq_analysis = encoding.analyze_frequency_components()
                
                freq_df = pd.DataFrame({
                    'Dimension': range(len(freq_analysis['frequencies'])),
                    'Frequency': freq_analysis['frequencies'].cpu().numpy(),
                    'Wavelength': freq_analysis['wavelengths'].cpu().numpy()
                })
                
                fig_freq = px.line(freq_df, x='Dimension', y='Frequency', 
                                  title='Frequency Components by Dimension',
                                  log_y=True)
                st.plotly_chart(fig_freq, use_container_width=True)
        
        # Pattern Analysis
        st.markdown("### 🔍 Pattern Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Position Similarities")
            if st.button("Compute Similarities"):
                similarities = self._compute_position_similarities(encoding, analysis_seq_len)
                
                fig_sim = go.Figure(data=go.Heatmap(
                    z=similarities.cpu().numpy(),
                    colorscale='RdBu',
                    zmid=0
                ))
                fig_sim.update_layout(
                    title="Position Similarity Matrix",
                    xaxis_title="Position",
                    yaxis_title="Position"
                )
                st.plotly_chart(fig_sim, use_container_width=True)
        
        with col2:
            st.markdown("#### Sequence Length Effects")
            if st.button("Analyze Length Effects"):
                length_effects = self._analyze_sequence_length_effects(encoding)
                
                # Create visualization of length effects
                fig_length = self._create_length_effects_plot(length_effects)
                st.plotly_chart(fig_length, use_container_width=True)
        
        with col3:
            st.markdown("#### Extrapolation Test")
            if st.button("Test Extrapolation"):
                if st.session_state.model_config.encoding_type in ["sinusoidal", "rope"]:
                    extrap_results = self._test_extrapolation(encoding)
                    
                    extrap_df = pd.DataFrame(extrap_results)
                    fig_extrap = px.line(extrap_df, x='sequence_length', y='quality_score',
                                        title='Extrapolation Quality')
                    st.plotly_chart(fig_extrap, use_container_width=True)
                else:
                    st.warning("Extrapolation testing only supported for sinusoidal and RoPE encodings")
    
    def attention_patterns_tab(self):
        """Attention patterns analysis tab."""
        st.markdown('<h2 class="section-header">🔍 Attention Patterns</h2>', 
                   unsafe_allow_html=True)
        
        # Tokenize input
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(st.session_state.input_text)
        
        if len(tokens) > st.session_state.model_config.max_seq_len:
            tokens = tokens[:st.session_state.model_config.max_seq_len]
            st.warning(f"Text truncated to {st.session_state.model_config.max_seq_len} tokens")
        
        # Get or create model
        model = self._get_or_create_model()
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Analysis Controls")
            
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
            show_attention_values = st.checkbox("Show Attention Values", value=False)
            
            attention_threshold = st.slider(
                "Attention Threshold",
                0.0, 1.0, 0.1,
                step=0.05,
                help="Minimum attention weight to highlight"
            )
            
            # Compute attention
            if st.button("🔍 Analyze Attention"):
                with st.spinner("Computing attention patterns..."):
                    attention_data = self._compute_attention_patterns(model, tokens)
                    st.session_state.attention_data = attention_data
        
        with col1:
            st.markdown("### 🎯 Attention Visualization")
            
            if hasattr(st.session_state, 'attention_data'):
                attention_weights = st.session_state.attention_data['attention_weights']
                
                if len(attention_weights) > layer_idx:
                    # Main attention heatmap
                    fig_attn = self._create_attention_heatmap(
                        attention_weights[layer_idx], tokens, head_idx, 
                        show_token_labels, show_attention_values, attention_threshold
                    )
                    st.plotly_chart(fig_attn, use_container_width=True)
                    
                    # Multi-head comparison
                    st.markdown("### 👥 Multi-Head Comparison")
                    
                    query_pos = st.selectbox(
                        "Query Position",
                        range(len(tokens)),
                        format_func=lambda x: f"{tokens[x]} (pos {x})"
                    )
                    
                    fig_multihead = self._create_multihead_comparison(
                        attention_weights[layer_idx], tokens, query_pos
                    )
                    st.plotly_chart(fig_multihead, use_container_width=True)
            else:
                st.info("Click 'Analyze Attention' to generate attention patterns")
        
        # Advanced Analysis
        if hasattr(st.session_state, 'attention_data'):
            st.markdown("### 🔬 Advanced Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Head Specialization", "Layer Evolution", "Pattern Statistics"])
            
            with tab1:
                head_analysis = self._analyze_head_specialization(st.session_state.attention_data)
                
                # Check if we have valid data
                if 'error' in head_analysis or not head_analysis.get('entropies'):
                    st.warning("No attention data available for head specialization analysis.")
                    st.info("Please generate attention patterns first in the main attention tab.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Attention Entropy by Head")
                        entropy_df = pd.DataFrame({
                            'Head': range(len(head_analysis['entropies'])),
                            'Entropy': head_analysis['entropies']
                        })
                        fig_entropy = px.bar(entropy_df, x='Head', y='Entropy')
                        st.plotly_chart(fig_entropy, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Head Similarity Matrix")
                        if head_analysis.get('similarities') is not None and head_analysis['similarities'].numel() > 0:
                            fig_head_sim = go.Figure(data=go.Heatmap(
                                z=head_analysis['similarities'].cpu().numpy(),
                                colorscale='RdBu',
                                zmid=0
                            ))
                            fig_head_sim.update_layout(title="Head Similarity Matrix")
                            st.plotly_chart(fig_head_sim, use_container_width=True)
                        else:
                            st.info("No similarity data available")
            
            with tab2:
                st.markdown("#### Attention Evolution Across Layers")
                evolution_fig = self._create_layer_evolution_plot(st.session_state.attention_data)
                st.plotly_chart(evolution_fig, use_container_width=True)
            
            with tab3:
                stats = self._compute_attention_statistics(st.session_state.attention_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Attention", f"{stats['mean_attention']:.4f}")
                    st.metric("Attention Sparsity", f"{stats['sparsity']:.4f}")
                
                with col2:
                    st.metric("Max Attention", f"{stats['max_attention']:.4f}")
                    st.metric("Effective Span", f"{stats['effective_span']:.1f}")
                
                with col3:
                    st.metric("Entropy", f"{stats['entropy']:.4f}")
                    st.metric("Distance", f"{stats['distance']:.2f}")
    
    def method_comparison_tab(self):
        """Method comparison tab."""
        st.markdown('<h2 class="section-header">⚖️ Method Comparison</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("Compare different positional encoding methods side by side.")
        
        # Method selection
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 🎯 Comparison Settings")
            
            available_methods = ["sinusoidal", "learned", "rope", "relative"]
            selected_methods = st.multiselect(
                "Select Methods to Compare",
                available_methods,
                default=["sinusoidal", "rope"],
                help="Choose 2-4 methods for comparison"
            )
            
            comparison_seq_len = st.slider(
                "Sequence Length",
                8, 128, 64,
                key="comparison_seq_len"
            )
            
            comparison_type = st.selectbox(
                "Comparison Type",
                ["Encoding Patterns", "Position Similarities", "Attention Effects", "Performance Metrics"]
            )
            
            if st.button("🔍 Run Comparison"):
                if len(selected_methods) < 2:
                    st.error("Please select at least 2 methods for comparison")
                else:
                    with st.spinner("Running comparison..."):
                        comparison_results = self._run_method_comparison(
                            selected_methods, comparison_seq_len, comparison_type
                        )
                        st.session_state.comparison_results = comparison_results
        
        with col1:
            st.markdown("### 📊 Comparison Results")
            
            if hasattr(st.session_state, 'comparison_results'):
                results = st.session_state.comparison_results
                
                if comparison_type == "Encoding Patterns":
                    fig = self._create_encoding_comparison_plot(results)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif comparison_type == "Position Similarities":
                    fig = self._create_similarity_comparison_plot(results)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif comparison_type == "Attention Effects":
                    fig = self._create_attention_comparison_plot(results)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif comparison_type == "Performance Metrics":
                    self._display_performance_comparison(results)
            else:
                st.info("Select methods and click 'Run Comparison' to see results")
        
        # Detailed comparison table
        if hasattr(st.session_state, 'comparison_results'):
            st.markdown("### 📋 Detailed Comparison")
            
            comparison_df = self._create_comparison_table(st.session_state.comparison_results)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download comparison results
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison CSV",
                data=csv,
                file_name="positional_encoding_comparison.csv",
                mime="text/csv"
            )
    
    def interactive_lab_tab(self):
        """Interactive lab for real-time experimentation."""
        st.markdown('<h2 class="section-header">🎯 Interactive Lab</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("Experiment with parameters in real-time and see immediate effects!")
        
        # Real-time parameter controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Live Controls")
            
            # Live parameter adjustment
            live_seq_len = st.slider(
                "Sequence Length",
                8, 64, 32,
                key="live_seq_len"
            )
            
            live_d_model = st.selectbox(
                "Model Dimension",
                [64, 128, 256, 512],
                index=1,
                key="live_d_model"
            )
            
            live_encoding_type = st.selectbox(
                "Encoding Type",
                ["sinusoidal", "rope"],
                key="live_encoding_type"
            )
            
            # Encoding-specific parameters
            if live_encoding_type == "rope":
                live_rope_theta = st.slider(
                    "RoPE Base (θ)",
                    1000.0, 50000.0, 10000.0,
                    step=1000.0,
                    key="live_rope_theta"
                )
            
            live_viz_type = st.selectbox(
                "Visualization",
                ["Heatmap", "3D Surface", "Line Plot"],
                key="live_viz_type"
            )
            
            # Live update toggle
            auto_update = st.checkbox("🔄 Auto Update", value=True, key="auto_update")
            
            if not auto_update:
                update_button = st.button("🔄 Update Visualization")
        
        with col2:
            st.markdown("### 📊 Live Visualization")
            
            # Create live encoding configuration
            live_config = ModelConfig(
                d_model=live_d_model,
                encoding_type=live_encoding_type,
                max_seq_len=live_seq_len * 2
            )
            
            if live_encoding_type == "rope":
                live_config.rope_theta = live_rope_theta
            
            # Generate live visualization
            if auto_update or (not auto_update and 'update_button' in locals() and update_button):
                encoding = get_positional_encoding(live_config)
                
                fig = self._create_encoding_visualization(
                    encoding, live_seq_len, live_viz_type
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="live_plot")
        
        # Real-time metrics
        st.markdown("### 📈 Live Metrics")
        
        if auto_update or (not auto_update and 'update_button' in locals() and update_button):
            metrics = self._compute_encoding_metrics(encoding, live_seq_len)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Encoding Variance", f"{metrics.get('encoding_variance', 0.0):.4f}")
            
            with col2:
                st.metric("Position Distinguishability", f"{metrics.get('distinguishability', 0.0):.4f}")
            
            with col3:
                st.metric("Dimension Utilization", f"{metrics.get('dimension_utilization', 0.0):.4f}")
            
            with col4:
                st.metric("Periodicity Score", f"{metrics.get('periodicity_score', 0.0):.4f}")
        
        # Parameter exploration
        st.markdown("### 🔬 Parameter Exploration")
        
        exploration_type = st.selectbox(
            "Exploration Type",
            ["Sequence Length Effects", "Dimension Effects", "Frequency Analysis"]
        )
        
        if st.button("🚀 Start Exploration"):
            with st.spinner("Running parameter exploration..."):
                exploration_results = self._run_parameter_exploration(
                    live_config, exploration_type
                )
                
                fig_exploration = self._create_exploration_plot(exploration_results, exploration_type)
                st.plotly_chart(fig_exploration, use_container_width=True)
    
    def export_reports_tab(self):
        """Export and reports tab."""
        st.markdown('<h2 class="section-header">📄 Export & Reports</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("Generate comprehensive reports and export your analysis results.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Export Settings")
            
            export_format = st.selectbox(
                "Export Format",
                ["PNG", "PDF", "SVG", "HTML"],
                key="export_format"
            )
            
            export_dpi = st.slider(
                "Image Quality (DPI)",
                100, 600, 300,
                step=50,
                key="export_dpi"
            )
            
            include_data = st.checkbox(
                "Include Raw Data",
                value=True,
                help="Include analysis data in export"
            )
            
            include_config = st.checkbox(
                "Include Configuration",
                value=True,
                help="Include model configuration in export"
            )
            
            report_sections = st.multiselect(
                "Report Sections",
                ["Summary", "Encoding Analysis", "Attention Patterns", "Method Comparison", "Performance Metrics"],
                default=["Summary", "Encoding Analysis"],
                help="Select sections to include in the report"
            )
        
        with col1:
            st.markdown("### 📊 Report Generation")
            
            # Report preview
            st.markdown("#### 📋 Report Preview")
            
            if st.button("🔍 Generate Preview"):
                preview_content = self._generate_report_preview(report_sections)
                st.markdown(preview_content, unsafe_allow_html=True)
            
            # Full report generation
            st.markdown("#### 📄 Full Report")
            
            if st.button("📄 Generate Complete Report"):
                with st.spinner("Generating comprehensive report..."):
                    report_data = self._generate_complete_report(
                        report_sections, include_data, include_config
                    )
                    
                    # Display download link
                    report_html = self._create_html_report(report_data)
                    
                    st.success("Report generated successfully!")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=report_html,
                        file_name="positional_encoding_analysis_report.html",
                        mime="text/html"
                    )
        
        # Export individual components
        st.markdown("### 🎯 Individual Exports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Current Visualization")
            if st.button("Export Current Plot"):
                # Export the current visualization
                st.info("Exporting current visualization...")
        
        with col2:
            st.markdown("#### 📈 Analysis Data")
            if st.button("Export Analysis Data"):
                # Export analysis data as CSV/JSON
                analysis_data = self._collect_analysis_data()
                
                if analysis_data:
                    csv_data = pd.DataFrame(analysis_data).to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        "analysis_data.csv",
                        "text/csv"
                    )
        
        with col3:
            st.markdown("#### ⚙️ Configuration")
            if st.button("Export Configuration"):
                config_data = {
                    'model_config': st.session_state.model_config.__dict__,
                    'viz_config': st.session_state.viz_config.__dict__
                }
                
                config_json = json.dumps(config_data, indent=2)
                st.download_button(
                    "📥 Download Config",
                    config_json,
                    "model_config.json",
                    "application/json"
                )
        
        # Batch export
        st.markdown("### 🔄 Batch Export")
        
        if st.button("📦 Generate Analysis Package"):
            with st.spinner("Creating analysis package..."):
                package_data = self._create_analysis_package()
                
                st.success("Analysis package created!")
                st.info("Package includes: visualizations, data, configuration, and detailed report")
    
    # Helper methods
    def _get_or_create_model(self):
        """Get or create model from cache."""
        config_key = self._get_config_key()
        
        if config_key not in st.session_state.model_cache:
            with st.spinner("Creating model..."):
                model = TransformerEncoder(st.session_state.model_config)
                st.session_state.model_cache[config_key] = model
        
        return st.session_state.model_cache[config_key]
    
    def _get_config_key(self):
        """Generate cache key from current configuration."""
        config = st.session_state.model_config
        return f"{config.d_model}_{config.n_heads}_{config.n_layers}_{config.encoding_type}"
    
    def _get_model_info(self):
        """Get model information for display."""
        config = st.session_state.model_config
        
        # Calculate approximate parameters
        d_model = config.d_model
        n_heads = config.n_heads
        n_layers = config.n_layers
        
        # Rough parameter estimation
        params_per_layer = (
            4 * d_model * d_model +  # QKV + output projections
            2 * d_model * config.d_ff +  # FFN
            2 * d_model  # Layer norms
        )
        
        total_params = n_layers * params_per_layer + config.vocab_size * d_model
        
        return {
            "Total Parameters": f"{total_params:,}",
            "Memory (est.)": f"{total_params * 4 / (1024*1024):.1f} MB",
            "Sequence Length": config.max_seq_len,
            "Attention Heads": config.n_heads
        }
    
    def _create_encoding_visualization(self, encoding, seq_len, viz_type):
        """Create encoding visualization based on type."""
        try:
            if viz_type == "Heatmap":
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    if isinstance(enc_output, dict):
                        matrix = enc_output.get('encoding', enc_output.get('output', None))
                        if matrix is not None:
                            matrix = matrix.squeeze(0).detach().cpu().numpy()
                        else:
                            matrix = np.zeros((1, 1))
                    else:
                        matrix = enc_output.squeeze(0).detach().cpu().numpy()
                elif hasattr(encoding, '__call__'):
                    matrix = encoding(seq_len).detach().cpu().numpy()
                else:
                    return None
                
                fig = go.Figure(data=go.Heatmap(
                    z=matrix.T,
                    colorscale='Viridis',
                    hovertemplate='Position: %{x}<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{st.session_state.model_config.encoding_type.title()} Encoding Heatmap",
                    xaxis_title="Position",
                    yaxis_title="Dimension",
                    height=500
                )
                
                return fig
            
            elif viz_type == "3D Surface":
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    if isinstance(enc_output, dict):
                        matrix = enc_output.get('encoding', enc_output.get('output', None))
                        if matrix is not None:
                            matrix = matrix.squeeze(0).detach().cpu().numpy()
                        else:
                            matrix = np.zeros((1, 1))
                    else:
                        matrix = enc_output.squeeze(0).detach().cpu().numpy()
                else:
                    return None
                
                # Limit dimensions for 3D visualization
                max_dims = min(32, matrix.shape[1])
                matrix_subset = matrix[:, :max_dims]
                
                x = np.arange(seq_len)
                y = np.arange(max_dims)
                X, Y = np.meshgrid(x, y)
                
                fig = go.Figure(data=[go.Surface(
                    x=X, y=Y, z=matrix_subset.T,
                    colorscale='Viridis'
                )])
                
                fig.update_layout(
                    title=f"{st.session_state.model_config.encoding_type.title()} 3D Surface",
                    scene=dict(
                        xaxis_title="Position",
                        yaxis_title="Dimension",
                        zaxis_title="Value"
                    ),
                    height=600
                )
                
                return fig
            
            elif viz_type == "Line Plot":
                if hasattr(encoding, 'forward'):
                    enc_output = encoding.forward(seq_len, encoding.d_model)
                    if isinstance(enc_output, dict):
                        matrix = enc_output.get('encoding', enc_output.get('output', None))
                        if matrix is not None:
                            matrix = matrix.squeeze(0).detach().cpu().numpy()
                        else:
                            matrix = np.zeros((1, 1))
                    else:
                        matrix = enc_output.squeeze(0).detach().cpu().numpy()
                else:
                    return None
                
                fig = go.Figure()
                
                # Plot first few dimensions
                for dim in range(min(8, matrix.shape[1])):
                    fig.add_trace(go.Scatter(
                        x=list(range(seq_len)),
                        y=matrix[:, dim],
                        mode='lines+markers',
                        name=f'Dim {dim}',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{st.session_state.model_config.encoding_type.title()} Line Plot",
                    xaxis_title="Position",
                    yaxis_title="Encoding Value",
                    height=500
                )
                
                return fig
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _compute_encoding_metrics(self, encoding, seq_len):
        """Compute encoding quality metrics."""
        try:
            from src.utils.metrics import EncodingMetrics
            
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, encoding.d_model)
                if isinstance(enc_output, dict):
                    matrix = enc_output.get('encoding', enc_output.get('output', None))
                    if matrix is not None:
                        matrix = matrix.squeeze(0)
                    else:
                        matrix = torch.zeros((1, 1))
                else:
                    matrix = enc_output.squeeze(0)
            else:
                return {}
            
            metrics_computer = EncodingMetrics()
            return metrics_computer.compute_encoding_quality(matrix)
            
        except Exception as e:
            st.error(f"Error computing metrics: {str(e)}")
            return {}
    
    def _compute_position_similarities(self, encoding, seq_len):
        """Compute position similarity matrix."""
        try:
            from src.utils.metrics import EncodingMetrics
            
            if hasattr(encoding, 'forward'):
                enc_output = encoding.forward(seq_len, encoding.d_model)
                if isinstance(enc_output, dict):
                    matrix = enc_output.get('encoding', enc_output.get('output', None))
                    if matrix is not None:
                        matrix = matrix.squeeze(0)
                    else:
                        matrix = torch.zeros((1, 1))
                else:
                    matrix = enc_output.squeeze(0)
            else:
                return torch.zeros(seq_len, seq_len)
            
            metrics_computer = EncodingMetrics()
            return metrics_computer.compute_position_similarity(matrix)
            
        except Exception as e:
            st.error(f"Error computing similarities: {str(e)}")
            return torch.zeros(seq_len, seq_len)
    
    def _compute_attention_patterns(self, model, tokens):
        """Compute attention patterns for the given model and tokens."""
        try:
            import torch
            from src.utils.tokenizer import SimpleTokenizer
            
            # Tokenize input if needed
            if isinstance(tokens, str):
                tokenizer = SimpleTokenizer()
                tokens = tokenizer.tokenize(tokens)
            
            # Convert tokens to input IDs
            input_ids = torch.tensor([list(range(len(tokens)))])
            
            # Run model forward pass with attention storage
            with torch.no_grad():
                outputs = model.forward(input_ids, store_visualizations=True)
            
            # Extract attention weights
            attention_weights = outputs.get('attention_weights', [])
            
            if not attention_weights:
                return {
                    'attention_weights': [],
                    'tokens': tokens,
                    'error': 'No attention weights available'
                }
            
            # Process attention weights for visualization
            processed_weights = []
            for layer_idx, layer_attention in enumerate(attention_weights):
                if layer_attention is not None:
                    # Handle different attention weight formats
                    if layer_attention.dim() == 4:  # (batch, heads, seq, seq)
                        processed_weights.append(layer_attention[0])  # Remove batch dimension
                    elif layer_attention.dim() == 3:  # (heads, seq, seq)
                        processed_weights.append(layer_attention)
                    else:
                        processed_weights.append(layer_attention)
            
            return {
                'attention_weights': processed_weights,
                'tokens': tokens,
                'num_layers': len(processed_weights),
                'num_heads': processed_weights[0].shape[0] if processed_weights else 0,
                'sequence_length': len(tokens)
            }
            
        except Exception as e:
            st.error(f"Error computing attention patterns: {str(e)}")
            return {
                'attention_weights': [],
                'tokens': tokens if 'tokens' in locals() else [],
                'error': str(e)
            }
    
    def _analyze_head_specialization(self, attention_data):
        """Analyze attention head specialization patterns."""
        try:
            import torch
            import numpy as np
            
            if 'error' in attention_data or not attention_data.get('attention_weights'):
                return {
                    'entropies': [],
                    'focuses': [],
                    'specializations': [],
                    'error': 'No attention data available'
                }
            
            attention_weights = attention_data['attention_weights']
            if not attention_weights:
                return {
                    'entropies': [],
                    'focuses': [],
                    'specializations': [],
                    'error': 'Empty attention weights'
                }
            
            # Flatten all heads across all layers for overall analysis
            all_entropies = []
            all_focuses = []
            all_specializations = []
            
            # Analyze each layer
            for layer_idx, layer_attention in enumerate(attention_weights):
                if layer_attention is None or layer_attention.numel() == 0:
                    continue
                
                # Handle different tensor shapes
                if layer_attention.dim() == 3:  # (heads, seq, seq)
                    heads_attention = layer_attention
                elif layer_attention.dim() == 4:  # (batch, heads, seq, seq)
                    heads_attention = layer_attention[0]  # Remove batch dimension
                else:
                    continue
                
                num_heads, seq_len, _ = heads_attention.shape
                
                # Compute metrics for each head
                for head_idx in range(num_heads):
                    head_attn = heads_attention[head_idx]
                    
                    # Compute entropy (higher = more uniform attention)
                    attn_probs = head_attn + 1e-8  # Add small epsilon
                    entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean()
                    all_entropies.append(entropy.item())
                    
                    # Compute focus (max attention weight)
                    max_attn = head_attn.max(dim=-1)[0].mean()
                    all_focuses.append(max_attn.item())
                    
                    # Compute specialization (variance in attention patterns)
                    attn_variance = head_attn.var(dim=-1).mean()
                    all_specializations.append(attn_variance.item())
            
            # Compute head similarities (simplified version)
            similarities = torch.eye(len(all_entropies)) if all_entropies else torch.tensor([])
            
            return {
                'entropies': all_entropies,
                'focuses': all_focuses,
                'specializations': all_specializations,
                'similarities': similarities,
                'total_heads': len(all_entropies)
            }
            
        except Exception as e:
            return {
                'entropies': [],
                'focuses': [],
                'specializations': [],
                'similarities': torch.tensor([]),
                'error': str(e)
            }
    
    def _create_layer_evolution_plot(self, attention_data):
        """Create a plot showing attention evolution across layers."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if 'error' in attention_data or not attention_data.get('attention_weights'):
                # Return empty plot
                fig = go.Figure()
                fig.add_annotation(text="No attention data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            attention_weights = attention_data['attention_weights']
            if not attention_weights:
                fig = go.Figure()
                fig.add_annotation(text="Empty attention weights", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Compute mean attention strength per layer
            layer_means = []
            for layer_idx, layer_attention in enumerate(attention_weights):
                if layer_attention is not None and layer_attention.numel() > 0:
                    if layer_attention.dim() == 3:  # (heads, seq, seq)
                        mean_attn = layer_attention.mean().item()
                    elif layer_attention.dim() == 4:  # (batch, heads, seq, seq)
                        mean_attn = layer_attention[0].mean().item()
                    else:
                        mean_attn = 0.0
                    layer_means.append(mean_attn)
                else:
                    layer_means.append(0.0)
            
            fig = go.Figure(data=go.Scatter(
                x=list(range(len(layer_means))),
                y=layer_means,
                mode='lines+markers',
                name='Mean Attention'
            ))
            
            fig.update_layout(
                title="Attention Evolution Across Layers",
                xaxis_title="Layer",
                yaxis_title="Mean Attention Strength",
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating plot: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _compute_attention_statistics(self, attention_data):
        """Compute attention statistics."""
        try:
            if 'error' in attention_data or not attention_data.get('attention_weights'):
                return {
                    'mean_attention': 0.0,
                    'std_attention': 0.0,
                    'max_attention': 0.0,
                    'min_attention': 0.0,
                    'sparsity': 0.0,
                    'effective_span': 0.0,
                    'entropy': 0.0,
                    'distance': 0.0
                }
            
            attention_weights = attention_data['attention_weights']
            if not attention_weights:
                return {
                    'mean_attention': 0.0,
                    'std_attention': 0.0,
                    'max_attention': 0.0,
                    'min_attention': 0.0,
                    'sparsity': 0.0,
                    'effective_span': 0.0,
                    'entropy': 0.0,
                    'distance': 0.0
                }
            
            # Flatten all attention weights
            all_weights = []
            for layer_attention in attention_weights:
                if layer_attention is not None and layer_attention.numel() > 0:
                    if layer_attention.dim() == 3:  # (heads, seq, seq)
                        all_weights.extend(layer_attention.flatten().tolist())
                    elif layer_attention.dim() == 4:  # (batch, heads, seq, seq)
                        all_weights.extend(layer_attention[0].flatten().tolist())
            
            if not all_weights:
                return {
                    'mean_attention': 0.0,
                    'std_attention': 0.0,
                    'max_attention': 0.0,
                    'min_attention': 0.0,
                    'sparsity': 0.0,
                    'effective_span': 0.0,
                    'entropy': 0.0,
                    'distance': 0.0
                }
            
            import numpy as np
            weights_array = np.array(all_weights)
            
            # Compute sparsity (fraction of near-zero values)
            threshold = 0.01  # Values below this are considered sparse
            sparse_count = np.sum(np.abs(weights_array) < threshold)
            sparsity = sparse_count / len(weights_array)
            
            # Compute effective span (simplified - based on attention weight distribution)
            # This is a heuristic measure of how "spread out" the attention is
            attention_std = np.std(weights_array)
            attention_mean = np.mean(weights_array)
            effective_span = min(attention_std * 2, 1.0)  # Cap at 1.0
            
            # Compute entropy (information content of attention distribution)
            # Normalize weights to probabilities
            weights_normalized = weights_array / (np.sum(weights_array) + 1e-8)
            entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-8))
            
            # Compute distance (average distance from mean)
            distance = np.mean(np.abs(weights_array - attention_mean))
            
            return {
                'mean_attention': float(np.mean(weights_array)),
                'std_attention': float(np.std(weights_array)),
                'max_attention': float(np.max(weights_array)),
                'min_attention': float(np.min(weights_array)),
                'sparsity': float(sparsity),
                'effective_span': float(effective_span),
                'entropy': float(entropy),
                'distance': float(distance)
            }
            
        except Exception as e:
            return {
                'mean_attention': 0.0,
                'std_attention': 0.0,
                'max_attention': 0.0,
                'min_attention': 0.0,
                'sparsity': 0.0,
                'effective_span': 0.0
            }
    
    def _analyze_sequence_length_effects(self, encoding):
        """Analyze how encoding quality changes with sequence length."""
        try:
            import torch
            import numpy as np
            
            if encoding is None:
                return {
                    'lengths': [],
                    'metrics': {},
                    'error': 'No encoding provided'
                }
            
            # Test different sequence lengths
            test_lengths = [10, 20, 50, 100, 200, 500]
            results = {
                'lengths': test_lengths,
                'variance': [],
                'distinguishability': [],
                'periodicity': []
            }
            
            for length in test_lengths:
                try:
                    # Generate encoding for this length
                    if hasattr(encoding, 'forward'):
                        # If it's a model, create dummy input
                        dummy_input = torch.zeros(1, length)
                        with torch.no_grad():
                            enc_output = encoding.forward(dummy_input)
                            if isinstance(enc_output, dict):
                                enc_matrix = enc_output.get('encoding', enc_output.get('output', None))
                            else:
                                enc_matrix = enc_output
                    else:
                        # If it's a matrix, truncate it
                        enc_matrix = encoding[:length, :] if encoding.size(0) >= length else encoding
                    
                    if enc_matrix is not None:
                        # Compute metrics for this length
                        variance = torch.var(enc_matrix).item()
                        results['variance'].append(variance)
                        
                        # Compute distinguishability (simplified)
                        if enc_matrix.size(0) > 1:
                            similarities = torch.mm(enc_matrix, enc_matrix.t()) / enc_matrix.size(-1)
                            off_diagonal = similarities - torch.eye(similarities.size(0))
                            distinguishability = -torch.mean(off_diagonal).item()
                        else:
                            distinguishability = 0.0
                        results['distinguishability'].append(distinguishability)
                        
                        # Compute periodicity (simplified)
                        if enc_matrix.size(0) > 2:
                            # Look for repeating patterns
                            first_half = enc_matrix[:enc_matrix.size(0)//2]
                            second_half = enc_matrix[enc_matrix.size(0)//2:]
                            if first_half.size(0) == second_half.size(0):
                                periodicity = torch.cosine_similarity(
                                    first_half.flatten(), second_half.flatten(), dim=0
                                ).item()
                            else:
                                periodicity = 0.0
                        else:
                            periodicity = 0.0
                        results['periodicity'].append(periodicity)
                    else:
                        results['variance'].append(0.0)
                        results['distinguishability'].append(0.0)
                        results['periodicity'].append(0.0)
                        
                except Exception as e:
                    results['variance'].append(0.0)
                    results['distinguishability'].append(0.0)
                    results['periodicity'].append(0.0)
            
            return results
            
        except Exception as e:
            return {
                'lengths': [],
                'metrics': {},
                'error': str(e)
            }
    
    def _run_parameter_exploration(self, base_config, exploration_type, param_ranges=None):
        """Run parameter exploration to find optimal settings."""
        try:
            import torch
            import numpy as np
            from src.models.transformer import TransformerModel
            from src.positional_encoding import (
                SinusoidalEncoding, RelativePositionalEncoding, RoPEEncoding
            )
            
            results = {
                'parameters': [],
                'metrics': [],
                'best_config': None,
                'best_score': -float('inf')
            }
            
            # Get encoding type from base_config
            encoding_type = base_config.get('encoding_type', 'sinusoidal')
            
            # Define parameter ranges if not provided
            if not param_ranges:
                if encoding_type == 'sinusoidal':
                    param_ranges = {
                        'd_model': [64, 128, 256, 512],
                        'max_length': [100, 200, 500, 1000]
                    }
                elif encoding_type == 'rope':
                    param_ranges = {
                        'd_model': [64, 128, 256, 512],
                        'max_length': [100, 200, 500, 1000],
                        'theta': [10000, 20000, 50000, 100000]
                    }
                else:  # relative
                    param_ranges = {
                        'd_model': [64, 128, 256, 512],
                        'max_length': [100, 200, 500, 1000],
                        'num_buckets': [16, 32, 64, 128]
                    }
            
            # Generate all parameter combinations
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())
            
            # Create cartesian product of parameters
            import itertools
            param_combinations = list(itertools.product(*param_values))
            
            for i, combination in enumerate(param_combinations[:20]):  # Limit to 20 combinations
                try:
                    # Create config for this combination
                    config = base_config.copy()
                    for j, param_name in enumerate(param_names):
                        config[param_name] = combination[j]
                    
                    # Create encoding with these parameters
                    if encoding_type == 'sinusoidal':
                        encoding = SinusoidalEncoding(
                            d_model=config['d_model'],
                            max_length=config['max_length']
                        )
                    elif encoding_type == 'rope':
                        encoding = RoPEEncoding(
                            d_model=config['d_model'],
                            max_length=config['max_length'],
                            theta=config.get('theta', 10000)
                        )
                    else:  # relative
                        encoding = RelativePositionalEncoding(
                            d_model=config['d_model'],
                            max_length=config['max_length'],
                            num_buckets=config.get('num_buckets', 32)
                        )
                    
                    # Generate encoding matrix
                    seq_len = min(config['max_length'], 200)  # Limit for performance
                    dummy_input = torch.zeros(1, seq_len)
                    
                    with torch.no_grad():
                        if hasattr(encoding, 'forward'):
                            enc_output = encoding.forward(dummy_input)
                            if isinstance(enc_output, dict):
                                enc_matrix = enc_output.get('encoding', enc_output.get('output', None))
                            else:
                                enc_matrix = enc_output
                        else:
                            enc_matrix = encoding(dummy_input)
                    
                    if enc_matrix is not None:
                        # Compute quality metrics
                        variance = torch.var(enc_matrix).item()
                        
                        # Compute distinguishability
                        if enc_matrix.size(0) > 1:
                            similarities = torch.mm(enc_matrix, enc_matrix.t()) / enc_matrix.size(-1)
                            off_diagonal = similarities - torch.eye(similarities.size(0))
                            distinguishability = -torch.mean(off_diagonal).item()
                        else:
                            distinguishability = 0.0
                        
                        # Compute periodicity
                        if enc_matrix.size(0) > 2:
                            first_half = enc_matrix[:enc_matrix.size(0)//2]
                            second_half = enc_matrix[enc_matrix.size(0)//2:]
                            if first_half.size(0) == second_half.size(0):
                                periodicity = torch.cosine_similarity(
                                    first_half.flatten(), second_half.flatten(), dim=0
                                ).item()
                            else:
                                periodicity = 0.0
                        else:
                            periodicity = 0.0
                        
                        # Overall quality score (weighted combination)
                        quality_score = (
                            0.4 * variance + 
                            0.4 * distinguishability + 
                            0.2 * periodicity
                        )
                        
                        results['parameters'].append(config.copy())
                        results['metrics'].append({
                            'variance': variance,
                            'distinguishability': distinguishability,
                            'periodicity': periodicity,
                            'quality_score': quality_score
                        })
                        
                        # Track best configuration
                        if quality_score > results['best_score']:
                            results['best_score'] = quality_score
                            results['best_config'] = config.copy()
                    
                except Exception as e:
                    # Skip this combination if it fails
                    continue
            
            return results
            
        except Exception as e:
            return {
                'parameters': [],
                'metrics': [],
                'best_config': None,
                'best_score': -float('inf'),
                'error': str(e)
            }
    
    def _create_length_effects_plot(self, length_effects):
        """Create a plot showing sequence length effects."""
        try:
            import plotly.graph_objects as go
            
            if 'error' in length_effects or not length_effects.get('lengths'):
                # Return empty plot
                fig = go.Figure()
                fig.add_annotation(text="No length effects data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            lengths = length_effects['lengths']
            
            fig = go.Figure()
            
            # Add variance trace
            if length_effects.get('variance'):
                fig.add_trace(go.Scatter(
                    x=lengths,
                    y=length_effects['variance'],
                    mode='lines+markers',
                    name='Variance',
                    line=dict(color='blue')
                ))
            
            # Add distinguishability trace
            if length_effects.get('distinguishability'):
                fig.add_trace(go.Scatter(
                    x=lengths,
                    y=length_effects['distinguishability'],
                    mode='lines+markers',
                    name='Distinguishability',
                    line=dict(color='red')
                ))
            
            # Add periodicity trace
            if length_effects.get('periodicity'):
                fig.add_trace(go.Scatter(
                    x=lengths,
                    y=length_effects['periodicity'],
                    mode='lines+markers',
                    name='Periodicity',
                    line=dict(color='green')
                ))
            
            fig.update_layout(
                title="Encoding Quality vs Sequence Length",
                xaxis_title="Sequence Length",
                yaxis_title="Metric Value",
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating plot: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _run_method_comparison(self, selected_methods, seq_len, comparison_config):
        """Run comparison between different encoding methods."""
        try:
            import torch
            import numpy as np
            from src.positional_encoding import (
                SinusoidalEncoding, RelativePositionalEncoding, RoPEEncoding
            )
            
            results = {
                'methods': selected_methods,
                'metrics': {},
                'encodings': {},
                'similarities': {}
            }
            
            # Generate encodings for each method
            for method in selected_methods:
                try:
                    if method == 'sinusoidal':
                        encoding = SinusoidalEncoding(
                            d_model=comparison_config.get('d_model', 256),
                            max_length=seq_len
                        )
                    elif method == 'rope':
                        encoding = RoPEEncoding(
                            d_model=comparison_config.get('d_model', 256),
                            max_length=seq_len,
                            theta=comparison_config.get('theta', 10000)
                        )
                    elif method == 'relative':
                        encoding = RelativePositionalEncoding(
                            d_model=comparison_config.get('d_model', 256),
                            max_length=seq_len,
                            num_buckets=comparison_config.get('num_buckets', 32)
                        )
                    else:
                        continue
                    
                    # Generate encoding matrix
                    dummy_input = torch.zeros(1, seq_len)
                    with torch.no_grad():
                        if hasattr(encoding, 'forward'):
                            enc_output = encoding.forward(dummy_input)
                            if isinstance(enc_output, dict):
                                enc_matrix = enc_output.get('encoding', enc_output.get('output', None))
                            else:
                                enc_matrix = enc_output
                        else:
                            enc_matrix = encoding(dummy_input)
                    
                    if enc_matrix is not None:
                        results['encodings'][method] = enc_matrix
                        
                        # Compute metrics
                        variance = torch.var(enc_matrix).item()
                        
                        # Compute distinguishability
                        if enc_matrix.size(0) > 1:
                            similarities = torch.mm(enc_matrix, enc_matrix.t()) / enc_matrix.size(-1)
                            off_diagonal = similarities - torch.eye(similarities.size(0))
                            distinguishability = -torch.mean(off_diagonal).item()
                        else:
                            distinguishability = 0.0
                        
                        # Compute periodicity
                        if enc_matrix.size(0) > 2:
                            first_half = enc_matrix[:enc_matrix.size(0)//2]
                            second_half = enc_matrix[enc_matrix.size(0)//2:]
                            if first_half.size(0) == second_half.size(0):
                                periodicity = torch.cosine_similarity(
                                    first_half.flatten(), second_half.flatten(), dim=0
                                ).item()
                            else:
                                periodicity = 0.0
                        else:
                            periodicity = 0.0
                        
                        results['metrics'][method] = {
                            'variance': variance,
                            'distinguishability': distinguishability,
                            'periodicity': periodicity,
                            'overall_score': (variance + distinguishability + periodicity) / 3
                        }
                
                except Exception as e:
                    results['metrics'][method] = {
                        'variance': 0.0,
                        'distinguishability': 0.0,
                        'periodicity': 0.0,
                        'overall_score': 0.0,
                        'error': str(e)
                    }
            
            # Compute pairwise similarities between methods
            method_names = list(results['encodings'].keys())
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names[i+1:], i+1):
                    try:
                        enc1 = results['encodings'][method1]
                        enc2 = results['encodings'][method2]
                        
                        # Compute cosine similarity
                        similarity = torch.cosine_similarity(
                            enc1.flatten(), enc2.flatten(), dim=0
                        ).item()
                        
                        results['similarities'][f"{method1}_vs_{method2}"] = similarity
                    except Exception as e:
                        results['similarities'][f"{method1}_vs_{method2}"] = 0.0
            
            return results
            
        except Exception as e:
            return {
                'methods': selected_methods,
                'metrics': {},
                'encodings': {},
                'similarities': {},
                'error': str(e)
            }
    
    def _test_extrapolation(self, encoding):
        """Test extrapolation capabilities of the encoding."""
        try:
            import torch
            import numpy as np
            
            if encoding is None:
                return []
            
            # Test different sequence lengths beyond training
            test_lengths = [50, 100, 200, 500, 1000]
            results = []
            
            for length in test_lengths:
                try:
                    if hasattr(encoding, 'forward'):
                        # If it's a model, create dummy input
                        dummy_input = torch.zeros(1, length)
                        with torch.no_grad():
                            enc_output = encoding.forward(dummy_input)
                            if isinstance(enc_output, dict):
                                enc_matrix = enc_output.get('encoding', enc_output.get('output', None))
                            else:
                                enc_matrix = enc_output
                    else:
                        # If it's a matrix, truncate or pad it
                        if encoding.size(0) >= length:
                            enc_matrix = encoding[:length, :]
                        else:
                            # Pad with zeros
                            padding = torch.zeros(length - encoding.size(0), encoding.size(1))
                            enc_matrix = torch.cat([encoding, padding], dim=0)
                    
                    if enc_matrix is not None:
                        # Compute quality metrics
                        variance = torch.var(enc_matrix).item()
                        
                        # Compute distinguishability
                        if enc_matrix.size(0) > 1:
                            similarities = torch.mm(enc_matrix, enc_matrix.t()) / enc_matrix.size(-1)
                            off_diagonal = similarities - torch.eye(similarities.size(0))
                            distinguishability = -torch.mean(off_diagonal).item()
                        else:
                            distinguishability = 0.0
                        
                        # Overall quality score
                        quality_score = (variance + distinguishability) / 2
                        
                        results.append({
                            'sequence_length': length,
                            'variance': variance,
                            'distinguishability': distinguishability,
                            'quality_score': quality_score
                        })
                    else:
                        results.append({
                            'sequence_length': length,
                            'variance': 0.0,
                            'distinguishability': 0.0,
                            'quality_score': 0.0
                        })
                        
                except Exception as e:
                    results.append({
                        'sequence_length': length,
                        'variance': 0.0,
                        'distinguishability': 0.0,
                        'quality_score': 0.0,
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            return [{'sequence_length': 0, 'variance': 0.0, 'distinguishability': 0.0, 'quality_score': 0.0, 'error': str(e)}]
    
    def _create_attention_heatmap(self, attention_weights, tokens, head_idx, show_token_labels, show_attention_values, attention_threshold):
        """Create attention heatmap visualization."""
        try:
            import plotly.graph_objects as go
            
            if attention_weights is None or attention_weights.numel() == 0:
                fig = go.Figure()
                fig.add_annotation(text="No attention data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Handle different tensor shapes
            if attention_weights.dim() == 3:  # (heads, seq, seq)
                attn_matrix = attention_weights[head_idx].cpu().numpy()
            elif attention_weights.dim() == 4:  # (batch, heads, seq, seq)
                attn_matrix = attention_weights[0, head_idx].cpu().numpy()
            else:
                attn_matrix = attention_weights.cpu().numpy()
            
            # Apply threshold if specified
            if attention_threshold > 0:
                attn_matrix = np.where(attn_matrix < attention_threshold, 0, attn_matrix)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attn_matrix,
                colorscale='Viridis',
                showscale=True,
                hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
            ))
            
            # Add token labels if requested
            if show_token_labels and tokens:
                fig.update_layout(
                    xaxis=dict(tickmode='array', tickvals=list(range(len(tokens))), ticktext=tokens),
                    yaxis=dict(tickmode='array', tickvals=list(range(len(tokens))), ticktext=tokens)
                )
            
            fig.update_layout(
                title=f"Attention Heatmap - Head {head_idx}",
                xaxis_title="Key Position",
                yaxis_title="Query Position",
                height=500
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating heatmap: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _create_encoding_comparison_plot(self, results):
        """Create encoding comparison plot."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            if not results or 'encodings' not in results:
                fig = go.Figure()
                fig.add_annotation(text="No comparison data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            encodings = results['encodings']
            methods = list(encodings.keys())
            
            if len(methods) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No encodings to compare", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=len(methods),
                subplot_titles=methods,
                shared_yaxes=True
            )
            
            for i, method in enumerate(methods):
                encoding = encodings[method]
                if encoding is not None:
                    # Take a subset for visualization
                    subset = encoding[:50, :50] if encoding.size(0) > 50 else encoding
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=subset.cpu().numpy().T,
                            colorscale='Viridis',
                            showscale=(i == len(methods) - 1),
                            hovertemplate=f'{method}<br>Position: %{{x}}<br>Dimension: %{{y}}<br>Value: %{{z:.3f}}<extra></extra>'
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                title="Encoding Comparison",
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating comparison plot: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _create_attention_comparison_plot(self, results):
        """Create attention comparison plot."""
        try:
            import plotly.graph_objects as go
            
            if not results or 'metrics' not in results:
                fig = go.Figure()
                fig.add_annotation(text="No attention comparison data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            metrics = results['metrics']
            methods = list(metrics.keys())
            
            if len(methods) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No methods to compare", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Extract metrics
            variance_values = [metrics[method].get('variance', 0) for method in methods]
            distinguishability_values = [metrics[method].get('distinguishability', 0) for method in methods]
            periodicity_values = [metrics[method].get('periodicity', 0) for method in methods]
            
            fig = go.Figure()
            
            # Add bar traces for each metric
            fig.add_trace(go.Bar(
                name='Variance',
                x=methods,
                y=variance_values,
                marker_color='blue'
            ))
            
            fig.add_trace(go.Bar(
                name='Distinguishability',
                x=methods,
                y=distinguishability_values,
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                name='Periodicity',
                x=methods,
                y=periodicity_values,
                marker_color='green'
            ))
            
            fig.update_layout(
                title="Attention Metrics Comparison",
                xaxis_title="Encoding Method",
                yaxis_title="Metric Value",
                barmode='group',
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating attention comparison plot: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _create_exploration_plot(self, exploration_results, exploration_type=None):
        """Create parameter exploration plot."""
        try:
            import plotly.graph_objects as go
            
            if not exploration_results or 'parameters' not in exploration_results:
                fig = go.Figure()
                fig.add_annotation(text="No exploration data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            parameters = exploration_results['parameters']
            metrics = exploration_results['metrics']
            
            if not parameters or not metrics:
                fig = go.Figure()
                fig.add_annotation(text="No exploration results to plot", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Extract data for plotting
            quality_scores = [m.get('quality_score', 0) for m in metrics]
            
            # Create scatter plot
            fig = go.Figure(data=go.Scatter(
                x=list(range(len(quality_scores))),
                y=quality_scores,
                mode='markers+lines',
                marker=dict(size=8, color=quality_scores, colorscale='Viridis'),
                hovertemplate='Experiment: %{x}<br>Quality Score: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Parameter Exploration Results",
                xaxis_title="Experiment Index",
                yaxis_title="Quality Score",
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating exploration plot: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _generate_report_preview(self, report_sections):
        """Generate a preview of the report."""
        try:
            preview = "# Positional Encoding Analysis Report Preview\n\n"
            
            for section in report_sections:
                preview += f"## {section}\n\n"
                preview += f"*Content for {section} will be generated here.*\n\n"
            
            preview += "---\n"
            preview += "*This is a preview. Full report will include detailed analysis, visualizations, and data.*\n"
            
            return preview
            
        except Exception as e:
            return f"# Error generating report preview\n\nError: {str(e)}"
    
    def _generate_complete_report(self, report_sections, include_data, include_config):
        """Generate a complete analysis report."""
        try:
            report = {
                'title': 'Positional Encoding Analysis Report',
                'sections': {},
                'metadata': {
                    'generated_at': str(pd.Timestamp.now()),
                    'include_data': include_data,
                    'include_config': include_config
                }
            }
            
            for section in report_sections:
                report['sections'][section] = {
                    'content': f"Analysis content for {section}",
                    'data': {} if include_data else None,
                    'config': {} if include_config else None
                }
            
            return report
            
        except Exception as e:
            return {
                'title': 'Error Report',
                'sections': {'Error': {'content': str(e)}},
                'metadata': {'error': True}
            }
    
    def _collect_analysis_data(self):
        """Collect all analysis data for export."""
        try:
            data = {
                'session_data': {},
                'encodings': {},
                'attention_data': {},
                'metrics': {}
            }
            
            # Collect session state data
            if hasattr(st.session_state, 'generated_encodings'):
                data['encodings'] = st.session_state.generated_encodings
            
            if hasattr(st.session_state, 'attention_data'):
                data['attention_data'] = st.session_state.attention_data
            
            if hasattr(st.session_state, 'comparison_results'):
                data['metrics']['comparison'] = st.session_state.comparison_results
            
            return data
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_analysis_package(self):
        """Create a complete analysis package."""
        try:
            package = {
                'metadata': {
                    'created_at': str(pd.Timestamp.now()),
                    'version': '1.0',
                    'description': 'Positional Encoding Analysis Package'
                },
                'data': self._collect_analysis_data(),
                'visualizations': {},
                'reports': {}
            }
            
            return package
            
        except Exception as e:
            return {'error': str(e), 'metadata': {'error': True}}
    
    def _create_multihead_comparison(self, attention_weights, tokens, query_pos):
        """Create multi-head attention comparison plot."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            if attention_weights is None or attention_weights.numel() == 0:
                fig = go.Figure()
                fig.add_annotation(text="No attention data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Handle different tensor shapes
            if attention_weights.dim() == 3:  # (heads, seq, seq)
                num_heads = attention_weights.size(0)
                seq_len = attention_weights.size(1)
            elif attention_weights.dim() == 4:  # (batch, heads, seq, seq)
                num_heads = attention_weights.size(1)
                seq_len = attention_weights.size(2)
            else:
                fig = go.Figure()
                fig.add_annotation(text="Invalid attention tensor shape", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Create subplots for each head
            fig = make_subplots(
                rows=1, cols=min(num_heads, 8),  # Limit to 8 heads for display
                subplot_titles=[f"Head {i}" for i in range(min(num_heads, 8))],
                shared_yaxes=True
            )
            
            for head_idx in range(min(num_heads, 8)):
                try:
                    if attention_weights.dim() == 3:
                        head_attention = attention_weights[head_idx, query_pos, :].cpu().numpy()
                    else:
                        head_attention = attention_weights[0, head_idx, query_pos, :].cpu().numpy()
                    
                    fig.add_trace(
                        go.Bar(
                            x=list(range(len(head_attention))),
                            y=head_attention,
                            name=f"Head {head_idx}",
                            showlegend=False
                        ),
                        row=1, col=head_idx+1
                    )
                except Exception as e:
                    # Add empty trace if there's an error
                    fig.add_trace(
                        go.Bar(x=[], y=[], name=f"Head {head_idx}"),
                        row=1, col=head_idx+1
                    )
            
            fig.update_layout(
                title=f"Multi-Head Attention Comparison - Query Position {query_pos}",
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating multi-head comparison: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _create_comparison_table(self, comparison_results):
        """Create a comparison table from results."""
        try:
            import pandas as pd
            
            if not comparison_results or 'metrics' not in comparison_results:
                return pd.DataFrame({'Method': [], 'Variance': [], 'Distinguishability': [], 'Periodicity': [], 'Overall Score': []})
            
            metrics = comparison_results['metrics']
            methods = list(metrics.keys())
            
            data = []
            for method in methods:
                method_metrics = metrics[method]
                data.append({
                    'Method': method,
                    'Variance': method_metrics.get('variance', 0.0),
                    'Distinguishability': method_metrics.get('distinguishability', 0.0),
                    'Periodicity': method_metrics.get('periodicity', 0.0),
                    'Overall Score': method_metrics.get('overall_score', 0.0)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            return pd.DataFrame({'Error': [str(e)]})
    
    # Additional helper methods would continue here...
    # (Due to length constraints, I'm showing the key structure and main methods)


def run_streamlit_app():
    """Run the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    run_streamlit_app()
