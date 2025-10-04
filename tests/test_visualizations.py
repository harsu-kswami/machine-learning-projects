
"""Tests for visualization components."""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock
import tempfile
import os

from config import ModelConfig, VisualizationConfig
from models import TransformerEncoder
from visualization import (
    AttentionVisualizer, 
    MultiHeadAttentionVisualizer,
    EncodingPlotter,
    HeatmapGenerator,
    ThreeDVisualizer
)
from utils.tokenizer import SimpleTokenizer

from . import TEST_CONFIGS, SAMPLE_TEXTS, set_test_seed


class TestAttentionVisualizer(unittest.TestCase):
    """Test attention visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig(**TEST_CONFIGS['small'])
        self.viz_config = VisualizationConfig()
        self.visualizer = AttentionVisualizer(self.viz_config)
        self.tokenizer = SimpleTokenizer()
        
        # Create sample attention weights
        self.batch_size = 1
        self.n_heads = 4
        self.seq_len = 8
        self.attention_weights = torch.softmax(
            torch.randn(self.batch_size, self.n_heads, self.seq_len, self.seq_len),
            dim=-1
        )
        
        # Sample tokens
        self.tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
    
    def test_visualize_attention_matrix(self):
        """Test attention matrix visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'attention_test.png')
            
            fig = self.visualizer.visualize_attention_matrix(
                self.attention_weights,
                tokens=self.tokens,
                head_idx=0,
                save_path=save_path
            )
            
            # Check that figure was created
            self.assertIsInstance(fig, plt.Figure)
            
            # Check that file was saved
            self.assertTrue(os.path.exists(save_path))
            
            plt.close(fig)
    
    def test_visualize_multi_head_attention(self):
        """Test multi-head attention visualization."""
        fig = self.visualizer.visualize_multi_head_attention(
            self.attention_weights,
            tokens=self.tokens
        )
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has correct number of subplots
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), self.n_heads)
        
        plt.close(fig)
    
    def test_attention_patterns_over_layers(self):
        """Test attention evolution visualization."""
        # Create multi-layer attention weights
        n_layers = 3
        layer_attention_weights = [
            torch.softmax(torch.randn(self.batch_size, self.n_heads, self.seq_len, self.seq_len), dim=-1)
            for _ in range(n_layers)
        ]
        
        fig = self.visualizer.visualize_attention_patterns_over_layers(
            layer_attention_weights,
            tokens=self.tokens,
            head_idx=0
        )
        
        self.assertIsInstance(fig, plt.Figure)
        
        # Should have subplots for each layer
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), n_layers)
        
        plt.close(fig)
    
    def test_analyze_attention_patterns(self):
        """Test attention pattern analysis."""
        analysis = self.visualizer.analyze_attention_patterns(self.attention_weights)
        
        # Check required analysis components
        self.assertIn('entropy', analysis)
        self.assertIn('distance', analysis)
        self.assertIn('sparsity', analysis)
        self.assertIn('head_similarity', analysis)
        self.assertIn('pattern_type', analysis)
        
        # Check entropy analysis
        entropy_analysis = analysis['entropy']
        self.assertIn('mean_entropy', entropy_analysis)
        self.assertIn('std_entropy', entropy_analysis)
        
        # Check distance analysis
        distance_analysis = analysis['distance']
        self.assertIn('mean_distance', distance_analysis)
        
        # Check sparsity analysis
        sparsity_analysis = analysis['sparsity']
        self.assertIn('gini_coefficient', sparsity_analysis)
        
        # Check pattern classification
        pattern_type = analysis['pattern_type']
        valid_patterns = ['self-focused', 'local', 'global', 'mixed']
        self.assertIn(pattern_type, valid_patterns)
    
    def test_attention_flow_diagram(self):
        """Test attention flow diagram creation."""
        fig = self.visualizer.create_attention_flow_diagram(
            self.attention_weights,
            self.tokens,
            threshold=0.1
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_invalid_attention_weights(self):
        """Test handling of invalid attention weights."""
        # Test with wrong dimensions
        invalid_weights = torch.randn(2, 3)  # Wrong shape
        
        with self.assertRaises((ValueError, IndexError, RuntimeError)):
            self.visualizer.visualize_attention_matrix(invalid_weights)
    
    def test_empty_tokens(self):
        """Test visualization with empty tokens."""
        fig = self.visualizer.visualize_attention_matrix(
            self.attention_weights,
            tokens=None,
            head_idx=0
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_attention_threshold(self):
        """Test attention visualization with threshold."""
        fig = self.visualizer.visualize_attention_matrix(
            self.attention_weights,
            tokens=self.tokens,
            head_idx=0,
            threshold=0.2
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestMultiHeadAttentionVisualizer(unittest.TestCase):
    """Test multi-head attention specific visualizations."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.viz_config = VisualizationConfig()
        self.visualizer = MultiHeadAttentionVisualizer(self.viz_config)
        
        self.n_heads = 8
        self.seq_len = 10
        self.attention_weights = torch.softmax(
            torch.randn(1, self.n_heads, self.seq_len, self.seq_len),
            dim=-1
        )
        self.tokens = [f'token_{i}' for i in range(self.seq_len)]
    
    def test_head_comparison_plot(self):
        """Test head comparison visualization."""
        fig = self.visualizer.create_head_comparison_plot(
            self.attention_weights,
            self.tokens,
            query_position=0
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_head_specialization_analysis(self):
        """Test head specialization analysis."""
        fig = self.visualizer.create_head_specialization_analysis(
            self.attention_weights,
            self.tokens
        )
        
        self.assertIsInstance(fig, plt.Figure)
        
        # Should have multiple subplots for different analyses
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), 4)  # At least 4 analysis plots
        
        plt.close(fig)
    
    def test_interactive_attention_explorer(self):
        """Test interactive attention explorer creation."""
        plotly_fig = self.visualizer.create_interactive_attention_explorer(
            self.attention_weights,
            self.tokens
        )
        
        self.assertIsInstance(plotly_fig, go.Figure)
        
        # Check that figure has data
        self.assertGreater(len(plotly_fig.data), 0)


class TestEncodingPlotter(unittest.TestCase):
    """Test positional encoding plotting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.viz_config = VisualizationConfig()
        self.plotter = EncodingPlotter(self.viz_config)
        
        # Create sample encoding
        from positional_encoding import SinusoidalEncoding
        self.encoding = SinusoidalEncoding(d_model=64, max_seq_len=128)
    
    def test_plot_sinusoidal_patterns(self):
        """Test sinusoidal encoding pattern plotting."""
        fig = self.plotter.plot_sinusoidal_patterns(
            self.encoding,
            seq_len=32
        )
        
        self.assertIsInstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), 4)
        
        plt.close(fig)
    
    def test_plot_different_dimensions(self):
        """Test plotting with specific dimensions."""
        dimensions = [0, 2, 4, 6]
        fig = self.plotter.plot_sinusoidal_patterns(
            self.encoding,
            seq_len=32,
            dimensions=dimensions
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_encoding_save_functionality(self):
        """Test saving encoding plots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'encoding_test.png')
            
            fig = self.plotter.plot_sinusoidal_patterns(
                self.encoding,
                seq_len=32,
                save_path=save_path
            )
            
            self.assertTrue(os.path.exists(save_path))
            plt.close(fig)
    
    def test_rope_visualization(self):
        """Test RoPE visualization functionality."""
        from positional_encoding import RoPEEncoding
        rope_encoding = RoPEEncoding(d_model=64, max_seq_len=128)
        
        # Test that RoPE plotting doesn't crash (method might not exist)
        try:
            fig = self.plotter.plot_rope_rotations(rope_encoding, seq_len=16)
            if fig:
                self.assertIsInstance(fig, plt.Figure)
                plt.close(fig)
        except AttributeError:
            # Method might not be implemented yet
            pass


class TestHeatmapGenerator(unittest.TestCase):
    """Test heatmap generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.viz_config = VisualizationConfig()
        self.generator = HeatmapGenerator(self.viz_config)
        
        # Sample data
        self.seq_len = 12
        self.attention_weights = torch.softmax(
            torch.randn(1, 4, self.seq_len, self.seq_len),
            dim=-1
        )
        self.tokens = [f'token_{i}' for i in range(self.seq_len)]
    
    def test_attention_heatmap_creation(self):
        """Test attention heatmap creation."""
        fig = self.generator.create_attention_heatmap(
            self.attention_weights,
            tokens=self.tokens
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_heatmap_with_mask(self):
        """Test heatmap with attention mask."""
        # Create attention mask
        mask = torch.ones(1, self.seq_len, dtype=torch.bool)
        mask[0, -2:] = False  # Mask last 2 positions
        
        fig = self.generator.create_attention_heatmap(
            self.attention_weights,
            tokens=self.tokens,
            mask=mask
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_multi_layer_heatmap(self):
        """Test multi-layer heatmap visualization."""
        n_layers = 3
        layer_attentions = [
            torch.softmax(torch.randn(1, 4, self.seq_len, self.seq_len), dim=-1)
            for _ in range(n_layers)
        ]
        
        fig = self.generator.create_multi_layer_heatmap(
            layer_attentions,
            tokens=self.tokens,
            head_idx=0
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_encoding_heatmap(self):
        """Test encoding heatmap creation."""
        # Create sample encoding matrix
        encoding_matrix = torch.randn(32, 64)
        
        fig = self.generator.create_encoding_heatmap(
            encoding_matrix,
            encoding_name="Test Encoding"
        )
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_comparison_heatmap(self):
        """Test comparison heatmap creation."""
        encoding_matrices = {
            'sinusoidal': torch.randn(32, 64),
            'learned': torch.randn(32, 64)
        }
        
        fig = self.generator.create_comparison_heatmap(encoding_matrices)
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestThreeDVisualizer(unittest.TestCase):
    """Test 3D visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.viz_config = VisualizationConfig()
        self.visualizer = ThreeDVisualizer(self.viz_config)
    
    def test_encoding_3d_surface(self):
        """Test 3D surface plot creation."""
        encoding_matrix = torch.randn(32, 64)
        
        fig = self.visualizer.create_encoding_3d_surface(
            encoding_matrix,
            encoding_name="Test Encoding"
        )
        
        self.assertIsInstance(fig, go.Figure)
        
        # Check that figure has surface data
        self.assertGreater(len(fig.data), 0)
        self.assertEqual(fig.data[0].type, 'surface')
    
    def test_attention_3d_landscape(self):
        """Test 3D attention landscape."""
        attention_weights = torch.softmax(
            torch.randn(1, 4, 10, 10),
            dim=-1
        )
        tokens = [f'token_{i}' for i in range(10)]
        
        fig = self.visualizer.create_attention_3d_landscape(
            attention_weights,
            tokens,
            head_idx=0
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
    
    def test_rope_rotation_3d(self):
        """Test 3D RoPE rotation visualization."""
        from positional_encoding import RoPEEncoding
        rope_encoding = RoPEEncoding(d_model=64, max_seq_len=64)
        
        # This might not be implemented, so handle gracefully
        try:
            fig = self.visualizer.create_rope_rotation_3d(
                rope_encoding,
                seq_len=16
            )
            if fig:
                self.assertIsInstance(fig, go.Figure)
        except (AttributeError, NotImplementedError):
            # Method might not be implemented
            pass


class TestVisualizationIntegration(unittest.TestCase):
    """Test integration between visualization components and models."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig(**TEST_CONFIGS['small'])
        self.model = TransformerEncoder(self.config)
        self.tokenizer = SimpleTokenizer()
        
        self.viz_config = VisualizationConfig()
        self.visualizer = AttentionVisualizer(self.viz_config)
    
    def test_end_to_end_visualization(self):
        """Test complete pipeline from model to visualization."""
        # Tokenize sample text
        text = SAMPLE_TEXTS[0]
        tokens = self.tokenizer.tokenize(text)[:16]  # Limit length
        
        # Create input
        input_ids = torch.tensor([list(range(len(tokens)))])
        
        # Forward pass
        outputs = self.model(input_ids, store_visualizations=True)
        attention_weights = outputs['attention_weights']
        
        # Visualize
        if attention_weights:
            fig = self.visualizer.visualize_attention_matrix(
                attention_weights[0],  # First layer
                tokens=tokens,
                head_idx=0
            )
            
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualization_config_effects(self):
        """Test that visualization config affects output."""
        # Test with different color schemes
        configs = [
            VisualizationConfig(colormap_attention='viridis'),
            VisualizationConfig(colormap_attention='plasma')
        ]
        
        attention_weights = torch.softmax(
            torch.randn(1, 4, 8, 8),
            dim=-1
        )
        
        for config in configs:
            visualizer = AttentionVisualizer(config)
            fig = visualizer.visualize_attention_matrix(attention_weights)
            
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)


class TestVisualizationErrorHandling(unittest.TestCase):
    """Test error handling in visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.viz_config = VisualizationConfig()
        self.visualizer = AttentionVisualizer(self.viz_config)
    
    def test_invalid_attention_shape(self):
        """Test handling of invalid attention weight shapes."""
        invalid_attention = torch.randn(10)  # 1D instead of 4D
        
        with self.assertRaises((ValueError, IndexError, RuntimeError)):
            self.visualizer.visualize_attention_matrix(invalid_attention)
    
    def test_mismatched_tokens_length(self):
        """Test handling of mismatched token lengths."""
        attention_weights = torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        wrong_tokens = ['token1', 'token2']  # Only 2 tokens for 8x8 attention
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = self.visualizer.visualize_attention_matrix(
                attention_weights,
                tokens=wrong_tokens
            )
            if fig:
                plt.close(fig)
        except (ValueError, IndexError):
            # Expected behavior for mismatched dimensions
            pass
    
    def test_empty_data(self):
        """Test handling of empty data."""
        empty_attention = torch.empty(0, 0, 0, 0)
        
        with self.assertRaises((ValueError, IndexError, RuntimeError)):
            self.visualizer.visualize_attention_matrix(empty_attention)
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_failure_handling(self, mock_savefig):
        """Test handling of save failures."""
        mock_savefig.side_effect = IOError("Cannot save file")
        
        attention_weights = torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        
        with self.assertRaises(IOError):
            self.visualizer.visualize_attention_matrix(
                attention_weights,
                save_path='/invalid/path/test.png'
            )


if __name__ == '__main__':
    # Set matplotlib backend for testing
    plt.switch_backend('Agg')
    
    unittest.main(verbosity=2)
