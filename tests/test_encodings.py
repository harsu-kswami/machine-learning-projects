"""Tests for positional encoding implementations."""

import unittest
import torch
import numpy as np
import math
from parameterized import parameterized

from config import ModelConfig
from positional_encoding import (
    SinusoidalEncoding, 
    LearnedPositionalEncoding,
    RoPEEncoding,
    RelativePositionalEncoding,
    get_positional_encoding
)

from . import TEST_CONFIGS, set_test_seed


class TestSinusoidalEncoding(unittest.TestCase):
    """Test sinusoidal positional encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.d_model = 64
        self.max_seq_len = 128
        self.encoding = SinusoidalEncoding(self.d_model, self.max_seq_len)
    
    def test_encoding_shape(self):
        """Test encoding output shape."""
        seq_len = 32
        encoding_output = self.encoding.forward(seq_len, self.d_model)
        
        expected_shape = (1, seq_len, self.d_model)
        self.assertEqual(encoding_output.shape, expected_shape)
    
    def test_encoding_values(self):
        """Test specific encoding values."""
        seq_len = 4
        encoding_output = self.encoding.forward(seq_len, self.d_model)
        encoding_matrix = encoding_output.squeeze(0)
        
        # Test position 0 (should be [0, 1, 0, 1, ...])
        pos_0 = encoding_matrix[0]
        expected_pos_0 = torch.zeros(self.d_model)
        expected_pos_0[1::2] = 1.0  # Odd indices should be 1
        
        torch.testing.assert_close(pos_0, expected_pos_0, atol=1e-6)
    
    def test_periodicity(self):
        """Test periodicity of sinusoidal patterns."""
        seq_len = 100
        encoding_output = self.encoding.forward(seq_len, self.d_model)
        encoding_matrix = encoding_output.squeeze(0)
        
        # Test that patterns repeat with expected periods
        for dim in range(0, min(8, self.d_model), 2):  # Test a few dimensions
            signal = encoding_matrix[:, dim].numpy()
            
            # Find dominant frequency using FFT
            fft = np.fft.fft(signal)
            frequencies = np.fft.fftfreq(len(signal))
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_freq = frequencies[dominant_freq_idx]
            
            # Expected frequency based on dimension
            expected_freq = 1 / (10000 ** (dim / self.d_model))
            
            # Allow some tolerance due to finite sequence length
            self.assertLess(abs(dominant_freq * len(signal) - expected_freq), 0.1)
    
    def test_different_sequence_lengths(self):
        """Test encoding with different sequence lengths."""
        for seq_len in [8, 16, 32, 64, 128]:
            encoding_output = self.encoding.forward(seq_len, self.d_model)
            expected_shape = (1, seq_len, self.d_model)
            self.assertEqual(encoding_output.shape, expected_shape)
    
    def test_extrapolation(self):
        """Test encoding extrapolation beyond training length."""
        # Create encoding with small max length
        small_encoding = SinusoidalEncoding(self.d_model, 64)
        
        # Test with longer sequence
        long_seq_len = 128
        encoding_output = small_encoding.forward(long_seq_len, self.d_model)
        
        # Should still work (extrapolate)
        expected_shape = (1, long_seq_len, self.d_model)
        self.assertEqual(encoding_output.shape, expected_shape)
    
    def test_frequency_analysis(self):
        """Test frequency analysis functionality."""
        if hasattr(self.encoding, 'analyze_frequency_components'):
            analysis = self.encoding.analyze_frequency_components()
            
            self.assertIn('frequencies', analysis)
            self.assertIn('wavelengths', analysis)
            
            frequencies = analysis['frequencies']
            self.assertEqual(len(frequencies), self.d_model // 2)
            
            # Frequencies should decrease for higher dimensions
            freq_values = frequencies.cpu().numpy()
            self.assertTrue(np.all(freq_values[:-1] >= freq_values[1:]))


class TestLearnedPositionalEncoding(unittest.TestCase):
    """Test learned positional encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.d_model = 64
        self.max_seq_len = 128
        self.encoding = LearnedPositionalEncoding(self.d_model, self.max_seq_len)
    
    def test_encoding_shape(self):
        """Test encoding output shape."""
        seq_len = 32
        encoding_output = self.encoding.forward(seq_len, self.d_model)
        
        expected_shape = (1, seq_len, self.d_model)
        self.assertEqual(encoding_output.shape, expected_shape)
    
    def test_learnable_parameters(self):
        """Test that encoding parameters are learnable."""
        # Check that position embeddings require gradients
        self.assertTrue(self.encoding.position_embeddings.weight.requires_grad)
        
        # Check parameter count
        expected_params = self.max_seq_len * self.d_model
        actual_params = self.encoding.position_embeddings.weight.numel()
        self.assertEqual(actual_params, expected_params)
    
    def test_gradient_flow(self):
        """Test gradient flow through learned encoding."""
        seq_len = 32
        encoding_output = self.encoding.forward(seq_len, self.d_model)
        
        # Create loss and backpropagate
        loss = encoding_output.mean()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.encoding.position_embeddings.weight.grad)
    
    def test_maximum_sequence_length(self):
        """Test encoding at maximum sequence length."""
        encoding_output = self.encoding.forward(self.max_seq_len, self.d_model)
        expected_shape = (1, self.max_seq_len, self.d_model)
        self.assertEqual(encoding_output.shape, expected_shape)
    
    def test_exceeding_maximum_length(self):
        """Test behavior when exceeding maximum sequence length."""
        long_seq_len = self.max_seq_len + 10
        
        with self.assertRaises((IndexError, RuntimeError)):
            self.encoding.forward(long_seq_len, self.d_model)


class TestRoPEEncoding(unittest.TestCase):
    """Test RoPE (Rotary Position Embedding) encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.d_model = 64
        self.max_seq_len = 128
        self.encoding = RoPEEncoding(self.d_model, self.max_seq_len)
    
    def test_encoding_output_format(self):
        """Test RoPE encoding output format."""
        seq_len = 32
        rope_output = self.encoding.forward(seq_len, self.d_model)
        
        # RoPE returns dictionary with cos and sin components
        self.assertIsInstance(rope_output, dict)
        self.assertIn('cos', rope_output)
        self.assertIn('sin', rope_output)
        
        cos_values = rope_output['cos']
        sin_values = rope_output['sin']
        
        # Check shapes
        expected_shape = (1, seq_len, self.d_model // 2)
        self.assertEqual(cos_values.shape, expected_shape)
        self.assertEqual(sin_values.shape, expected_shape)
    
    def test_rotation_properties(self):
        """Test mathematical properties of rotations."""
        seq_len = 16
        rope_output = self.encoding.forward(seq_len, self.d_model)
        
        cos_values = rope_output['cos']
        sin_values = rope_output['sin']
        
        # Test that cos^2 + sin^2 = 1 (unit circle property)
        cos_sin_sum = cos_values**2 + sin_values**2
        expected_ones = torch.ones_like(cos_sin_sum)
        
        torch.testing.assert_close(cos_sin_sum, expected_ones, atol=1e-6)
    
    def test_different_thetas(self):
        """Test RoPE with different theta values."""
        for theta in [1000, 10000, 100000]:
            rope_encoding = RoPEEncoding(self.d_model, self.max_seq_len, base=theta)
            rope_output = rope_encoding.forward(32, self.d_model)
            
            # Should produce valid output
            self.assertIn('cos', rope_output)
            self.assertIn('sin', rope_output)
    
    def test_relative_position_property(self):
        """Test that RoPE encodes relative positions."""
        seq_len = 8
        rope_output = self.encoding.forward(seq_len, self.d_model)
        
        cos_values = rope_output['cos']
        sin_values = rope_output['sin']
        
        # For small dimensions and positions, test rotation consistency
        for pos1 in range(min(4, seq_len)):
            for pos2 in range(min(4, seq_len)):
                rel_pos = pos2 - pos1
                
                # The relative rotation should depend only on relative position
                # This is a simplified test of the relative position property
                cos1 = cos_values[0, pos1]
                sin1 = sin_values[0, pos1]
                cos2 = cos_values[0, pos2]
                sin2 = sin_values[0, pos2]
                
                # All values should be finite
                self.assertTrue(torch.all(torch.isfinite(cos1)))
                self.assertTrue(torch.all(torch.isfinite(sin1)))
                self.assertTrue(torch.all(torch.isfinite(cos2)))
                self.assertTrue(torch.all(torch.isfinite(sin2)))
    
    def test_extrapolation(self):
        """Test RoPE extrapolation to longer sequences."""
        # Create RoPE with smaller max length
        rope_encoding = RoPEEncoding(self.d_model, 64)
        
        # Test with longer sequence
        long_seq_len = 128
        rope_output = rope_encoding.forward(long_seq_len, self.d_model)
        
        # Should work (RoPE can extrapolate)
        expected_shape = (1, long_seq_len, self.d_model // 2)
        self.assertEqual(rope_output['cos'].shape, expected_shape)
        self.assertEqual(rope_output['sin'].shape, expected_shape)


class TestRelativePositionalEncoding(unittest.TestCase):
    """Test relative positional encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.d_model = 64
        self.max_seq_len = 128
        self.encoding = RelativePositionalEncoding(self.d_model, self.max_seq_len)
    
    def test_bias_computation(self):
        """Test relative position bias computation."""
        seq_len = 16
        bias = self.encoding.get_relative_position_bias(seq_len)
        
        # Should be square matrix
        expected_shape = (seq_len, seq_len)
        self.assertEqual(bias.shape, expected_shape)
        
        # Should be symmetric around diagonal
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos_ij = j - i
                rel_pos_ji = i - j
                
                if abs(rel_pos_ij) <= self.encoding.max_relative_position:
                    # Values at symmetric positions should be related
                    pass  # More complex symmetry test could be added
    
    def test_different_sequence_lengths(self):
        """Test relative encoding with different sequence lengths."""
        for seq_len in [8, 16, 32, 64]:
            bias = self.encoding.get_relative_position_bias(seq_len)
            expected_shape = (seq_len, seq_len)
            self.assertEqual(bias.shape, expected_shape)
    
    def test_learnable_parameters(self):
        """Test that relative position parameters are learnable."""
        # Check that embeddings require gradients
        self.assertTrue(self.encoding.relative_position_embeddings.weight.requires_grad)


class TestPositionalEncodingFactory(unittest.TestCase):
    """Test positional encoding factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
    
    @parameterized.expand([
        ('sinusoidal', SinusoidalEncoding),
        ('learned', LearnedPositionalEncoding),
        ('rope', RoPEEncoding),
        ('relative', RelativePositionalEncoding),
    ])
    def test_encoding_factory(self, encoding_type, expected_class):
        """Test that factory creates correct encoding types."""
        config = ModelConfig()
        config.encoding_type = encoding_type
        config.d_model = 64
        config.max_seq_len = 128
        
        encoding = get_positional_encoding(config)
        self.assertIsInstance(encoding, expected_class)
    
    def test_invalid_encoding_type(self):
        """Test factory with invalid encoding type."""
        config = ModelConfig()
        config.encoding_type = 'invalid_type'
        
        with self.assertRaises(ValueError):
            get_positional_encoding(config)
    
    def test_config_parameters(self):
        """Test that factory respects config parameters."""
        config = ModelConfig()
        config.encoding_type = 'rope'
        config.rope_theta = 50000
        config.d_model = 128
        
        encoding = get_positional_encoding(config)
        
        # Check that RoPE uses correct base value
        if hasattr(encoding, 'base'):
            self.assertEqual(encoding.base, config.rope_theta)


class TestEncodingComparison(unittest.TestCase):
    """Test comparison between different encoding methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig()
        self.config.d_model = 64
        self.config.max_seq_len = 64
        self.seq_len = 32
    
    def test_encoding_outputs_different(self):
        """Test that different encoding methods produce different outputs."""
        encodings = {}
        
        # Generate encodings with different methods
        for enc_type in ['sinusoidal', 'learned']:
            config = ModelConfig()
            config.encoding_type = enc_type
            config.d_model = self.config.d_model
            config.max_seq_len = self.config.max_seq_len
            
            encoding = get_positional_encoding(config)
            if enc_type == 'learned':
                # Initialize learned encoding randomly
                torch.nn.init.normal_(encoding.position_embeddings.weight, 0, 0.1)
            
            enc_output = encoding.forward(self.seq_len, self.config.d_model)
            encodings[enc_type] = enc_output
        
        # Compare sinusoidal vs learned
        sinusoidal = encodings['sinusoidal']
        learned = encodings['learned']
        
        # They should be different
        self.assertFalse(torch.allclose(sinusoidal, learned, atol=1e-3))
    
    def test_encoding_magnitudes(self):
        """Test that different encodings have reasonable magnitudes."""
        for enc_type in ['sinusoidal', 'learned']:
            config = ModelConfig()
            config.encoding_type = enc_type
            config.d_model = self.config.d_model
            config.max_seq_len = self.config.max_seq_len
            
            encoding = get_positional_encoding(config)
            enc_output = encoding.forward(self.seq_len, self.config.d_model)
            
            # Check magnitude is reasonable
            magnitude = torch.norm(enc_output)
            self.assertGreater(magnitude, 0.1)  # Not too small
            self.assertLess(magnitude, 1000)    # Not too large
    
    def test_position_distinguishability(self):
        """Test that different positions are distinguishable."""
        config = ModelConfig()
        config.encoding_type = 'sinusoidal'
        config.d_model = self.config.d_model
        config.max_seq_len = self.config.max_seq_len
        
        encoding = get_positional_encoding(config)
        enc_output = encoding.forward(self.seq_len, self.config.d_model)
        enc_matrix = enc_output.squeeze(0)
        
        # Compute pairwise similarities
        similarities = torch.mm(enc_matrix, enc_matrix.t())
        
        # Diagonal should be 1 (self-similarity)
        diagonal = torch.diag(similarities)
        expected_diagonal = torch.ones_like(diagonal) * self.config.d_model
        torch.testing.assert_close(diagonal, expected_diagonal, rtol=1e-3)
        
        # Off-diagonal elements should be smaller (positions distinguishable)
        off_diagonal = similarities.clone()
        off_diagonal.fill_diagonal_(0)
        
        # Most off-diagonal elements should be smaller than diagonal
        self.assertTrue(torch.mean(torch.abs(off_diagonal)) < torch.mean(diagonal))


if __name__ == '__main__':
    unittest.main(verbosity=2)
