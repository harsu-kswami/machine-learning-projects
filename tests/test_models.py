"""Tests for transformer models and components."""

import unittest
import torch
import torch.nn as nn
import numpy as np
from parameterized import parameterized

from config import ModelConfig
from models import TransformerEncoder, MultiHeadAttention, FeedForward, TransformerEncoderLayer
from positional_encoding import get_positional_encoding

from . import TEST_CONFIGS, SAMPLE_TEXTS, set_test_seed


class TestTransformerComponents(unittest.TestCase):
    """Test individual transformer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig(**TEST_CONFIGS['small'])
        self.batch_size = 2
        self.seq_len = 16
        
        # Create test input
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.config.d_model)
        self.test_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
    
    def test_multi_head_attention_forward(self):
        """Test MultiHeadAttention forward pass."""
        mha = MultiHeadAttention(self.config)
        
        # Test forward pass
        output, attention_weights = mha(
            self.test_input, self.test_input, self.test_input, self.test_mask
        )
        
        # Check output shape
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Check attention weights shape
        expected_attn_shape = (self.batch_size, self.config.n_heads, self.seq_len, self.seq_len)
        self.assertEqual(attention_weights.shape, expected_attn_shape)
        
        # Check attention weights sum to 1
        attn_sums = attention_weights.sum(dim=-1)
        torch.testing.assert_close(attn_sums, torch.ones_like(attn_sums), atol=1e-6)
    
    def test_multi_head_attention_masked(self):
        """Test MultiHeadAttention with masking."""
        mha = MultiHeadAttention(self.config)
        
        # Create mask with some positions masked
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[0, -2:] = False  # Mask last 2 positions for first sample
        
        output, attention_weights = mha(
            self.test_input, self.test_input, self.test_input, mask
        )
        
        # Check that masked positions have zero attention
        masked_attention = attention_weights[0, :, :, -2:]
        self.assertTrue(torch.allclose(masked_attention, torch.zeros_like(masked_attention)))
    
    def test_feed_forward(self):
        """Test FeedForward layer."""
        ff = FeedForward(self.config)
        
        output = ff(self.test_input)
        
        # Check output shape
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Check that output is different from input (non-identity)
        self.assertFalse(torch.allclose(output, self.test_input))
    
    def test_transformer_encoder_layer(self):
        """Test TransformerEncoderLayer."""
        layer = TransformerEncoderLayer(self.config)
        
        output, attention_weights = layer(self.test_input, self.test_mask)
        
        # Check output shape
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Check attention weights shape
        expected_attn_shape = (self.batch_size, self.config.n_heads, self.seq_len, self.seq_len)
        self.assertEqual(attention_weights.shape, expected_attn_shape)
    
    @parameterized.expand([
        ('small', TEST_CONFIGS['small']),
        ('medium', TEST_CONFIGS['medium']),
    ])
    def test_different_model_sizes(self, name, config_dict):
        """Test components with different model sizes."""
        config = ModelConfig(**config_dict)
        
        # Adjust test input size
        test_input = torch.randn(self.batch_size, self.seq_len, config.d_model)
        
        # Test MultiHeadAttention
        mha = MultiHeadAttention(config)
        output, _ = mha(test_input, test_input, test_input, self.test_mask)
        self.assertEqual(output.shape, test_input.shape)
        
        # Test FeedForward
        ff = FeedForward(config)
        output = ff(test_input)
        self.assertEqual(output.shape, test_input.shape)


class TestTransformerEncoder(unittest.TestCase):
    """Test complete TransformerEncoder model."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig(**TEST_CONFIGS['small'])
        self.model = TransformerEncoder(self.config)
        
        self.batch_size = 2
        self.seq_len = 16
        self.test_input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
    
    def test_model_forward(self):
        """Test model forward pass."""
        outputs = self.model(self.test_input_ids)
        
        # Check output keys
        expected_keys = ['last_hidden_state', 'hidden_states', 'attention_weights']
        for key in expected_keys:
            self.assertIn(key, outputs)
        
        # Check output shapes
        last_hidden_state = outputs['last_hidden_state']
        expected_shape = (self.batch_size, self.seq_len, self.config.d_model)
        self.assertEqual(last_hidden_state.shape, expected_shape)
        
        # Check hidden states
        hidden_states = outputs['hidden_states']
        self.assertEqual(len(hidden_states), self.config.n_layers + 1)  # +1 for embedding
        
        # Check attention weights
        attention_weights = outputs['attention_weights']
        self.assertEqual(len(attention_weights), self.config.n_layers)
    
    def test_model_with_attention_mask(self):
        """Test model with attention mask."""
        # Create attention mask
        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        attention_mask[0, -3:] = False  # Mask last 3 positions
        
        outputs = self.model(self.test_input_ids, attention_mask=attention_mask)
        
        # Check that model runs without error
        self.assertIn('last_hidden_state', outputs)
        
        # Check attention weights respect mask
        attention_weights = outputs['attention_weights'][0]  # First layer
        masked_attention = attention_weights[0, :, :, -3:]  # First sample, last 3 positions
        self.assertTrue(torch.allclose(masked_attention, torch.zeros_like(masked_attention)))
    
    @parameterized.expand([
        ('sinusoidal',),
        ('learned',),
        ('rope',),
    ])
    def test_different_positional_encodings(self, encoding_type):
        """Test model with different positional encodings."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        config.encoding_type = encoding_type
        
        model = TransformerEncoder(config)
        outputs = model(self.test_input_ids)
        
        # Check that model works with different encodings
        self.assertIn('last_hidden_state', outputs)
        
        expected_shape = (self.batch_size, self.seq_len, config.d_model)
        self.assertEqual(outputs['last_hidden_state'].shape, expected_shape)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        outputs = self.model(self.test_input_ids)
        
        # Create a simple loss
        loss = outputs['last_hidden_state'].mean()
        loss.backward()
        
        # Check that embedding layer has gradients
        self.assertIsNotNone(self.model.embeddings.token_embeddings.weight.grad)
        
        # Check that encoder layers have gradients
        for layer in self.model.encoder.layers:
            self.assertIsNotNone(layer.self_attention.query_projection.weight.grad)
    
    def test_model_evaluation_mode(self):
        """Test model in evaluation mode."""
        self.model.eval()
        
        with torch.no_grad():
            outputs1 = self.model(self.test_input_ids)
            outputs2 = self.model(self.test_input_ids)
        
        # Outputs should be identical in eval mode
        torch.testing.assert_close(
            outputs1['last_hidden_state'], 
            outputs2['last_hidden_state']
        )
    
    def test_model_training_mode(self):
        """Test model in training mode with dropout."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        config.dropout = 0.1  # Enable dropout
        model = TransformerEncoder(config)
        model.train()
        
        outputs1 = model(self.test_input_ids)
        outputs2 = model(self.test_input_ids)
        
        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(
            outputs1['last_hidden_state'], 
            outputs2['last_hidden_state']
        ))
    
    def test_model_device_handling(self):
        """Test model device handling."""
        device = torch.device('cpu')
        self.model.to(device)
        
        # Test with inputs on same device
        test_input = self.test_input_ids.to(device)
        outputs = self.model(test_input)
        
        # Check outputs are on correct device
        self.assertEqual(outputs['last_hidden_state'].device, device)
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # All parameters should be trainable by default
        self.assertEqual(total_params, trainable_params)
        
        # Parameter count should be reasonable for small model
        self.assertGreater(total_params, 1000)  # At least 1k parameters
        self.assertLess(total_params, 1000000)  # Less than 1M parameters for small model


class TestModelIntegration(unittest.TestCase):
    """Integration tests for complete model pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig(**TEST_CONFIGS['medium'])
        self.model = TransformerEncoder(self.config)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from text to model output."""
        from utils.tokenizer import SimpleTokenizer
        
        tokenizer = SimpleTokenizer()
        
        # Tokenize sample text
        text = SAMPLE_TEXTS[0]
        tokens = tokenizer.tokenize(text)
        
        # Create simple vocabulary
        vocab = {token: i for i, token in enumerate(set(tokens))}
        vocab.update({'<PAD>': len(vocab), '<UNK>': len(vocab)})
        
        # Encode tokens
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Pad to model requirements
        max_len = self.config.max_seq_len
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids.extend([vocab['<PAD>']] * (max_len - len(token_ids)))
        
        # Convert to tensor and add batch dimension
        input_ids = torch.tensor([token_ids])
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Check outputs
        self.assertIn('last_hidden_state', outputs)
        self.assertEqual(outputs['last_hidden_state'].shape[0], 1)  # Batch size 1
        self.assertEqual(outputs['last_hidden_state'].shape[1], max_len)  # Sequence length
    
    def test_batch_processing(self):
        """Test model with batch of different length sequences."""
        from utils.data_preprocessing import SequenceProcessor
        
        processor = SequenceProcessor()
        
        # Create sequences of different lengths
        sequences = [
            torch.randint(0, 100, (10,)),
            torch.randint(0, 100, (15,)),
            torch.randint(0, 100, (8,)),
        ]
        
        # Pad sequences
        padded_batch = processor.pad_sequences([seq.tolist() for seq in sequences])
        
        # Forward pass
        outputs = self.model(padded_batch)
        
        # Check outputs
        self.assertEqual(outputs['last_hidden_state'].shape[0], len(sequences))
    
    def test_visualization_integration(self):
        """Test integration with visualization components."""
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Forward pass with visualization storage
        outputs = self.model(input_ids, store_visualizations=True)
        
        # Check attention weights are available for visualization
        self.assertIn('attention_weights', outputs)
        attention_weights = outputs['attention_weights']
        
        # Test that attention weights can be processed by visualization tools
        from visualization import AttentionVisualizer
        from config import VisualizationConfig
        
        viz_config = VisualizationConfig()
        visualizer = AttentionVisualizer(viz_config)
        
        # Should not raise exception
        analysis = visualizer.analyze_attention_patterns(attention_weights[0])
        self.assertIn('entropy', analysis)


class TestModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
    
    def test_empty_input(self):
        """Test model with empty input."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        model = TransformerEncoder(config)
        
        # Test with minimal input
        input_ids = torch.randint(0, config.vocab_size, (1, 1))
        outputs = model(input_ids)
        
        self.assertEqual(outputs['last_hidden_state'].shape, (1, 1, config.d_model))
    
    def test_maximum_sequence_length(self):
        """Test model with maximum sequence length."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        model = TransformerEncoder(config)
        
        # Test with maximum length
        max_len = config.max_seq_len
        input_ids = torch.randint(0, config.vocab_size, (1, max_len))
        
        outputs = model(input_ids)
        self.assertEqual(outputs['last_hidden_state'].shape, (1, max_len, config.d_model))
    
    def test_invalid_config(self):
        """Test model creation with invalid configuration."""
        # Test with invalid head count
        config = ModelConfig()
        config.d_model = 64
        config.n_heads = 7  # Should not divide d_model evenly
        
        with self.assertRaises((ValueError, AssertionError)):
            TransformerEncoder(config)
    
    def test_model_state_dict(self):
        """Test model state dict save/load."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        model1 = TransformerEncoder(config)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = TransformerEncoder(config)
        model2.load_state_dict(state_dict)
        
        # Test that models produce same output
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            outputs1 = model1(input_ids)
            outputs2 = model2(input_ids)
        
        torch.testing.assert_close(
            outputs1['last_hidden_state'],
            outputs2['last_hidden_state']
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
