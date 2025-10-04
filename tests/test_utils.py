"""Tests for utility functions and components."""

import unittest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, mock_open

from config import ModelConfig, VisualizationConfig
from utils.tokenizer import SimpleTokenizer, BPETokenizer, WordPieceTokenizer, create_tokenizer
from utils.data_preprocessing import TextPreprocessor, SequenceProcessor, DatasetBuilder
from utils.metrics import AttentionMetrics, EncodingMetrics, PerformanceMetrics
from utils.export_utils import FigureExporter, DataExporter, ReportGenerator
from utils.performance_profiler import PerformanceProfiler, ModelProfiler

from . import TEST_CONFIGS, SAMPLE_TEXTS, set_test_seed


class TestTokenizers(unittest.TestCase):
    """Test tokenization utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.sample_texts = SAMPLE_TEXTS
        self.simple_tokenizer = SimpleTokenizer()
    
    def test_simple_tokenizer_basic(self):
        """Test basic SimpleTokenizer functionality."""
        text = "Hello world! This is a test."
        tokens = self.simple_tokenizer.tokenize(text)
        
        # Should split on whitespace and handle punctuation
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Test encoding
        token_ids = self.simple_tokenizer.encode(text)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), len(tokens))  # Due to special tokens
    
    def test_simple_tokenizer_decode(self):
        """Test SimpleTokenizer decoding."""
        text = "Hello world test"
        
        # Build vocabulary
        self.simple_tokenizer.build_vocab([text])
        
        # Encode and decode
        token_ids = self.simple_tokenizer.encode(text, add_special_tokens=False)
        decoded_text = self.simple_tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Should be similar (might not be identical due to preprocessing)
        self.assertIn('hello', decoded_text.lower())
        self.assertIn('world', decoded_text.lower())
    
    def test_simple_tokenizer_vocabulary_building(self):
        """Test vocabulary building."""
        self.simple_tokenizer.build_vocab(self.sample_texts)
        
        vocab = self.simple_tokenizer.get_vocab()
        
        # Should have special tokens
        self.assertIn('<PAD>', vocab)
        self.assertIn('<UNK>', vocab)
        self.assertIn('<BOS>', vocab)
        self.assertIn('<EOS>', vocab)
        
        # Should have reasonable size
        self.assertGreater(len(vocab), 10)
        self.assertLess(len(vocab), 1000)
    
    def test_simple_tokenizer_save_load(self):
        """Test SimpleTokenizer save/load functionality."""
        # Build vocabulary
        self.simple_tokenizer.build_vocab(self.sample_texts[:2])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = f.name
        
        try:
            # Save tokenizer
            self.simple_tokenizer.save(save_path)
            
            # Load tokenizer
            loaded_tokenizer = SimpleTokenizer.load(save_path)
            
            # Test that they work the same
            test_text = "test sentence"
            original_tokens = self.simple_tokenizer.encode(test_text)
            loaded_tokens = loaded_tokenizer.encode(test_text)
            
            self.assertEqual(original_tokens, loaded_tokens)
        finally:
            os.unlink(save_path)
    
    def test_bpe_tokenizer_basic(self):
        """Test basic BPE tokenizer functionality."""
        bpe_tokenizer = BPETokenizer(vocab_size=1000)
        
        # Train on sample texts
        bpe_tokenizer.train(self.sample_texts[:3])
        
        # Test tokenization
        text = "hello world"
        tokens = bpe_tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        
        # Test encoding/decoding
        token_ids = bpe_tokenizer.encode(text)
        decoded_text = bpe_tokenizer.decode(token_ids)
        
        # Should preserve most of the original text
        self.assertTrue(any(word in decoded_text.lower() for word in ['hello', 'world']))
    
    def test_wordpiece_tokenizer_basic(self):
        """Test basic WordPiece tokenizer functionality."""
        wp_tokenizer = WordPieceTokenizer(vocab_size=1000)
        
        # Train on sample texts
        wp_tokenizer.train(self.sample_texts[:3])
        
        # Test tokenization
        text = "hello world"
        tokens = wp_tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        
        # Test encoding/decoding
        token_ids = wp_tokenizer.encode(text)
        decoded_text = wp_tokenizer.decode(token_ids)
        
        self.assertIsInstance(decoded_text, str)
    
    def test_tokenizer_factory(self):
        """Test tokenizer factory function."""
        # Test different tokenizer types
        tokenizers = {
            'simple': SimpleTokenizer,
            'bpe': BPETokenizer,
            'wordpiece': WordPieceTokenizer
        }
        
        for tokenizer_type, expected_class in tokenizers.items():
            tokenizer = create_tokenizer(tokenizer_type, vocab_size=500)
            self.assertIsInstance(tokenizer, expected_class)
    
    def test_tokenizer_factory_invalid_type(self):
        """Test tokenizer factory with invalid type."""
        with self.assertRaises(ValueError):
            create_tokenizer('invalid_tokenizer_type')


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.preprocessor = TextPreprocessor()
        self.sequence_processor = SequenceProcessor()
        self.sample_texts = SAMPLE_TEXTS
    
    def test_text_preprocessor_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "Hello World! Visit https://example.com or email test@example.com"
        cleaned_text = self.preprocessor.clean_text(dirty_text)
        
        # Should replace URLs and emails
        self.assertIn('[URL]', cleaned_text)
        self.assertIn('[EMAIL]', cleaned_text)
        
        # Should be lowercase if enabled
        if self.preprocessor.lowercase:
            self.assertEqual(cleaned_text, cleaned_text.lower())
    
    def test_text_preprocessor_tokenization(self):
        """Test text tokenization."""
        text = "Hello world! This is a test."
        tokens = self.preprocessor.tokenize_simple(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # All tokens should be strings
        self.assertTrue(all(isinstance(token, str) for token in tokens))
    
    def test_text_preprocessor_vocabulary_creation(self):
        """Test vocabulary creation."""
        vocab = self.preprocessor.create_vocabulary(self.sample_texts)
        
        # Should have special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for token in special_tokens:
            self.assertIn(token, vocab)
        
        # Should have reasonable size
        self.assertGreater(len(vocab), 10)
    
    def test_text_preprocessor_encoding(self):
        """Test text encoding."""
        vocab = self.preprocessor.create_vocabulary(self.sample_texts)
        encoded_sequences = self.preprocessor.encode_texts(
            self.sample_texts[:2], 
            vocab
        )
        
        self.assertEqual(len(encoded_sequences), 2)
        
        # All encoded sequences should contain integers
        for seq in encoded_sequences:
            self.assertTrue(all(isinstance(token_id, int) for token_id in seq))
    
    def test_sequence_processor_padding(self):
        """Test sequence padding."""
        sequences = [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10]
        ]
        
        padded = self.sequence_processor.pad_sequences(sequences)
        
        # Should be a tensor
        self.assertIsInstance(padded, torch.Tensor)
        
        # All sequences should have same length
        self.assertEqual(padded.shape[0], len(sequences))
        expected_length = max(len(seq) for seq in sequences)
        self.assertEqual(padded.shape[1], expected_length)
    
    def test_sequence_processor_attention_mask(self):
        """Test attention mask creation."""
        sequences = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0]
        ])
        
        mask = self.sequence_processor.create_attention_mask(sequences, pad_token_id=0)
        
        # Check mask shape
        self.assertEqual(mask.shape, sequences.shape)
        
        # Check mask values
        expected_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0]
        ])
        torch.testing.assert_close(mask, expected_mask)
    
    def test_sequence_processor_position_ids(self):
        """Test position ID creation."""
        sequences = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        
        position_ids = self.sequence_processor.create_position_ids(sequences)
        
        # Check shape
        self.assertEqual(position_ids.shape, sequences.shape)
        
        # Check values
        expected_positions = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ])
        torch.testing.assert_close(position_ids, expected_positions)
    
    def test_dataset_builder(self):
        """Test dataset building."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        builder = DatasetBuilder(config)
        
        dataset = builder.build_from_texts(
            self.sample_texts[:3],
            test_split=0.2,
            validation_split=0.2
        )
        
        # Check dataset structure
        expected_keys = ['train', 'validation', 'test', 'vocab', 'vocab_size']
        for key in expected_keys:
            self.assertIn(key, dataset)
        
        # Check data types
        self.assertIsInstance(dataset['train'], torch.Tensor)
        self.assertIsInstance(dataset['vocab'], dict)
        self.assertIsInstance(dataset['vocab_size'], int)
    
    def test_dataset_builder_save_load(self):
        """Test dataset save/load functionality."""
        config = ModelConfig(**TEST_CONFIGS['small'])
        builder = DatasetBuilder(config)
        
        dataset = builder.build_from_texts(self.sample_texts[:2])
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = f.name
        
        try:
            # Save dataset
            builder.save_dataset(dataset, save_path)
            
            # Load dataset
            loaded_dataset = builder.load_dataset(save_path)
            
            # Check that key components are preserved
            self.assertEqual(dataset['vocab_size'], loaded_dataset['vocab_size'])
            torch.testing.assert_close(dataset['train'], loaded_dataset['train'])
        finally:
            os.unlink(save_path)


class TestMetrics(unittest.TestCase):
    """Test metrics computation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.attention_metrics = AttentionMetrics()
        self.encoding_metrics = EncodingMetrics()
        self.performance_metrics = PerformanceMetrics()
    
    def test_attention_entropy(self):
        """Test attention entropy computation."""
        # Create sample attention weights
        attention_weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        entropy_results = self.attention_metrics.compute_attention_entropy(attention_weights)
        
        # Check result structure
        self.assertIn('entropy', entropy_results)
        self.assertIn('mean_entropy', entropy_results)
        self.assertIn('std_entropy', entropy_results)
        
        # Check values are reasonable
        mean_entropy = entropy_results['mean_entropy']
        self.assertGreater(mean_entropy, 0)
        self.assertLess(mean_entropy, 10)  # Reasonable upper bound
    
    def test_attention_distance(self):
        """Test attention distance computation."""
        attention_weights = torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        
        distance_results = self.attention_metrics.compute_attention_distance(attention_weights)
        
        self.assertIn('attention_distances', distance_results)
        self.assertIn('mean_distance', distance_results)
        
        mean_distance = distance_results['mean_distance']
        self.assertGreaterEqual(mean_distance, 0)
        self.assertLess(mean_distance, 8)  # Max possible distance
    
    def test_attention_sparsity(self):
        """Test attention sparsity computation."""
        attention_weights = torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        
        sparsity_results = self.attention_metrics.compute_attention_sparsity(attention_weights)
        
        self.assertIn('gini_coefficient', sparsity_results)
        self.assertIn('l1_l2_ratio', sparsity_results)
        self.assertIn('effective_span', sparsity_results)
        
        # Gini coefficient should be between 0 and 1
        gini = sparsity_results['gini_coefficient']
        self.assertGreaterEqual(gini, 0)
        self.assertLessEqual(gini, 1)
    
    def test_head_similarity(self):
        """Test attention head similarity computation."""
        attention_weights = torch.softmax(torch.randn(2, 6, 8, 8), dim=-1)
        
        similarities = self.attention_metrics.compute_head_similarity(attention_weights)
        
        # Should be square matrix of head similarities
        self.assertEqual(similarities.shape, (2, 6, 6))
        
        # Diagonal should be zero (self-similarity not computed)
        diagonal = torch.diagonal(similarities, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))
    
    def test_attention_pattern_analysis(self):
        """Test comprehensive attention pattern analysis."""
        attention_weights = torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        
        analysis = self.attention_metrics.analyze_attention_patterns(attention_weights)
        
        # Check all analysis components
        expected_keys = ['entropy', 'distance', 'sparsity', 'head_similarity', 'attention_sparsity']
        for key in expected_keys:
            if key in analysis:  # Some keys might be optional
                self.assertIsInstance(analysis[key], dict)
    
    def test_position_similarity(self):
        """Test position similarity computation."""
        encoding_matrix = torch.randn(32, 64)
        
        similarities = self.encoding_metrics.compute_position_similarity(encoding_matrix)
        
        # Should be square matrix
        self.assertEqual(similarities.shape, (32, 32))
        
        # Diagonal should be 1 (cosine similarity of vector with itself)
        diagonal = torch.diag(similarities)
        expected_diagonal = torch.ones_like(diagonal)
        torch.testing.assert_close(diagonal, expected_diagonal, atol=1e-6)
    
    def test_encoding_quality(self):
        """Test encoding quality metrics."""
        encoding_matrix = torch.randn(32, 64)
        
        quality_metrics = self.encoding_metrics.compute_encoding_quality(encoding_matrix)
        
        # Check expected metrics
        expected_keys = [
            'distinguishability', 
            'encoding_variance', 
            'dimension_utilization', 
            'periodicity_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, quality_metrics)
            self.assertIsInstance(quality_metrics[key], float)
    
    def test_encoding_comparison(self):
        """Test encoding method comparison."""
        encoding_matrices = {
            'method1': torch.randn(32, 64),
            'method2': torch.randn(32, 64),
            'method3': torch.randn(32, 64)
        }
        
        comparison_results = self.encoding_metrics.compare_encodings(encoding_matrices)
        
        # Should have results for each method
        for method_name in encoding_matrices.keys():
            self.assertIn(method_name, comparison_results)
        
        # Should have pairwise similarities
        self.assertIn('pairwise_similarities', comparison_results)
    
    def test_performance_timing(self):
        """Test performance timing functionality."""
        def dummy_function(x):
            return torch.sum(x ** 2)
        
        timing_results = self.performance_metrics.measure_encoding_time(
            dummy_function,
            sequence_lengths=[8, 16, 32],
            num_runs=5
        )
        
        # Should have results for each sequence length
        for seq_len in [8, 16, 32]:
            self.assertIn(seq_len, timing_results)
            
            result = timing_results[seq_len]
            self.assertIn('mean_time', result)
            self.assertIn('std_time', result)
            
            # Times should be positive
            self.assertGreater(result['mean_time'], 0)


class TestExportUtils(unittest.TestCase):
    """Test export utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.viz_config = VisualizationConfig()
        self.figure_exporter = FigureExporter(self.viz_config)
        self.data_exporter = DataExporter()
    
    def test_figure_exporter_matplotlib(self):
        """Test matplotlib figure export."""
        import matplotlib.pyplot as plt
        
        # Create sample figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title('Test Figure')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name
        
        try:
            # Export figure
            result_path = self.figure_exporter.export_matplotlib_figure(fig, save_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(result_path, save_path)
        finally:
            plt.close(fig)
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_data_exporter_attention_weights(self):
        """Test attention weights export."""
        attention_weights = [
            torch.softmax(torch.randn(1, 4, 8, 8), dim=-1),
            torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        ]
        tokens = ['token1', 'token2', 'token3', 'token4', 'token5', 'token6', 'token7', 'token8']
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            save_path = f.name
        
        try:
            # Export attention weights
            result_path = self.data_exporter.export_attention_weights(
                attention_weights, 
                save_path, 
                tokens=tokens,
                format='hdf5'
            )
            
            # Check file was created
            self.assertTrue(os.path.exists(result_path))
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_data_exporter_metrics(self):
        """Test metrics export."""
        metrics = {
            'accuracy': 0.95,
            'loss': 0.05,
            'attention_entropy': [2.1, 2.3, 1.8],
            'nested_metrics': {
                'precision': 0.92,
                'recall': 0.89
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_path = f.name
        
        try:
            # Export metrics
            result_path = self.data_exporter.export_metrics(metrics, save_path)
            
            # Check file was created and is valid JSON
            self.assertTrue(os.path.exists(result_path))
            
            with open(result_path, 'r') as f:
                loaded_metrics = json.load(f)
                
            # Check that data was preserved
            self.assertEqual(loaded_metrics['accuracy'], 0.95)
            self.assertIn('export_date', loaded_metrics)
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_report_generator(self):
        """Test report generation."""
        report_generator = ReportGenerator(self.viz_config)
        
        # Sample analysis results and figures
        analysis_results = {
            'model_config': {'d_model': 64, 'n_heads': 4},
            'attention_metrics': {'mean_entropy': 2.1, 'sparsity': 0.3}
        }
        
        figures = {}  # Empty figures for testing
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            save_path = f.name
        
        try:
            # Generate report
            result_path = report_generator.generate_html_report(
                analysis_results,
                figures,
                save_path,
                title="Test Report"
            )
            
            # Check file was created
            self.assertTrue(os.path.exists(result_path))
            
            # Check that it's valid HTML
            with open(result_path, 'r') as f:
                content = f.read()
                self.assertIn('<html', content.lower())
                self.assertIn('test report', content.lower())
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiling utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.profiler = PerformanceProfiler(device='cpu')
    
    def test_basic_profiling(self):
        """Test basic performance profiling."""
        def test_function():
            # Simple computation
            x = torch.randn(100, 100)
            return torch.mm(x, x.t())
        
        with self.profiler.profile("test_operation"):
            result = test_function()
        
        # Check that profiling recorded the operation
        self.assertGreater(len(self.profiler.results), 0)
        
        latest_result = self.profiler.results[-1]
        self.assertEqual(latest_result.operation_name, "test_operation")
        self.assertGreater(latest_result.execution_time, 0)
    
    def test_profile_function_wrapper(self):
        """Test function profiling wrapper."""
        def compute_sum(tensor):
            return torch.sum(tensor)
        
        test_tensor = torch.randn(1000, 1000)
        result, profiling_result = self.profiler.profile_function(compute_sum, test_tensor)
        
        # Check that function executed correctly
        self.assertIsInstance(result, torch.Tensor)
        
        # Check profiling result
        self.assertIsNotNone(profiling_result)
        self.assertGreater(profiling_result.execution_time, 0)
    
    def test_profiling_summary(self):
        """Test profiling summary generation."""
        # Profile multiple operations
        operations = ['op1', 'op2', 'op1']  # op1 appears twice
        
        for op_name in operations:
            with self.profiler.profile(op_name):
                torch.randn(100, 100)
        
        summary = self.profiler.get_summary()
        
        # Check summary structure
        self.assertIn('op1', summary)
        self.assertIn('op2', summary)
        
        # Check that op1 has count of 2
        self.assertEqual(summary['op1']['count'], 2)
        self.assertEqual(summary['op2']['count'], 1)
        
        # Check summary metrics
        for op_summary in summary.values():
            self.assertIn('mean_time', op_summary)
            self.assertIn('total_time', op_summary)
    
    def test_model_profiler(self):
        """Test model-specific profiling."""
        from models import TransformerEncoder
        
        config = ModelConfig(**TEST_CONFIGS['small'])
        model = TransformerEncoder(config)
        model_profiler = ModelProfiler(model, device='cpu')
        
        def input_generator(batch_size, seq_len):
            return torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Profile forward passes
        results = model_profiler.profile_forward_pass(
            input_generator,
            batch_sizes=[1, 2],
            sequence_lengths=[8, 16]
        )
        
        # Should have results for each combination
        expected_ops = ['forward_batch1_seq8', 'forward_batch1_seq16', 
                       'forward_batch2_seq8', 'forward_batch2_seq16']
        
        for op_name in expected_ops:
            self.assertIn(op_name, results)
            self.assertGreater(results[op_name].execution_time, 0)
    
    def test_profiler_clear_results(self):
        """Test clearing profiler results."""
        # Add some results
        with self.profiler.profile("test_op"):
            pass
        
        self.assertGreater(len(self.profiler.results), 0)
        
        # Clear results
        self.profiler.clear_results()
        self.assertEqual(len(self.profiler.results), 0)
    
    def test_profiler_save_results(self):
        """Test saving profiler results."""
        # Profile an operation
        with self.profiler.profile("save_test_op"):
            torch.randn(50, 50)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_path = f.name
        
        try:
            # Save results
            self.profiler.save_results(save_path)
            
            # Check file was created and is valid
            self.assertTrue(os.path.exists(save_path))
            
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            # Check structure
            self.assertIn('results', saved_data)
            self.assertIn('summary', saved_data)
            self.assertIn('device', saved_data)
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestUtilsIntegration(unittest.TestCase):
    """Test integration between utility components."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_test_seed()
        self.config = ModelConfig(**TEST_CONFIGS['small'])
    
    def test_tokenizer_data_preprocessing_integration(self):
        """Test integration between tokenizer and data preprocessing."""
        # Create tokenizer and build vocabulary
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(SAMPLE_TEXTS[:3])
        
        # Use with data preprocessing
        preprocessor = TextPreprocessor()
        sequence_processor = SequenceProcessor()
        
        # Preprocess and tokenize
        clean_texts = [preprocessor.clean_text(text) for text in SAMPLE_TEXTS[:2]]
        encoded_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in clean_texts]
        
        # Pad sequences
        padded_sequences = sequence_processor.pad_sequences(encoded_sequences)
        
        # Should work without errors
        self.assertIsInstance(padded_sequences, torch.Tensor)
        self.assertEqual(padded_sequences.shape[0], len(clean_texts))
    
    def test_metrics_visualization_integration(self):
        """Test integration between metrics and visualization."""
        # Create sample data
        attention_weights = torch.softmax(torch.randn(1, 4, 8, 8), dim=-1)
        
        # Compute metrics
        attention_metrics = AttentionMetrics()
        analysis = attention_metrics.analyze_attention_patterns(attention_weights)
        
        # Should be able to use metrics in visualization
        # (This is more of an interface test)
        self.assertIsInstance(analysis, dict)
        self.assertIn('entropy', analysis)
    
    def test_export_profiler_integration(self):
        """Test integration between export utilities and profiler."""
        # Profile some operations
        profiler = PerformanceProfiler()
        
        with profiler.profile("export_test_op"):
            torch.randn(100, 100)
        
        # Export profiling results
        data_exporter = DataExporter()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_path = f.name
        
        try:
            # Convert profiler results to exportable format
            summary = profiler.get_summary()
            data_exporter.export_metrics(summary, save_path)
            
            # Should work without errors
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
