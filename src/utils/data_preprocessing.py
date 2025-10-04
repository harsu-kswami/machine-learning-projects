"""Data preprocessing utilities for text and sequence handling."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, Iterator
import re
import string
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path

from config import ModelConfig


class TextPreprocessor:
    """Text preprocessing utilities for transformer input preparation."""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # Common preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        """Clean raw text for processing.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' [EMAIL] ', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        cleaned_text = self.clean_text(text)
        return cleaned_text.split()
    
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """Preprocess batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tokenized texts
        """
        return [self.tokenize_simple(text) for text in texts]
    
    def create_vocabulary(
        self, 
        texts: List[str], 
        min_freq: int = 2,
        max_vocab_size: Optional[int] = None
    ) -> Dict[str, int]:
        """Create vocabulary from texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for inclusion
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            Vocabulary dictionary mapping tokens to IDs
        """
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_simple(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        filtered_tokens = {
            token: count for token, count in token_counts.items() 
            if count >= min_freq
        }
        
        # Sort by frequency (descending)
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        if max_vocab_size:
            sorted_tokens = sorted_tokens[:max_vocab_size - 4]  # Reserve space for special tokens
        
        # Create vocabulary with special tokens
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Add regular tokens
        for i, (token, _) in enumerate(sorted_tokens):
            vocab[token] = i + 4
        
        return vocab
    
    def encode_texts(
        self, 
        texts: List[str], 
        vocab: Dict[str, int],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[List[int]]:
        """Encode texts to token IDs.
        
        Args:
            texts: List of text strings
            vocab: Vocabulary dictionary
            max_length: Maximum sequence length
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of encoded sequences
        """
        encoded_sequences = []
        
        for text in texts:
            tokens = self.tokenize_simple(text)
            
            # Convert tokens to IDs
            token_ids = []
            if add_special_tokens:
                token_ids.append(vocab['<BOS>'])
            
            for token in tokens:
                token_id = vocab.get(token, vocab['<UNK>'])
                token_ids.append(token_id)
            
            if add_special_tokens:
                token_ids.append(vocab['<EOS>'])
            
            # Truncate if necessary
            if max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if add_special_tokens:
                    token_ids[-1] = vocab['<EOS>']  # Ensure EOS at end
            
            encoded_sequences.append(token_ids)
        
        return encoded_sequences


class SequenceProcessor:
    """Utilities for processing sequences for transformer input."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def pad_sequences(
        self, 
        sequences: List[List[int]], 
        max_length: Optional[int] = None,
        padding: str = 'post'
    ) -> torch.Tensor:
        """Pad sequences to uniform length.
        
        Args:
            sequences: List of token ID sequences
            max_length: Maximum length (auto-detect if None)
            padding: 'pre' or 'post' padding
            
        Returns:
            Padded tensor of shape (batch_size, max_length)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) >= max_length:
                # Truncate
                padded_seq = seq[:max_length]
            else:
                # Pad
                pad_length = max_length - len(seq)
                if padding == 'post':
                    padded_seq = seq + [self.pad_token_id] * pad_length
                else:  # 'pre'
                    padded_seq = [self.pad_token_id] * pad_length + seq
            
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences, dtype=torch.long)
    
    def create_attention_mask(
        self, 
        sequences: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Create attention mask for padded sequences.
        
        Args:
            sequences: Padded sequences tensor
            pad_token_id: Padding token ID
            
        Returns:
            Attention mask tensor (1 for real tokens, 0 for padding)
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        return (sequences != pad_token_id).long()
    
    def create_position_ids(self, sequences: torch.Tensor) -> torch.Tensor:
        """Create position IDs for sequences.
        
        Args:
            sequences: Input sequences tensor
            
        Returns:
            Position IDs tensor
        """
        batch_size, seq_len = sequences.shape
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        return position_ids
    
    def create_batches(
        self,
        sequences: List[List[int]],
        batch_size: int,
        shuffle: bool = True
    ) -> Iterator[torch.Tensor]:
        """Create batches from sequences.
        
        Args:
            sequences: List of token sequences
            batch_size: Batch size
            shuffle: Whether to shuffle sequences
            
        Yields:
            Batched and padded tensors
        """
        if shuffle:
            import random
            sequences = sequences.copy()
            random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            padded_batch = self.pad_sequences(batch_sequences)
            yield padded_batch
    
    def split_sequences(
        self,
        sequences: List[List[int]],
        window_size: int,
        stride: int = None
    ) -> List[List[int]]:
        """Split long sequences into smaller windows.
        
        Args:
            sequences: List of token sequences
            window_size: Size of each window
            stride: Stride between windows (defaults to window_size)
            
        Returns:
            List of windowed sequences
        """
        if stride is None:
            stride = window_size
        
        windowed_sequences = []
        
        for seq in sequences:
            if len(seq) <= window_size:
                windowed_sequences.append(seq)
            else:
                for start_idx in range(0, len(seq) - window_size + 1, stride):
                    window = seq[start_idx:start_idx + window_size]
                    windowed_sequences.append(window)
        
        return windowed_sequences


class DatasetBuilder:
    """Build datasets for transformer training and evaluation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.text_preprocessor = TextPreprocessor()
        self.sequence_processor = SequenceProcessor(pad_token_id=config.pad_token_id)
        
    def build_from_texts(
        self,
        texts: List[str],
        test_split: float = 0.1,
        validation_split: float = 0.1,
        min_seq_length: int = 5,
        max_seq_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Build dataset from raw texts.
        
        Args:
            texts: List of text strings
            test_split: Fraction for test set
            validation_split: Fraction for validation set
            min_seq_length: Minimum sequence length
            max_seq_length: Maximum sequence length
            
        Returns:
            Dictionary containing train/val/test tensors
        """
        if max_seq_length is None:
            max_seq_length = self.config.max_seq_len
        
        # Create vocabulary
        vocab = self.text_preprocessor.create_vocabulary(
            texts, 
            max_vocab_size=self.config.vocab_size
        )
        
        # Encode texts
        encoded_sequences = self.text_preprocessor.encode_texts(
            texts, vocab, max_length=max_seq_length
        )
        
        # Filter by length
        filtered_sequences = [
            seq for seq in encoded_sequences
            if min_seq_length <= len(seq) <= max_seq_length
        ]
        
        # Split into train/val/test
        n_sequences = len(filtered_sequences)
        n_test = int(n_sequences * test_split)
        n_val = int(n_sequences * validation_split)
        n_train = n_sequences - n_test - n_val
        
        train_sequences = filtered_sequences[:n_train]
        val_sequences = filtered_sequences[n_train:n_train + n_val]
        test_sequences = filtered_sequences[n_train + n_val:]
        
        # Pad sequences
        dataset = {
            'train': self.sequence_processor.pad_sequences(train_sequences, max_seq_length),
            'validation': self.sequence_processor.pad_sequences(val_sequences, max_seq_length),
            'test': self.sequence_processor.pad_sequences(test_sequences, max_seq_length),
            'vocab': vocab,
            'vocab_size': len(vocab)
        }
        
        return dataset
    
    def build_position_prediction_dataset(
        self,
        texts: List[str],
        corruption_prob: float = 0.15
    ) -> Dict[str, torch.Tensor]:
        """Build dataset for position prediction task.
        
        Args:
            texts: List of text strings
            corruption_prob: Probability of corrupting position
            
        Returns:
            Dataset for position prediction
        """
        dataset = self.build_from_texts(texts)
        
        # Create corrupted position labels
        for split_name in ['train', 'validation', 'test']:
            sequences = dataset[split_name]
            batch_size, seq_len = sequences.shape
            
            # Create position labels (0 = correct, 1 = corrupted)
            position_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
            
            # Randomly corrupt some positions
            corruption_mask = torch.rand(batch_size, seq_len) < corruption_prob
            position_labels[corruption_mask] = 1
            
            # Apply corruption by shuffling tokens at corrupted positions
            corrupted_sequences = sequences.clone()
            for b in range(batch_size):
                corrupted_positions = torch.where(corruption_mask[b])[0]
                if len(corrupted_positions) > 1:
                    # Shuffle corrupted positions
                    shuffled_indices = torch.randperm(len(corrupted_positions))
                    corrupted_sequences[b, corrupted_positions] = sequences[b, corrupted_positions[shuffled_indices]]
            
            dataset[f'{split_name}_corrupted'] = corrupted_sequences
            dataset[f'{split_name}_labels'] = position_labels
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], save_path: str):
        """Save dataset to disk.
        
        Args:
            dataset: Dataset dictionary
            save_path: Path to save dataset
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for complex objects, JSON for simple data
        if save_path.suffix == '.json':
            # Convert tensors to lists for JSON serialization
            json_dataset = {}
            for key, value in dataset.items():
                if isinstance(value, torch.Tensor):
                    json_dataset[key] = value.tolist()
                else:
                    json_dataset[key] = value
            
            with open(save_path, 'w') as f:
                json.dump(json_dataset, f, indent=2)
        else:
            # Use pickle for full tensor support
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
    
    def load_dataset(self, load_path: str) -> Dict[str, Any]:
        """Load dataset from disk.
        
        Args:
            load_path: Path to dataset file
            
        Returns:
            Loaded dataset dictionary
        """
        load_path = Path(load_path)
        
        if load_path.suffix == '.json':
            with open(load_path, 'r') as f:
                json_dataset = json.load(f)
            
            # Convert lists back to tensors
            dataset = {}
            for key, value in json_dataset.items():
                if isinstance(value, list) and key != 'vocab':
                    dataset[key] = torch.tensor(value)
                else:
                    dataset[key] = value
            
            return dataset
        else:
            with open(load_path, 'rb') as f:
                return pickle.load(f)


# Convenience functions
def preprocess_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> List[str]:
    """Preprocess single text string.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation
        
    Returns:
        List of tokens
    """
    preprocessor = TextPreprocessor(lowercase=lowercase, remove_punctuation=remove_punctuation)
    return preprocessor.tokenize_simple(text)


def create_sequences(
    texts: List[str],
    max_length: int,
    pad_token_id: int = 0
) -> torch.Tensor:
    """Create padded sequences from texts.
    
    Args:
        texts: List of text strings
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        
    Returns:
        Padded sequences tensor
    """
    preprocessor = TextPreprocessor()
    processor = SequenceProcessor(pad_token_id=pad_token_id)
    
    # Simple encoding (token index in vocabulary)
    tokenized_texts = preprocessor.preprocess_batch(texts)
    
    # Create simple vocabulary
    all_tokens = [token for tokens in tokenized_texts for token in tokens]
    vocab = {token: i + 1 for i, token in enumerate(set(all_tokens))}
    vocab['<PAD>'] = pad_token_id
    
    # Encode to IDs
    encoded_sequences = []
    for tokens in tokenized_texts:
        encoded_seq = [vocab.get(token, 0) for token in tokens]
        encoded_sequences.append(encoded_seq)
    
    return processor.pad_sequences(encoded_sequences, max_length)


def build_vocabulary(
    texts: List[str],
    min_freq: int = 2,
    max_vocab_size: Optional[int] = None
) -> Dict[str, int]:
    """Build vocabulary from texts.
    
    Args:
        texts: List of text strings
        min_freq: Minimum token frequency
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        Vocabulary dictionary
    """
    preprocessor = TextPreprocessor()
    return preprocessor.create_vocabulary(texts, min_freq, max_vocab_size)
