"""Tokenization utilities for text processing."""

import torch
from typing import List, Dict, Tuple, Optional, Union, Set
import re
from collections import defaultdict, Counter
import json
from pathlib import Path
import pickle


class SimpleTokenizer:
    """Simple whitespace-based tokenizer with basic preprocessing."""
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        lowercase: bool = True,
        add_special_tokens: bool = True
    ):
        self.lowercase = lowercase
        self.add_special_tokens = add_special_tokens
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        if vocab is not None:
            self.vocab = vocab
            self.reverse_vocab = {v: k for k, v in vocab.items()}
        else:
            self.vocab = self.special_tokens.copy()
            self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        # Simple whitespace tokenization with basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
        tokens = text.split()
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: Optional[bool] = None) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Override class setting for special tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.vocab['<BOS>'])
        
        for token in tokens:
            token_id = self.vocab.get(token, self.vocab['<UNK>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.vocab['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id, '<UNK>')
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab_size: Optional[int] = None):
        """Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for inclusion
            max_vocab_size: Maximum vocabulary size
        """
        # Tokenize all texts and count frequencies
        token_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # Filter by minimum frequency
        filtered_tokens = {
            token: count for token, count in token_counts.items()
            if count >= min_freq
        }
        
        # Sort by frequency
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        if max_vocab_size:
            sorted_tokens = sorted_tokens[:max_vocab_size - len(self.special_tokens)]
        
        # Build vocabulary
        self.vocab = self.special_tokens.copy()
        
        for i, (token, _) in enumerate(sorted_tokens):
            self.vocab[token] = i + len(self.special_tokens)
        
        # Update reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def save(self, path: str):
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        tokenizer_data = {
            'vocab': self.vocab,
            'lowercase': self.lowercase,
            'add_special_tokens': self.add_special_tokens,
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
            
        Returns:
            Loaded tokenizer instance
        """
        with open(path, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(
            vocab=tokenizer_data['vocab'],
            lowercase=tokenizer_data['lowercase'],
            add_special_tokens=tokenizer_data['add_special_tokens']
        )
        
        return tokenizer
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab.copy()


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Get initial word-level tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens with character separation
        """
        words = text.lower().split()
        word_tokens = []
        
        for word in words:
            # Add end-of-word marker and split into characters
            word_chars = list(word) + ['</w>']
            word_tokens.append(word_chars)
        
        return word_tokens
    
    def _get_pairs(self, word_tokens: List[List[str]]) -> Counter:
        """Get all adjacent pairs and their frequencies.
        
        Args:
            word_tokens: List of tokenized words
            
        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()
        
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_tokens: List[List[str]]) -> List[List[str]]:
        """Merge the most frequent pair in vocabulary.
        
        Args:
            pair: Pair to merge
            word_tokens: Current word tokens
            
        Returns:
            Updated word tokens with merged pairs
        """
        new_word_tokens = []
        
        for word in word_tokens:
            new_word = []
            i = 0
            
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    # Merge the pair
                    merged_token = word[i] + word[i + 1]
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_tokens.append(new_word)
        
        return new_word_tokens
    
    def train(self, texts: List[str]):
        """Train BPE tokenizer on texts.
        
        Args:
            texts: List of training texts
        """
        # Initialize with character-level vocabulary
        all_chars = set()
        word_tokens = []
        
        for text in texts:
            tokens = self._get_word_tokens(text)
            word_tokens.extend(tokens)
            
            for word in tokens:
                all_chars.update(word)
        
        # Build initial vocabulary
        self.vocab = self.special_tokens.copy()
        
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Perform BPE merges
        self.merges = []
        
        for _ in range(self.vocab_size - len(self.vocab)):
            pairs = self._get_pairs(word_tokens)
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Merge the pair
            word_tokens = self._merge_vocab(best_pair, word_tokens)
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = len(self.vocab)
            self.merges.append(best_pair)
        
        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _apply_bpe_to_word(self, word: str) -> List[str]:
        """Apply BPE merges to a single word.
        
        Args:
            word: Input word
            
        Returns:
            List of BPE tokens
        """
        # Start with character-level tokens
        tokens = list(word) + ['</w>']
        
        # Apply each merge in order
        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge_pair:
                    # Apply merge
                    merged_token = tokens[i] + tokens[i + 1]
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using trained BPE.
        
        Args:
            text: Input text
            
        Returns:
            List of BPE tokens
        """
        words = text.lower().split()
        all_tokens = []
        
        for word in words:
            word_tokens = self._apply_bpe_to_word(word)
            all_tokens.extend(word_tokens)
        
        return all_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.vocab['<BOS>'])
        
        for token in tokens:
            token_id = self.vocab.get(token, self.vocab['<UNK>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.vocab['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id, '<UNK>')
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = text.strip()
        
        return text


class WordPieceTokenizer:
    """WordPiece tokenizer similar to BERT."""
    
    def __init__(self, vocab_size: int = 30000, unk_token: str = '[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def train(self, texts: List[str]):
        """Train WordPiece tokenizer.
        
        Args:
            texts: List of training texts
        """
        # Initialize vocabulary with special tokens and characters
        self.vocab = self.special_tokens.copy()
        
        # Collect all characters
        all_chars = set()
        for text in texts:
            cleaned_text = self._clean_text(text)
            all_chars.update(cleaned_text)
        
        # Add characters to vocabulary
        for char in sorted(all_chars):
            if char not in self.vocab and char != ' ':
                self.vocab[char] = len(self.vocab)
        
        # Collect word frequencies
        word_counts = Counter()
        for text in texts:
            words = self._clean_text(text).split()
            word_counts.update(words)
        
        # Generate subword candidates
        subword_counts = Counter()
        
        for word, count in word_counts.items():
            # Generate all possible subwords
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    subword = word[i:j]
                    if i > 0:
                        subword = '##' + subword  # WordPiece continuation marker
                    subword_counts[subword] += count
        
        # Select top subwords
        sorted_subwords = sorted(subword_counts.items(), key=lambda x: x[1], reverse=True)
        
        for subword, _ in sorted_subwords:
            if len(self.vocab) >= self.vocab_size:
                break
            
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)
        
        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using WordPiece algorithm.
        
        Args:
            word: Input word
            
        Returns:
            List of WordPiece tokens
        """
        if len(word) == 0:
            return []
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # Find longest subword match
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                
                end -= 1
            
            if cur_substr is None:
                # Cannot tokenize, return UNK
                return [self.unk_token]
            
            tokens.append(cur_substr)
            start = end
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using WordPiece.
        
        Args:
            text: Input text
            
        Returns:
            List of WordPiece tokens
        """
        cleaned_text = self._clean_text(text)
        words = cleaned_text.split()
        
        all_tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            all_tokens.extend(word_tokens)
        
        return all_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.vocab['[CLS]'])
        
        for token in tokens:
            token_id = self.vocab.get(token, self.vocab['[UNK]'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.vocab['[SEP]'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id, '[UNK]')
            
            if skip_special_tokens and token in ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']:
                continue
            
            tokens.append(token)
        
        # Join tokens and clean WordPiece markers
        text = ' '.join(tokens)
        text = text.replace(' ##', '')
        
        return text


def create_tokenizer(
    tokenizer_type: str,
    vocab_size: int = 10000,
    **kwargs
) -> Union[SimpleTokenizer, BPETokenizer, WordPieceTokenizer]:
    """Factory function to create tokenizers.
    
    Args:
        tokenizer_type: Type of tokenizer ('simple', 'bpe', 'wordpiece')
        vocab_size: Vocabulary size
        **kwargs: Additional tokenizer arguments
        
    Returns:
        Tokenizer instance
    """
    if tokenizer_type == 'simple':
        return SimpleTokenizer(**kwargs)
    elif tokenizer_type == 'bpe':
        return BPETokenizer(vocab_size=vocab_size, **kwargs)
    elif tokenizer_type == 'wordpiece':
        return WordPieceTokenizer(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
