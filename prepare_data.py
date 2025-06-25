# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4
# Preparing training data script

# ------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from typing import Tuple
from datasets import load_dataset
import tiktoken

class Tokenizer():
    def __init__(self, vocab: list):
        """
        Base tokenizer class that provides common functionality for encoding and decoding.

        Parameters
        ----------
        vocab : list
            A list of tokens that form the vocabulary for tokenization.
        """

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list:
        """
        Encodes text into a list of token indices.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        list
            A list of token indices.
        """

        raise NotImplementedError("Subclasses should implement this method")

    def decode(self, tokens: list) -> str:
        """
        Decodes a list of token indices back into text.

        Parameters
        ----------
        tokens : list
            A list of token indices to decode.

        Returns
        -------
        str
            The decoded text.
        """

        raise NotImplementedError("Subclasses should implement this method")


class CharTokenizer(Tokenizer):
    def __init__(self, text: str):
        """ 
        Manually initializes the tokenizer with a given vocabulary.

        Parameters
        ----------
        vocab : list
            A list of characters that form the vocabulary for tokenization.
        """

        self.vocab = sorted(list(set(text))) # Create a vocabulary from the text
        super().__init__(self.vocab)

    def encode(self, text: str) -> list:
        """
        Encodes text into a list of token indices.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        list
            A list of token indices.
        """

        return [self.stoi[c] for c in text]  

    def decode(self, tokens: list) -> str:
        """
        Decodes a list of token indices back into text.

        Parameters
        ----------
        tokens : list
            A list of token indices to decode.

        Returns
        -------
        str
            The decoded text.
        """

        return ''.join([self.itos[i] for i in tokens])
    
class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int = 64):
        """
        Creates a character-level dataset for language modeling.

        Parameters
        ----------
        text : str
            The input text data to create the dataset from.

        block_size : int, optional
            The size of each input sequence (default is 64).
        """

        self.block_size = block_size
        self.tokenizer = CharTokenizer(text)
        self.data = self.tokenizer.encode(text)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """

        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input and target sequences for a given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the input sequence (x) and the target sequence (y).
        """

        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

class TikTokenTokenizer(Tokenizer):
    def __init__(self, text: str):
        """
        Tokenizer that uses tiktoken encoding but with a limited vocabulary.

        Parameters
        ----------
        text : str
            The text to initialize the tokenizer with.
        """

        self.tiktoken_enc = tiktoken.get_encoding("gpt2")
        
        # Create a limited vocabulary from the text
        unique_tokens = set(self.tiktoken_enc.encode(text))
        vocab_tokens = [self.tiktoken_enc.decode([token]) for token in unique_tokens]
        
        # Add special tokens
        special_tokens = ['<UNK>', '<PAD>']
        vocab_tokens = special_tokens + vocab_tokens

        self.vocab = vocab_tokens
        
        super().__init__(self.vocab)

    def encode(self, text: str) -> list:
        """
        Encodes text into a list of token indices using tiktoken encoding.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        list
            A list of token indices.
        """

        # Encode using tiktoken and map to limited vocabulary
        orig_token_idx = self.tiktoken_enc.encode(text)
        text_tokens = [self.tiktoken_enc.decode([idx]) for idx in orig_token_idx]

        return [self.stoi.get(text_token, self.stoi['<UNK>']) for text_token in text_tokens]

    def decode(self, tokens: list) -> str:
        """
        Decodes a list of token indices back into text using tiktoken decoding.

        Parameters
        ----------
        tokens : list
            A list of token indices to decode.

        Returns
        -------
        str
            The decoded text.
        """

        # Map limited vocabulary indices back to original tokens
        text_tokens = [self.itos[token] for token in tokens]

        return ''.join(text_tokens)

class TinyStoriesDataset(Dataset):
    def __init__(self, split: str = 'train', block_size: int = 64, max_samples: int = None):
        """
        Creates a dataset from TinyStories using tiktoken tokenization with limited vocabulary.

        Parameters
        ----------
        split : str, optional
            The dataset split to use ('train' or 'validation', default is 'train').
        block_size : int, optional
            The size of each input sequence (default is 64).
        max_samples : int, optional
            Maximum number of samples to load (default is None for all samples).
        """

        self.block_size = block_size
        
        # Load TinyStories dataset
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        
        # Limit samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        # Sample some text to build vocabulary
        sample_size = min(1000, len(self.dataset))
        sample_texts = [self.dataset[i]['text'] for i in range(sample_size)]
        sample_text = '\n'.join(sample_texts)
        
        # Initialize tokenizer with vocabulary from sample
        self.tokenizer = TikTokenTokenizer(sample_text)
        
        print(f"TinyStoriesDataset loaded with {len(self.dataset)} stories and vocabulary size {self.tokenizer.vocab_size}")

    def __len__(self) -> int:
        """
        Returns the number of stories in the dataset.
        """
        
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input and target sequences for a given story index.

        Parameters
        ----------
        idx : int
            The index of the story to retrieve from the TinyStories dataset.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the input sequence (x) and the target sequence (y).
        """
        
        # Get the story text at the given index
        story_text = self.dataset[idx]['text']
        
        # Tokenize the story
        tokens = self.tokenizer.encode(story_text)
        
        # If story is shorter than block_size, pad it
        if len(tokens) < self.block_size + 1:
            print(f"Warning: Story at index {idx} is shorter than block_size + 1. Padding with <PAD> token.")
            # Pad with the PAD token index
            pad_token_idx = self.tokenizer.stoi['<PAD>']
            tokens.extend([pad_token_idx] * (self.block_size + 1 - len(tokens)))
        
        # Take only block_size + 1 tokens (for input and target)
        chunk = tokens[:self.block_size + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # Input sequence
        y = torch.tensor(chunk[1:], dtype=torch.long)   # Target sequence (shifted by 1)
        
        return x, y



