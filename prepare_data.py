# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4
# Preparing training data script

# ------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

class Tokenizer:
    def __init__(self, vocab: list):
        """ 
        Manually initializes the tokenizer with a given vocabulary.

        Parameters
        ----------
        vocab : list
            A list of characters that form the vocabulary for tokenization.
        """

        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list) -> str:
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

        chars = sorted(list(set(text)))
        self.tokenizer = Tokenizer(chars)
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

if __name__ == "__main__":
    # input_path = Path("tiny_shakespeare.txt")
    # text = input_path.read_text(encoding="utf-8")

    text = """
        Alice was beginning to get very tired of sitting by her sister on the
        bank, and of having nothing to do: once or twice she had peeped into the
        book her sister was reading, but it had no pictures or conversations in
        it, 'and what is the use of a book,' thought Alice 'without pictures or
        conversation?'
        So she was considering in her own mind (as well as she could, for the
        hot day made her feel very sleepy and stupid), whether the pleasure
        of making a daisy-chain would be worth the trouble of getting up and
        picking the daisies, when suddenly a White Rabbit with pink eyes ran
        close by her.
        """

    dataset = CharDataset(text)
    torch.save(dataset, "./dataset/alice_sample_dataset.pt")
    
    print("Saved dataset with vocab size:", dataset.vocab_size)