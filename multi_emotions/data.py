"""
Data loading, tokenization, and preprocessing for multi-emotion recognition.

SmartTokenizer, Vocabulary, and process_text are identical to those in
`emotions/data.py`, they are text utilities with no dependency on
the specific dataset or number of classes.

NOTE :
    Multi-class vs multi-label
    ---
    Multi-class  : each sample has exactly ONE emotion label (the most common one).
                Labels are integers in range [0, NUM_CLASSES[.
                Loss: CrossEntropyLoss is used

    Multi-label  : each sample CAN have several emotions at once 
                (e.g. "I am happy but also a bit worried").
                Labels are binary vectors of length NUM_CLASSES.
                Loss: BCEWithLogitsLoss (probably).

For now, we assume multi-class (nearly the same setup as the binary model,
just with more classes). A comment marks where we have to change if multi-label 
is needed.
"""

import re
from collections import Counter
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset


class SmartTokenizer:
    """Regex-based tokenizer."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)

class Vocabulary:
    """Mapping between tokens (strings) and indices (integers).

    Parameters
    ----------
    pad_idx : int
        Index reserved for the padding token <pad>.
    unk_idx : int
        Index reserved for unknown tokens <unk>.
    """

    def __init__(self, pad_idx: int = 0, unk_idx: int = 1):
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.token_to_idx = {"<pad>": pad_idx, "<unk>": unk_idx}
        self.idx_to_token = {pad_idx: "<pad>", unk_idx: "<unk>"}

        self.vocab_size = 2  # Start with <pad> and <unk>

    def build_from_samples(self, samples: List[str], max_vocab_size: int = 20000) -> None:
        """Build vocabulary from a list of raw text samples.

        Parameters
        ----------
        samples : List[str]
            Raw text samples (training set or a subset of it).
        max_vocab_size : int
            Maximum number of tokens to keep (most frequent ones).
        """
        tokenizer = SmartTokenizer()
        word_cnt = Counter()

        for text in samples:
            tokens = tokenizer.tokenize(text)
            word_cnt.update(tokens)

        most_common = word_cnt.most_common(max_vocab_size - 2)
        for word, _ in most_common:
            self.token_to_idx[word] = len(self.token_to_idx)
            self.idx_to_token[len(self.idx_to_token)] = word

        self.vocab_size = len(self.token_to_idx)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of token strings to a list of indices."""
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert a list of indices back to token strings."""
        return [self.idx_to_token.get(idx, "<unk>") for idx in indices]


def process_text(
    text: str, vocab: Vocabulary, max_length: int, pad_idx: int = 0,
    ) -> torch.Tensor:
    """Tokenize, encode and pad/truncate a single text to a fixed length.

    Parameters
    ----------
    text : str
        Raw input text.
    vocab : Vocabulary
        Vocabulary used for encoding tokens.
    max_length : int
        Target sequence length after padding / truncation.
    pad_idx : int
        Index to use for padding.

    Returns
    -------
    torch.Tensor
        1-D LongTensor of shape (max_length,).
    """
    tokenizer = SmartTokenizer()
    tokens = tokenizer.tokenize(text)

    # Handle both Vocabulary object and dict
    if isinstance(vocab, dict):
        unk_idx = vocab.get("<unk>", 1)
        ids = [vocab.get(token, unk_idx) for token in tokens]
    else:
        ids = vocab.encode(tokens)

    if len(ids) < max_length:
        ids = ids + [pad_idx] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    return torch.tensor(ids, dtype=torch.long)
