"""Data loading utilities for multi-label emotion recognition.

This module keeps the same tokenization and vocabulary utilities used in the
binary workflow, but it creates multi-hot labels for GoEmotions.
"""

import re
from collections import Counter
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

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


class MultiEmotionDataLoader:
    """Load and preprocess multi-label data from GoEmotions."""

    def __init__(self):
        self.train_split = None
        self.test_split = None
        self.vocab = None
        # Compact 6-label mapping used in this project.
        self.goemotions_id_to_class = {
            2: 0,   # anger
            14: 1,  # fear
            17: 2,  # joy
            25: 3,  # sadness
            26: 4,  # surprise
            27: 5,  # neutral
        }
        self.train_records = []
        self.test_records = []

    def _to_multihot(self, labels: List[int]) -> Optional[List[float]]:
        """Convert GoEmotions label ids into a compact multi-hot vector.

        Parameters
        ----------
        labels : List[int]
            Raw label ids from GoEmotions for one sample.

        Returns
        -------
        Optional[List[float]]
            Multi-hot vector for the mapped labels.
            Returns None if no selected labels exist for this sample.
        """
        vector = [0.0] * len(self.goemotions_id_to_class)
        for raw_label in labels:
            mapped = self.goemotions_id_to_class.get(raw_label)
            if mapped is not None:
                vector[mapped] = 1.0

        if sum(vector) == 0:
            return None
        return vector

    def _prepare_records(self, split) -> List[Tuple[str, List[float]]]:
        """Convert one split into (text, multi_hot_labels) records."""
        records = []
        for row in split:
            label_vector = self._to_multihot(row["labels"])
            if label_vector is None:
                continue
            records.append((row["text"], label_vector))
        return records

    def load_dataset(self, seed: int = 42) -> None:
        """Load the dataset from HuggingFace datasets library.

        Parameters
        ----------
        seed : int
            Random seed for shuffling the dataset.

        Notes
        -----
        This method should be implemented to load the specific dataset from HuggingFace.
        """
        print("Loading GoEmotions dataset...")
        dataset = load_dataset("go_emotions")
        self.train_split = dataset["train"].shuffle(seed=seed)
        self.test_split = dataset["test"].shuffle(seed=seed)
        self.train_records = self._prepare_records(self.train_split)
        self.test_records = self._prepare_records(self.test_split)

        print(
            f"Prepared multi-label samples -> "
            f"Train: {len(self.train_records)}, Test: {len(self.test_records)}"
        )

    def build_vocabulary(
        self,
        num_samples: int = 10000,
        max_vocab_size: int = 20000,
    ) -> Vocabulary:
        """Build vocabulary from training samples.

        Parameters
        ----------
        num_samples : int
            How many training samples to use for building the vocabulary.
        max_vocab_size : int
            Maximum vocabulary size.

        Returns
        -------
        Vocabulary
            The built vocabulary, also stored in self.vocab.
        """
        if not self.train_records:
            raise ValueError("Load dataset first with load_dataset()")

        num_samples = min(num_samples, len(self.train_records))

        print(f"Building vocabulary from {num_samples} samples...")

        samples = [text for text, _ in self.train_records[:num_samples]]

        self.vocab = Vocabulary()
        self.vocab.build_from_samples(samples, max_vocab_size)

        print(f"Vocabulary size: {self.vocab.vocab_size}")
        return self.vocab

    def process_and_create_loaders(
        self,
        max_seq_length: int,
        batch_size: int,
        train_samples: int,
        test_samples: int,
        verbose: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        """Process raw data and return train / test DataLoaders.

        Parameters
        ----------
        max_seq_length : int
            Sequence length after padding / truncation.
        batch_size : int
            Mini-batch size.
        train_samples : int
            Number of training samples to use.
        test_samples : int
            Number of test samples to use.
        verbose : bool
            Print progress every 1000 samples if True.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            (train_loader, test_loader)

        Notes
        -----
        Labels are returned as multi-hot float vectors.
        """
        if self.vocab is None:
            raise ValueError("Call build_vocabulary() first.")

        if not self.train_records or not self.test_records:
            raise ValueError("No records available. Call load_dataset() first.")

        train_samples = min(train_samples, len(self.train_records))
        test_samples = min(test_samples, len(self.test_records))

        print(f"Processing {train_samples} training samples...")
        train_texts, train_labels = [], []

        for i in range(train_samples):
            text, label = self.train_records[i]
            processed = process_text(text, self.vocab, max_seq_length, self.vocab.pad_idx)
            train_texts.append(processed)
            train_labels.append(label)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  {i + 1} processed")

        train_texts = torch.stack(train_texts)
        train_labels = torch.tensor(train_labels, dtype=torch.float)

        print(f"Processing {test_samples} test samples...")
        test_texts, test_labels = [], []

        for i in range(test_samples):
            text, label = self.test_records[i]
            processed = process_text(text, self.vocab, max_seq_length, self.vocab.pad_idx)
            test_texts.append(processed)
            test_labels.append(label)
            if verbose and (i + 1) % 500 == 0:
                print(f"  {i + 1} processed")

        test_texts = torch.stack(test_texts)
        test_labels = torch.tensor(test_labels, dtype=torch.float)

        train_loader = DataLoader(
            TensorDataset(train_texts, train_labels),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(test_texts, test_labels),
            batch_size=batch_size,
            shuffle=False,
        )

        print("DataLoaders successfully created!")
        print(f"Train: {len(train_loader)}, Test: {len(test_loader)}")

        return train_loader, test_loader
