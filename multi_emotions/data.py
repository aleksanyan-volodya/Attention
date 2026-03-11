"""
Data loading, tokenization, and preprocessing for multi-emotion recognition.

SmartTokenizer, Vocabulary, and process_text are identical to those in
`emotions/data.py`, they are text utilities with no dependency on
the specific dataset or number of classes.

The dataset-specific part is isolated in MultiEmotionDataLoader.
That class is a placeholder: its internal logic (which HuggingFace dataset
to load, how labels are encoded, whether it is multi-class or multi-label)
will be filled in once the dataset is decided.

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


# TODO complete once dataset is chosen
class MultiEmotionDataLoader:
    """Load and preprocess a multi-emotion text dataset.

    This class is intentionally left as a skeleton. The exact implementation
    (which dataset to load, how to read labels, train/test split names, etc.)
    depends on the dataset that will be chosen.

    Once a dataset is chosen, need to implement:
        1. load_dataset()     -> fill self.train_split and self.test_split
        2. build_vocabulary() -> same as in `emotions/data.py`
        3. process_and_create_loaders() -> encode texts, create DataLoaders

    For multi-label datasets, labels should be float tensors of shape
    (num_samples, NUM_CLASSES) and the loss function must be BCEWithLogitsLoss.
    For multi-class datasets (one label per sample), labels stay as LongTensors
    and the loss is CrossEntropyLoss which is same as the binary model
    """

    def __init__(self):
        self.train_split = None
        self.test_split = None
        self.vocab = None

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
        self.train_split = dataset["train"]
        self.test_split = dataset["test"]

        print(f"Train: {len(self.train_split)}, Test: {len(self.test_split)}")

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
        if self.train_split is None:
            raise ValueError("Load dataset first with load_dataset()")

        print(f"Building vocabulary from {num_samples} samples...")

        # TODO: adjust the field name "text" to match the dataset's column name.
        samples = [self.train_split[i]["text"] for i in range(num_samples)]

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
        TODO: once the dataset is loaded:
          - Adjust the column names ("text", "label") to the actual dataset fields.
          - For multi-label datasets, convert labels to float tensors and
            use BCEWithLogitsLoss instead of CrossEntropyLoss.
        """
        if self.vocab is None:
            raise ValueError("Call build_vocabulary() first.")

        print(f"Processing {train_samples} training samples...")
        train_texts, train_labels = [], []

        for i in range(train_samples):
            processed = process_text(
                self.train_split[i]["text"],   # TODO: check column name
                self.vocab,
                max_seq_length,
                self.vocab.pad_idx,
            )
            train_texts.append(processed)
            train_labels.append(self.train_split[i]["label"])  # TODO: check column name
            if verbose and (i + 1) % 1000 == 0:
                print(f"  {i + 1} processed")

        train_texts = torch.stack(train_texts)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        # For multi-label: torch.tensor(train_labels, dtype=torch.float)

        print(f"Processing {test_samples} test samples...")
        test_texts, test_labels = [], []

        for i in range(test_samples):
            processed = process_text(
                self.test_split[i]["text"],   # TODO: check column name
                self.vocab,
                max_seq_length,
                self.vocab.pad_idx,
            )
            test_texts.append(processed)
            test_labels.append(self.test_split[i]["label"])  # TODO: check column name
            if verbose and (i + 1) % 500 == 0:
                print(f"  {i + 1} processed")

        test_texts = torch.stack(test_texts)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        # For multi-label: torch.tensor(test_labels, dtype=torch.float)

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
