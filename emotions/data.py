"""Data loading, tokenization, and preprocessing."""

import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


class SmartTokenizer:
    """Regex-based tokenizer."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)

class Vocabulary:
    """Vocabulary mapping tokens to indices."""

    def __init__(self, pad_idx: int = 0, unk_idx: int = 1):
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.token_to_idx = {"<pad>": pad_idx, "<unk>": unk_idx}
        self.idx_to_token = {pad_idx: "<pad>", unk_idx: "<unk>"}

        self.vocab_size = 2 # Start with pad and unk


    def build_from_samples(self, samples: List[str], max_vocab_size: int = 20000) -> None:
        """Build vocabulary from text samples."""
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
        """Encode tokens to indices"""
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Decode indices to tokens"""
        return [self.idx_to_token.get(idx, "<unk>") for idx in indices]
    

def process_text(
    text: str, vocab: Vocabulary, max_length: int, pad_idx: int = 0
    ) -> torch.Tensor:
    """Convert text to padded token tensor"""
    tokenizer = SmartTokenizer()
    tokens = tokenizer.tokenize(text)
    ids = vocab.encode(tokens)

    if len(ids) < max_length:
        ids = ids + [pad_idx] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    return torch.tensor(ids, dtype=torch.long)


class IMDBDataLoader:
    """Load and preprocess IMDB dataset speciffically for sentiment analysis"""

    def __init__(self):
        self.train_split = None
        self.test_split = None
        self.vocab = None

    def load_dataset(self, seed: int = 42) -> None:
        """Load IMDB dataset from HuggingFace datasets library"""
        print("Loading IMDB dataset...")

        dataset = load_dataset("imdb")
        self.train_split = dataset["train"].shuffle(seed=seed)
        self.test_split = dataset["test"].shuffle(seed=seed)

        print(f"Train: {len(self.train_split)}, Test: {len(self.test_split)}")

    def build_vocabulary(self, num_samples: int = 10000, max_vocab_size: int = 20000) -> Vocabulary:
        """Build vocabulary from training samples."""
        if self.train_split is None:
            raise ValueError("Load dataset first with load_dataset()")
        
        print(f"Building vocabulary from {num_samples} samples....")
        samples = [self.train_split[i]["text"] for i in range(num_samples)]
        
        self.vocab = Vocabulary()
        self.vocab.build_from_samples(samples, max_vocab_size)
        print(f"Vocabulary size: {self.vocab.vocab_size}")
        return self.vocab
    
    def process_and_create_loaders(self, max_seq_length: int,train_samples: int = 5000,test_samples: int = 2000,
    ) -> Tuple[DataLoader, DataLoader]:
        """Process data and create DataLoaders."""

        print(f"Processing {train_samples} training samples...")

        train_texts = [] 
        train_labels = []

        for i in range(train_samples):
            processed = process_text(
                self.train_split[i]["text"],
                self.vocab,
                max_seq_length,
                self.vocab.pad_idx,
            )
            train_texts.append(processed)
            train_labels.append(self.train_split[i]["label"])
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1} processed")
