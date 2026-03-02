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