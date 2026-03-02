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
