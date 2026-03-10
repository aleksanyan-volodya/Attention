"""
Training script for the multi-emotion classifier.

This script mirrors `emotions/train_example.py`.
Before running it we must:
  1. Choose a dataset and implement MultiEmotionDataLoader.load_dataset() in data.py.
  2. Set NUM_CLASSES and EMOTION_LABELS in config.py.
  3. Set TRAIN_SUBSET_SIZE and TEST_SUBSET_SIZE in config.py.

Everything else below is already ok.
"""

import torch.nn as nn
import torch.optim as optim

from config import *
from data import MultiEmotionDataLoader
from train import (
    train_model,
    save_model,
    save_vocabulary,
    plot_training_results,
)

import sys
sys.path.append("..")
from transformerNew import Transformer

# Making sure config values are filled in before starting
assert NUM_CLASSES is not None, "Set NUM_CLASSES in config.py before training."
assert EMOTION_LABELS is not None, "Set EMOTION_LABELS in config.py before training."
assert TRAIN_SUBSET_SIZE is not None, "Set TRAIN_SUBSET_SIZE in config.py."
assert TEST_SUBSET_SIZE is not None, "Set TEST_SUBSET_SIZE in config.py."

print(f"Device: {DEVICE}")
print(f"Emotions ({NUM_CLASSES} classes): {EMOTION_LABELS}\n")

# Load the Data
data_loader = MultiEmotionDataLoader()
data_loader.load_dataset(seed=RANDOM_SEED)
data_loader.build_vocabulary(VOCAB_BUILD_SIZE, VOCAB_SIZE)

train_loader, test_loader = data_loader.process_and_create_loaders(
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    TRAIN_SUBSET_SIZE,
    TEST_SUBSET_SIZE,
    verbose=False,
)

# Create model
# tgt_vocab_size is set to NUM_CLASSES so the output layer has N outputs.
model = Transformer(
    src_vocab_size=data_loader.vocab.vocab_size,
    tgt_vocab_size=NUM_CLASSES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT,
    pad_token_id=PAD_IDX,
    mask=False,
    encoder_only=True,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")
