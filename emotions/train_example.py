"""Simple training script."""

import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from data import IMDBDataLoader
from train import train_model, save_model, plot_training_results
import sys
sys.path.append("..")
from transformerNew import Transformer

# Load the Data
data_loader = IMDBDataLoader()
data_loader.load_dataset(seed=RANDOM_SEED)
data_loader.build_vocabulary(VOCAB_BUILD_SIZE, VOCAB_SIZE)

train_loader, test_loader = data_loader.process_and_create_loaders(
    MAX_SEQ_LENGTH, BATCH_SIZE, TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE
)

# Create model
model = Transformer(
    src_vocab_size=data_loader.vocab.vocab_size,
    tgt_vocab_size=2,  # Binary classification
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT,
    pad_token_id=PAD_IDX,
    mask=False
).to(DEVICE)
