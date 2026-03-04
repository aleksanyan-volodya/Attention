"""Configuration and hyperparameters for the IMDB Negatif/Positif review model."""

import torch

# Device and Basic Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Data Processing Parameters
MAX_SEQ_LENGTH = 256
VOCAB_SIZE = 20000
PAD_IDX = 0
UNK_IDX = 1

# Subsets of data for faster training/testing
TRAIN_SUBSET_SIZE = 5000
TEST_SUBSET_SIZE = 2000
VOCAB_BUILD_SIZE = 10000

# Model Architecture Parameters
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 3
D_FF = 512
DROPOUT = 0.4
NUM_CLASSES = 2  # Binary classification: Negative, Positive

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
NUM_EPOCHS = 40

# File Paths
MODEL_SAVE_PATH = "transformer_imdb_classifier.pt"
RESULTS_PLOT_PATH = "training_results.png"
VOCAB_SAVE_PATH = "vocabulary.pkl"
