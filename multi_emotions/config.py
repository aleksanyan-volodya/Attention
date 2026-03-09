"""
Hyperparameters and configuration for the multi-emotion recognition model.

Most architecture settings carry over from the binary sentiment model in
`emotions/config.py`. The main unknowns that depend on the dataset are
marked with TODO comments.
"""

import torch

# Device and Basic Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Data Processing Parameters
MAX_SEQ_LENGTH = 256   # Max number of tokens per sample after truncation/padding
VOCAB_SIZE = 20000     # Maximum vocabulary size
PAD_IDX = 0            # Index used for padding tokens
UNK_IDX = 1            # Index used for unknown tokens

# Subsets of data for faster training/testing
# TODO: set these once dataset is chosen and its size is known
TRAIN_SUBSET_SIZE = None
TEST_SUBSET_SIZE = None
# TODO: adjust once dataset size is known
VOCAB_BUILD_SIZE = 10000

# Model Architecture Parameters
D_MODEL = 256     # Embedding / hidden dimension
NUM_HEADS = 8     # Number of attention heads (D_MODEL must be divisible by this)
NUM_LAYERS = 3    # Number of transformer encoder layers
D_FF = 512        # Feed-forward inner dimension inside each encoder layer
DROPOUT = 0.4     # Dropout rate (regularization)
# TODO: set this once the dataset is chosen.
NUM_CLASSES = None  # MUST be set before building the model

# TODO: need tp fill this in once the dataset label mapping is known.
# Example :
#   EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EMOTION_LABELS = None  # List[str], length must equal NUM_CLASSES !

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
NUM_EPOCHS = 40

# File Paths
MODEL_SAVE_PATH = "multi_emotion_classifier.pt"
RESULTS_PLOT_PATH = "training_results.png"
VOCAB_SAVE_PATH = "vocabulary.pkl"
