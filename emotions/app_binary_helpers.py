"""
Helper fonctions for binary sentiment workflow in the app
"""
from typing import Any, Dict, Tuple

import torch.nn as nn
import torch.optim as optim

from emotions.config import *
from emotions.data import IMDBDataLoader
from emotions.train import load_model, load_vocabulary, train_model
from transformerNew import Transformer


def build_pretrained_binary_model() -> Tuple[Transformer, Any]:
    """Load the pretrained binary model and vocabulary

    Returns
    -------
    Tuple[Transformer, Any]
        Transformer model with exosting weights and vocabulary object
    """
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=NUM_CLASSES,
        d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH, dropout=DROPOUT,
        pad_token_id=PAD_IDX, mask=False, encoder_only=True,
    ).to(DEVICE)

    model = load_model(model, f"emotions/{MODEL_SAVE_PATH}", DEVICE)
    vocab = load_vocabulary(f"emotions/{VOCAB_SAVE_PATH}")
    # if verbose:
    #     print(model succefully loaded !)

    return model, vocab

def validate_transformer_dimensions(d_model: int, num_heads: int) -> bool:
    """Check if transformer has valid dimensions

    Parameters
    ----------
    d_model : int
        Embedding size.
    num_heads : int
        Number of attention heads.

    Returns
    -------
    bool
        True if settings are valid.
    """
    return d_model % num_heads == 0

def get_prediction_artifacts(session_state: Any) -> Tuple[Transformer, Any, int]:
    """Choose model and vocabulary for prediction

    If a custom model is present in streamlit session, then use it
    Otherwise load the pretrained model

    Parameters
    ----------
    session_state : Any
        Streamlit session_state object

    Returns
    -------
    Tuple[Transformer, Any, int]
        (model, vocabulary, max_seq_length)
    """
    has_custom_model = (
        "custom_binary_model" in session_state
        and "custom_binary_vocab" in session_state
    )

    if has_custom_model:
        model = session_state["custom_binary_model"]
        vocab = session_state["custom_binary_vocab"]
        max_len = int(session_state.get("custom_binary_max_seq_len", MAX_SEQ_LENGTH))
    else:
        model, vocab = build_pretrained_binary_model()
        max_len = MAX_SEQ_LENGTH
    
    return model, vocab, max_len

def train_custom_binary_model(
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_seq_length: int,
    train_samples: int,
    test_samples: int,
    vocab_build_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    dropout: float,
) -> Tuple[Transformer, Any, Dict[str, float]]:
    """Train a binary sentiment model from user parameters

    Parameters
    ----------
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for Adam optimizer
    batch_size : int
        Batch size
    max_seq_length : int
        Token sequence length after padding
    train_samples : int
        Number of train samples to use
    test_samples : int
        Number of test samples to use
    vocab_build_size : int
        Number of samples used to build vocabulary
    d_model : int
        Embedding size
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of encoder layers
    d_ff : int
        Feed-forward hidden size
    dropout : float
        Dropout proba

    Returns
    -------
    Tuple[Transformer, Any, Dict[str, float]]
        Trained model, vocabulary, metrics dictionary.
    """
    data_loader = IMDBDataLoader()
    data_loader.load_dataset(seed=RANDOM_SEED)
    data_loader.build_vocabulary(num_samples=vocab_build_size, max_vocab_size=VOCAB_SIZE,)

    train_loader, test_loader = data_loader.process_and_create_loaders(
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        train_samples=train_samples,
        test_samples=test_samples,
        verbose=False,
    )

    model = Transformer(
        src_vocab_size=data_loader.vocab.vocab_size,
        tgt_vocab_size=NUM_CLASSES,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        pad_token_id=PAD_IDX,
        mask=False,
        encoder_only=True,
    ).to(DEVICE)
    
    crierion = ...
    optimizer = ...
    vocab = ...
    metrics = ...
    return model, vocab, metrics