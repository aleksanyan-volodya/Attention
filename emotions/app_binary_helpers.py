"""
Helper fonctions for binary sentiment workflow in the app
"""
from typing import Any, Dict, Tuple

from emotions.config import *
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

def validate_transfrmer_dimensions(d_model: int, num_heads: int) -> bool:
    """Check if transformer is has valid dimensions

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
