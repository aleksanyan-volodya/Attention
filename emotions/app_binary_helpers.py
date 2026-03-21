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