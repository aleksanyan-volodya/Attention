"""
Simple inference script.

This script mirrors `emotions/predict_example.py`.
Before running it we must:
  1. Have a trained model saved at MODEL_SAVE_PATH.
  2. Have a vocabulary saved at VOCAB_SAVE_PATH.
  3. Have NUM_CLASSES and EMOTION_LABELS set correctly in config.py
     (must match what was used during training).

The script shows two usage patterns:
  - Single prediction  : one text at a time, shows full probability distribution.
  - Batch prediction   : many texts at once, more efficient for large inputs.
"""

import torch

from config import *
import sys
sys.path.append("..")
from transformerNew import Transformer
from train import predict_sentiment, batch_predict_sentiment, load_model, load_vocabulary

# Config must be complete before loading the model
assert NUM_CLASSES is not None, "Set NUM_CLASSES in config.py."
assert EMOTION_LABELS is not None, "Set EMOTION_LABELS in config.py."

# Load model and vocabulary
print("Loading model and vocabulary...")

model = Transformer(
    src_vocab_size=VOCAB_SIZE,
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

model = load_model(model, MODEL_SAVE_PATH, DEVICE)
vocab = load_vocabulary(VOCAB_SAVE_PATH)

# Test predictions
# TODO: replace with real examples that cover the emotions in your dataset.
test_texts = [
    "I can't believe how happy I am today, everything is perfect!",
    "I am so angry, this is completely unacceptable, and no one should be excused like this.",
    "I'm really scared about what might happen next. This is a very strange feeling",
    "This is so sad, I feel like crying, but I overwellmed it hits me even more.",
    "Wow, I did not expect that at all! It is very surprising to me and gamechanging",
]

print("\n=== Single Predictions ===")
for text in test_texts:
    emotion, conf, probs = predict_emotion(
        text, model, vocab, DEVICE, EMOTION_LABELS, MAX_SEQ_LENGTH
    )
    print(f"\n  Text      : {text[:60]}")
    print(f"  Predicted : {emotion} ({conf:.2%} confidence)")
    print("  All probs :", {label: f"{p:.2%}" for label, p in zip(EMOTION_LABELS, probs)})

print("\n=== Batch Predictions ===")
emotions, confidences, _ = batch_predict_emotion(
    test_texts, model, vocab, DEVICE, EMOTION_LABELS, MAX_SEQ_LENGTH
)
for text, emotion, conf in zip(test_texts, emotions, confidences):
    print(f"  {text[:50]:50} -> {emotion} ({conf:.2%})")
