"""Simple inference script."""

import torch

from config import *
import sys
sys.path.append("..")
from transformerNew import Transformer
from train import predict_sentiment, batch_predict_sentiment, explain_prediction, load_model, load_vocabulary

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
    encoder_only=True
).to(DEVICE)   

model = load_model(model, MODEL_SAVE_PATH, DEVICE)
vocab = load_vocabulary(VOCAB_SAVE_PATH)

# Test predictions
test_reviews = [
    "This movie was absolutely fantastic! I loved every minute.",
    "Terrible waste of time. One of the worst films ever made.",
    "It was okay, nothing special but watchable.",
]

print("\n=== Single Predictions ===")
for review in test_reviews:
    label, conf, probs = predict_sentiment(review, model, vocab, DEVICE)
    print(f"{review[:40]:40} -> {label} ({conf:.2%})")

print("\n=== Batch Predictions ===")
labels, confidences, _ = batch_predict_sentiment(test_reviews, model, vocab, DEVICE)
for review, label, conf in zip(test_reviews, labels, confidences):
    print(f"{review[:40]:40} -> {label} ({conf:.2%})")

print("\n=== Token Importance (top 5 tokens) ===")
for review in test_reviews:
    label, conf, top_tokens = explain_prediction(review, model, vocab, DEVICE, top_k=5)
    print(f"\nText   : {review[:60]}")
    print(f"Result : {label} ({conf:.2%})")
    print(f"Tokens :")
    for token, score in top_tokens:
        bar = int(score * 20)
        print(f"{token:<20} {'█' * bar} {score:.2f}")

