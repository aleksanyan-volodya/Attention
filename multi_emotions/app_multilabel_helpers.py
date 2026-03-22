"""Helpers for train-your-own multi-label workflow in Streamlit."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from multi_emotions.config import *
from multi_emotions.data import MultiEmotionDataLoader, Vocabulary, process_text
from transformerNew import Transformer


def validate_transformer_dimensions(d_model: int, num_heads: int) -> bool:
    """Return True when d_model is divisible by num_heads."""
    return d_model % num_heads == 0


def _accumulate_micro_counts(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> Tuple[float, float, float]:
    """Compute TP/FP/FN for micro-F1 from one batch."""
    preds = (probs >= threshold).float()

    tp = float((preds * labels).sum().item())
    fp = float((preds * (1.0 - labels)).sum().item())
    fn = float(((1.0 - preds) * labels).sum().item())
    return tp, fp, fn


def _micro_f1_from_counts(tp: float, fp: float, fn: float) -> float:
    """Convert aggregated TP/FP/FN counts into micro-F1."""
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def _run_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    threshold: float,
    is_train: bool,
) -> Tuple[float, float]:
    """Run one train/eval epoch and return (loss, micro_f1)."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    tp_total, fp_total, fn_total = 0.0, 0.0, 0.0

    for texts, labels in data_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(texts)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        tp, fp, fn = _accumulate_micro_counts(probs, labels, threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    avg_loss = total_loss / max(len(data_loader), 1)
    micro_f1 = _micro_f1_from_counts(tp_total, fp_total, fn_total)
    return avg_loss, micro_f1


def train_custom_multilabel_model(
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
    threshold: float,
) -> Tuple[Transformer, Vocabulary, Dict[str, float]]:
    """Train a multi-label model from user-selected hyperparameters."""
    data_loader = MultiEmotionDataLoader()
    data_loader.load_dataset(seed=RANDOM_SEED)
    data_loader.build_vocabulary(
        num_samples=vocab_build_size,
        max_vocab_size=VOCAB_SIZE,
    )

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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: List[float] = []
    test_losses: List[float] = []
    train_f1s: List[float] = []
    test_f1s: List[float] = []

    for _ in range(epochs):
        train_loss, train_f1 = _run_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            threshold=threshold,
            is_train=True,
        )
        test_loss, test_f1 = _run_one_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            threshold=threshold,
            is_train=False,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)

    metrics = {
        "final_test_f1": float(test_f1s[-1]) if test_f1s else 0.0,
        "final_test_loss": float(test_losses[-1]) if test_losses else 0.0,
        "epochs": float(epochs),
        "train_samples": float(train_samples),
        "test_samples": float(test_samples),
        "max_seq_length": float(max_seq_length),
        "threshold": float(threshold),
    }

    return model, data_loader.vocab, metrics


def predict_multilabel(
    text: str,
    model: nn.Module,
    vocab: Vocabulary,
    threshold: float,
    max_length: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Predict one or more emotion labels for a single text."""
    model.eval()

    processed = process_text(text, vocab, max_length, pad_idx=vocab.pad_idx)
    processed = processed.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(processed)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    prob_dict = {
        label: float(prob)
        for label, prob in zip(EMOTION_LABELS, probs)
    }

    predicted = [label for label, prob in prob_dict.items() if prob >= threshold]

    # Keep one label when all probabilities are below threshold.
    if not predicted:
        top_label = max(prob_dict, key=prob_dict.get)
        predicted = [top_label]

    return predicted, prob_dict
