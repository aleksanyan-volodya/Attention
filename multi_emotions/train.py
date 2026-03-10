"""
Training, evaluation, inference, and persistence utilities.

Nearly everything (training loop, loss, optimizer, save/load) is the same
as in the binary model. The right loss function depends on the task type:
  - Multi-class (one label per sample)  -> CrossEntropyLoss  (same as before)
  - Multi-label (multiple labels/sample) -> BCEWithLogitsLoss (change noted below)
"""

from typing import Dict, List, Optional, Tuple
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data import Vocabulary, process_text

import sys
sys.path.append("..")
from transformerNew import Transformer

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full pass over the training data.

    Parameters
    ----------
    model : nn.Module
        The transformer classifier.
    data_loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function (CrossEntropyLoss for multi-class).
    optimizer : torch.optim.Optimizer
        Optimizer (e.g. Adam).
    device : torch.device
        CPU or CUDA.

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy) for this epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in tqdm(data_loader, desc="Training"):
        texts = texts.to(device)
        labels = labels.to(device)

        outputs = model(texts)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(data_loader), correct / total


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    emotion_labels: Optional[List[str]] = None,
) -> Tuple[float, float, Optional[Dict[str, float]]]:
    """Evaluate the model on val/test data.

    Compared to the binary version this function also computes per-emotion
    accuracy when emotion_labels is provided. That extra breakdown tells you
    which emotions are easy and which ones the model struggles with.

    Parameters
    ----------
    model : nn.Module
        The transformer classifier.
    data_loader : DataLoader
        Validation or test data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        CPU or CUDA.
    emotion_labels : List[str], optional
        Human-readable name for each class index (from config.EMOTION_LABELS).
        If None, per-emotion stats are not computed.

    Returns
    -------
    Tuple[float, float, Optional[Dict[str, float]]]
        (average_loss, overall_accuracy, per_emotion_accuracy_dict)
        per_emotion_accuracy_dict is None when emotion_labels is not passed.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Tracks correct and total counts per class for per-emotion accuracy
    num_classes = len(emotion_labels) if emotion_labels else 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for texts, labels in tqdm(data_loader, desc="Evaluating"):
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class stats (only when emotion_labels is given)
            if emotion_labels:
                for cls_idx in range(num_classes):
                    mask = labels == cls_idx
                    class_total[cls_idx] += mask.sum().item()
                    class_correct[cls_idx] += (predicted[mask] == cls_idx).sum().item()

    overall_acc = correct / total

    per_emotion_acc = None
    if emotion_labels:
        per_emotion_acc = {
            emotion_labels[i]: (
                class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            )
            for i in range(num_classes)
        }

    return total_loss / len(data_loader), overall_acc, per_emotion_acc