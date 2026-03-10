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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    emotion_labels: Optional[List[str]] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train for multiple epochs and print progress.

    Parameters
    ----------
    model : nn.Module
        The transformer classifier.
    train_loader : DataLoader
        Training data loader.
    test_loader : DataLoader
        Test/validation data loader.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        CPU or CUDA.
    num_epochs : int
        Total number of training epochs.
    emotion_labels : List[str], optional
        Passed to evaluate() to get per-emotion accuracy each epoch.

    Returns
    -------
    Tuple[List[float], List[float], List[float], List[float]]
        (train_losses, train_accuracies, test_losses, test_accuracies)
    """

    print("Starting training...\n")

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc, per_emotion = evaluate(
            model, test_loader, criterion, device, emotion_labels
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Accuracy: {test_acc:.4f}")

        # Print per-emotion breakdown every 5 epochs to track hard emotions
        if per_emotion and epoch % 5 == 0:
            print("  Per-emotion accuracy:")
            for emotion, acc in per_emotion.items():
                print(f"    {emotion:<12}: {acc:.4f}")
        print()

    print("Training completed.")
    return train_losses, train_accuracies, test_losses, test_accuracies


def predict_emotion(
    text: str,
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    emotion_labels: List[str],
    max_length: int = 256,
) -> Tuple[str, float, np.ndarray]:
    """Predict the dominant emotion for a single input text.

    Parameters
    ----------
    text : str
        Raw input text from the user.
    model : nn.Module
        Trained transformer classifier.
    vocab : Vocabulary
        Vocabulary used during training.
    device : torch.device
        CPU or CUDA.
    emotion_labels : List[str]
        Human-readable emotion names in class-index order.
    max_length : int
        Sequence length used during training (must match MAX_SEQ_LENGTH).

    Returns
    -------
    Tuple[str, float, np.ndarray]
        (predicted_emotion, confidence, all_class_probabilities)
    """
    model.eval()
    processed = process_text(text, vocab, max_length, pad_idx=vocab.pad_idx)
    processed = processed.unsqueeze(0).to(device)  # Shape: (1, max_length)

    with torch.no_grad():
        output = model(processed)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    emotion = emotion_labels[prediction]
    confidence = probs[0, prediction].item()
    probs_array = probs[0].detach().cpu().numpy()

    return emotion, confidence, probs_array

def batch_predict_emotion(
    texts: List[str],
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    emotion_labels: List[str],
    max_length: int = 256,
    batch_size: int = 32,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Predict emotions for a list of texts.

    Parameters
    ----------
    texts : List[str]
        List of raw input texts.
    model : nn.Module
        Trained transformer classifier.
    vocab : Vocabulary
        Vocabulary used during training.
    device : torch.device
        CPU or CUDA.
    emotion_labels : List[str]
        Human-readable emotion names in class-index order.
    max_length : int
        Sequence length used during training.
    batch_size : int
        How many texts to process at once.

    Returns
    -------
    Tuple[List[str], np.ndarray, np.ndarray]
        (predicted_emotions, confidences, all_class_probabilities)
        - predicted_emotions : List of emotion name strings, one per input.
        - confidences        : 1-D array, confidence for the top prediction.
        - all_class_probs    : 2-D array of shape (len(texts), NUM_CLASSES).
    """
    model.eval()
    all_emotions, all_confidences, all_probs = [], [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        processed = torch.stack(
            [process_text(t, vocab, max_length, pad_idx=vocab.pad_idx) for t in batch]
        ).to(device)

        with torch.no_grad():
            outputs = model(processed)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

        for j, pred in enumerate(predictions):
            emotion = emotion_labels[pred.item()]
            confidence = probs[j, pred].item()
            all_emotions.append(emotion)
            all_confidences.append(confidence)
            all_probs.append(probs[j].detach().cpu().numpy())

    return (
        all_emotions, 
        np.array(all_confidences), 
        np.array(all_probs),
    )

# Utility functions
def save_model(model: nn.Module, filepath: str) -> None:
    """Save model weights to file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str, device: torch.device) -> nn.Module:
    """Load model weights from file.

    Parameters
    ----------
    model : nn.Module
        Model instance with the same architecture as the saved one.
    filepath : str
        Path to the saved .pt file.
    device : torch.device
        Device to map the weights to.

    Returns
    -------
    nn.Module
        The model with loaded weights, in eval mode.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


def save_vocabulary(vocab: Vocabulary, filepath: str) -> None:
    """Save vocabulary to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {filepath}")


def load_vocabulary(filepath: str) -> Vocabulary:
    """Load vocabulary from disk.

    Returns
    -------
    Vocabulary
        The deserialized Vocabulary object.
    """
    with open(filepath, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab

# Visualization

def plot_training_results(
    train_losses: List[float],
    train_accuracies: List[float],
    test_losses: List[float],
    test_accuracies: List[float],
    save_path = None,
) -> None:
    """Plot loss and accuracy curves and save to file.

    Parameters
    ----------
    train_losses : List[float]
    train_accuracies : List[float]
    test_losses : List[float]
    test_accuracies : List[float]
    save_path : str
        Where to save the PNG image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, label="Train Loss", marker="o")
    axes[0].plot(test_losses, label="Test Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()
    axes[0.grid(True)]

    axes[1].plot(train_accuracies, label="Train Accuracy", marker="o")
    axes[1].plot(test_accuracies, label="Test Accuracy", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Test Accuracy")
    axes[1].legend()
    axes[1].grid(True)


    plt.tight_layout()

    if save_path: 
        plt.savefig(save_path, dpi=100)
        print(f"Plots saved to '{save_path}'")

    plt.show()
