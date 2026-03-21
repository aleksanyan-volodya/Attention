"""Training, evaluation, and inference utilities."""

from typing import Tuple, List
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


# Training
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
        Loss function (CrossEntropyLoss).
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
    total_loss = 0
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
) -> Tuple[float, float]:
    
    """Evaluate model on test/val data
    
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

    Returns
    -------
    Tuple[float, float] 
        (average_loss, accuracy) on the evaluation set
"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

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

    return total_loss / len(data_loader), correct / total

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train model for multiple epochs.

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

    Returns
    -------
    Tuple[List[float], List[float], List[float], List[float]]
        (train_losses, train_accuracies, test_losses, test_accuracies)
    """

    print("Starting training...\n")

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_acc:.4f}\n")

    print("Training completed")
    return train_losses, train_accuracies, test_losses, test_accuracies


def predict_sentiment(
    text: str,
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 128,
) -> Tuple[str, float, np.ndarray]:
    """Predict sentiment for input text.

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
    max_length : int
        Sequence length used during training (must match MAX_SEQ_LENGTH).

    Returns
    -------
    Tuple[str, float, np.ndarray]
        (predicted_emotion, confidence, both_class_probabilities)
    """
    model.eval()

    # Handle both Vocabulary object and dict
    if isinstance(vocab, dict):
        pad_idx = vocab.get("<pad>", 0)
    else:
        pad_idx = vocab.pad_idx
    
    processed_text = process_text(text, vocab, max_length, pad_idx=pad_idx)
    processed_text = processed_text.unsqueeze(0).to(device) # Shape: (1, max_length)

    with torch.no_grad():
        output = model(processed_text)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    label = "Positive" if prediction == 1 else "Negative"
    confidence = probs[0, prediction].item()
    probs_array = probs[0].detach().cpu().numpy()

    return label, confidence, probs_array

def batch_predict_sentiment(
    texts: List[str],
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 32,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Predict sentiment for multiple texts.
    
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
        - predicted_emotions : Emotion name strings, one per input
        - confidences        : 1-D array, confidence for the top prediction.
        - all_class_probs    : 2-D array of shape (len(texts), 2).
    """
    model.eval()

    # Handle both Vocabulary object and dict
    if isinstance(vocab, dict):
        pad_idx = vocab.get("<pad>", 0)
    else:
        pad_idx = vocab.pad_idx

    all_labels = []
    all_confidences = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        processed = torch.stack(
            [process_text(t, vocab, max_length, pad_idx=pad_idx) for t in batch]
        ).to(device)

        with torch.no_grad():
            outputs = model(processed)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

        for j, pred in enumerate(predictions):
            label = "Positive" if pred.item() == 1 else "Negative"
            confidence = probs[j, pred].item()
            all_labels.append(label)
            all_confidences.append(confidence)
            all_probs.append(probs[j].detach().cpu().numpy())

    return (
        all_labels,
        np.array(all_confidences),
        np.array(all_probs),
    )


def explain_prediction(
    text: str,
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 128,
    top_k: int = 5,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Predict sentiment and return the tokens that most influenced the prediction.

    Use gradient x input : the gradient of the predicted class score
    with respect to each token embedding, multiplied element-wise by the embedding
    itself, then summed across the embedding. This gives one importance
    score per token, and no model architecture changes needed.

    Parameters
    ----------
    text : str
        Raw input text.
    model : nn.Module
        Trained transformer classifier (encoder_only=True).
    vocab : Vocabulary
        Vocabulary used during training.
    device : torch.device
        CPU or CUDA
    max_length : int
        Sequence length used during training (must match MAX_SEQ_LENGTH).
    top_k : int
        How many top tokens to return.

    Returns
    -------
    Tuple[str, float, List[Tuple[str, float]]]
        (label, confidence, top_tokens) where top_tokens is a list of
        (token_string, importance_score) pairs sorted by importance (highest first).
        Importance scores are normalized to [0, 1] relative to the most important token.
    """
    model.eval()

    if isinstance(vocab, dict):
        pad_idx_local = vocab.get("<pad>", 0)
    else:
        pad_idx_local = vocab.pad_idx

    processed_text = process_text(text, vocab, max_length, pad_idx=pad_idx_local)
    input_ids = processed_text.unsqueeze(0).to(device)

    # Hooking to capture the embedding output tensor and keep its gradient
    # (Intermediate tensors lose their grad after backward by default, so we need to capture it here.)
    captured = {}

    def _capture_hook(module, inp, out):
        captured["emb"] = out
        out.retain_grad()

    hook_handle = model.embedding.register_forward_hook(_capture_hook)

    # Forward pass with gradients enabled
    output = model(input_ids)
    probs = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

    # Backpropagate through the predicted class score
    model.zero_grad()
    probs[0, predicted_class].backward()

    hook_handle.remove()

    emb = captured["emb"]   # (1, max_length, d_model)
    grad = emb.grad          # (1, max_length, d_model)

    # Gradient x input: how much each embedding is contributing.
    # Sum across d_model and take abs(value) => ONE scalar per token position.
    importance = (grad * emb).sum(dim=-1).abs().squeeze(0)  # (max_length,)

    # Drop padding positions
    non_pad_mask = processed_text != pad_idx_local
    indices = processed_text[non_pad_mask].tolist()
    if isinstance(vocab, dict):
        idx_to_token = {v: k for k, v in vocab.items()}
        tokens = [idx_to_token.get(i, "<unk>") for i in indices]
    else:
        tokens = vocab.decode(indices)
    scores = importance[non_pad_mask].detach().cpu().numpy()

    # Normalize to so scores are easy to read
    if scores.max() > 0:
        scores = scores / scores.max()

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_tokens = [(tokens[i], float(scores[i])) for i in top_indices]

    label = "Positive" if predicted_class == 1 else "Negative"
    confidence = float(probs[0, predicted_class].item())

    return label, confidence, top_tokens


# Utility functions
def save_model(model: nn.Module, filepath: str) -> None:
    """Save model state to file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to '{filepath}'")


def load_model(model: nn.Module, filepath: str, device: torch.device) -> nn.Module:
    """Load model state from file.
    
    Parameters
    ----------
    model : nn.Module
        Model instance with the same architecture as the saved one.
    filepath : str
        Path to the saved .pt file
    device : torch.device
        Device to map the weights to.

    Returns
    -------
    nn.Module
        The model with loaded weights, in eval mode.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from '{filepath}'")
    return model

def save_vocabulary(vocab: Vocabulary, filepath: str) -> None:
    """Save vocabulary to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to '{filepath}'")


def load_vocabulary(filepath: str) -> Vocabulary:
    """Load vocabulary from pickle file.
    
    Returns
    -------
    Vocabulary
        The deserialized vocabulary object
    """
    with open(filepath, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from '{filepath}'")
    return vocab

# Visualization

def plot_training_results(
    train_losses: List[float],
    train_accuracies: List[float],
    test_losses: List[float],
    test_accuracies: List[float],
    save_path=None,
) -> None:
    """Plot training and test metrics.
        Parameters
    ----------
    train_losses : List[float]
    train_accuracies : List[float]
    test_losses : List[float]
    test_accuracies : List[float]
    save_path : str
        Where to save the PNG image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", marker="o")
    axes[0].plot(test_losses, label="Test Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Test Loss")
    axes[0].legend()
    axes[0].grid(True)

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
        print(f"Plot saved to '{save_path}'")

    plt.show()