"""Training, evaluation, and inference utilities."""

from typing import Tuple, List

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
    """Train model for one epoch."""
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
    
    """Evaluate model on test/val data"""
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
    """Train model for multiple epochs."""
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
    """Predict sentiment for input text."""
    model.eval()

    processed_text = process_text(text, vocab, max_length, pad_idx=vocab.pad_idx)
    processed_text = processed_text.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(processed_text)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    label = "Positive" if prediction == 1 else "Negative"
    confidence = probs[0, prediction].item()
    probs_array = probs[0].detach().cpu().numpy()

    return label, confidence, probs_array

def batch_predict_sentiment(
    texts: list,
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 32,
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Predict sentiment for multiple texts."""
    model.eval()

    all_labels = []
    all_confidences = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        processed_texts = []
        for text in batch_texts:
            processed_text = process_text(
                text, vocab, max_length, pad_idx=vocab.pad_idx
            )
            processed_texts.append(processed_text)

        batch_tensor = torch.stack(processed_texts).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
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


# Utility functions
def save_model(model: nn.Module, filepath: str) -> None:
    """Save model state to file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to '{filepath}'")


def load_model(model: nn.Module, filepath: str, device: torch.device) -> nn.Module:
    """Load model state from file."""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from '{filepath}'")
    return model