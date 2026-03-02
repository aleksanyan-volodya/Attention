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
