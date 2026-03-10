"""Simple training script."""

import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from data import IMDBDataLoader
from train import (
    train_model, 
    save_model,
    save_vocabulary,
    plot_training_results
)
import sys
sys.path.append("..")
from transformerNew import Transformer

print(f"Device used :{DEVICE}\n")

# Load the Data
data_loader = IMDBDataLoader()
data_loader.load_dataset(seed=RANDOM_SEED)
data_loader.build_vocabulary(VOCAB_BUILD_SIZE, VOCAB_SIZE)

train_loader, test_loader = data_loader.process_and_create_loaders(
    MAX_SEQ_LENGTH, BATCH_SIZE, TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE, verbose=False
)

# Create model
model = Transformer(
    src_vocab_size=data_loader.vocab.vocab_size,
    tgt_vocab_size=NUM_CLASSES,  # Binary classification
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


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
# they are using Adam optim orginally, but maybe other optim could also be ok


train_losses, train_accs, test_losses, test_accs = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
)

#Save results
save_model(model, MODEL_SAVE_PATH)
save_vocabulary(data_loader.vocab, VOCAB_SAVE_PATH)
plot_training_results(train_losses, train_accs, test_losses, test_accs, RESULTS_PLOT_PATH)
print(f"\nFinal Test Accuracy: {test_accs[-1]:.4f}")

