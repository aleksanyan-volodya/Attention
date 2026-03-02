# This file is the row version of the model 
# trained IMDB dataset for neg/pos recognition

import re	# maybe change if other tokenization algo is used
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

import sys
sys.path.append("..")
from transformerNew import Transformer

# using HF datasets
from datasets import load_dataset




def load_split_data(seed=42, dataset_name="imdb"):
	dataset = load_dataset(dataset_name)

	train = dataset["train"].shuffle(seed)
	test = dataset["test"].shuffle(seed)

	print("Dataset succefully loaded")

	print(f"Train samples: {len(train)}")
	print(f"Test samples: {len(test)}")

	return train, test


def main():
	train, test = load_split_data()



if __name__ == '__main__':
	print(f"GPU available: {torch.cuda.is_available()}")
	main()
	print("end :)")