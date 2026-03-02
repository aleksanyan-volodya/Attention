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

#HYPERPARAM
VOCAB_SIZE = 20000


def load_split_data(seed=42, dataset_name="imdb"):
	dataset = load_dataset(dataset_name)

	train = dataset["train"].shuffle(seed)
	test = dataset["test"].shuffle(seed)

	print("Dataset succefully loaded")
	print(f"Train samples: {len(train)}")
	print(f"Test samples: {len(test)}")

	return train, test

def tokenizer(text, train):
	PAD_IDX = 0
	UNK_IDX = 1

	# using regex tokenizer: words + contractions + numbers
	def __regex_tokenizer():
		text = text.lower()
		return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)

	print("Building vocabulary")
	word_counter = Counter()

	for i, sample in enumerate(train):
		if i>= 10000: 	# use subset to build vocab more quickly
			break
		tokens = __regex_tokenizer(sample["text"])
		word_counter.update(tokens)

	most_commun = word_counter.most_commun(VOCAB_SIZE-2)
	vocab_dict = {"<pad>": PAD_IDX,
					"unk>": UNK_IDX}
	for word, _ in most_commun:
		vocab_dict[word] = len(vocab_dict)

	MODEL_VOCAB_SIZE = len(vocab_dict)


def main():
	train, test = load_split_data()



if __name__ == '__main__':
	print(f"GPU available: {torch.cuda.is_available()}")
	main()
	print("end :)")