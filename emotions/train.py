# This file is the row version of the model 
# trained IMDB dataset for neg/pos recognition

import re	# maybe change if other tokenization algo is used
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm


if __name__ == '__main__':
	print("end :)")