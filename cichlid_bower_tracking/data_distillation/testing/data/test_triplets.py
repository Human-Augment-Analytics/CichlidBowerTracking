from typing import Tuple

from torch.utils.data import Dataset
import torch.nn as nn
import torch

import random

class TestTriplets(Dataset):
    def __init__(self, batch_size=16, num_batches=10, num_channels=3, dim=224):
        self.__version__ = '0.1.0'

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.num_channels = num_channels
        self.dim = dim

    def __len__(self) -> int:
        return self.batch_size * self.num_batches
    
    def __getitem__(self, index: int) -> Tuple:
        anchor = torch.randn(self.num_channels, self.dim, self.dim)
        positive = torch.randn(self.num_channels, self.dim, self.dim)
        negative = torch.randn(self.num_channels, self.dim, self.dim)

        y_true = int(random.random() * 2)

        return anchor, positive, negative, y_true
