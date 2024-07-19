from typing import Tuple

from torch.utils.data import Dataset
import torch.nn as nn
import torch

import random

class TestTriplets(Dataset):
    def __init__(self, batch_size=16, num_batches=10, num_channels=3, dim=224):
        '''
        Initializes an instance of the TestTriplets dataset class.

        Inputs:
            batch_size: the number of images per batch in the wrapping DataLoader; defaults to 16.
            num_batches: the number of image batches to be created; defaults to 10.
            num_channels: the number of channels to include in each image (1 for greyscale, 3 for RGB); defaults to 3.
            dim: the dimension of each image; defaults to 224.
        '''

        self.__version__ = '0.1.0'

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.num_channels = num_channels
        self.dim = dim

    def __len__(self) -> int:
        '''
        Returns the number of "images" in the dataset.

        Inputs: None.
        
        Returns: self.batch_size * self.num_batches.
        '''

        return self.batch_size * self.num_batches
    
    def __getitem__(self, index: int) -> Tuple:
        '''
        Returns triplets of "images" as well as a (random) binary "ground truth" class label.

        Inputs:
            index: the index of the triplet of "images" and the associated "ground truth" label; doesn't really do anything, but useful for iteration.

        Returns:
            anchor: anchor "images".
            positive: positive "images" that're "similar" to the anchors.
            negative: negative "images" that're "dissimilar" from the anchors.
            y_true: the (random) binary "ground truth" class labels for the anchor "images".
        '''

        anchor = torch.randn(self.num_channels, self.dim, self.dim)
        positive = torch.randn(self.num_channels, self.dim, self.dim)
        negative = torch.randn(self.num_channels, self.dim, self.dim)

        y_true = int(random.random() * 2)

        return anchor, positive, negative, y_true
