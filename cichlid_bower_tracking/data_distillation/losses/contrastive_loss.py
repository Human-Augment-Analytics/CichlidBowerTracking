import torch.nn.functional as F
import torch.nn as nn
import torch

import math

class ContrastiveLoss(nn.Module):
    def __init__(self, distance_metric: nn.Module, margin=1.0):
        '''
        Initializes an instance of the ContrastiveLoss PyTorch module.

        Inputs:
            distance_metric: a PyTorch module to be used in calculating the distance between the input feature embeddings.
            margin: the minimum distance between a pair of dissimilar embeddings.
        '''

        super(ContrastiveLoss, self).__init__()

        self.__version__ = '0.1.0'

        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, y: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the contrastive loss for a pair of feature embedding batches, given a batch of similarity labels (0 for similar, 1 for dissimilar).

        Inputs:
            y: a PyTorch Tensor representing the batch of similarity labels (0 for similar, 1 for dissimilar).
            z1: a PyTorch Tensor representing a batch of feature embeddings.
            z2: a PyTorch Tensor representing another batch of feature embeddings.

        Returns:
            loss: the contrastive loss between each pair of feature embeddings in the batch.
        '''

        distance = self.distance_metric(z1, z2)
        contrastive_loss = (1 - y) * 0.5 * math.pow(distance, 2) + 0.5 * y * math.pow(F.relu(self.margin - distance), 2)

        return contrastive_loss
