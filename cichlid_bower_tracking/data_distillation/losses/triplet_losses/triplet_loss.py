import torch.nn.functional as F
import torch.nn as nn
import torch

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p_norm=2):
        '''
        Initializes an instance of the TripletLoss class.

        Inputs:
            margin: a float representing the margin parameter of the triplet loss.
            p_norm: an int representing the value of p to be used in computing the pairwise distance between the anchor-positive and anchor-negative pairs.
        '''
        
        super(TripletLoss, self).__init__()

        self.__version__ = '0.1.0'

        self.margin = margin
        self.p_norm = p_norm

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor, z_negative: torch.Tensor) -> torch.Tensor:
        '''
        Computes and returns the triplet loss, given three embeddings: an anchor, a positive, and a negative.

        Inputs:
            z_anchor: the embedding of the anchor image.
            z_positive: the embedding of the positive image.
            z_negative: the embedding of the negative image.

        Returns:
            loss: the mean-reduced triplet loss between the three image embeddings.
        '''

        sqr_distance_1 = torch.pow(F.pairwise_distance(z_anchor, z_positive, p=self.p_norm), self.p_norm)
        sqr_distance_2 = torch.pow(F.pairwise_distance(z_anchor, z_negative, p=self.p_norm), self.p_norm)

        loss = F.relu(sqr_distance_1 - sqr_distance_2 + self.margin).mean()

        return loss