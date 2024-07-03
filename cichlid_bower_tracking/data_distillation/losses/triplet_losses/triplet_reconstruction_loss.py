import torch.nn as nn
import torch

class TripletReconstructionLoss(nn.Module):
    def __init__(self, criterion: nn.Module):
        '''
        Initializes an instance of the TripletReconstructionLoss class.

        Inputs:
            criterion: a PyTorch module representing the reconstruction loss to be used.
        '''
        
        self.__version__ = '0.1.0'

        self.criterion = criterion

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, \
                anchor_reconstruction: torch.Tensor, positive_reconstruction: torch.Tensor, negative_reconstruction: torch.Tensor) -> torch.Tensor:
        
        '''
        Computes the triplet reconstruction loss (sum of individual reconstruction losses).

        Inputs:
            anchor: the anchor images Tensor.
            positive: the positive images Tensor (similar to anchor).
            negative: the negative images Tensor (dissimilar to anchor).
            anchor_reconstruction: the reconstructions Tensor of the anchor images.
            positive_reconstruction: the reconstructions Tensor of the positive images.
            negative_reconstruction: the reconstructions Tensor of the negative images.

        Returns:
            triplet_reconstruction_loss: the sum of all individual reconstruction losses between images and their reconstructions.
        '''
        
        anchor_loss = self.criterion(anchor, anchor_reconstruction)
        positive_loss = self.criterion(positive, positive_reconstruction)
        negative_loss  =self.criterion(negative, negative_reconstruction)

        triplet_reconstruction_loss = anchor_loss + positive_loss + negative_loss

        return triplet_reconstruction_loss