import torch.nn as nn
import torch

class PairLoss(nn.Module):
    def __init__(self, criterion: nn.Module):
        '''
        Initializes an instance of the PairLoss PyTorch module.

        Inputs:
            criterion: a PyTorch module representing the reconstructive loss to be used in evaluating the image reconstructions.
        '''

        super(PairLoss, self).__init__()
        
        self.__version__ = '0.1.0'

        self.criterion = criterion

    def forward(self, x1_reconstruction: torch.Tensor, x2_reconstruction: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the pair loss (the sum of reconstruction losses from a pair of images) for a batch of image pairs.

        Inputs:
            x1_reconstruction: the set of reconstructed images associated with x1.
            x2_reconstruction: the set of reconstructed images associated with x2.
            x1: one set of original images in the batch.
            x2: another set of original images in the batch.

        Returns:
            pair_loss: the pair loss for a batch of image pairs.
        '''
        
        loss1 = self.criterion(x1_reconstruction, x1)
        loss2 = self.criterion(x2_reconstruction, x2)

        pair_loss = loss1 + loss2

        return pair_loss