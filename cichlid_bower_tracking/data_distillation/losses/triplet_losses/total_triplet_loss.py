from triplet_loss import TripletLoss
from triplet_reconstruction_loss import TripletReconstructionLoss

import torch.nn as nn
import torch

class TotalTripletLoss(nn.Module):
    def __init__(self, reconstruction_criterion: str, margin=1.0, p_norm=2):
        '''
        Initializes an instance of the TotalTripletLoss class.

        Inputs:
            reconstruction_criterion: a string indicating which reconstructive loss metric to use in the pair loss; must either be 'l2'/'mse' or 'l1'.
            margin: the value to be used for the margin hyperparameter of the triplet loss; defaults to 1.0.
            p_norm: the p-value for the norm used by pairwise distance; defaults to 2.
        '''
        
        self.__version__ = '0.1.0'

        self.margin = margin
        self.p_norm = p_norm

        if reconstruction_criterion == 'l2' or reconstruction_criterion == 'mse':
            self.reconstruction_criterion = nn.MSELoss()
        elif reconstruction_criterion == 'l1':
            self.reconstruction_criterion = nn.L1Loss()
        else:
            raise Exception(f'value passed to reconstruction_criterion hyperparameter must either be \'l2\'/\'mse\' or \'l1\'... ({reconstruction_criterion} passed).')
        
        self.triplet_loss = TripletLoss(margin=self.margin, p_norm=self.p_norm)
        self.triplet_reconstruction_loss = TripletReconstructionLoss(criterion=self.reconstruction_criterion)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, \
                z_anchor: torch.Tensor, z_positive: torch.Tensor, z_negative: torch.Tensor, \
                anchor_reconstruction: torch.Tensor, positive_reconstruction: torch.Tensor, negative_reconstruction: torch.Tensor) -> torch.Tensor:
        
        '''
        Computes the total triplet loss (sum of triplet loss and triplet reconstruction loss) for each triplet of images.

        Inputs:
            anchor: the anchor images Tensor.
            positive: the positive images Tensor (similar to the anchor).
            negative: the negative images Tensor (dissimilar to the anchor).
            z_anchor: the anchor embeddings Tensor.
            z_positive: the positive embeddings Tensor.
            z_negative: the negative embeddings Tensor.
            anchor_reconstruction: the reconstructions Tensor of the anchor images.
            positive_reconstruction: the reconstructions Tensor of the positive images.
            negative_reconstruction: the reconstructions Tensor of the negative images.

        Returns:
            total_triplet_loss: the computed total triplet loss (sum of triplet loss and triplet reconstruction loss). 
        '''

        reconstruction_loss = self.triplet_reconstruction_loss(anchor=anchor, positive=positive, negative=negative, \
                                                               anchor_reconstruction=anchor_reconstruction, positive_reconstruction=positive_reconstruction, negative_reconstruction=negative_reconstruction)
        
        triplet_loss = self.triplet_loss(z_anchor=z_anchor, z_positive=z_positive, z_negative=z_negative)

        total_triplet_loss = reconstruction_loss + triplet_loss
        
        return total_triplet_loss