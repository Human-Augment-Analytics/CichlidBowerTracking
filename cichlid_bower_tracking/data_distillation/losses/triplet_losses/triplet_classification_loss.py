from triplet_losses.triplet_loss import TripletLoss

import torch.nn as nn
import torch

class TripletClassificationLoss(nn.Module):
    def __init__(self, margin=1.0, p_norm=2):
        '''
        Initializes an instance of the TripletClassificationLoss class.

        Inputs:
            margin: the margin parameter to be used in the triplet loss.
            p_norm: the norm to be used in the triplet loss.
        '''

        self.__version__ = '0.1.0'

        self.triplet_loss = TripletLoss(margin=margin, p_norm=p_norm)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor, z_negative: torch.Tensor, \
                y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        '''
        Computes the sum of the triplet loss (using the image embeddings) and the cross entropy loss (using the predicted anchor class).

        Inputs:
            z_anchor: the anchor image embeddings.
            z_positive: the positive image embeddings (similar to the anchor images).
            z_negative: the negative_image_embeddings (dissimilar from the anchor images).
            y_pred: the predicted classes for the anchor images.
            y_true: the true classes of the anchor images.

        Returns:
            total_loss: the sum of the triplet loss and cross entropy loss.
        '''
        
        triplet_loss = self.triplet_loss(z_anchor, z_positive, z_negative)
        ce_loss = self.ce_loss(y_pred, y_true)

        total_loss = triplet_loss + ce_loss

        return total_loss