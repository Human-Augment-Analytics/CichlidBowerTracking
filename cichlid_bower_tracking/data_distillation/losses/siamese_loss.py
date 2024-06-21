from contrastive_loss import ContrastiveLoss
from pair_loss import PairLoss

import torch.nn as nn
import torch

class SiameseLoss(nn.Module):
    def __init__(self, reconstruction_criterion: str, distance_metric: str, p_norm=2.0, distance_eps=1e-6, cosine_eps=1e-8, margin=1.0):
        '''
        Initializes an instance of the SiameseLoss PyTorch module.

        Inputs:
            reconstruction_criterion: a string indicating which reconstructive loss metric to use in the pair loss; must either be 'l2' or 'l1'.
            distance_metric: a string indicating which distance metric to use in contrastive loss; must either be 'pairwise' or 'cosine'.
            p_norm: the p-value for the norm used by pairwise distance; only useful if distance_metric == 'pairwise'.
            distance_eps: the value to be used for the eps hyperparameter of the pairwise distance; only useful if distance_metric == 'pairwise'.
            cosine_eps: the value to be used for the eps hyperparameter of the cosine similarity; only useful if distance_metric == 'cosine'.
            margin: the value to be used for the margin hyperparameter of the contrastive loss.
        '''

        super(SiameseLoss, self).__init__()

        self.p_norm = p_norm
        self.distance_eps = distance_eps
        self.cosine_eps = cosine_eps
        self.margin = margin

        if reconstruction_criterion == 'l2':
            self.reconstruction_criterion = nn.MSELoss()
        elif reconstruction_criterion == 'l1':
            self.reconstruction_criterion = nn.L1Loss()
        else:
            raise Exception(f'value passed to reconstruction_criterion hyperparameter must either be \'l2\' or \'l1\' ({reconstruction_criterion} passed).')
        
        if distance_metric == 'pairwise':
            self.distance_metric = nn.PairwiseDistance(p=self.p_norm, eps=self.distance_eps)
        elif distance_metric == 'cosine':
            self.distance_metric = nn.CosineSimilarity(eps=self.cosine_eps)
        else:
            raise Exception(f'value passed to distance_metric hyperparameter must either be \'pairwise\' or \'cosine\' ({distance_metric}) passed.')

        self.pair_loss = PairLoss(self.reconstruction_criterion)
        self.contrastive_loss = ContrastiveLoss(self.distance_metric, margin=self.margin)

    def forward(self, y: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor, x1_reconstruction: torch.Tensor, x2_reconstruction: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Siamese loss (sum of the pair loss and contrastive loss) for each pair of images.

        Inputs:
            y: a PyTorch Tensor representing the batch of similarity labels (0 for similar/same fish, 1 for dissimilar/different fish) for the feature embeddings.
            x1: a PyTorch Tensor representing a set of images from a batch.
            x2: a PyTorch Tensor representing another set of images from the batch.
            z1: a PyTorch Tensor representing the set of feature embeddings associated with image set x1.
            z2: a PyTorch Tensor representing the set of feature embeddings associated with image set x2.
            x1_reconstruction: a PyTorch Tensor representing the reconstructions of the images in set x1.
            x2_reconstruction: a PyTorch Tensor representing the reconstructions of the images in set x2.

        Returns:
            siamese_loss: the Siamese loss between a batch of reconstructed image and feature embedding pairs.
        '''
        
        reconstruction_loss = self.pair_loss(x1_reconstruction, x2_reconstruction, x1, x2)
        contrastive_loss = self.contrastive_loss(y, z1, z2)

        siamese_loss = reconstruction_loss + contrastive_loss

        return siamese_loss