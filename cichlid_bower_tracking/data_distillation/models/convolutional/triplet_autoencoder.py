from typing import Tuple

from encoder import Encoder
from decoder import Decoder

import torch.nn as nn
import torch

class TripletAutoencoder(nn.Module):
    def __init__(self, features: int, batch_size=16, img_channels=3, img_dim=256, encoder_dropout=0.5):
        '''
        Initializes an instance of the TripletAutoencoder class.

        Inputs:
            features: essentially the size of the intermediate feature embedding.
            batch_size: the number of images in the input batch; defaults to 16 [deprecated: has no effect on output, soon to be removed].
            img_channels: the number of channels in the input images and their reconstructions (1 for greyscale, 3 for RGB); defaults to 256.
            img_dim: the dimension of the input images and their reconstructions; defaults to 256.
            encoder_dropout: the dropout probability of the dropout layer in the encoder.
        '''
        super(TripletAutoencoder, self).__init__()

        self.__version__ = '0.1.1'

        self.features = features

        self.batch_size = batch_size
        self.img_channels = img_channels
        self.img_dim = img_dim

        if 0.0 < encoder_dropout < 1.0:
            self.encoder_dropout = encoder_dropout
        else:
            raise Exception(f'Input to encoder_dropout hyperparameter must be between 0.0 and 1.0 ({encoder_dropout} passed).')
        
        self.encoder = Encoder(self.features, self.batch_size, self.img_channels, self.img_dim, self.encoder_dropout)
        self.decoder = Decoder(self.features, self.batch_size, self.img_channels, self.img_dim)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        '''
        Performs distillation and reconstruction using the passed triplet of image batches.

        Inputs:
            anchor: a PyTorch Tensor representing a batch of images.
            positive: a PyTorch Tensor representing a batch of images similar to the anchor images.
            negative: a PyTorch Tensor representing a batch of images dissimilar from the anchor images.
        
        Returns:
            z_anchor: a PyTorch Tensor representing the batch of feature embedded anchor images.
            z_positive: a PyTorch Tensor representing the batch of feature embedded positive images.
            z_negative: a PyTorch Tensor representing the batch of feature embedded negative images.
            anchor_reconstruction: a PyTorch Tensor containing the reconstructed anchor images.
            positive_reconstruction: a PyTorch Tensor containing the reconstructed positive images.
            negative_reconstruction: a PyTorch Tensor containing the reconstructed negatove images.
        '''
        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)
        z_negative = self.encoder(negative)

        anchor_reconstruction = self.decoder(z_anchor)
        positive_representation = self.decoder(z_positive)
        negative_representation = self.decoder(z_negative)

        return z_anchor, z_positive, z_negative, anchor_reconstruction, positive_representation, negative_representation
    
    def distill(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Given input image(s), peforms distillation and returns embedding(s).

        Inputs:
            x: a PyTorch Tensor representing image(s).

        Returns:
            z: a PyTorch Tensor representing image embedding(s).
        '''

        z = self.encoder(x)

        return z
    
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Given input embedding(s), returns reconstruction(s).
        
        Inputs:
            z: a PyTorch Tensor representing image embedding(s).
        
        Returns:
            x_reconstruction: a PyTorch Tensor representing image reconstruction(s).
        '''
        
        x_reconstruction = self.decoder(z)

        return x_reconstruction