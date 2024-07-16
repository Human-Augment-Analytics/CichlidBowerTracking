from typing import Tuple

from data_distillation.models.convolutional.encoder import Encoder
from data_distillation.models.convolutional.decoder import Decoder

import torch.nn as nn
import torch

class SiameseAutoencoder(nn.Module):
    def __init__(self, features: int, batch_size=16, img_channels=3, img_dim=256, encoder_dropout=0.5):
        '''
        Initializes an instance of the SiameseAutoencoder PyTorch module.

        Inputs:
            features: an integer indicating the number of features to which the input should be compressed by the encoder.
            batch_size: an integer indicating the number of images included in each batch during training and evaluation; defaults to 32 [deprecated: has no effect on output, soon to be removed].
            img_channels: an integer indicating the number of channels in the input images; defaults to 3 (assumes RGB over greyscale).
            img_dim: an integer indicating the input images' shared height and width; defaults to 128.
            encoder_dropout: a float indicating what probability should be used in the dropout layer of the encoder; defaults to 0.5, must be in the interval (0, 1).
        '''

        super(SiameseAutoencoder, self).__init__()

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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Performs encoder-based distillation and decoder-based reconstruction on the passed pair of inputs for a forward pass of the model.

        Inputs:
            x1: a PyTorch Tensor representing a batch of images, each with shape (self.batch_size, self.img_channels, self.img_dim, self.img_dim).
            x2: a PyTorch Tensor representing another batch of images, each with shape (self.batch_size, self.img_channels, self.img_dim, self.img_dim).

        Returns:
            z1: the feature embeddings of the input x1, as created by the encoder.
            z2: the feature embeddings of the input x2, as created by the encoder.
            x1_reconstruction: the reconstructions of input x1, as created by the decoder.
            x2_reconstruction: the reconstructions of input x2, as created by the decoder.
        '''

        assert x1.shape == (self.batch_size, self.img_channels, self.img_dim, self.img_dim)
        assert x2.shape == (self.batch_size, self.img_channels, self.img_dim, self.img_dim)

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        x1_reconstruction = self.decoder(z1)
        x2_reconstruction = self.decoder(z2)

        return z1, z2, x1_reconstruction, x2_reconstruction
    
    def distill(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs distillation by passing the input through the encoder to obtain feature embeddings.

        Inputs:
            x: a PyTorch tensor representing a batch of images, each with shape (self.batch_size, self.img_channels, self.img_dim, self.img_dim), to be distilled.

        Returns:
            z: the feature embeddings of the input x, as created by the encoder.
        '''
        
        assert x.shape == (self.batch_size, self.img_channels, self.img_dim, self.img_dim)

        z = self.encoder(x)

        return z
    
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Performs reconstruction by passing the input embedding through the decoder to obtain an image reconstruction.

        Inputs:
            z: a PyTorch tensor representing a batch of feature embeddings, each created by the encoder.

        Returns:
            x_reconstruction: the image reconstructed using the feature embedding z.
        '''

        assert z.shape == (self.batch_size, self.features)

        x_reconstruction = self.decoder(z)

        return x_reconstruction