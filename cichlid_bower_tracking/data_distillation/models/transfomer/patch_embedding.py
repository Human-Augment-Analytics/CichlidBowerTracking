import torch.nn as nn
import torch

import math

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int, batch_size: int, in_channels=3, in_dim=256, patch_dim=16):
        '''
        Initializes an instance of the PatchEmbedding class.

        Inputs:
            embed_dim: essentially the number of output channels of the patch embedding comvolution.
            batch_size: the number of images per batch.
            in_channels: the number of channels in each input image (1 of greyscale, 3 if RGB); defaults to 3.
            in_dim: the dimension of each input image; defaults to 256.
            patch_dim: essentially the kernel size of the patch embedding convolution.
        '''
        
        super(PatchEmbedding, self).__init__()

        self.__version__ = '0.1.0'

        self.batch_size = batch_size
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim

        self.patch_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_dim, stride=self.patch_dim)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs patch embedding on the input image batch Tensor.

        Inputs:
            x: a PyTorch Tensor representing a batch of images to be used in patch embedding.

        Outputs:
            out: the computed patch embeddings PyTorch Tensor.
        '''

        assert x.shape == (self.batch_size, self.in_channels, self.in_dim, self.in_dim) # shape (N, C_in, D_in, D_in)

        out = self.patch_conv(x) # shape (N, C_embed, D_out, D_out)

        self.out_dim = out.shape[-1]

        out = self.flatten(out) # shape (N, C_embed, D_out * D_out)
        out = torch.transpose(out, dim0=1, dim1=2) # shape (N, D_out * D_out, C_embed)

        self.npatches = out.shape[1]
        
        return out