import torch.nn as nn
import torch

import math

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int, batch_size=16, in_channels=3, patch_dim=16, add_norm=False):
        '''
        Initializes an instance of the PatchEmbedding class.

        Inputs:
            embed_dim: essentially the number of output channels of the patch embedding comvolution.
            batch_size: the number of images per batch; defaults to 16 [deprecated: has no effect on output, soon to be removed].
            in_channels: the number of channels in each input image (1 of greyscale, 3 if RGB); defaults to 3.
            in_dim: the dimension of each input image; defaults to 256.
            patch_dim: essentially the kernel size of the patch embedding convolution.
            add_norm: a Boolean indicating whether or not the output patch embedding should be passed through a layer norm before reshaping; defaults to False.
        '''
        
        super(PatchEmbedding, self).__init__()

        self.__version__ = '0.1.3'

        self.batch_size = batch_size
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.add_norm = add_norm

        self.patch_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_dim, stride=self.patch_dim)
        if self.add_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.embed_dim)

        self.flatten = nn.Flatten(start_dim=2)

    def get_num_patches(self, new_dim: int) -> int:
        '''
        Given a new dimension, calculates the new number of patches.

        Inputs:
            new_dim: an int representing the new image dimension.

        Returns:
            new_num_patches: the number of patches in an image with the new dimension.
        '''

        assert new_dim % self.patch_dim == 0
        new_num_patches = int(math.pow(new_dim // self.patch_dim, 2))
        
        return new_num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs patch embedding on the input image batch Tensor.

        Inputs:
            x: a PyTorch Tensor representing a batch of images to be used in patch embedding.

        Outputs:
            out: the computed patch embeddings PyTorch Tensor.
        '''

        assert len(x.shape) == 4 
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == x.shape[3]

        # embed
        out = self.patch_conv(x) # shape (N, C_embed, D_out, D_out)
        if self.add_norm:
            out = self.norm(out)

        # reshape
        out = self.flatten(out) # shape (N, C_embed, D_out * D_out)
        out = torch.transpose(out, dim0=1, dim1=2) # shape (N, D_out * D_out, C_embed)
        
        return out