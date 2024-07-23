from typing import Tuple

import torch.nn as nn
import torch

import math

class SpatialReductionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads=8, dropout=0.1, sr_ratio=2, batch_first=True):
        '''
        Initializes an instance of the SpatialReductionAttention class, inspired by "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions" by Wang et al. (2021).

        Inputs:
            embed_dim: the embedding dimension.
            num_heads: the number of heads to use in the attention mechansim; defaults to 8.
            dropout: the dropout probability to be used in the attention mechansim; defaults to 0.1.
            sr_ratio: the spatial reduction ratio (kernel size and stride for a 2D convolution); defaults to 2.
            batch_first: indicates that the batch size is the first dimension of the input Tensor's shape; defaults to True.
        '''

        super(SpatialReductionAttention, self).__init__()

        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.sr_ratio = sr_ratio
        self.batch_first = batch_first

        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(normalized_shape=self.embed_dim)

    def _apply_sr(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies spatial reduction to the input Tensor.

        Inputs:
            x: the Tensor to be spatially reduced.
        
        Returns:
            x_red: the spatially reduced x Tensor.
        '''

        batch_size, num_patches, init_embed_dim = x.shape
        dim = int(math.sqrt(num_patches))

        print(f'(B, N, C) = ({batch_size}, {num_patches}, {init_embed_dim})')

        x_red = x.permute(0, 2, 1).reshape(batch_size, init_embed_dim, dim, dim)
        x_red = self.sr(x_red).reshape(batch_size, init_embed_dim, -1).permute(0, 2, 1)
        x_red = self.norm(x_red)

        return x_red        

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple:
        '''
        Applies spatial reduction attention (SRA) to the passed Tensor.

        Inputs:
            q: a query Tensor.
            k: a key Tensor.
            v: a value Tensor.

        Returns:
            out: the result of applying SRA to the input query, key, value triplet.
        '''
        
        if self.sr_ratio > 1:
            k = self._apply_sr(k)
            v = self._apply_sr(v)

        out = self.attention(q, k, v)

        return out