from typing import Union

from embeddings.patch_embedding import PatchEmbedding
from embeddings.mini_patch_embedding import MiniPatchEmbedding
from embeddings.positional_encoding import PositonalEncoding
from transformer_encoder import TransformerEncoder

import torch.nn as nn
import torch

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, patcher: Union[PatchEmbedding, MiniPatchEmbedding], n_encoders=8, p_dropout=0.1, mlp_ratio=4.0):
        '''
        Initializes an instance of the ViTEncoder class.

        Inputs:
            embed_dim: the embedding dimension of the output.
            n_heads: the number of heads to be used by each TransformerEncoder's Multi-Head Attention.
            patcher: a PyTorch module used in obtaining patch embeddings; must be an instance of either the PatchEmbedding or MiniPatchEmbedding classes.
            n_encoders: the number of TransformerEncoders to include in the encoder stack (n_encoders - 1 if isinstance(patcher, MiniPatchEmbedding), otherwise n_encoders); defaults to 8.
            p_dropout: the dropout probability parameter for each TransformerEncoder; defaults to 0.1. 
            mlp_ratio: indicates the size of the MLP's hidden layer relative to the embedding dimension in each TransformerEncoder; defaults to 4.0.
        '''
        
        super(ViTEncoder, self).__init__()

        assert isinstance(patcher, PatchEmbedding) or isinstance(patcher, MiniPatchEmbedding)

        self.__version__ = '0.1.0'
        
        self.embed_dim = embed_dim
        self.nheads = n_heads

        self.patcher = patcher
        self.pos_enc = PositonalEncoding(embed_dim=self.embed_dim, n_patches=self.patcher.npatches)

        self.p_dropout = p_dropout
        self.mlp_ratio = mlp_ratio

        self.use_minipatch = isinstance(self.patcher, MiniPatchEmbedding)
        self.nencoders = (n_encoders - 1) if self.use_minipatch else n_encoders
        
        self.encoder_stack = [TransformerEncoder(embed_dim=embed_dim, n_heads=n_heads, p_dropout=self.p_dropout,mlp_ratio=self.mlp_ratio) for _ in range(self.nencoders)]

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Generates a feature embedding for each image in the passed batch.

        Inputs:
            x: a PyTorch Tensor representing a batch of images.

        Returns:
            z: a PyTorch Tensor representing a batch of feature embeddings associated with each image in the input.
        '''

        out = self.patcher(x)
        out = self.pos_enc(out)

        for i in range(self.nencoders):
            out = self.encoder_stack[i](out)

        z = self.norm(out)        

        return z