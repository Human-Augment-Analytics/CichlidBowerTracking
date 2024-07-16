from typing import Union

from data_distillation.models.transformer.embeddings.patch_transpose import PatchTranspose
from data_distillation.models.transformer.embeddings.mini_patch_transpose import MiniPatchTranspose

from data_distillation.models.transformer.transformer_encoder import TransformerEncoder

import torch.nn as nn
import torch

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, transposer: Union[PatchTranspose, MiniPatchTranspose], n_decoders=8, p_dropout=0.1, mlp_ratio=4.0):
        '''
        Initializes an instance of the ViTDecoder class.

        Inputs:
            embed_dim: the embedding dimension of the input.
            n_heads: the number of heads to be used by each TransformerEncoder's Multi-Head Attention.
            transposer: a PyTorch module used in reverse patch embeddings; must be an instance of either the PatchTranspose or MiniPatchTranspose classes.
            n_decoders: the number of TransformerEncoders to include in the decoder stack (n_decoders - 1 if isinstance(transposer, MiniPatchTranspose), otherwise n_decoders); defaults to 8.
            p_dropout: the dropout probability parameter for each TransformerEncoder; defaults to 0.1. 
            mlp_ratio: indicates the size of the MLP's hidden layer relative to the embedding dimension in each TransformerEncoder; defaults to 4.0.
        '''
        
        super(ViTDecoder, self).__init__()

        assert isinstance(transposer, PatchTranspose) or isinstance(transposer, MiniPatchTranspose)
        
        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.nheads = n_heads

        self.p_dropout = p_dropout
        self.mlp_ratio = mlp_ratio

        self.use_minipatch = isinstance(self.transposer, MiniPatchTranspose)
        self.ndecoders = (n_decoders - 1) if self.use_minipatch else n_decoders

        self.decoder_stack = [TransformerEncoder(embed_dim=self.embed_dim, n_heads=self.nheads, p_dropout=self.p_dropout, mlp_ratio=self.mlp_ratio) for _ in range(self.nencoders)]
        self.norm = nn.LayerNorm(self.embed_dim)

        self.transposer = transposer

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Reconstructs each image represented by the embeddings in the passed Tensor.

        Inputs:
            z: a PyTorch Tensor representing a batch of feature embeddings.

        Returns:
            x_reconstruction: a PyTorch Tensor representing a batch of image reconstructions based on the passed embeddings.
        '''

        out = self.decoder_stack[0](z)
        for i in range(1, self.ndecoders):
            out = self.decoder_stack[i](out)

        out = self.norm(out)
        x_reconstruction = self.transposer(out)

        return x_reconstruction