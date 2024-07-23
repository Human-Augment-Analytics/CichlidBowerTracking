from data_distillation.models.transformer.attention_mechs.spatial_reduction_attention import SpatialReductionAttention as SRA
from data_distillation.models.transformer.embeddings.positional_encoding import PositionalEncoding

import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, p_dropout=0.1, mlp_ratio=4.0, use_sra=False, sr_ratio=2):
        '''
        Initializes an instance of the TransformerEncoder class.

        Inputs:
            embed_dim: the embedding dimension of the input and output.
            n_heads: the number of heads to be used by the Multi-Head Attention.
            p_dropout: the dropout probability parameter; defaults to 0.1.
            mlp_ratio: indicates the size of the MLP's hidden layer relative to the embedding dimension; defaults to 4.0.
            use_sra: indicates that SRA should be used instead of standard self-attention; defaults to False.
            sr_ratio: the spatial reduction ratio used by SRA; defaults to 2.
        '''

        super(TransformerBlock, self).__init__()

        self.__version__ = '0.2.0'
        
        self.embed_dim = embed_dim
        self.nheads = n_heads
        self.p_dropout = p_dropout
        self.mlp_ratio = mlp_ratio
        self.use_sra = use_sra
        self.sr_ratio = sr_ratio

        self.norm1 = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.nheads, dropout=self.p_dropout) if not self.use_sra else SRA(embed_dim=self.embed_dim, num_heads=self.nheads, dropout=self.p_dropout, sr_ratio=self.sr_ratio)
        self.norm2 = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=int(self.embed_dim * self.mlp_ratio)),
            nn.GELU(),
            nn.Linear(in_features=int(self.embed_dim * self.mlp_ratio), out_features=self.embed_dim),
            nn.Dropout(p=self.p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass of the TransformerEncoder class, implemented as depicted in 'Vision Transformer for Contrastive Clustering' by Ling et al. (2022).

        Inputs:
            x: a PyTorch Tensor representing a batch of image embeddings (post patch/mini-patch embedding and positional encoding).

        Returns:
            out: a PyTorch Tensor representing the output of the TransformerEncoder, applied to the input Tensor x.
        '''

        out1 = self.norm1(x)
        out1, _ = self.attention(out1, out1, out1)

        out1 += x

        out2 = self.norm2(out1)
        out2 = self.mlp(out2)

        out = out2 + out1

        return out