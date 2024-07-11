from data_distillation.models.transformer.transformer_encoder import TransformerEncoder

from collections import OrderedDict

import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_classes: int, depth=2, dropout=0.1, mlp_ratio=4.0):
        '''
        Initializes and instance of the Classifier class.

        Inputs:
            embed_dim: the embedding dimension of the input.
            num_heads: the number of heads to be used by the self-attention mechanisms in the transformer blocks.
            num_classes: the number of classes in the dataset being trained on.
            depth: the number of transformer blocks to pass the input embedding through; defaults to 2.
            dropout: the dropout probability used by each transformer block; defaults to 0.1.
            mlp_ratio: the size of the hidden layer in each transformer block's MLP, also used for scaling the MLP in the head of this classifier; defaults to 4.0.
        '''

        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth

        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        self.blocks = nn.Sequential(*[TransformerEncoder(embed_dim=self.embed_dim, n_heads=self.num_heads, p_dropout=self.dropout, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)])

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=int(self.embed_dim * self.mlp_ratio)),
            nn.BatchNorm1d(num_features=int(self.embed_dim * self.mlp_ratio)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.embed_dim * self.mlp_ratio), out_features=int(self.embed_dim * (self.mlp_ratio ** 2))),
            nn.BatchNorm1d(num_features=int(self.embed_dim * (self.mlp_ratio ** 2))),
            nn.reLU(),
            nn.Linear(in_features=int(self.embed_dim * (self.mlp_ratio ** 2)), out_features=self.num_classes)
        )

    def forward(self, z_anchor: torch.Tensor) -> torch.Tensor:
        '''
        Uses the anchor embeddings from the extractor and predicts its class.

        Inputs:
            z_anchor: the anchor image embeddings batch output by the extractor.
        
        Returns:
            pred: the predicted classes of the images in the anchor batch.
        '''
        
        for block in self.blocks:
            z_anchor = block(z_anchor)

        z_anchor = z_anchor[:, 0]
        pred = self.mlp(z_anchor)

        return pred