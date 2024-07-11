import torch.nn as nn
import torch

class PositonalEncoding(nn.Module):
    def __init__(self, embed_dim: int, n_patches: int, add_one=False):
        '''
        Initializes an instance of the PositionalEncoding class.

        Inputs:
            embed_dim: the embedding dimension of the input and output.
            n_patches: an int value representing the total number of patches in the passed embedding.
        '''
        
        super(PositonalEncoding, self).__init__()

        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.add_one = add_one

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1 * self.add_one, self.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Adds a positional encoding to the input embedding.

        Inputs:
            x: a PyTorch Tensor representing a batch of embeddings.
        
        Returns:
            out: a PyTorch Tensor representing the position-encoded embeddings batch.
        '''

        out = x + self.pos_embedding

        return out