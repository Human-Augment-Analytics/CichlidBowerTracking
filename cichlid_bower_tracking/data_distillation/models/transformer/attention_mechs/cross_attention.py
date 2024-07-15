import torch.nn as nn
import torch

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.1, batch_first=True):
        '''
        Initializes an instance of the CrossAttention class.

        Inputs:
            embed_dim: the dimension used by the input and output embeddings.
            num_heads: the number of heads to use in the attention mechanism.
            dropout: the dropout probability to use in the attention mechanism; defaults to 0.5.
            batch_first: indicates if the batch size is the first dimension of the inut; defaults to True.
        '''

        super(CrossAttention, self).__init__()

        self.__version__ = '0.1.2'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        self.norm = nn.LayerNorm(normalized_shape=self.embed_dim)

    def forward(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        '''
        Performs cross-attention (and layer norm + addition) using the input query and context (key and value).
        
        Inputs:
            q: the query to be used in the attention mechanism.
            c: the context (key and value) to be used in the attention mechanism.

        Returns:
            out: the cross-attention (and layer norm + addition) between q and c.
        '''

        out, _ = self.attention(q, c, c)
        out = out + q
        out = self.norm(out)

        return out
