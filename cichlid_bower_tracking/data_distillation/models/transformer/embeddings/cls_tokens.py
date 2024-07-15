import torch.nn as nn
import torch

class CLSTokens(nn.Module):
    def __init__(self, embed_dim: int):
        '''
        Initializes an instance of the CLSTokens class.

        Inputs:
            embed_dim: the embedding dimension to be used.
        '''

        super(CLSTokens, self).__init__()

        self.__version__ = '0.1.1'
        self.embed_dim = embed_dim

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Adds the stored CLS tokens to the input.

        Inputs:
            x: the output of the patch embedding.

        Returns:
            out: the patch_embedding with CLS tokens prepended.
        '''

        batch_size = x.shape[0]
        
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        out = torch.cat((cls_tokens, x), dim=1)

        return out