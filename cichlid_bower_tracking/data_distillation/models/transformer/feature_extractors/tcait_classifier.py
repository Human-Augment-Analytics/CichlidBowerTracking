from typing import Optional
from collections import OrderedDict

from data_distillation.models.transformer.transformer_block import TransformerBlock

import torch.nn as nn
import torch

class TCAiTClassifier(nn.Module):
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

        super(TCAiTClassifier, self).__init__()

        self.__version__ = '0.1.2'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth

        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim=self.embed_dim, n_heads=self.num_heads, p_dropout=self.dropout, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)])

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=int(self.embed_dim * self.mlp_ratio)),
            nn.BatchNorm1d(num_features=int(self.embed_dim * self.mlp_ratio)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.embed_dim * self.mlp_ratio), out_features=int(self.embed_dim * (self.mlp_ratio ** 2))),
            nn.BatchNorm1d(num_features=int(self.embed_dim * (self.mlp_ratio ** 2))),
            nn.ReLU(),
            nn.Linear(in_features=int(self.embed_dim * (self.mlp_ratio ** 2)), out_features=self.num_classes)
        )

    def __str__(self) -> str:
        '''
        Returns a string representation of the Classifier component.

        Inputs: None.

        Returns:
            string: a string representation of the Classifier component.
        '''

        string = f'Classifier\n{"=" * 90}\n'
        string += f'{"Name":50s} | {"# Params":12s} | {"Size":20s}\n'
        string += f'{"-" * 90}\n'

        total_num_params = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            num_params = param.numel()
            total_num_params += num_params

            string += f'{name:50s} | {(num_params):12d} | {str(tuple(param.size())):20s}\n'

        string += f'{"-" * 90}\n'
        string += f'{"TOTAL CLASSIFIER # PARAMS":50s} | {total_num_params:35d}\n'

        self.num_params = total_num_params

        return string

    def prepare_for_finetuning(self, new_num_classes: int, new_mlp_ratio: Optional[float]=None) -> int:
        '''
        Prepares a pre-trained Classifier for fine-tuning by replacing the head of the MLP.

        Inputs:
            new_num_classes: the number of classes in the fine-tuning dataset.
            new_mlp_ratio: the ratio to be used in resizing MLP hidden layers during fine-tuning.

        Returns:
            old_num_classes: the number of classes in the pre-training dataset.
        '''

        old_num_classes = self.num_classes
        
        old_params = OrderedDict()
        for name, param in self.mlp.named_parameters():
            old_params[name] = param.data.clone()

        new_mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=int(self.embed_dim * new_mlp_ratio)),
            nn.BatchNorm1d(num_features=int(self.embed_dim * new_mlp_ratio)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.embed_dim * self.mlp_ratio), out_features=int(self.embed_dim * (new_mlp_ratio ** 2))),
            nn.BatchNorm1d(num_features=int(self.embed_dim * (new_mlp_ratio ** 2))),
            nn.ReLU(),
            nn.Linear(in_features=int(self.embed_dim * (new_mlp_ratio ** 2)), out_features=new_num_classes)
        )

        with torch.no_grad():
            for (_, old_param), (_, new_param) in zip(old_params.items(), new_mlp.named_parameters()):
                if new_param.shape == old_param.shape:
                    new_param.copy_(old_param)
                elif len(new_param.shape) == len(old_param.shape) == 2:
                    min_in = min(old_param.shape[1], new_param.shape[1])
                    min_out = min(old_param.shape[0], new_param.shape[0])
                    new_param[:min_out, :min_in].copy_(old_param[:min_out, :min_in])

        self.mlp = new_mlp
        self.num_classes = new_num_classes

        return old_num_classes

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