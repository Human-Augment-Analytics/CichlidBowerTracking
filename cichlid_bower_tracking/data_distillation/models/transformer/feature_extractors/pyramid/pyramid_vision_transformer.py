from typing import Tuple

import torch.nn as nn
import torch

import timm
import math

class PyramidVisionTransformer(nn.Module):
    def __init__(self, name: str, num_classes: int, drop_rate=0.0, pretrained=False):
        '''
        Initializes the PyramidVisionTransformer (PVT) wrapper class.

        Inputs:
            name: the name of the PVT variant to use as the backbone.
            num_classes: the number of classes to be used in prediction.
            drop_rate: the dropout rate to use in the MLP head.
            pretrained: indicates whether or not a pre-trained backbone PVT should be used.
        '''

        super(PyramidVisionTransformer, self).__init__()
        assert len(name) > 6 and name.startswith('pvt_v2', 0, 6), f'Invalid PVT model name (got \"{name}\").'

        self.name = name
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.pretrained = pretrained

        self.model = timm.create_model(self.name, pretrained=pretrained, features_only=True)
        self.head_dropout = nn.Dropout(self.drop_rate)
        self.head = nn.Linear(self.model.embed_dims[-1], self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        '''
        Initializes the weights used in the passed PyTorch module.

        Inputs:
            m: a PyTorch module to have its weights initialized.

        Returns: Nothing.
        '''

        if isinstance(m, nn.Linear):
            timm.layers.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Passes the anchor, positive, and negative images through the backbone PVT and then generates predictions for the anchor using the defined MLP head.

        Inputs:
            anchor: the anchor image.
            positive: the positive image that is similar to the anchor.
            negative: the negative image that is dissimilar from the anchor.

        Returns:
            z_anchor: the output anchor embedding from the last stage of the PVT backbone.
            z_positive: the output positive embedding from the last stage of the PVT backbone.
            z_negative: the output negative embedding from the last stage of the PVT backbone.
            pred: the output predictions for the anchor from the MLP head.
        '''
        
        z_anchor = self.model(anchor)[-1]
        z_positive = self.model(positive)[-1]
        z_negative = self.model(negative)[-1]

        pred = self.head(self.head_dropout(z_anchor.mean(dim=(-1, -2))))

        return z_anchor, z_positive, z_negative, pred