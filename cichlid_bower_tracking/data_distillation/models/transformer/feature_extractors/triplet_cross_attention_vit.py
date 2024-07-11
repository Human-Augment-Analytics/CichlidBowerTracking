from typing import Optional

from data_distillation.models.transformer.feature_extractors.extractor import Extractor
from data_distillation.models.transformer.feature_extractors.classifier import Classifier

import torch.nn as nn
import torch

class TripletCrossAttentionViT(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_extractor_heads: int, num_classifier_heads: int, in_channels=3, in_dim=256, extractor_depth=8, extractor_dropout=0.1, \
                 extractor_mlp_ratio=4.0, extractor_patch_dim: Optional[int]=16, extractor_patch_kernel_size: Optional[int]=3, extractor_patch_stride: Optional[int]=2, \
                 extractor_patch_ratio: Optional[float]=8.0, extractor_patch_ratio_decay: Optional[float]=0.5, extractor_patch_n_convs: Optional[int]=5, extractor_use_minipatch=False, \
                 classifier_depth=2, classifier_dropout=0.1, classifier_mlp_ratio=4.0):
        
        '''
        Initializes an instance of the TripletCrossAttentionViT (T-CAiT) class.

        Inputs:
            embed_dim: the embedding dimension to be used.
            num_classes: the number of classes in the dataset being trained on.
            num_extractor_heads: the number of heads to be used in the extractor's self-attention and cross attention mechanisms.
            num_classifier_heads: the number of heads to be used in the classifier's self-attention mechansims.
            in_channels: the number of channels in the input image(s); 1 for greyscale and 3 for RGB, defaults to 3.
            in_dim: the size of the input image(s); defaults to 256.
            extractor_depth: the number of encoder blocks to be used in the extractor (depth - 1 if mini-patch embedding, otherwise depth); defaults to 8.
            extractor_dropout: the dropout to be used by the extractor's self-attention and cross attention mechanisms; defaults to 0.1.
            extractor_mlp_ratio: the size of the hidden layer of each transformer block's MLP in the extractor, relative to the embed_dim; defaults to 4.
            extractor_patch_dim: the patch size to be used in extractor's standard patch embedding; defaults to 16 (only effective if use_minipatch is False).
            extractor_patch_kernel_size: the kernel size to be used in extractor's mini-patch embedding; defaults to 3 (only effective if use_minipatch is True).
            extractor_patch_stride: the stride to be used in extractor's mini-patch embedding; defaults to 2 (only effective if use_minipatch is True).
            extractor_patch_ratio: the starting rate at which the number of channels increases in the extractor's mini-patch embedding; defaults to 4.0 (only effective if use_minipatch is True).
            extractor_patch_ratio_decay: the rate at which the patch_ratio decays in the extractor's mini-patch embedding; defaults to 0.5 (only effective if use_minipatch is True).
            extractor_patch_n_convs: the number of convolutions to be used in the extractor's mini-patch embedding; defaults to 5 (only effective if use_minipatch is True).
            extractor_use_minipatch: indicates whether or not the extractor uses mini-patch embedding instead of standard patch embedding; defaults to False.
            classifier_depth: the number of transformer blocks to pass the input embedding through in the classifier; defaults to 2.
            classifier_dropout: the dropout probability used by each transformer block in the classifier; defaults to 0.1.
            classifier_mlp_ratio: the size of the hidden layer in each transformer block's MLP in the classifier, also used for scaling the MLP in the head of the classifier; defaults to 4.0.
        '''
        
        self.__version__ = '0.1.0'

        self.extractor = Extractor(embed_dim=embed_dim, num_heads=num_extractor_heads, in_channels=in_channels, in_dim=in_dim, depth=extractor_depth, dropout=extractor_dropout, \
                                   mlp_ratio=extractor_mlp_ratio, patch_dim=extractor_patch_dim, patch_kernel_size=extractor_patch_kernel_size, patch_stride=extractor_patch_stride, \
                                   patch_ratio=extractor_patch_ratio, patch_ratio_decay=extractor_patch_ratio_decay, patch_n_convs=extractor_patch_n_convs, use_minipatch=extractor_use_minipatch)
        self.classifier = Classifier(embed_dim=embed_dim, num_heads=num_classifier_heads, num_classes=num_classes, depth=classifier_depth, dropout=classifier_dropout, mlp_ratio=classifier_mlp_ratio)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        '''
        Extracts features from the input image batches and predicts the class of the anchor image batch.

        Inputs:
            anchor: the anchor images batch.
            positive: the positive images batch.
            negative: the negative images batch.

        Returns:
            z_anchor: the anchor image embeddings batch.
            z_positive: the positive image embeddings batch.
            z_negative: the negative image embeddings batch.
            pred: the predicted classes for the anchor images batch. 
        '''
        
        z_anchor, z_positive, z_negative = self.extractor(anchor, positive, negative)
        pred = self.classifier(z_anchor)

        return z_anchor, z_positive, z_negative, pred