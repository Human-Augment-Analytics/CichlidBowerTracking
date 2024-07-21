import os

from typing import Optional, Tuple
from collections import OrderedDict

from data_distillation.models.transformer.feature_extractors.extractor import Extractor
from data_distillation.models.transformer.feature_extractors.classifier import Classifier

import torch.nn as nn
import torch

class TripletCrossAttentionViT(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_extractor_heads: int, num_classifier_heads: int, in_channels=3, in_dim=256, extractor_depth=8, extractor_dropout=0.1, \
                 extractor_mlp_ratio=4.0, extractor_patch_dim: Optional[int]=16, extractor_patch_kernel_size: Optional[int]=3, extractor_patch_stride: Optional[int]=2, \
                 extractor_patch_ratio: Optional[float]=8.0, extractor_patch_ratio_decay: Optional[float]=0.5, extractor_patch_n_convs: Optional[int]=5, extractor_sr_ratio: Optional[int]=2, \
                 extractor_use_minipatch=False, extractor_use_sra=False, classifier_depth=2, classifier_dropout=0.1, classifier_mlp_ratio=4.0):
        
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
            extractor_sr_ratio: the spatial reduction ratio to be used by SRA in the extractor; defaults to 2 (only effective if use_sra is True).
            extractor_use_minipatch: indicates whether or not the extractor uses mini-patch embedding instead of standard patch embedding; defaults to False.
            extractor_use_sra: indicates whether or not the extractor should use SRA instead of standard self-attention; defaults to False.
            classifier_depth: the number of transformer blocks to pass the input embedding through in the classifier; defaults to 2.
            classifier_dropout: the dropout probability used by each transformer block in the classifier; defaults to 0.1.
            classifier_mlp_ratio: the size of the hidden layer in each transformer block's MLP in the classifier, also used for scaling the MLP in the head of the classifier; defaults to 4.0.
        '''

        super(TripletCrossAttentionViT, self).__init__()
        
        self.__version__ = '0.1.1'

        self.extractor = Extractor(embed_dim=embed_dim, num_heads=num_extractor_heads, in_channels=in_channels, in_dim=in_dim, depth=extractor_depth, dropout=extractor_dropout, sr_ratio=extractor_sr_ratio, \
                                   mlp_ratio=extractor_mlp_ratio, patch_dim=extractor_patch_dim, patch_kernel_size=extractor_patch_kernel_size, patch_stride=extractor_patch_stride, \
                                   patch_ratio=extractor_patch_ratio, patch_ratio_decay=extractor_patch_ratio_decay, patch_n_convs=extractor_patch_n_convs, use_minipatch=extractor_use_minipatch, use_sra=extractor_use_sra)
        self.classifier = Classifier(embed_dim=embed_dim, num_heads=num_classifier_heads, num_classes=num_classes, depth=classifier_depth, dropout=classifier_dropout, mlp_ratio=classifier_mlp_ratio)

    def __str__(self) -> str:
        '''
        Returns a string representation of the entire T-CAiT model.

        Inputs: None.

        Returns:
            string: a string representation of the entire T-CAiT model.
        '''

        extractor_string = str(self.extractor)
        classifier_string = str(self.classifier)

        self.num_params = self.extractor.num_params + self.classifier.num_params

        string = extractor_string + '\n' + classifier_string + '\n'
        string += f'{"=" * 90}\n'
        string += f'{"TRIPLET CROSS ATTENTION ViT # PARAMS":50s} | {self.num_params:35d}\n'

        return string
    
    def _unfreeze_extractor(self) -> None:
        '''
        Unfreezes the parameters in the extractor.

        Inputs: None.
        '''

        for param in self.extractor.parameters():
            param.requires_grad = True
    
    def freeze_extractor(self) -> None:
        '''
        Freezes the parameters in the extractor.

        Inputs: None.
        '''

        for param in self.extractor.parameters():
            param.requires_grad = False

    def save_weights(self, filepath: str) -> None:
        '''
        Saves the weights of the full T-CAiT model to the passed filepath.

        Inputs:
            filepath: a system path to the file where the model weights will be saved.
        '''

        filedir = filepath.rstrip('/ ').split('/')[:-1]
        assert os.path.exists(filedir), f'Invalid filepath: filedir {filedir} does not exist!'

        print(f'Saving model weights to {filepath}...')
        torch.save(self.state_dict(), filepath)
        print(f'Model weights saved successfully to {filepath}!')

    def save_extractor_weights(self, filepath: str) -> None:
        '''
        Saves the weights of just the extractor component to the passed filepath.

        Inputs:
            filepath: a system path to the file where the extractor weights will be saved.
        '''

        filedir = filepath.rstrip('/ ').split('/')[:-1]
        assert os.path.exists(filedir), f'Invalid filepath: filedir {filedir} does not exist!'

        print(f'Saving extractor weights to {filepath}...')
        torch.save(self.extractor.state_dict(), filepath)
        print(f'Extractor weights saved successfully to {filepath}!')
    
    def load_extractor_weights(self, filepath: str) -> None:
        '''
        Loads the extractor weights stored at the passed filepath into the extractor.

        Inputs:
            filepath: a system path to the file where the extractor weights are saved.
        '''

        assert os.path.exists(filepath), f'Invalid filepath: {filepath} does not exist!'

        print(f'Loading extractor weights from {filepath}...')
        self.extractor.load_state_dict(torch.load(filepath))
        print(f'Extractor weights loaded successfully from {filepath}!')

    def prepare_for_finetuning(self, new_dim: int, new_num_classes: int, new_mlp_ratio: float) -> None:
        '''
        Prepares the Classifier of a pre-trained model for fine-tuning by replacing the head of the MLP.

        Inputs:
            new_dim: the dimension of the images to be used in fine-tuning.
            new_num_classes: the number of classes in the fine-tuning dataset.
            new_mlp_ratio: the ratio to be used in scaling the hidden layers of the classifier's MLP.
        '''

        old_img_dim = self.extractor.prepare_for_finetuning(new_dim=new_dim)
        print(f'Extractor prepared for fine-tuning: image dimension changed from {old_img_dim} to {new_dim}.')

        old_num_classes = self.classifier.prepare_for_finetuning(new_num_classes=new_num_classes, new_mlp_ratio=new_mlp_ratio)
        print(f'Classifier prepared for fine-tuning: number of classes changed from {old_num_classes} to {new_num_classes}.')

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