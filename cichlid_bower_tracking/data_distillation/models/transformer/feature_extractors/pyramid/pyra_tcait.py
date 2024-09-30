from typing import List, Optional, Tuple
from collections import OrderedDict

from data_distillation.models.transformer.feature_extractors.pyramid.pyra_tcait_stage import PyraTCAiTStage as Stage
from data_distillation.models.transformer.feature_extractors.pyramid.pyra_tcait_stage_2 import PyraTCAiTStage2 as Stage2

import torch.nn as nn
import torch

import timm
import math

class PyraTCAiT(nn.Module):
    def __init__(self, embed_dims: List[int], head_counts: List[int], mlp_ratios: List[int], sr_ratios: List[int], depths: List[int], num_stages=4, dropout=0.1, first_patch_dim=4, in_channels=3, in_dim=224, init_alpha=0.0, init_beta=0.0, add_classifier=True, use_improved=False, classification_intent=False, num_classes: Optional[int]=None):
        '''
        Initializes an instance of the PyraTCAiT class; inspired by "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions" by Wang et al. (2021.)

        Inputs:
            embed_dims: a list of embedding dimensions for each stage.
            head_counts: a list of the number of attention heads to use in each stage.
            mlp_ratios: a list of the MLP expansion ratios to use in each stage.
            sr_ratios: a list of the SRA reduction ratios to use in each stage.
            depths: a list of depths for each stage.
            num_stages: the number of stages to be used in the model; defaults to 4.
            dropout: the dropout probability to be used throughout each stage; defaults to 0.1.
            first_patch_dim: the patch size to be used in the first patch embedding; defaults to 4.
            in_channels: the number of channels in the input image for the first stage (1 for greyscale, 3 for RGB); defaults to 3.
            in_dim: the dimension of the input image for the first stage; defaults to 224.
            add_classifier: a Boolean indicating whether or not to use a classifier layer at the end of the model; defaults to False.
            num_classes: the (optional) number of classes to use in the classifier; defaults to None, only effective if add_classifier is True.
        '''

        super(PyraTCAiT, self).__init__()
        
        self.__version__ = '0.1.0'

        self.num_stages = num_stages
        assert len(embed_dims) == self.num_stages, f'Invalid embed_dims input: should be of length {self.num_stages} (got length {len(embed_dims)})'
        assert len(head_counts) == self.num_stages, f'Invalid head_counts input: should be of length {self.num_stages} (got length {len(head_counts)})'
        assert len(mlp_ratios) == self.num_stages, f'Invalid mlp_ratios input: should be of length {self.num_stages} (got length {len(mlp_ratios)})'
        assert len(sr_ratios) == self.num_stages, f'Invalid sr_ratios input: should be of length {self.num_stages} (got length {len(sr_ratios)})'
        assert len(depths) == self.num_stages, f'Invalid depths input: should be of length {self.num_stages} (got length {len(depths)})'

        self.embed_dims = embed_dims
        self.head_counts = head_counts
        self.mlp_ratios = mlp_ratios
        self.sr_ratios = sr_ratios
        self.depths = depths
        self.dropout = dropout

        self.first_patch_dim = first_patch_dim
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.init_alpha = init_alpha
        self.init_beta = init_beta

        self.add_classifier = add_classifier
        self.use_improved = use_improved
        self.classification_intent = classification_intent
        self.num_classes = num_classes

        stages = []
        for i in range(self.num_stages):
            stage_i = Stage(embed_dim=self.embed_dims[i],
                            in_channels=self.in_channels if i == 0 else self.embed_dims[i - 1],
                            in_dim=self.in_dim if i == 0 else int(self.in_dim // math.pow(2, i + 1)),
                            num_heads=self.head_counts[i],
                            depth=self.depths[i], 
                            stage_num=i,
                            patch_dim=self.first_patch_dim if i == 0 else 2,
                            dropout=self.dropout,
                            mlp_ratio=self.mlp_ratios[i],
                            sr_ratio=self.sr_ratios[i],
                            add_cls=(self.add_classifier and i == self.num_stages - 1)) if not self.use_improved \
                      else Stage2(embed_dim=self.embed_dims[i],
                                  in_channels=self.in_channels if i == 0 else self.embed_dims[i - 1],
                                  in_dim=self.in_dim if i == 0 else int(self.in_dim // math.pow(2, i + 1)),
                                  num_heads=self.head_counts[i],
                                  depth=self.depths[i], 
                                  stage_num=i,
                                  patch_dim=self.first_patch_dim if i == 0 else 2,
                                  dropout=self.dropout,
                                  mlp_ratio=self.mlp_ratios[i],
                                  sr_ratio=self.sr_ratios[i],
                                  add_cls=(self.add_classifier and i == self.num_stages - 1),
                                  cls_intent=self.classification_intent)
                        
            stages.append(stage_i)

        self.stages = nn.Sequential(*stages)

        if self.add_classifier:
            assert self.num_classes is not None and self.num_classes > 0, f'Invalid num_classes input: should be an integer greater than 0 to use classifier (got {self.num_classes})'

            self.drop = nn.Dropout(self.dropout)
            self.mlp = nn.Linear(in_features=self.embed_dims[-1], out_features=self.num_classes)

        self.apply(self._init_weights)

    def __str__(self) -> str:
        '''
        Returns a string representation of the current PyraTCAiT model.

        Inputs: None.

        Returns:
            string: a string representation of the current PyraTCAiT model.
        '''

        stages_string, self.num_params = '', 0
        for stage in self.stages:
            stages_string += str(stage) + '\n'
            self.num_params += stage.num_params

        footer_string = f'{"=" * 110}\n'
        footer_string += f'{("PyraT-CAiT # PARAMS" if not self.add_classifier else "Pre-classifier PyraT-CAiT # PARAMS"):70s} | {self.num_params:35d}\n'

        classifier_string = ''
        if self.add_classifier:
            classifier_string = f'\nClassifier\n{"=" * 110}\n'
            classifier_string += f'{"Name":70s} | {"Params":12s} | {"Size":20s}\n'
            classifier_string += f'{"-" * 110}\n'

            total_num_params = 0
            for name, param in self.mlp.named_parameters():
                if not param.requires_grad:
                    continue
                
                num_params = param.numel()
                total_num_params += num_params

                classifier_string += f'{name:70s} | {(num_params):12d} | {str(tuple(param.size())):20s}\n'
            
            classifier_string += f'{"-" * 110}\n'
            classifier_string += f'{"CLASSIFIER # PARAMS":70s} | {total_num_params:35d}\n'

            self.num_params += total_num_params

            classifier_string += f'\n{"=" * 110}\n'
            classifier_string += f'{"FULL PyraT-CAiT # PARAMS":70s} | {self.num_params:35d}\n'

        string = stages_string + footer_string + classifier_string

        return string
    
    def _init_weights(self, m: nn.Module) -> None:
        '''
        Given a module, initializes the weights.

        Input:
            m: a PyTorch module.

        Returns: nothing.
        '''

        if isinstance(m, nn.Linear):
            timm.layers.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups

            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()
    
    def _replace_classifier(self, new_num_classes: int) -> int:
        '''
        Replaces the MLP clasifier head in preparation for fine-tuning.

        Inputs:
            new_num_classes: the new number of classes to be used during fine-tuning.

        Returns:
            old_num_classes: the old number of classes used during pre-training.
        '''
        
        old_num_classes = self.num_classes
        
        old_params = OrderedDict()
        for name, param in self.mlp.named_parameters():
            old_params[name] = param.data.clone()

        new_mlp = nn.Linear()

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
        
    def prepare_for_finetuning(self, new_dim: int, new_num_classes: Optional[int]=None) -> None:
        '''
        Prepares the whole model for fine-tuning.

        Inputs:
            new_dim: the new dimension of images in the fine-tuning datset.
            new_num_classes: the new number of classes in the fine-tuning dataset.
        '''

        if self.add_classifier:
            assert new_num_classes is not None and self.num_classes > 0, f'Invalid num_classes input: should be an integer greater than 0 to use classifier (got {new_num_classes})'

        for i in range(self.depth):
            stage_i = self.stages[i]
            old_in_dim = stage_i.prepare_for_finetuning(new_dim=new_dim if i == 0 else int(math.pow(self.in_dim // math.pow(2, i + 1), 2)))

            print(f'Stage {i} prepared for fine-tuning: input image dimension changed from {old_in_dim} to {int(math.pow(self.in_dim // math.pow(2, i + 1), 2))}.')

        if self.add_classifier:
            old_num_classes = self._replace_classifier(new_num_classes=new_num_classes)
            print(f'Classifier prepared for fine-tuning: number of classes changed from {old_num_classes} to {new_num_classes}.')

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        '''
        Passes the input triplet of images through the full PyraTCAiT model.

        Inputs:
            anchor: a batch of anchor images.
            positive: a batch of positive images (similar to the anchor images).
            negative: a batch of negative images (dissimilar from the anchor images).

        Returns:
            z_anchor: the embedded anchor batch.
            z_positive: the embedded positive batch.
            z_negative: the embedded negative batch.
            pred: the output of the MLP classifier head if self.add_classifier is True, otherwise None. 
        '''

        for stage in self.stages:
            anchor, positive, negative = stage(anchor, positive, negative)

        pred = None
        if self.add_classifier:
            # anchor_cls = anchor[:, 0]
            # pred = self.mlp(anchor_cls)

            # print(f'anchor shape: {anchor.shape}')
            # print(f'mlp.weight shape: {self.mlp.weight.shape}')

            pred = self.mlp(self.drop(anchor.mean(dim=1)))

        z_anchor, z_positive, z_negative = anchor, positive, negative

        return z_anchor, z_positive, z_negative, pred