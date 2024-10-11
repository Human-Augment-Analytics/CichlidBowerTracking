from typing import Tuple, Dict

import pandas as pd
import numpy as np
import timm

class TripletMiner:
    def __init__(self, num_triplets: int, diff_ratios=[0.25, 0.5, 0.25], pretr_model=None):
        _valid_pretr_models = {'resnet152', 'vit_large_patch16_224_in21k', 'vit_large_patch14_224_clip_laion2b', 'tf_efficientnet_b5.ns_jft_in1k', 'tf_efficientnet_b6.ns_jft_in1k', 'deit_base_distilled_patch16_224.fb_in1k', 'seresnext101d_32x8d.ah_in1k'}

        assert sum(diff_ratios) == 0, f'Invalid Difficulty Ratios: need to sum to zero (got sum {sum(diff_ratios)})'
        if pretr_model is not None:
            assert pretr_model in _valid_pretr_models, f'Invalid Pre-trained Model: must pick from {_valid_pretr_models} (got {pretr_model})'

        self.num_triplets = num_triplets
        self.diff_ratios = diff_ratios
        self.pretr_model = pretr_model

        self.model = None
        if pretr_model == 'resnet152':
            self.model = timm.create_model(model_name='resnet152', pretrained=True, features_only=True)
        elif pretr_model == 'vit_large_patch16_224_in21k':
            self.model = timm.create_model(model_name='vit_large_patch16_224_in21k', pretrained=True, features_only=True)
        elif pretr_model == 'vit_large_patch14_224_clip_laion2b':
            self.model = timm.create_model(model_name='vit_large_patch14_224_clip_laion2b', pretrained=True, features_only=True)
        elif pretr_model == 'tf_efficientnet_b5.ns_jft_in1k':
            self.model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k', pretrained=True, features_only=True)
        elif pretr_model == 'tf_efficientnet_b6.ns_jft_in1k':
            self.model = timm.create_model('tf_efficientnet_b6.ns_jft_in1k', pretrained=True, features_only=True)
        elif pretr_model == 'deit_base_distilled_patch16_224.fb_in1k':
            self.model = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=True, features_only=True)
        elif pretr_model == 'seresnext101d_32x8d.ah_in1k':
            self.model = timm.create_model('seresnext101d_32x8d.ah_in1k', pretrained=True, features_only=True)

    def _hard_mine(self, embeddings: Dict[int, np.ndarray], anchor: int) -> Tuple[np.ndarray, np.ndarray]:
        tmp = {identity: embedding for identity, embedding in embeddings.items() if identity != anchor}

        hard_pos, hard_neg = None, None
        # ===================================================
        # TODO: Implement hard-mining for...
        #   - positives
        #   - negatives
        # ===================================================



        del tmp
        return hard_pos, hard_neg
    
    def embed()