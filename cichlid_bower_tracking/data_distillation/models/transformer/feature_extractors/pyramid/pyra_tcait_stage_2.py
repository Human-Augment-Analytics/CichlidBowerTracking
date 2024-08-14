from typing import Tuple

from data_distillation.models.transformer.embeddings.patch_embedding import PatchEmbedding
from data_distillation.models.transformer.embeddings.cls_tokens import CLSTokens
from data_distillation.models.transformer.embeddings.positional_encoding import PositionalEncoding
from data_distillation.models.transformer.transformer_block import TransformerBlock
from data_distillation.models.transformer.triplet_cross_attention_transformer_block import TripletCrossAttentionTransformerBlock as TCABlock

import torch.nn.functional as F
import torch.nn as nn
import torch

import math

class PyraTCAiTStage2(nn.Module):
    def __init__(self, embed_dim: int, in_channels: int, in_dim: int, num_heads: int, depth: int, stage_num: int, patch_dim=4, dropout=0.1, mlp_ratio=8.0, sr_ratio=8, init_alpha=0.1, init_beta=0.1, add_cls=False, cls_intent=False):
        '''
        Initializes an instance of the PyraTCAiTStage class.

        Inputs:
            embed_dim: the embedding dimension (number of channels) of the stage's output.
            in_channels: the number of channels in the stage's input.
            in_dim: the dimension of the stage's input.
            num_heads: the number of attention heads to use in SRA and Cross Attention.
            depth: the number of transformer blocks to be used in the stage.
            stage_num: the stage number, reflective of its position in the whole model.
            patch_dim: the patch size to be used for patch embedding; defaults to 4.
            dropout: the dropout probability to be used throughout the model; defaults to 0.1.
            mlp_ratio: the expansion ratio of the MLP's hidden layer in each transformer block; defaults to 8.0.
            sr_ratio: the spatial reduction ratio used by SRA in the transformer blocks; defaults to 8.
            add_cls: a Boolean indicating that CLS tokens should be added to the embeddings; defaults to False.
        '''

        super(PyraTCAiTStage2, self).__init__()

        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.depth = depth
        self.stage_num = stage_num
        self.patch_dim = patch_dim
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.sr_ratio = sr_ratio
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.add_cls = add_cls
        self.cls_intent = cls_intent

        self.patcher = PatchEmbedding(embed_dim=self.embed_dim, in_channels=self.in_channels, patch_dim=self.patch_dim, add_norm=True)
        if self.add_cls:
            self.anchor_cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
            self.positive_cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
            self.negative_cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
        
        self.anchor_pos_encoder = PositionalEncoding(embed_dim=self.embed_dim, n_patches=self.patcher.get_num_patches(self.in_dim), add_one=self.add_cls)
        self.positive_pos_encoder = PositionalEncoding(embed_dim=self.embed_dim, n_patches=self.patcher.get_num_patches(self.in_dim), add_one=self.add_cls)
        self.negative_pos_encoder = PositionalEncoding(embed_dim=self.embed_dim, n_patches=self.patcher.get_num_patches(self.in_dim), add_one=self.add_cls)

        self.transformer_stack = nn.Sequential(*[TransformerBlock(embed_dim=self.embed_dim, n_heads=self.num_heads, p_dropout=self.dropout, mlp_ratio=self.mlp_ratio, use_sra=True, sr_ratio=self.sr_ratio) for _ in range(self.depth - 1)])
        self.tca_block = TCABlock(embed_dim=self.embed_dim, n_patches=self.patcher.get_num_patches(self.in_dim), n_heads=self.num_heads, p_dropout=self.dropout, mlp_ratio=self.mlp_ratio, init_alpha=self.init_alpha, init_beta=self.init_beta, add_one=self.add_cls)

    def __str__(self) -> str:
        '''
        Returns a string representation of the i-th stage of a PyraT-CAiT.

        Inputs: None.

        Returns:
            string: a string representation of the i-th stage of a PyraT-CAiT.
        '''

        string = f'Stage {self.stage_num}\n{"=" * 110}\n'
        string += f'{"Name":70s} | {"Params":12s} | {"Size":20s}\n'
        string += f'{"-" * 110}\n'

        total_num_params = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            num_params = param.numel()
            total_num_params += num_params

            string += f'{name:70s} | {(num_params):12d} | {str(tuple(param.size())):20s}\n'

        string += f'{"-" * 110}\n'
        string += f'{"STAGE " + str(self.stage_num) + " TOTAL # PARAMS":70s} | {total_num_params:35d}\n'

        self.num_params = total_num_params

        return string
    
    def _embed(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        '''
        Performs the initial embedding on the input Tensor triplet.

        Inputs:
            anchor: a batch of anchor images.
            positive: a batch of positive images (similar to the anchor batch).
            negative: a batch of negative images (dissimilar from the anchor batch).

        Returns:
            anchor_embed: the embedded anchor batch.
            positive_embed: the embedded positive batch.
            negative_embed: the embedded negative batch.
        '''

        anchor_embed = self.patcher(anchor)
        positive_embed = self.patcher(positive)
        negative_embed = self.patcher(negative)

        if self.add_cls:
            anchor_embed = self.anchor_cls_tokenizer(anchor_embed)
            positive_embed = self.positive_cls_tokenizer(positive_embed)
            negative_embed = self.negative_cls_tokenizer(negative_embed)

        anchor_embed = self.anchor_pos_encoder(anchor_embed)
        positive_embed = self.positive_pos_encoder(positive_embed)
        negative_embed = self.negative_pos_encoder(negative_embed)

        return anchor_embed, positive_embed, negative_embed
    
    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Reshapes the input Tensor into a standard image structure.

        Inputs:
            x: an embedding Tensor.
        
        Returns:
            out: the reshaped embedding Tensor with a standard image structure.
        '''

        batch_size, num_patches, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        out_dim = int(math.sqrt(num_patches))

        out = torch.transpose(x, dim0=1, dim1=2)
        out = out.reshape(batch_size, embed_dim, out_dim, out_dim)

        return out
    
    def _interpolate_pos_encoding(self, new_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Interpolates the positional encoding parameter to prepare for fine-tuning on larger image data.

        Inputs:
            new_dim: the new image data dimension.
        
        Returns:
            anchor_new_pos_embedding: the new positional embedding for the anchor positional encoder.
            positive_new_pos_embedding: the new positional embedding for the positive positional encoder.
            negative_new_pos_embedding: the new positional embedding for the negative positional encoder.
        '''

        new_n_patches = self.patcher.get_num_patches(new_dim=new_dim)
        old_n_patches = self.anchor_pos_encoder.pos_embedding.shape[1] - 1

        if new_n_patches == old_n_patches:
            return self.anchor_pos_encoder.pos_embedding, self.positive_pos_encoder.pos_embedding, self.negative_pos_encoder.pos_embedding
        
        # interpolate anchor positional encoding weights
        anchor_class_pos_embedding = self.anchor_pos_encoder.pos_embedding[:, 0]
        anchor_old_patch_pos_embedding = self.anchor_pos_encoder.pos_embedding[:, 1:]

        old_size = int(math.sqrt(old_n_patches))
        new_size = int(math.sqrt(new_n_patches))

        anchor_new_patch_pos_embedding = F.interpolate(
            anchor_old_patch_pos_embedding.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2),
            size=(new_size, new_size),
            mode='bicubic'
        )

        anchor_new_patch_pos_embedding = anchor_new_patch_pos_embedding.premute(0, 2, 3, 1).view(1, -1, self.embed_dim)
        anchor_new_pos_embedding = torch.cat((anchor_class_pos_embedding.unsqueeze(0), anchor_new_patch_pos_embedding), dim=1)

        # interpolate positive positional encoding weights
        positive_class_pos_embedding = self.positive_pos_encoder.pos_embedding[:, 0]
        positive_old_patch_pos_embedding = self.positive_pos_encoder.pos_embedding[:, 1:]

        old_size = int(math.sqrt(old_n_patches))
        new_size = int(math.sqrt(new_n_patches))

        positive_new_patch_pos_embedding = F.interpolate(
            positive_old_patch_pos_embedding.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2),
            size=(new_size, new_size),
            mode='bicubic'
        )

        positive_new_patch_pos_embedding = positive_new_patch_pos_embedding.premute(0, 2, 3, 1).view(1, -1, self.embed_dim)
        positive_new_pos_embedding = torch.cat((positive_class_pos_embedding.unsqueeze(0), positive_new_patch_pos_embedding), dim=1)

        # interpolate negative positional encoding weights
        negative_class_pos_embedding = self.negative_pos_encoder.pos_embedding[:, 0]
        negative_old_patch_pos_embedding = self.negative_pos_encoder.pos_embedding[:, 1:]

        old_size = int(math.sqrt(old_n_patches))
        new_size = int(math.sqrt(new_n_patches))

        negative_new_patch_pos_embedding = F.interpolate(
            negative_old_patch_pos_embedding.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2),
            size=(new_size, new_size),
            mode='bicubic'
        )

        negative_new_patch_pos_embedding = negative_new_patch_pos_embedding.premute(0, 2, 3, 1).view(1, -1, self.embed_dim)
        negative_new_pos_embedding = torch.cat((negative_class_pos_embedding.unsqueeze(0), negative_new_patch_pos_embedding), dim=1)
        
        # return new positional embeddings
        return anchor_new_pos_embedding, positive_new_pos_embedding, negative_new_pos_embedding
    
    def _intent_gate(self, z_anchor_pure: torch.Tensor, z_anchor_mixed: torch.Tensor) -> torch.Tensor:
        '''
        Modeled after a logic gate, returns the appropriate anchor embedding depending on the value of self.cls_intent.

        Inputs:
            z_anchor_pure: The pure anchor embedding (absent any cross attention tainting), useful for classification.
            z_anchor_mixed: The anchor embedding mixed with positive and negative cross attention, useful for reID.

        Returns:
            z_anchor: the appropriate anchor embedding given the value of self.cls_intent.
        '''

        z_anchor = z_anchor_mixed if not self.cls_intent else z_anchor_pure

        return z_anchor
    
    def prepare_for_finetuning(self, new_dim: int) -> int:
        '''
        Prepares the i-th Stage for fine-tuning.

        Inputs:
            new_dim: the new image data dimension to be used in fine-tuning.

        Returns:
            old_in_dim: the old image data dimension from pre-training.
        '''

        assert new_dim % self.patch_dim == 0

        new_num_patches = self.patcher.get_num_patches(self.in_dim)
        self.anchor_pos_encoder.n_patches = new_num_patches
        self.positive_pos_encoder.n_patches = new_num_patches
        self.negative_pos_encoder.n_patches = new_num_patches

        anchor_pos_embedding, positive_pos_embedding, negative_pos_embedding = self._interpolate_pos_encoding(new_dim=self.in_dim)
        self.anchor_pos_encoder.pos_embedding = nn.Parameter(anchor_pos_embedding)
        self.positive_pos_encoder.pos_embedding = nn.Parameter(positive_pos_embedding)
        self.negative_pos_encoder.pos_embedding = nn.Parameter(negative_pos_embedding)

        old_in_dim = self.in_dim
        self.in_dim = new_dim

        return old_in_dim

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Passes the input triplet through the i-th Stage of a PyraT-CAiT model.

        Inputs:
            anchor: a batch of anchor images.
            positive: a batch of positive images (similar to the anchor images).
            negative: a batch of negative images (dissimilar from the anchor images).

        Returns:
            z_anchor: the embedded anchor batch.
            z_positive: the embedded positive batch.
            z_negative: the embedded negative batch.
        '''
        
        anchor, positive, negative = self._embed(anchor, positive, negative)

        for block in self.transformer_stack:
            anchor = block(anchor)
            positive = block(positive)
            negative = block(negative)

        z_anchor_pure, z_anchor_mixed, z_positive, z_negative = self.tca_block(anchor, positive, negative)
        z_anchor = self._intent_gate(z_anchor_pure, z_anchor_mixed)
        
        z_anchor = self._reshape_output(z_anchor) if not self.add_cls else z_anchor
        z_positive = self._reshape_output(z_positive) if not self.add_cls else z_positive
        z_negative = self._reshape_output(z_negative) if not self.add_cls else z_negative

        return z_anchor, z_positive, z_negative