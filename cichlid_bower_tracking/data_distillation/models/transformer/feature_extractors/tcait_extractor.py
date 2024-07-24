from typing import Optional, Tuple
import math

from data_distillation.models.transformer.embeddings.patch_embedding import PatchEmbedding
from data_distillation.models.transformer.embeddings.mini_patch_embedding import MiniPatchEmbedding
from data_distillation.models.transformer.embeddings.positional_encoding import PositionalEncoding
from data_distillation.models.transformer.embeddings.cls_tokens import CLSTokens

from data_distillation.models.transformer.attention_mechs.cross_attention import CrossAttention
from data_distillation.models.transformer.transformer_block import TransformerBlock

import torch.nn.functional as F
import torch.nn as nn
import torch

class TCAiTExtractor(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, in_channels=3, in_dim=256, depth=8, dropout=0.1, mlp_ratio=4.0, patch_dim: Optional[int]=16, patch_kernel_size: Optional[int]=3, \
                 patch_stride: Optional[int]=2, patch_ratio: Optional[float]=8.0, patch_ratio_decay: Optional[float]=0.5, patch_n_convs: Optional[int]=5, use_minipatch=False):
        '''
        Initializes an instance of the Extractor class.

        Inputs:
            embed_dim: the embedding dimension to be used.
            num_heads: the number of heads to be used in the self-attention and cross attention mechanisms.
            in_channels: the number of channels in the input image(s); 1 for greyscale and 3 for RGB, defaults to 3.
            in_dim: the size of the input image(s); defaults to 256.
            depth: the number of encoder blocks to be used (depth - 1 if mini-patch embedding, otherwise depth); defaults to 8.
            dropout: the dropout to be used by the self-attention and cross attention mechanisms; defaults to 0.1.
            mlp_ratio: the size of the hidden layer of each transformer block's MLP, relative to the embed_dim; defaults to 4.
            patch_dim: the patch size to be used in standard patch embedding; defaults to 16 (only effective if use_minipatch is False).
            patch_kernel_size: the kernel size to be used in mini-patch embedding; defaults to 3 (only effective if use_minipatch is True).
            patch_stride: the stride to be used in mini-patch embedding; defaults to 2 (only effective if use_minipatch is True).
            patch_ratio: the starting rate at which the number of channels increases in mini-patch embedding; defaults to 4.0 (only effective if use_minipatch is True).
            patch_ratio_decay: the rate at which the patch_ratio decays in mini-patch embedding; defaults to 0.5 (only effective if use_minipatch is True).
            patch_n_convs: the number of convolutions to be used in mini-patch embedding; defaults to 5 (only effective if use_minipatch is True).
            use_minipatch: indicates whether or not mini-patch embedding is used instead of standard patch embedding; defaults to False.
        '''

        super(TCAiTExtractor, self).__init__()
        
        self.__version__ = '0.2.1'

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.in_channels = in_channels
        self.in_dim = in_dim

        self.depth = depth if not use_minipatch else (depth - 1)
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        
        self.patch_dim = patch_dim
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_ratio = patch_ratio
        self.patch_ratio_decay = patch_ratio_decay
        self.patch_n_convs = patch_n_convs
        
        self.use_minipatch = use_minipatch
        if not self.use_minipatch:
            self.patcher = PatchEmbedding(embed_dim=self.embed_dim, in_channels=self.in_channels, patch_dim=self.patch_dim)
        else:
            self.patcher = MiniPatchEmbedding(embed_dim=self.embed_dim, in_channels=self.in_channels, in_dim=self.in_dim, kernel_size=self.patch_kernel_size, \
                                              stride=self.patch_stride, ratio=self.patch_ratio, ratio_decay=self.patch_ratio_decay, n_convs=self.patch_n_convs)

        self.anchor_cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
        self.positive_cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
        self.negative_cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
        
        n_patches = self.patcher.get_num_patches(self.in_dim) if not self.use_minipatch else self.patcher.get_num_patches_and_dims_list(self.in_dim)[0]
        self.anchor_pos_encoder = PositionalEncoding(embed_dim=self.embed_dim, n_patches=n_patches, add_one=True)
        self.positive_pos_encoder = PositionalEncoding(embed_dim=self.embed_dim, n_patches=n_patches, add_one=True)
        self.negative_pos_encoder = PositionalEncoding(embed_dim=self.embed_dim, n_patches=n_patches, add_one=True)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_dim=self.embed_dim, n_heads=self.num_heads, p_dropout=self.dropout, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)])
        
        self.positive_cross_attn = CrossAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)
        self.negative_cross_attn = CrossAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)

    def __str__(self) -> str:
        '''
        Returns a string representation of the Extractor component.

        Inputs: None.

        Returns:
            string: a string representation of the Extractor component.
        '''

        string = f'Extractor\n{"=" * 110}\n'
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
        string += f'{"TOTAL EXTRACTOR # PARAMS":70s} | {total_num_params:35d}\n'

        self.num_params = total_num_params

        return string

    def _embed(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Uses the patcher, cls_tokenizers, and pos_encoders (in that order) to embed the input.

        Inputs:
            anchor: the raw anchor images batch.
            positive: the raw positive images batch.
            negative: the raw negative images batch.
        
        Returns:
            anchor_embed: the embedded anchors batch.
            positive_embed: the embedded positives batch.
            negative_embed: the embedded negatives batch.
        '''
        
        anchor_embed = self.patcher(anchor)
        positive_embed = self.patcher(positive)
        negative_embed = self.patcher(negative)

        anchor_embed = self.anchor_cls_tokenizer(anchor_embed)
        positive_embed = self.positive_cls_tokenizer(positive_embed)
        negative_embed = self.negative_cls_tokenizer(negative_embed)

        anchor_embed = self.anchor_pos_encoder(anchor_embed)
        positive_embed = self.positive_pos_encoder(positive_embed)
        negative_embed = self.negative_pos_encoder(negative_embed)

        return anchor_embed, positive_embed, negative_embed
    
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

        new_n_patches = self.patcher.get_num_patches(new_dim=new_dim) if not self.use_minipatch else self.patcher.get_num_patches_and_dims_list(new_dim=new_dim)[0]
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
    
    def prepare_for_finetuning(self, new_dim: int) -> int:
        '''
        Prepares the Extractor for fine-tuning.

        Inputs:
            new_dim: the new image data dimension to be used in fine-tuning.

        Returns:
            old_in_dim: the old image data dimension from pre-training.
        '''

        assert new_dim % self.patch_dim == 0

        new_num_patches = self.patcher.get_num_patches(self.in_dim) if not self.use_minipatch else self.patcher.get_num_patches_and_dims_list(self.in_dim)[0]
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
        Extracts features from the input image batches by passing them through an encoder.

        Inputs:
            anchor: a batch of images.
            positive: a batch of images similar to the images in the anchor batch.
            negative: a batch of images dissimilar from the images in the anchor batch.

        Returns:
            anchor: the anchor image embeddings.
            positive: the positive image embeddings.
            negative: the negative image embeddings.
        '''
        
        anchor, positive, negative = self._embed(anchor, positive, negative)

        for block in self.transformer_blocks:
            anchor = block(anchor)
            positive = block(positive)
            negative = block(negative)

        anchor = self.positive_cross_attn(anchor, positive) + self.negative_cross_attn(anchor, negative)

        return anchor, positive, negative
