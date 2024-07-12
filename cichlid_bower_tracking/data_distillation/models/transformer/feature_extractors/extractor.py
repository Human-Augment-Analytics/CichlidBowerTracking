from typing import Optional, Tuple
import math

from data_distillation.models.transformer.embeddings.patch_embedding import PatchEmbedding
from data_distillation.models.transformer.embeddings.mini_patch_embedding import MiniPatchEmbedding
from data_distillation.models.transformer.embeddings.positional_encoding import PositonalEncoding
from data_distillation.models.transformer.embeddings.cls_tokens import CLSTokens

from data_distillation.models.transformer.attention_mechs.cross_attention import CrossAttention
from data_distillation.models.transformer.transformer_encoder import TransformerEncoder

import torch.nn.functional as F
import torch.nn as nn
import torch

class Extractor(nn.Module):
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
        
        self.__version__ = '0.1.0'

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
            self.patcher = PatchEmbedding(embed_dim=self.embed_dim, in_channels=self.in_channels, in_dim=self.in_dim, patch_dim=self.patch_dim)
        else:
            self.patcher = MiniPatchEmbedding(embed_dim=self.embed_dim, in_channels=self.in_channels, in_dim=self.in_dim, kernel_size=self.patch_kernel_size, \
                                              stride=self.patch_stride, ratio=self.patch_ratio, ratio_decay=self.patch_ratio_decay, n_convs=self.patch_n_convs)

        self.cls_tokenizer = CLSTokens(embed_dim=self.embed_dim)
        self.pos_encoder = PositonalEncoding(embed_dim=embed_dim, n_patches=(self.patcher.get_num_patches(self.in_dim) if not self.use_minipatch else self.patcher.get_num_patches_and_dims_list(self.in_dim)[0]), add_one=True)

        self.transformer_blocks = nn.Sequential(*[TransformerEncoder(embed_dim=self.embed_dim, n_heads=self.num_heads, p_dropout=self.dropout, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)])
        self.cross_attn = CrossAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)

    def __str__(self) -> str:
        '''
        Returns a string representation of the Extractor component.

        Inputs: None.

        Returns:
            string: a string representation of the Extractor component.
        '''

        string = f'Extractor\n{"=" * 80}\n'
        string += f'{"Name":30s} | {"Params":12s} | {"Size":30s}\n'

        total_num_params = 0

        for name, param in self.extractor.named_parameters():
            if not param.requires_grad():
                continue

            num_params = param.numel()
            total_num_params += num_params

            string += f'{name:30s} | {(num_params):12d} | {str(tuple(param.size())):30s}\n'

        string += f'{"-" * 80}\n'
        string += f'{"TOTAL EXTRACTOR # PARAMS":30s} | {total_num_params:45d}\n'

        self.num_params = total_num_params

        return string

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Uses the patcher, cls_tokenizer, and pos_encoder (in that order) to embed the input.

        Inputs:
            x: the image batch to be embedded.
        
        Returns:
            out_embed: the image embedding batch.
        '''
        
        out_embed = self.patcher(x)
        out_embed = self.cls_tokenizer(out_embed)
        out_embed = self.pos_encoder(out_embed)

        return out_embed
    
    def _interpolate_pos_encoding(self, new_dim: int) -> torch.Tensor:
        '''
        Interpolates the positional encoding parameter to prepare for fine-tuning on larger image data.

        Inputs:
            new_dim: the new image data dimension.
        
        Returns:
            new_pos_embedding: the interpolated positional embedding.
        '''

        new_n_patches = self.patcher.get_num_patches(new_dim=new_dim) if not self.use_minipatch else self.patcher.get_num_patches_and_dims_list(new_dim=new_dim)[0]
        old_n_patches = self.pos_encoder.pos_embedding.shape[1] - 1

        if new_n_patches == old_n_patches:
            return self.pos_encoder.pos_embedding
        
        class_pos_embedding = self.pos_encoder.pos_embedding[:, 0]
        old_patch_pos_embedding = self.pos_encoder.pos_embedding[:, 1:]

        old_size = int(math.sqrt(old_n_patches))
        new_size = int(math.sqrt(new_n_patches))

        new_patch_pos_embedding = F.interpolate(
            old_patch_pos_embedding.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2),
            size=(new_size, new_size),
            mode='bicubic'
        )

        new_patch_pos_embedding = new_patch_pos_embedding.premute(0, 2, 3, 1).view(1, -1, self.embed_dim)
        new_pos_embedding = torch.cat((class_pos_embedding.unsqueeze(0), new_patch_pos_embedding), dim=1)

        return new_pos_embedding
    
    def prepare_for_finetuning(self, new_dim: int) -> None:
        '''
        Prepares the Extractor for fine-tuning.

        Inputs:
            new_dim: the new image data dimension.
        '''
        
        assert new_dim % self.patch_dim == 0

        new_num_patches = self.patcher.get_num_patches(self.in_dim) if not self.use_minipatch else self.patcher.get_num_patches_and_dims_list(self.in_dim)[0]
        self.pos_encoder.n_patches = new_num_patches

        new_pos_embedding = self._interpolate_pos_encoding(shape=(1, new_num_patches + 1, self.embed_dim), new_dim=self.in_dim)
        self.pos_encoder.pos_embedding = nn.Parameter(new_pos_embedding)

        self.in_dim = new_dim
    
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
        
        anchor = self._embed(anchor)
        positive = self._embed(positive)
        negative = self._embed(negative)

        for block in self.transformer_blocks:
            anchor = block(anchor)
            anchor = self.cross_attn(anchor, positive) + self.cross_attn(anchor, negative)

            positive = block(positive)
            negative = block(negative)

        return anchor, positive, negative
