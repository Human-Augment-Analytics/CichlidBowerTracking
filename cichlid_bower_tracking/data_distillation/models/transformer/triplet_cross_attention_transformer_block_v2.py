from typing import Tuple

from data_distillation.models.transformer.attention_mechs.cross_attention import CrossAttention

import torch.nn.functional as F
import torch.nn as nn
import torch

class TripletCrossAttentionTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_patches: int, n_heads: int, p_dropout=0.1, mlp_ratio=4.0, init_lambda=0.5, init_alpha=0.1, init_beta=0.1, add_one=False):
        
        super(TripletCrossAttentionTransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.mlp_ratio = mlp_ratio
        self.init_lambda = init_lambda
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.add_one = add_one

        self.norm1 = nn.LayerNorm(normalized_shape=self.embed_dim)

        # self.alpha = nn.Parameter(torch.zeros(1, self.n_patches + 1 * self.add_one, self.embed_dim).fill_(torch.tensor(self.init_alpha)))
        # self.beta = nn.Parameter(torch.zeros(1, self.n_patches + 1 * self.add_one, self.embed_dim).fill_(torch.tensor(self.init_beta)))
        
        self.lam = nn.Parameter(torch.tensor(self.init_lambda))
        self.alpha = nn.Parameter(torch.tensor(self.init_alpha))
        self.beta = nn.Parameter(torch.tensor(self.init_beta))

        self.positive_cross_attn = CrossAttention(embed_dim=self.embed_dim, num_heads=self.n_heads, dropout=self.p_dropout)
        self.negative_cross_attn = CrossAttention(embed_dim=self.embed_dim, num_heads=self.n_heads, dropout=self.p_dropout)

        self.norm2 = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=int(self.embed_dim * self.mlp_ratio)),
            nn.GELU(),
            nn.Linear(in_features=int(self.embed_dim * self.mlp_ratio), out_features=self.embed_dim),
            nn.Dropout(p=self.p_dropout)
        )

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pass triplet through first layer norm
        z_anchor = self.norm1(anchor)
        z_positive = self.norm1(positive)
        z_negative = self.norm1(negative)

        # perform cross attention between (anchor, positive) and (anchor, negative) pairs
        positive_ca = self.positive_cross_attn(z_anchor, z_positive)
        negative_ca = self.negative_cross_attn(z_anchor, z_negative)

        # add cross attention outputs multiplied by alpha and beta parameters (alpha for pull, beta for push)
        # z_anchor_pure = z_anchor.clone() # clone pure anchor for classification
        # z_anchor_mixed = z_anchor + self.alpha * positive_ca + self.beta * negative_ca # generate mixed anchor for reID; pull (anchor, positive) together, push (anchor, negative) apart

        asym_z_anchor = anchor + self.alpha * positive_ca - self.beta * negative_ca
        asym_z_positive = positive + self.alpha * positive_ca
        asym_z_negative = negative - self.beta * negative_ca

        sym_z_anchor = z_anchor + self.alpha * positive_ca + self.beta * negative_ca 
        sym_z_positive = z_positive + self.alpha * positive_ca + self.beta * negative_ca
        sym_z_negative = z_negative + self.alpha * positive_ca + self.beta * negative_ca

        lam_sig = F.sigmoid(self.lam)
        z_anchor = lam_sig * asym_z_anchor + (1 - lam_sig) * sym_z_anchor
        z_positive = lam_sig * asym_z_positive + (1 - lam_sig) * sym_z_positive
        z_negative = lam_sig * asym_z_negative + (1 - lam_sig) * sym_z_negative

        # z_anchor = anchor + self.alpha * positive_ca + self.beta * negative_ca # generate mixed anchor for reID; pull (anchor, positive) together, push (anchor, negative) apart
        # z_positive = positive + self.alpha * positive_ca + self.beta * negative_ca # pull (positive, anchor) together, push (positive, negative) apart
        # z_negative = negative + self.alpha * positive_ca + self.beta * negative_ca # push (negative, positive) and (negative, anchor) apart

        # pass through second layer norm
        # z_anchor_pure = self.norm2(z_anchor_pure)
        # z_anchor_mixed = self.norm2(z_anchor_mixed)

        z_anchor = z_anchor + self.mlp(self.norm2(z_anchor))
        z_positive = z_positive + self.mlp(self.norm2(z_positive))
        z_negative = z_negative + self.mlp(self.norm2(z_negative))

        # pass through MLP
        # z_anchor_pure = self.mlp(z_anchor_pure)
        # z_anchor_mixed = self.mlp(z_anchor_mixed)

        # z_positive = self.mlp(z_positive)
        # z_negative = self.mlp(z_negative)

        # return all
        # return z_anchor_pure, z_anchor_mixed, z_positive, z_negative
        return z_anchor, z_positive, z_negative


