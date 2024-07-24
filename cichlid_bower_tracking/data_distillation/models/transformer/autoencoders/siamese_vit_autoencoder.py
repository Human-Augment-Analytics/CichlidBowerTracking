from typing import Tuple

from data_distillation.models.transformer.embeddings.patch_embedding import PatchEmbedding
from data_distillation.models.transformer.embeddings.patch_transpose import PatchTranspose
from data_distillation.models.transformer.embeddings.mini_patch_embedding import MiniPatchEmbedding
from data_distillation.models.transformer.embeddings.mini_patch_transpose import MiniPatchTranspose
from data_distillation.models.transformer.embeddings.positional_encoding import PositionalEncoding

from data_distillation.models.transformer.autoencoders.vit_encoder import ViTEncoder
from data_distillation.models.transformer.autoencoders.vit_decoder import ViTDecoder

import torch.nn as nn
import torch

class SiameseViTAutoencoder(nn.Module):
    def __init__(self, embed_dim: int, n_encoder_heads: int, n_decoder_heads: int, n_encoders=8, n_decoders=8, encoder_dropout=0.1, decoder_dropout=0.1, encoder_mlp_ratio=4.0, decoder_mlp_ratio=4.0, batch_size=16, img_channels=3, img_dim=256, \
                 patcher_kernel_size=3, patcher_stride=2, patcher_ratio=8.0, patcher_ratio_decay=0.5, patcher_n_convs=5, use_minipatching=False, \
                 transposer_kernel_size=3, transposer_stride=2, transposer_ratio=1.0, transposer_ratio_growth=2.0, transposer_n_deconvs=5, \
                 patcher_dim=16):
        '''
        Initializes an instance of the SiameseViTAutoencoder class.

        Inputs:
            embed_dim: the dimension to be used in creating the image embeddings.
            n_encoder_heads: the number of heads to use in the encoder's multi-head attention mechanisms.
            n_decoder_heads: the number of heads to use in the decoder's multi-head attention mechanisms.
            n_encoders: the number of TransformerEncoders to use in the encoder; default to 8.
            n_decoders: the number of TransformerEncoders to use in the decoder; defaults to 8.
            encoder_dropout: the dropout probability to be used in the encoder; defaults to 0.1.
            decoder_dropout: the dropout probability to be used in the decoder; defaults to 0.1.
            encoder_mlp_ratio: the size of the hidden layer in the MLP used by the encoder, relative to the embed_dim; defaults to 4.0.
            decoder_mlp_ratio: the size of the hidden layer in the MLP used by the decoder, relative to the embed_dim; defaults to 4.0.
            batch_size: the number of images per batch; defaults to 16 [deprecated: has no effect on output, soon to be removed].
            img_channels: the number of channels in the images (1 for greyscale, 3 for RGB); defaults to 3.
            img_dim: the size of each image (assumes square shape); defaults to 256.
            patcher_kernel_size: the kernel size to be used by a MiniPatchEmbedding; defaults to 3, no effect if use_minipatching == False.
            patcher_stride: the stride to be used by a MiniPatchEmbedding; defaults to 2, no effect if use_minipatching == False.
            patcher_ratio: the channel-scaling ratio to be used by a MiniPatchEmbedding; defaults to 8.0, no effect if use_minipatching == False.
            patcher_ratio_decay: the rate at which a MiniPatchEmbeddings channel-scaling ratio decreases; defaults to 0.5, no effect if use_minipatching == False.
            patcher_n_convs: the number of convolutions to be used in a MiniPatchEmbedding; defaults to 5, no effect if use_minipatching == False.
            use_minipatching: a Boolean indicating whether or not a MiniPatchEmbedding should be used in place of the regular PatchEmbedding; defaults to False.
            transposer_kernel_size: the kernel size to be used by a MiniPatchTranspose; defaults to 3, no effect if use_minipatching == False.
            transposer_stride: the stride to be used by a MiniPatchTranspose; defaults to 2, no effect if use_minipatching == False.
            transposer_ratio: the channel-scaling ratio to be used by a MiniPatchTranspose; defaults to 1.0, no effect if use_minipatching == False.
            transposer_ratio_growth: the rate at which a MiniPatchTranspose channel-scaling ratio increases; defaults to 2.0, no effect if use_minipatching == False.
            transposer_n_deconvs: the number of deconvolutions (transpose convolutions) to be used in a MiniPatchTranspose; defaults to 5, no effect if use_minipatching == False.
            patcher_dim: the patch size to be used by a PatchEmbedding and PatchTranspose; defaults to 16, no effect if use_minipatching == True.
        '''
        
        super(SiameseViTAutoencoder, self).__init__()

        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.n_encoder_heads = n_encoder_heads
        self.n_decoder_heads = n_decoder_heads
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.encoder_mlp_ratio = encoder_mlp_ratio
        self.decoder_mlp_ratio = decoder_mlp_ratio

        self.batch_size = batch_size
        self.img_channels = img_channels
        self.img_dim = img_dim

        self.use_minipatching = use_minipatching

        self.patcher_kernel_size = patcher_kernel_size
        self.patcher_stride = patcher_stride
        self.patcher_ratio = patcher_ratio
        self.patcher_ratio_decay = patcher_ratio_decay
        self.patcher_n_convs = patcher_n_convs

        self.transposer_kernel_size = transposer_kernel_size
        self.transposer_stride = transposer_stride
        self.transposer_ratio = transposer_ratio
        self.transposer_ratio_growth = transposer_ratio_growth
        self.transposer_n_deconvs = transposer_n_deconvs

        self.patcher_dim = patcher_dim

        if not self.use_minipatching:
            self.patcher = PatchEmbedding(embed_dim=self.embed_dim, batch_size=self.batch_size, in_channels=self.img_channels, patch_dim=self.patcher_dim)
            self.transposer = PatchTranspose(embed_dim=self.embed_dim, batch_size=self.batch_size, out_channels=self.img_channels, out_dim=self.img_dim, patch_dim=self.patcher_dim)
        else:
            self.patcher = MiniPatchEmbedding(embed_dim=self.embed_dim, batch_size=self.batch_size, in_channels=self.img_channels, kernel_size=self.patcher_kernel_size, \
                                              stride=self.patcher_stride, ratio=self.patcher_ratio, ratio_decay=self.patcher_ratio_decay, n_convs=self.patcher_n_convs)
            
            self.transposer = MiniPatchTranspose(embed_dim=self.embed_dim, patcher_dims_list=self.patcher.get_num_patches_and_dims_list(self.img_dim)[1], patcher_intermediate_channels=self.patcher.intermediate_channels,
                                                 n_patches=self.patcher.get_num_patches_and_dims_list(self.img_dim)[0], batch_size=self.batch_size, out_channels=self.img_channels, out_dim=self.img_dim, kernel_size=self.transposer_kernel_size, \
                                                 stride=self.transposer_stride, ratio=self.transposer_ratio, ratio_growth=self.transposer_ratio_growth, n_deconvs=self.transposer_n_deconvs)
            
        self.pos_enc = PositionalEncoding(embed_dim=self.embed_dim, n_patches=(self.patcher.get_num_patches(self.img_dim) if not self.use_minipatching else self.patcher.get_num_patches_and_dims_list(self.img)[0]))

        self.encoder = ViTEncoder(embed_dim=self.embed_dim, n_heads=self.n_encoder_heads, patcher=self.patcher, pos_enc=self.pos_enc, n_encoders=self.n_encoders, p_dropout=self.encoder_dropout, mlp_ratio=self.encoder_mlp_ratio)
        self.decoder = ViTDecoder(embed_dim=self.embed_dim, n_heads=self.n_decoder_heads, transposer=self.transposer, n_decoders=self.n_decoders, p_dropout=self.decoder_dropout, mlp_ratio=self.decoder_mlp_ratio)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Performs distillation and reconstruction on a pair of input image batches using shared encoder and decoder weights.

        Inputs:
            x1: a PyTorch Tensor representing one batch of images.
            x2: a PyTorch Tensor representing another batch of images.

        Returns:
            z1: a PyTorch Tensor representing the feature embeddings of input x1.
            z2: a PyTorch Tensor representing the feature embeddings of input x2.
            x1_reconstruction: a PyTorch Tensor representing reconstructions of the images in input x1.
            x2_reconstruction: a PyTorch Tensor representing reconstructions of the images in input x2.
        '''
        
        z1 = self.transformer_encoder(x1)
        z2 = self.transformer_decoder(x2)

        x1_reconstruction = self.transformer_decoder(z1)
        x2_reconstruction = self.transformer_decoder(z2)

        return z1, z2, x1_reconstruction, x2_reconstruction

    def distill(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Given a batch of images, distills them into feature embeddings.

        Inputs:
            x: a PyTorch Tensor representing a batch of images.

        Returns:
            z: a PyTorch Tensor representing the feature embeddings of input x.
        '''

        z = self.transformer_encoder(x)

        return z
    
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Given a batch of image embeddings, attempts to reconstruct the original images from which the embeddings are distilled.

        Inputs:
            z: a PyTorch Tensor representing a batch of image embeddings.

        Returns:
            x_reconstruction: a PyTorch Tensor representing a batch of image reconstructions.
        '''

        x_reconstruction = self.transformer_decoder(z)

        return x_reconstruction