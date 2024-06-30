import torch.nn as nn
import torch

class MiniPatchTranspose(nn.Module):
    def __init__(self, embed_dim: int, before_stack_dim: int, batch_size: int, n_patches: int, out_channels=3, out_dim=256, kernel_size=3, stride=2, ratio=8.0, ratio_growth=2.0, n_deconvs=5):
        '''
        Initializes an instance of the MiniBatchTranspose class.

        Inputs:
            embed_dim: essentially the number of channels in the input.
            before_stack_dim: the number of channels produced by the MiniPatchEmbedding's convolutional stack, before being passed through the final convolution.
            batch_size: the number of embeddings in the input Tensor.
            n_patches: the value of the MiniPatchEmbedding's npatches instance variable. 
            out_channels: the number of channels in the reconstructed image (1 for greyscale, 3 for RGB); defaults to 3.
            out_dim: the dimension of the reconstructed image; defaults to 256.
            kernel_size: the dimension of the kernel used in the deconvolutional stack; defaults to 3.
            stride: the stride used in the deconvolutional stack; defaults to 2.
            ratio: the channel multiplier used during the deconvolutional stack; defaults to 8.0.
            ratio_growth: the rate at which the ratio (channel multiplier) increases throughout the deconvolutional stack; defaults to 2.0.
            n_deconvs: the number of deconvolutions to be used (including the initial stride-one 1x1 deconvolution); defaults to 5.
        '''
        
        super(MiniPatchTranspose, self).__init__()

        self.__version__ = '0.1.0'

        self.embed_dim = embed_dim
        self.before_stack_dim = before_stack_dim
        self.batch_size = batch_size
        self.npatches = n_patches
        self.out_channels = out_channels
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.ratio = ratio
        self.ratio_growth = ratio_growth
        self.n_deconvs = n_deconvs

        self.unflatten = nn.Unflatten(1, (self.out_channels, self.in_dim, self.in_dim))
        self.init_deconv = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.before_stack_dim, kernel_size=1, stride=1)

        self.deconv_stack, init_channels = [], self.before_stack_dim
        for _ in range(self.n_deconvs - 1):
            b_norm_i = nn.BatchNorm2d(num_features=int(init_channels * ratio))
            deconv_i = nn.ConvTranspose2d(in_channels=init_channels, out_channels=int(init_channels * ratio))

            self.deconv_stack += [b_norm_i, deconv_i]

            init_channels = int(init_channels * self.ratio)
            self.ratio *= self.ratio_growth

        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Essentially reverses the mini-patch embedding process performed by the MiniPatchEmbedding.

        Inputs:
            z: a PyTorch Tensor representing a batch of image embeddings.
        
        Returns:
            x_reconstruction: a PyTorch Tensor representing a batch of reconstructed images.    
        '''
        
        assert z.shape == (self.batch_size, self.npatches, self.embed_dim) # shape (N, D_in * D_in, C_in)

        z = torch.transpose(z, dim0=1, dim1=2) # shape (N, C_embed, D_in * D_in)
        
        out = self.unflatten(z) # shape (N, C_embed, D_in, D_in)
        out = self.init_deconv(z) # shape (N, C_before, D_in, D_in)
        
        out = self.deconv_stack[0](out)
        for i in range(1, len(self.deconv_stack)):
            out = self.deconv_stack[i](out)

        x_reconstruction = self.sigmoid(out) # shape (N, C_out, D_out, D_out)

        return x_reconstruction