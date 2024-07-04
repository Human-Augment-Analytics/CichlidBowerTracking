import torch.nn as nn
import torch

class PatchTranspose(nn.Module):
    def __init__(self, embed_dim: int, n_patches: int, batch_size=16, out_channels=3, out_dim=256, patch_dim=16):
        '''
        Initializes an instance of the PatchTranspose class.

        Inputs:
            embed_dim: essentially the number of channels in the input.
            batch_size: the number of embeddings in the input Tensor; defaults to 16 [deprecated: has no effect on output, soon to be removed].
            n_patches: the value of the PatchEmbedding's npatches instance variable.
            out_channels: the number of channels in the reconstructed image (1 for greyscale, 3 for RGB); defaults to 3.
            out_dim: the dimension of the reconstructed image; defaults to 256.
            patch_dim: essentially the kernel size of the deconvolution.
        '''
        
        super(PatchTranspose, self).__init__()

        self.__version__ = '0.1.2'

        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.npatches = n_patches
        self.out_channels = out_channels
        self.out_dim = out_dim
        self.patch_dim = patch_dim

        self.unflatten = nn.Unflatten(1, (self.out_channels, self.out_dim // self.patch_dim, self.out_dim // self.patch_dim))
        self.patch_deconv = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.out_channels, kernel_size=self.patch_dim, stride=self.patch_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Essentially reverses the patch embedding process performed by the PatchEmbedding.

        Inputs:
            z: a PyTorch Tensor representing a batch of image embeddings.
        
        Returns:
            x_reconstruction: a PyTorch Tensor representing a batch of reconstructed images.    
        '''

        assert len(z.shape) == 3
        assert z.shape[:1] == (self.npatches, self.embed_dim) # shape (N, D_in * D_in, C_in)

        z = torch.transpose(z, dim0=1, dim1=2) # shape (N, C_embed, D_in * D_in)

        out = self.unflatten(z) # shape (N, C_embed, D_in, D_in)
        out = self.patch_deconv(out) # shape (N, C_out, D_out, D_out)

        x_reconstruction = self.sigmoid(out) # shape (N, C_out, D_out, D_out)

        return x_reconstruction