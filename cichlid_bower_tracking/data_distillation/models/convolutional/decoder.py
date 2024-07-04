import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, features: int, batch_size=16, img_channels=3, img_dim=256):
        '''
        Initializes an instance of the Decoder PyTorch module of the SiameseAutoencoder.

        Inputs:
            features: an integer indicating the number of features to which the input should be compressed by the encoder.
            batch_size: an integer indicating the number of images included in each batch during training and evaluation; defaults to 16 [deprecated: has no effect on output, soon to be removed].
            img_channels: an integer indicating the number of channels in the input images; defaults to 3 (assumes RGB over greyscale).
            img_dim: an integer indicating the input images' shared height and width; defaults to 128.
        '''

        super(Decoder, self).__init__()

        self.__version__ = '0.1.2'

        self.in_features = features

        self.batch_size = batch_size
        self.out_channels = img_channels
        self.out_dim = img_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=256 * 64 * 64),
            nn.Unflatten(1, (256, 64, 64))
        )

        self.deconv_2 = nn.sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        self.deconv_1 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Attempts to reconstruct the original image, represented by the passed batch of feature embeddings.

        Inputs:
            z: a PyTorch Tensor of shape (self.batch_size, self.in_features) representing a batch of feature embeddings, as created by the encoder.

        Returns:
            x_reconstruction: a PyTorch Tensor representing the batch of reconstructed images.
        '''
        
        assert len(z.shape) == 2
        assert z.shape[1:] == (self.in_features, )

        out = self.fc(z)
        out = self.deconv_2(out)
        x_reconstruction = self.deconv_1(out)

        return x_reconstruction
    
