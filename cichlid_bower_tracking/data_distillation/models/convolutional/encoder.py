import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, features: int, batch_size=16, img_channels=3, img_dim=256, p_dropout=0.5):
        '''
        Initializes an instance of the Encoder PyTorch module of the SiameseAutoencoder.

        Inputs:
            features: an integer indicating the number of features to which the input should be compressed by the encoder.
            batch_size: an integer indicating the number of images included in each batch during training and evaluation; defaults to 16 [deprecated: has no effect on output, soon to be removed].
            img_channels: an integer indicating the number of channels in the input images; defaults to 3 (assumes RGB over greyscale).
            img_dim: an integer indicating the input images' shared height and width; defaults to 128.
            p_dropout: a float indicating what probability should be used in the dropout layer; defaults to 0.5, should be in the interval (0, 1).
        '''

        super(Encoder, self).__init__()

        self.__version__ = '0.1.2'
        
        self.out_features = features
        
        self.batch_size = batch_size
        self.in_channels = img_channels
        self.in_dim = img_dim
        self.p_dropout = p_dropout

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_2 = nn.sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 64 * 64, out_features=self.out_features),
            nn.Dropout(p=self.p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Obtains feature embeddings for the passed input in a forward pass of the encoder.

        Inputs:
            x: a PyTorch Tensor representing a batch of images, each of shape (self.batch_size, self.in_channels, self.in_dim, self.in_dim).

        Returns:
            z: a PyTorch Tensor containing the feature embeddings for each image in the passed batch.
        '''

        assert len(x.shape) == 4
        assert x.shape[:1] == (self.in_channels, self.in_dim, self.in_dim)

        out = self.conv_1(x)
        out = self.conv_2(out)
        z = self.fc(out)

        return z