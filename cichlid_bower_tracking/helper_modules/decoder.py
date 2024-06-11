import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, features: int, batch_size=32, img_channels=3, img_dim=128):
        super(Decoder, self).__init__()

        self.__version__ = '0.1.0'

        self.in_features = features

        self.batch_size = batch_size
        self.out_channels = img_channels
        self.out_dim = img_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=256 * 32 * 32),
            nn.Unflatten(1, (256, 32, 32))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (self.batch_size, self.in_features)

        out = self.fc(x)
        out = self.deconv_2(out)
        out = self.deconv_1(out)

        return out
    
