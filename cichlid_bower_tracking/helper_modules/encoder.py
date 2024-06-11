import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, features: int, batch_size=32, img_channels=3, img_dim=128, p_dropout=0.5):
        super(Encoder, self).__init__()

        self.__version__ = '0.1.0'
        
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
            nn.Linear(in_features=256 * 32 * 32, out_features=self.out_features),
            nn.Dropout(p=self.p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (self.batch_size, self.in_channels, self.in_dim, self.in_dim)

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.fc(out)

        return out