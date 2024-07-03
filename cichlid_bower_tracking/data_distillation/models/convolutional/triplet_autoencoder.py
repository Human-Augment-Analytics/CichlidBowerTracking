from typing import Tuple

from encoder import Encoder
from decoder import Decoder

import torch.nn as nn
import torch

class TripletAutoencoder(nn.Module):
    def __init__(self):
        super(TripletAutoencoder, self).__init__()

        # =================================================        
        # TODO
        # =================================================

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)
        z_negative = self.encoder(negative)

        anchor_reconstruction = self.decoder(z_anchor)
        positive_representation = self.decoder(z_positive)
        negative_representation = self.decoder(z_negative)

        return z_anchor, z_positive, z_negative, anchor_reconstruction, positive_representation, negative_representation
    
    def distill(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        return z
    
    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        x_reconstruction = self.decoder(z)

        return x_reconstruction