import torch.nn as nn

from .encoder import EncoderPath
from .decoder import DecoderPath

class AResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderPath(in_channels=4, base_channels=2)
        self.decoder = DecoderPath(base_channels=2)

    def forward(self, x):
        bottleneck, skips, cache = self.encoder(x)
        x = self.decoder(bottleneck, skips, cache)
        return x