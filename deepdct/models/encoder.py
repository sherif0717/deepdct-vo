import torch
import torch.nn as nn

from .attention import AttentionDownBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        return x


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = EncoderBlock(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.attn = AttentionDownBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        x = self.encoder(x)      # Conv + ReLU

        skip = x                 # decoder skip: raw Conv+ReLU output
        x_attn = x               # attention branch input

        x_bn = self.batch_norm(x) # BN branch

        x_attn = self.attn(x_attn)

        x = x_bn + x_attn
        x = self.pool(x)

        return x, skip
    

class EncoderPath(nn.Module):
    def __init__(self, in_channels=4, base_channels=2):
        super().__init__()

        self.enc1 = EncoderStage(in_channels, base_channels)          # 4 -> 2
        self.enc2 = EncoderStage(base_channels, base_channels * 2)    # 2 -> 4
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4) # 4 -> 8

    def forward(self, x):
        x, skip1 = self.enc1(x)   # 120x120 -> 60x60
        x, skip2 = self.enc2(x)   # 60x60 -> 30x30
        x, skip3 = self.enc3(x)   # 30x30 -> 15x15

        return x, (skip1, skip2, skip3)