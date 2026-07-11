import torch
import torch.nn as nn


class AttentionDownBlock(nn.Module):
    """Encoder self-attention: AttentionGate(x, x)."""
    def __init__(self, channels):
        super().__init__()

        self.theta = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(channels)
        )

        self.phi = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(channels)
        )

        self.bn =  nn.BatchNorm2d(channels)

    def forward(self, x):
        # DeepDCT-style self-gating: AttentionGate(x, x)
        a = self.theta(x)
        b = self.phi(x)

        a = self.theta(a)
        b = self.phi(b)

        attn = torch.relu(a + b)
        attn = self.psi(attn)

        x = self.bn(x)

        return x * attn
    

class AttentionUpBlock(nn.Module):
    """
    Fig. 4-style self-gating attention over the concatenated
    decoder and encoder feature tensor.
    """

    def __init__(self, channels):
        super().__init__()

        self.input_bn = nn.BatchNorm2d(channels)

        self.theta = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(channels),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.phi = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(channels),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.psi_conv = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.psi_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        left = self.theta(x)
        right = self.phi(x)

        attn = torch.relu(left + right)
        attn = self.psi_conv(attn)

        #conventional, not Fig. 4-faithful ordering:
        attn = self.psi_bn(attn)
        attn = torch.sigmoid(attn)
        

        x = self.input_bn(x)

        return x * attn

class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = self.bn(x)
        return x

