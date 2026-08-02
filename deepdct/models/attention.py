from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AttentionDownBlock(nn.Module):
    """Encoder self-attention: AttentionGate(x, x)."""

    def __init__(self, channels):
        super().__init__()

        self.theta = nn.Sequential(
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
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
            nn.BatchNorm2d(channels),
        )

        self.bn = nn.BatchNorm2d(channels)

        # Runtime-only tensor used by interpretability utilities.
        self.last_attention_map: Optional[Tensor] = None

    def forward(self, x):
        # DeepDCT-style self-gating: AttentionGate(x, x)
        a = self.theta(x)
        b = self.phi(x)

        a = self.theta(a)
        b = self.phi(b)

        attn_features = torch.relu(a + b)

        # Expand the original Sequential explicitly so the true
        # sigmoid attention coefficients can be retained.
        attn_logits = self.psi[0](attn_features)
        attention_map = self.psi[1](attn_logits)

        # Save the [0, 1] coefficient map before BatchNorm.
        self.last_attention_map = attention_map

        # Preserve the original Conv -> Sigmoid -> BN behavior.
        attn = self.psi[2](attention_map)

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

        self.last_attention_map: Optional[Tensor] = None

    def forward(self, x):
        left = self.theta(x)
        right = self.phi(x)

        attn = torch.relu(left + right)
        attn = self.psi_conv(attn)

        #conventional, not Fig. 4-faithful ordering:
        attn = self.psi_bn(attn)
        attn = torch.sigmoid(attn)
        
        self.last_attention_map = attn

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

