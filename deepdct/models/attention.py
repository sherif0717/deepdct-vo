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

    def forward(self, x):
        # DeepDCT-style self-gating: AttentionGate(x, x)
        a = self.theta(x)
        b = self.phi(x)

        a = self.theta(a)
        b = self.phi(b)

        attn = torch.relu(a + b)
        attn = self.psi(attn)

        return x * attn