import torch
import torch.nn as nn

from .attention import AttentionUpBlock, OutputBlock


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_t = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

        torch.nn.init.kaiming_normal_(self.conv_t.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.conv_t(x)
        x = torch.relu(x)
        x = self.batch_norm(x)
        return x
    
class DecoderStage(nn.Module):
    def __init__(self, x_channels, skip_channels, out_channels):
        super().__init__()

        merged_channels = x_channels + skip_channels

        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.attn = AttentionUpBlock(merged_channels)
        self.decoder = DecoderBlock(
            merged_channels,
            out_channels,
        )

    def forward(self, x, skip, encoder_attn=None):
        x = self.up(x)

        if encoder_attn is not None:
            if x.shape != encoder_attn.shape:
                raise ValueError(
                    "Upsampled decoder feature and encoder-attention "
                    f"feature must match, got {x.shape} and "
                    f"{encoder_attn.shape}."
                )

            x = x + encoder_attn

        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "Decoder and skip spatial dimensions must match, "
                f"got {x.shape[-2:]} and {skip.shape[-2:]}."
            )

        x = torch.cat([x, skip], dim=1)
        x = self.attn(x)
        x = self.decoder(x)

        if x.shape != skip.shape:
            raise ValueError(
                "Decoder output and skip must match for residual "
                f"addition, got {x.shape} and {skip.shape}."
            )

        return x + skip

    
class DecoderPath(nn.Module):
    def __init__(self, base_channels=2):
        super().__init__()

        self.dec1 = DecoderStage(
            x_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )

        self.dec2 = DecoderStage(
            x_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.dec3 = DecoderStage(
            x_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        self.output = OutputBlock(
            in_channels=base_channels,
            out_channels=1,
        )

    def forward(self, x, skips, encoder_attn=None):
        skip1, skip2, skip3 = skips

        x = self.dec1(x, skip3, encoder_attn)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        x = self.output(x)

        return x
        
