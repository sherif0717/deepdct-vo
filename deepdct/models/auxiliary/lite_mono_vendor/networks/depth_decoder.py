"""Lite-Mono depth decoder adapted for package-local vendoring.

This implementation preserves the module names and state-dict structure used
by the official Lite-Mono ``DepthDecoder`` while avoiding the repository-root
import ``from layers import *``.

Expected use:

    decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    outputs = decoder(encoder_features)
    disparity = outputs[("disp", 0)]
"""

from collections import OrderedDict
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    # PyTorch provides an equivalent initializer in supported modern releases.
    from torch.nn.init import trunc_normal_


class Conv3x3(nn.Module):
    """Reflection-padded 3x3 convolution used by Lite-Mono/Monodepth2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_refl: bool = True,
    ) -> None:
        super().__init__()

        self.pad = (
            nn.ReflectionPad2d(1)
            if use_refl
            else nn.ZeroPad2d(1)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    """Conv3x3 followed by ELU, matching the official decoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = Conv3x3(
            in_channels,
            out_channels,
        )
        self.nonlin = nn.ELU(
            inplace=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.nonlin(self.conv(x))


def upsample(
    x: Tensor,
    scale_factor: int = 2,
    mode: str = "nearest",
) -> Tensor:
    """Upsample a feature map by a factor of two."""
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(
            x,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )

    return F.interpolate(
        x,
        scale_factor=scale_factor,
        mode=mode,
    )


class DepthDecoder(nn.Module):
    """Lite-Mono multi-scale disparity decoder.

    Args:
        num_ch_enc:
            Encoder channel counts. Lite-Mono exposes this through
            ``encoder.num_ch_enc``.

        scales:
            Decoder scales for which disparity maps are generated.
            The official single-image inference path uses ``range(3)``.

        num_output_channels:
            Number of disparity channels. Monocular depth uses one.

        use_skips:
            Concatenate encoder skip features at decoder levels 1 and 2.

    Input:
        Sequence of three encoder feature maps ordered from high to low
        spatial resolution.

    Output:
        Dictionary containing sigmoid disparity maps under keys
        ``("disp", scale)``.
    """

    def __init__(
        self,
        num_ch_enc: Sequence[int],
        scales: Iterable[int] = range(4),
        num_output_channels: int = 1,
        use_skips: bool = True,
    ) -> None:
        super().__init__()

        num_ch_enc_array = np.asarray(
            num_ch_enc,
            dtype=np.int64,
        )

        if num_ch_enc_array.ndim != 1 or len(num_ch_enc_array) < 3:
            raise ValueError(
                "num_ch_enc must contain at least three encoder "
                f"channel counts, but received {num_ch_enc}."
            )

        if np.any(num_ch_enc_array <= 0):
            raise ValueError(
                "All encoder channel counts must be positive."
            )

        self.num_output_channels = int(
            num_output_channels
        )
        self.use_skips = bool(
            use_skips
        )
        self.upsample_mode = "bilinear"
        self.scales = tuple(
            int(scale) for scale in scales
        )
        self.num_ch_enc = num_ch_enc_array

        if self.num_output_channels <= 0:
            raise ValueError(
                "num_output_channels must be positive."
            )

        invalid_scales = [
            scale
            for scale in self.scales
            if scale not in (0, 1, 2)
        ]
        if invalid_scales:
            raise ValueError(
                "DepthDecoder supports scales 0, 1, and 2; "
                f"received invalid scales {invalid_scales}."
            )

        # The official Lite-Mono decoder uses half of the corresponding
        # encoder channels at each decoder level.
        self.num_ch_dec = (
            self.num_ch_enc / 2
        ).astype("int")

        self.convs = OrderedDict()

        for i in range(2, -1, -1):
            # First convolution at this decoder level.
            if i == 2:
                num_ch_in = int(
                    self.num_ch_enc[-1]
                )
            else:
                num_ch_in = int(
                    self.num_ch_dec[i + 1]
                )

            num_ch_out = int(
                self.num_ch_dec[i]
            )

            self.convs[("upconv", i, 0)] = ConvBlock(
                num_ch_in,
                num_ch_out,
            )

            # Second convolution after upsampling and optional skip.
            num_ch_in = int(
                self.num_ch_dec[i]
            )

            if self.use_skips and i > 0:
                num_ch_in += int(
                    self.num_ch_enc[i - 1]
                )

            self.convs[("upconv", i, 1)] = ConvBlock(
                num_ch_in,
                num_ch_out,
            )

        for scale in self.scales:
            self.convs[("dispconv", scale)] = Conv3x3(
                int(self.num_ch_dec[scale]),
                self.num_output_channels,
            )

        # ModuleList registration preserves the official checkpoint key
        # hierarchy under ``decoder.<index>``.
        self.decoder = nn.ModuleList(
            list(self.convs.values())
        )

        self.sigmoid = nn.Sigmoid()
        self.outputs: Dict[Tuple[str, int], Tensor] = {}

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):
            trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0,
                )

    def forward(
        self,
        input_features: Sequence[Tensor],
    ) -> Dict[Tuple[str, int], Tensor]:
        """Decode encoder features into multi-scale sigmoid disparities."""
        self._validate_features(
            input_features
        )

        outputs: Dict[Tuple[str, int], Tensor] = {}
        x = input_features[-1]

        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](
                x
            )

            tensors = [
                upsample(x)
            ]

            if self.use_skips and i > 0:
                skip = input_features[i - 1]

                # Odd input dimensions can produce a one-pixel mismatch.
                if tensors[0].shape[-2:] != skip.shape[-2:]:
                    tensors[0] = F.interpolate(
                        tensors[0],
                        size=skip.shape[-2:],
                        mode="nearest",
                    )

                tensors.append(
                    skip
                )

            x = torch.cat(
                tensors,
                dim=1,
            )

            x = self.convs[("upconv", i, 1)](
                x
            )

            if i in self.scales:
                disparity = self.convs[
                    ("dispconv", i)
                ](x)

                disparity = upsample(
                    disparity,
                    mode=self.upsample_mode,
                )

                outputs[("disp", i)] = self.sigmoid(
                    disparity
                )

        self.outputs = outputs
        return outputs

    def _validate_features(
        self,
        input_features: Sequence[Tensor],
    ) -> None:
        if not isinstance(
            input_features,
            (list, tuple),
        ):
            raise TypeError(
                "DepthDecoder expects a list or tuple of feature maps."
            )

        if len(input_features) < 3:
            raise ValueError(
                "DepthDecoder expects at least three encoder feature maps, "
                f"but received {len(input_features)}."
            )

        batch_size = input_features[0].shape[0]

        for index, feature in enumerate(
            input_features
        ):
            if not torch.is_tensor(
                feature
            ):
                raise TypeError(
                    f"Feature {index} is not a torch.Tensor."
                )

            if feature.ndim != 4:
                raise ValueError(
                    f"Feature {index} must have shape [B, C, H, W], "
                    f"but received {tuple(feature.shape)}."
                )

            if feature.shape[0] != batch_size:
                raise ValueError(
                    "All encoder features must have the same batch size."
                )

            if not feature.is_floating_point():
                raise TypeError(
                    f"Feature {index} must be floating point."
                )
