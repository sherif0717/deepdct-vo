"""Rotation and directional-translation heads for DeepDCT-VO.

Fig. 2 defines two separate regression models:

Model R:
    cat(C_prev, C_curr, S_curr, D_curr)
        -> Conv2D(1, 3x3)
        -> ReLU
        -> Dropout(0.2)
        -> Flatten
        -> Dense(3)
        -> LeakyReLU
        -> rotation

Model T:
    cat(C_prev, C_curr, S_curr, D_curr, rotation_map)
        -> Conv2D(1, 3x3)
        -> ReLU
        -> Dropout(0.2)
        -> Flatten
        -> Dense(3)
        -> LeakyReLU
        -> directional translation

The two heads use the same internal topology but consume different
numbers of input channels.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RegressionHead(nn.Module):
    """Shared paper-style regression block.

    Args:
        in_channels:
            Number of channels in the concatenated fusion tensor.

        input_size:
            Spatial size used before flattening. The paper commonly uses
            120 x 120 for its lightweight configuration.

        dropout:
            Dropout probability used after the convolution.

        negative_slope:
            Negative slope for the final LeakyReLU.

    Input:
        Tensor shaped [B, in_channels, H, W].

    Output:
        Tensor shaped [B, 3].
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int] = (120, 120),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive, "
                f"but received {in_channels}."
            )

        if len(input_size) != 2:
            raise ValueError(
                "input_size must contain height and width."
            )

        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(
                "input_size values must be positive, "
                f"but received {input_size}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1, "
                f"but received {dropout}."
            )

        if negative_slope < 0.0:
            raise ValueError(
                "negative_slope cannot be negative."
            )

        self.in_channels = in_channels
        self.input_size = tuple(input_size)

        # CF2 in Eqs. (16) and (21): reduce fused channels to one.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

        # The paper states dropout=0.2 for the Conv2D stage.
        self.dropout = nn.Dropout2d(
            p=dropout,
        )

        flattened_size = (
            self.input_size[0]
            * self.input_size[1]
        )

        self.dense = nn.Linear(
            in_features=flattened_size,
            out_features=3,
            bias=True,
        )

        self.output_activation = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)

        if x.shape[-2:] != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.dense(x)
        x = self.output_activation(x)

        return x

    def _validate_input(self, x: Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError(
                "RegressionHead expects a torch.Tensor, "
                f"but received {type(x).__name__}."
            )

        if x.ndim != 4:
            raise ValueError(
                "RegressionHead expects [B, C, H, W], "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                "Unexpected fusion-channel count. "
                f"Expected {self.in_channels}, "
                f"received {x.shape[1]}."
            )

        if not x.is_floating_point():
            raise TypeError(
                "RegressionHead expects floating-point features, "
                f"but received {x.dtype}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "RegressionHead input contains NaN or infinite values."
            )


class RotationHead(RegressionHead):
    """Estimate relative roll, pitch, and yaw.

    Expected Model R fusion:

        cat(C_prev, C_curr, S_curr, D_curr)

    Output ordering:

        [roll, pitch, yaw]

    Use the same ordering as your generated rotation labels throughout
    training, evaluation, and coordinate reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int] = (120, 120),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_size=input_size,
            dropout=dropout,
            negative_slope=negative_slope,
        )


class DirectionalTranslationHead(RegressionHead):
    """Estimate local directional translation.

    Expected Model T fusion:

        cat(C_prev, C_curr, S_curr, D_curr, rotation_map)

    Output ordering should match the DCT-label generator, for example:

        [t_forward, t_lateral, t_vertical]

    or the exact axis ordering already used in your label files.
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int] = (120, 120),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_size=input_size,
            dropout=dropout,
            negative_slope=negative_slope,
        )