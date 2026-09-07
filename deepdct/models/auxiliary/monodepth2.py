"""Frozen Monodepth2 depth provider with the DeepDCT-VO depth-map contract.

Expected vendor layout::

    deepdct/models/auxiliary/monodepth2_vendor/
        __init__.py
        layers.py
        networks/
            __init__.py
            depth_decoder.py
            resnet_encoder.py

In the vendored ``networks/depth_decoder.py``, change the upstream absolute
import ``from layers import ...`` to ``from ..layers import ...``.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from .monodepth2_vendor.networks import DepthDecoder, ResnetEncoder
except ImportError as exc:
    raise ImportError(
        "Monodepth2 vendor code was not found. See the A6_monodepth2 "
        "placement guide for the required vendor layout."
    ) from exc


PathLike = Union[str, Path]


def disparity_to_depth(
    disparity: Tensor,
    min_depth: float,
    max_depth: float,
) -> Tuple[Tensor, Tensor]:
    """Match the Lite-Mono/Monodepth2 sigmoid-disparity conversion."""
    if min_depth <= 0.0:
        raise ValueError("min_depth must be positive.")
    if max_depth <= min_depth:
        raise ValueError("max_depth must be greater than min_depth.")

    min_disparity = 1.0 / max_depth
    max_disparity = 1.0 / min_depth
    scaled_disparity = min_disparity + (
        max_disparity - min_disparity
    ) * disparity
    return scaled_disparity, 1.0 / scaled_disparity


class Monodepth2DepthBranch(nn.Module):
    """Pretrained Monodepth2 adapter preserving DeepDCT-VO's cue contract."""

    SUPPORTED_OUTPUT_MODES = {
        "normalized_depth",
        "depth",
        "disparity",
        "scaled_disparity",
    }

    def __init__(
        self,
        checkpoint_dir: PathLike,
        num_layers: int = 18,
        feed_size: Tuple[int, int] = (192, 640),
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        normalization_depth: float = 80.0,
        output_mode: str = "normalized_depth",
        freeze_pretrained: bool = True,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if num_layers not in {18, 34, 50, 101, 152}:
            raise ValueError("Unsupported Monodepth2 ResNet depth.")
        if len(feed_size) != 2 or min(feed_size) <= 0:
            raise ValueError("feed_size must contain positive height and width.")
        if min_depth <= 0.0 or max_depth <= min_depth:
            raise ValueError("Require 0 < min_depth < max_depth.")
        if normalization_depth <= 0.0:
            raise ValueError("normalization_depth must be positive.")
        if output_mode not in self.SUPPORTED_OUTPUT_MODES:
            raise ValueError(
                "Unsupported output_mode. Choose from "
                f"{sorted(self.SUPPORTED_OUTPUT_MODES)}."
            )

        checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
        encoder_checkpoint = self._load_checkpoint(
            checkpoint_path / "encoder.pth"
        )
        decoder_checkpoint = self._load_checkpoint(
            checkpoint_path / "depth.pth"
        )

        self.feed_height = int(encoder_checkpoint.get("height", feed_size[0]))
        self.feed_width = int(encoder_checkpoint.get("width", feed_size[1]))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.normalization_depth = float(normalization_depth)
        self.output_mode = output_mode
        self.freeze_pretrained = bool(freeze_pretrained)

        self.encoder = ResnetEncoder(num_layers, pretrained=False)
        self.decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4),
        )

        self._load_filtered_state_dict(
            self.encoder, encoder_checkpoint, "encoder.pth", strict_checkpoint
        )
        self._load_filtered_state_dict(
            self.decoder, decoder_checkpoint, "depth.pth", strict_checkpoint
        )

        if self.freeze_pretrained:
            self.freeze()

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict[str, Tensor]:
        if not path.is_file():
            raise FileNotFoundError(f"Monodepth2 checkpoint not found: {path}")
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # Older PyTorch compatibility.
            checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{path.name} must contain a dictionary.")
        return checkpoint

    @staticmethod
    def _load_filtered_state_dict(
        module: nn.Module,
        checkpoint: Dict[str, Tensor],
        checkpoint_name: str,
        strict: bool,
    ) -> None:
        model_state = module.state_dict()
        filtered = {
            key: value
            for key, value in checkpoint.items()
            if key in model_state and torch.is_tensor(value)
        }
        if not filtered:
            raise RuntimeError(f"No compatible parameters in {checkpoint_name}.")
        incompatible = module.load_state_dict(filtered, strict=False)
        if strict and incompatible.missing_keys:
            raise RuntimeError(
                f"{checkpoint_name} missing parameters: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"{checkpoint_name} unexpected parameters: "
                f"{incompatible.unexpected_keys}"
            )

    def freeze(self) -> None:
        self.freeze_pretrained = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.decoder.eval()

    def unfreeze(self) -> None:
        self.freeze_pretrained = False
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "Monodepth2DepthBranch":
        super().train(mode)
        if self.freeze_pretrained:
            self.encoder.eval()
            self.decoder.eval()
        return self

    @staticmethod
    def _validate_input(image: Tensor) -> None:
        if not torch.is_tensor(image):
            raise TypeError("Monodepth2DepthBranch expects a torch.Tensor.")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB input shaped [B, 3, H, W].")
        if not image.is_floating_point():
            raise TypeError("Expected floating-point RGB input.")
        if not torch.isfinite(image).all():
            raise ValueError("RGB input contains NaN or infinite values.")

    def forward_disparity(self, image: Tensor) -> Tensor:
        """Return finest-scale sigmoid disparity at incoming VO resolution."""
        self._validate_input(image)
        original_size = image.shape[-2:]
        resized = F.interpolate(
            image,
            size=(self.feed_height, self.feed_width),
            mode="bilinear",
            align_corners=False,
        )
        outputs = self.decoder(self.encoder(resized))
        key = ("disp", 0)
        if key not in outputs:
            raise KeyError(
                "Monodepth2 decoder output lacks ('disp', 0); "
                f"received {list(outputs.keys())}."
            )
        disparity = outputs[key]
        if disparity.ndim != 4 or disparity.shape[1] != 1:
            raise RuntimeError(
                "Monodepth2 disparity must be [B, 1, H, W], received "
                f"{tuple(disparity.shape)}."
            )
        if disparity.shape[-2:] != original_size:
            disparity = F.interpolate(
                disparity,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return disparity

    def forward_all(self, image: Tensor) -> Dict[str, Tensor]:
        disparity = self.forward_disparity(image)
        scaled_disparity, depth = disparity_to_depth(
            disparity, self.min_depth, self.max_depth
        )
        normalized_depth = depth.clamp(
            min=0.0, max=self.normalization_depth
        ) / self.normalization_depth
        return {
            "disparity": disparity,
            "scaled_disparity": scaled_disparity,
            "depth": depth,
            "normalized_depth": normalized_depth,
        }

    def forward(self, image: Tensor) -> Tensor:
        return self.forward_all(image)[self.output_mode]
