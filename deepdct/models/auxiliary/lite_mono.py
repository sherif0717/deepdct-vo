"""Lite-Mono depth auxiliary branch for the DeepDCT-VO Fig. 2 pipeline.

This wrapper exposes:

    depth_curr = depth_model(image_curr)

Input:
    image_curr: [B, 3, H, W], floating-point RGB tensor in [0, 1].

Default output:
    depth_curr: [B, 1, H, W], clipped to 80 m and normalized to [0, 1].

Vendor requirement:
    Copy the official Lite-Mono ``networks`` package into

        deepdct/models/auxiliary/lite_mono_vendor/networks/

    The package must expose ``LiteMono`` and ``DepthDecoder``.

Checkpoint requirement:
    The pretrained checkpoint directory must contain

        encoder.pth
        depth.pth
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


try:
    from .lite_mono_vendor.networks import DepthDecoder, LiteMono
except ImportError as exc:
    raise ImportError(
        "Lite-Mono vendor code was not found. Copy the official "
        "Lite-Mono 'networks' package to "
        "'deepdct/models/auxiliary/lite_mono_vendor/networks/'."
    ) from exc



# deepdct/models/auxiliary/lite_mono.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

_DEFAULT_CHECKPOINT_DIR = (
    _DEFAULT_WEIGHTS_DIR / "lite-mono-tiny-640x192"
)


PathLike = Union[str, Path]


def disparity_to_depth(
    disparity: Tensor,
    min_depth: float,
    max_depth: float,
) -> Tuple[Tensor, Tensor]:
    """Convert sigmoid disparity to scaled disparity and inverse depth."""
    if min_depth <= 0.0:
        raise ValueError("min_depth must be positive.")

    if max_depth <= min_depth:
        raise ValueError("max_depth must be greater than min_depth.")

    min_disparity = 1.0 / max_depth
    max_disparity = 1.0 / min_depth

    scaled_disparity = (
        min_disparity
        + (max_disparity - min_disparity) * disparity
    )
    depth = 1.0 / scaled_disparity

    return scaled_disparity, depth


class LiteMonoDepthBranch(nn.Module):
    """Pretrained Lite-Mono depth-map generator."""

    SUPPORTED_MODELS = {
        "lite-mono",
        "lite-mono-small",
        "lite-mono-tiny",
        "lite-mono-8m",
    }

    SUPPORTED_OUTPUT_MODES = {
        "normalized_depth",
        "depth",
        "disparity",
        "scaled_disparity",
    }

    def __init__(
        self,
        checkpoint_dir: Optional[PathLike] = _DEFAULT_CHECKPOINT_DIR,
        model_name: str = "lite-mono-tiny",
        feed_size: Tuple[int, int] = (192, 640),
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        normalization_depth: float = 80.0,
        output_mode: str = "normalized_depth",
        freeze_pretrained: bool = True,
        strict_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self._validate_configuration(
            model_name=model_name,
            feed_size=feed_size,
            min_depth=min_depth,
            max_depth=max_depth,
            normalization_depth=normalization_depth,
            output_mode=output_mode,
        )

        self.model_name = model_name
        self.feed_height = int(feed_size[0])
        self.feed_width = int(feed_size[1])
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.normalization_depth = float(normalization_depth)
        self.output_mode = output_mode
        self.freeze_pretrained = freeze_pretrained

        encoder_checkpoint = None
        decoder_checkpoint = None

        if checkpoint_dir is not None:
            encoder_checkpoint, decoder_checkpoint = self._read_checkpoints(
                checkpoint_dir
            )

            self.feed_height = int(
                encoder_checkpoint.get("height", self.feed_height)
            )
            self.feed_width = int(
                encoder_checkpoint.get("width", self.feed_width)
            )

        self.encoder = LiteMono(
            model=self.model_name,
            height=self.feed_height,
            width=self.feed_width,
        )

        self.decoder = DepthDecoder(
            self.encoder.num_ch_enc,
            scales=range(3),
        )

        if encoder_checkpoint is not None:
            self._load_filtered_state_dict(
                module=self.encoder,
                checkpoint=encoder_checkpoint,
                checkpoint_name="encoder.pth",
                strict=strict_checkpoint,
            )

            self._load_filtered_state_dict(
                module=self.decoder,
                checkpoint=decoder_checkpoint,
                checkpoint_name="depth.pth",
                strict=strict_checkpoint,
            )

        if freeze_pretrained:
            self.freeze()

    def freeze(self) -> None:
        self.freeze_pretrained = True

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.encoder.eval()
        self.decoder.eval()

    def unfreeze(self) -> None:
        self.freeze_pretrained = False

        for parameter in self.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "LiteMonoDepthBranch":
        super().train(mode)

        if self.freeze_pretrained:
            self.encoder.eval()
            self.decoder.eval()

        return self

    def forward_disparity(self, image: Tensor) -> Tensor:
        """Return full-resolution sigmoid disparity [B, 1, H, W]."""
        self._validate_input(image)

        original_size = image.shape[-2:]

        resized = F.interpolate(
            image,
            size=(self.feed_height, self.feed_width),
            mode="bilinear",
            align_corners=False,
        )

        features = self.encoder(resized)
        outputs = self.decoder(features)

        key = ("disp", 0)
        if key not in outputs:
            raise KeyError(
                "Lite-Mono decoder output does not contain ('disp', 0). "
                f"Received keys: {list(outputs.keys())}."
            )

        disparity = outputs[key]

        if disparity.ndim != 4 or disparity.shape[1] != 1:
            raise RuntimeError(
                "Lite-Mono disparity must have shape [B, 1, H, W], "
                f"but received {tuple(disparity.shape)}."
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
            disparity=disparity,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

        normalized_depth = depth.clamp(
            min=0.0,
            max=self.normalization_depth,
        ) / self.normalization_depth

        return {
            "disparity": disparity,
            "scaled_disparity": scaled_disparity,
            "depth": depth,
            "normalized_depth": normalized_depth,
        }

    def forward(self, image: Tensor) -> Tensor:
        """Return the configured one-channel current-frame depth map."""
        outputs = self.forward_all(image)
        return outputs[self.output_mode]

    @staticmethod
    def _read_checkpoints(
        checkpoint_dir: PathLike,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
        encoder_path = checkpoint_path / "encoder.pth"
        decoder_path = checkpoint_path / "depth.pth"

        if not encoder_path.is_file():
            raise FileNotFoundError(
                f"Lite-Mono encoder checkpoint not found: {encoder_path}"
            )

        if not decoder_path.is_file():
            raise FileNotFoundError(
                f"Lite-Mono decoder checkpoint not found: {decoder_path}"
            )

        encoder_checkpoint = torch.load(
            str(encoder_path),
            map_location="cpu",
            weights_only=True,
        )
        decoder_checkpoint = torch.load(
            str(decoder_path),
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(encoder_checkpoint, dict):
            raise TypeError("encoder.pth must contain a dictionary.")

        if not isinstance(decoder_checkpoint, dict):
            raise TypeError("depth.pth must contain a dictionary.")

        return encoder_checkpoint, decoder_checkpoint

    @staticmethod
    def _load_filtered_state_dict(
        module: nn.Module,
        checkpoint: Dict[str, Tensor],
        checkpoint_name: str,
        strict: bool,
    ) -> None:
        model_state = module.state_dict()

        filtered_state = {
            key: value
            for key, value in checkpoint.items()
            if key in model_state and torch.is_tensor(value)
        }

        if not filtered_state:
            raise RuntimeError(
                f"No compatible parameters were found in {checkpoint_name}."
            )

        incompatible = module.load_state_dict(
            filtered_state,
            strict=False,
        )

        if strict and incompatible.missing_keys:
            raise RuntimeError(
                f"{checkpoint_name} is missing model parameters: "
                f"{incompatible.missing_keys}"
            )

        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"{checkpoint_name} contains unexpected model parameters: "
                f"{incompatible.unexpected_keys}"
            )

    @classmethod
    def _validate_configuration(
        cls,
        model_name: str,
        feed_size: Tuple[int, int],
        min_depth: float,
        max_depth: float,
        normalization_depth: float,
        output_mode: str,
    ) -> None:
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Lite-Mono model '{model_name}'. "
                f"Choose from {sorted(cls.SUPPORTED_MODELS)}."
            )

        if len(feed_size) != 2 or min(feed_size) <= 0:
            raise ValueError(
                "feed_size must contain positive height and width."
            )

        if min_depth <= 0.0:
            raise ValueError("min_depth must be positive.")

        if max_depth <= min_depth:
            raise ValueError(
                "max_depth must be greater than min_depth."
            )

        if normalization_depth <= 0.0:
            raise ValueError(
                "normalization_depth must be positive."
            )

        if output_mode not in cls.SUPPORTED_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported output_mode '{output_mode}'. "
                f"Choose from {sorted(cls.SUPPORTED_OUTPUT_MODES)}."
            )

    @staticmethod
    def _validate_input(image: Tensor) -> None:
        if not torch.is_tensor(image):
            raise TypeError(
                "LiteMonoDepthBranch expects a torch.Tensor, "
                f"but received {type(image).__name__}."
            )

        if image.ndim != 4:
            raise ValueError(
                "LiteMonoDepthBranch expects [B, 3, H, W], "
                f"but received {tuple(image.shape)}."
            )

        if image.shape[1] != 3:
            raise ValueError(
                "LiteMonoDepthBranch expects three RGB channels, "
                f"but received {image.shape[1]}."
            )

        if not image.is_floating_point():
            raise TypeError(
                "LiteMonoDepthBranch expects floating-point RGB tensors, "
                f"but received {image.dtype}."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                "LiteMonoDepthBranch input contains NaN or infinite values."
            )
