"""LR-ASPP semantic-map generator for the DeepDCT-VO Fig. 2 pipeline.

This wrapper uses Torchvision's pretrained LR-ASPP MobileNetV3-Large model
to generate a single-channel semantic map for each RGB frame.

Interfaces
----------
forward_logits(x)
    Return full-resolution segmentation logits: [B, K, H, W].

forward_labels(x)
    Return integer class labels: [B, 1, H, W].

forward_map(x)
    Return the normalized single-channel semantic map S_k used by Fig. 2:
    [B, 1, H, W].

forward(x)
    Alias of forward_map(x), so the module can be used directly as the
    semantic-map branch in DeepDCTVO.

The RGB tensor supplied to this module must contain floating-point values in
[0, 1]. ImageNet normalization is applied internally by default.
"""

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.segmentation import lraspp_mobilenet_v3_large

try:
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
    )
except ImportError:
    LRASPP_MobileNet_V3_Large_Weights = None


class LRASPPSemanticBranch(nn.Module):
    """Generate the one-channel semantic maps used by DeepDCT-VO.

    Args:
        pretrained:
            Load Torchvision's pretrained LR-ASPP segmentation weights.

        freeze_pretrained:
            Freeze the complete LR-ASPP network. This is the recommended
            setting when LR-ASPP is used only to generate fixed semantic
            maps for DeepDCT-VO.

        normalize_input:
            Apply ImageNet mean/std normalization internally. Set this to
            False only when the caller already supplies normalized images.

        normalize_map:
            Convert class IDs to floating-point values in [0, 1]. This is
            appropriate for concatenation with RGB images in the four-channel
            A-ResUNet input.

        progress:
            Display Torchvision's checkpoint download progress.

    Input:
        RGB image tensor [B, 3, H, W], floating point.

    Default output:
        Semantic map [B, 1, H, W], floating point in [0, 1].
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_pretrained: bool = True,
        normalize_input: bool = True,
        normalize_map: bool = True,
        progress: bool = True,
    ) -> None:
        super().__init__()

        self.freeze_pretrained = freeze_pretrained
        self.normalize_input = normalize_input
        self.normalize_map = normalize_map

        self.model = self._build_model(
            pretrained=pretrained,
            progress=progress,
        )

        self._validate_torchvision_structure()

        self.num_classes = int(
            self.model.classifier.low_classifier.out_channels
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_pretrained:
            self.freeze_pretrained_model()

    @staticmethod
    def _build_model(
        pretrained: bool,
        progress: bool,
    ) -> nn.Module:
        """Construct LR-ASPP across current and legacy Torchvision APIs."""
        if LRASPP_MobileNet_V3_Large_Weights is not None:
            weights = (
                LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                if pretrained
                else None
            )

            return lraspp_mobilenet_v3_large(
                weights=weights,
                # Prevent an additional backbone-weight download when
                # pretrained=False, which is useful for offline tests.
                weights_backbone=None,
                progress=progress,
            )

        # Compatibility with older Torchvision versions.
        return lraspp_mobilenet_v3_large(
            pretrained=pretrained,
            progress=progress,
        )

    def _validate_torchvision_structure(self) -> None:
        """Fail early if the installed Torchvision layout is incompatible."""
        if not hasattr(self.model, "backbone"):
            raise RuntimeError(
                "Torchvision LR-ASPP does not expose a 'backbone' module."
            )

        if not hasattr(self.model, "classifier"):
            raise RuntimeError(
                "Torchvision LR-ASPP does not expose a 'classifier' module."
            )

        if not hasattr(self.model.classifier, "low_classifier"):
            raise RuntimeError(
                "Torchvision LR-ASPP classifier does not expose "
                "'low_classifier'."
            )

    def freeze_pretrained_model(self) -> None:
        """Freeze LR-ASPP and keep its BatchNorm layers in evaluation mode."""
        self.freeze_pretrained = True

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def unfreeze_pretrained_model(self) -> None:
        """Enable LR-ASPP fine-tuning."""
        self.freeze_pretrained = False

        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "LRASPPSemanticBranch":
        """Keep a frozen LR-ASPP model in evaluation mode."""
        super().train(mode)

        if self.freeze_pretrained:
            self.model.eval()

        return self

    def prepare_input(self, x: Tensor) -> Tensor:
        """Validate and optionally apply ImageNet normalization."""
        self._validate_input(x)

        if not self.normalize_input:
            return x

        mean = self.image_mean.to(
            device=x.device,
            dtype=x.dtype,
        )
        std = self.image_std.to(
            device=x.device,
            dtype=x.dtype,
        )

        return (x - mean) / std

    def forward_logits(self, x: Tensor) -> Tensor:
        """Return full-resolution LR-ASPP class logits [B, K, H, W]."""
        x = self.prepare_input(x)
        output = self.model(x)

        if not isinstance(output, (dict, OrderedDict)):
            raise TypeError(
                "Torchvision LR-ASPP must return a mapping containing 'out'."
            )

        if "out" not in output:
            raise KeyError(
                "Torchvision LR-ASPP output does not contain 'out'. "
                f"Received keys: {list(output.keys())}."
            )

        logits = output["out"]

        if logits.ndim != 4:
            raise RuntimeError(
                "LR-ASPP logits must have shape [B, K, H, W], "
                f"but received {tuple(logits.shape)}."
            )

        return logits

    def forward_labels(self, x: Tensor) -> Tensor:
        """Return single-channel integer class IDs [B, 1, H, W]."""
        logits = self.forward_logits(x)

        return torch.argmax(
            logits,
            dim=1,
            keepdim=True,
        )

    def forward_map(self, x: Tensor) -> Tensor:
        """Return the one-channel semantic map S_k used in Fig. 2.

        The class-index map is converted to the input floating-point dtype.
        By default, class IDs are normalized to [0, 1] before concatenation
        with the corresponding RGB frame:

            CI_k = cat(I_k, S_k)

        Argmax is intentionally non-differentiable because this branch is
        intended to operate as a frozen pretrained map generator.
        """
        labels = self.forward_labels(x)
        semantic_map = labels.to(dtype=x.dtype)

        if self.normalize_map:
            denominator = float(max(self.num_classes - 1, 1))
            semantic_map = semantic_map / denominator

        return semantic_map

    def forward_all(self, x: Tensor) -> Dict[str, Tensor]:
        """Return logits, integer labels, and the normalized semantic map."""
        logits = self.forward_logits(x)

        labels = torch.argmax(
            logits,
            dim=1,
            keepdim=True,
        )

        semantic_map = labels.to(dtype=x.dtype)

        if self.normalize_map:
            denominator = float(max(self.num_classes - 1, 1))
            semantic_map = semantic_map / denominator

        return {
            "logits": logits,
            "labels": labels,
            "semantic_map": semantic_map,
        }

    def forward(self, x: Tensor) -> Tensor:
        """Return the Fig. 2-compatible one-channel semantic map."""
        return self.forward_map(x)

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError(
                "LRASPPSemanticBranch expects a torch.Tensor, "
                f"but received {type(x).__name__}."
            )

        if x.ndim != 4:
            raise ValueError(
                "LRASPPSemanticBranch expects [B, 3, H, W], "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                "LRASPPSemanticBranch expects three RGB channels, "
                f"but received {x.shape[1]}."
            )

        if not x.is_floating_point():
            raise TypeError(
                "LRASPPSemanticBranch expects floating-point RGB tensors, "
                f"but received {x.dtype}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "LRASPPSemanticBranch input contains NaN or infinite values."
            )
