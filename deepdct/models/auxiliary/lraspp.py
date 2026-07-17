"""LR-ASPP semantic auxiliary branch for DeepDCT-VO.

The module wraps Torchvision's LR-ASPP MobileNetV3-Large network and
provides three interfaces:

    forward_backbone(x)
        Returns the native low- and high-level MobileNetV3 feature maps.

    forward_features(x)
        Returns a compact semantic feature map intended for pose fusion.

    forward(x)
        Returns full-resolution semantic segmentation logits.

The module processes one RGB frame at a time. DeepDCT-VO should call it
separately for the two frames in an image pair.
"""

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.models.segmentation import lraspp_mobilenet_v3_large

try:
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
    )
except ImportError:
    # Allows the module to import under older Torchvision releases.
    LRASPP_MobileNet_V3_Large_Weights = None


class ConvBNReLU(nn.Sequential):
    """1x1 or 3x3 feature projection block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
    ) -> None:
        if kernel_size not in (1, 3):
            raise ValueError(
                "ConvBNReLU supports kernel sizes 1 and 3, "
                f"but received {kernel_size}."
            )

        padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LRASPPSemanticBranch(nn.Module):
    """Pretrained LR-ASPP semantic branch for pose feature extraction.

    The pretrained LR-ASPP model contributes:

        low:
            Relatively high-resolution MobileNetV3 features, normally at
            approximately 1/8 of the input spatial resolution.

        high:
            Deeper semantic features, normally at approximately 1/16 of
            the input spatial resolution.

        context:
            High-level features after LR-ASPP's pretrained context
            transformation and global gating operations.

    The low and context feature maps are projected, spatially aligned,
    concatenated, and compressed into ``pose_feature_channels`` channels.

    Args:
        pretrained:
            Load Torchvision's pretrained LR-ASPP segmentation weights.

        pose_feature_channels:
            Number of channels returned by ``forward_features``.

        freeze_pretrained:
            Freeze the MobileNetV3 backbone and LR-ASPP classifier.
            The pose-feature adapter remains trainable.

        progress:
            Display Torchvision's checkpoint-download progress.

    Input:
        Tensor shaped ``[B, 3, H, W]``.

    Outputs:
        ``forward_backbone``:
            Dictionary containing ``low`` and ``high``.

        ``forward_features``:
            Tensor shaped approximately
            ``[B, pose_feature_channels, H/8, W/8]``.

        ``forward``:
            Full-resolution segmentation logits shaped
            ``[B, 21, H, W]`` when the default pretrained weights are used.
    """

    # Torchvision LR-ASPP MobileNetV3-Large channel counts.
    LOW_CHANNELS = 40
    HIGH_CHANNELS = 960
    CONTEXT_CHANNELS = 128

    def __init__(
        self,
        pretrained: bool = True,
        pose_feature_channels: int = 64,
        freeze_pretrained: bool = True,
        progress: bool = True,
    ) -> None:
        super().__init__()

        if pose_feature_channels <= 0:
            raise ValueError(
                "pose_feature_channels must be positive, "
                f"but received {pose_feature_channels}."
            )

        self.pose_feature_channels = pose_feature_channels
        self.freeze_pretrained = freeze_pretrained

        self.model = self._build_model(
            pretrained=pretrained,
            progress=progress,
        )

        self._validate_torchvision_structure()

        # The low-level feature map retains more spatial detail.
        self.low_projection = ConvBNReLU(
            in_channels=self.LOW_CHANNELS,
            out_channels=pose_feature_channels,
            kernel_size=1,
        )

        # The context feature has already passed through the pretrained
        # LR-ASPP cbr and global scale branches.
        self.context_projection = ConvBNReLU(
            in_channels=self.CONTEXT_CHANNELS,
            out_channels=pose_feature_channels,
            kernel_size=1,
        )

        # Compress concatenated low-level and context features to a compact
        # representation for the DeepDCT-VO pose head.
        self.pose_feature_fusion = nn.Sequential(
            ConvBNReLU(
                in_channels=2 * pose_feature_channels,
                out_channels=pose_feature_channels,
                kernel_size=3,
            ),
            nn.Conv2d(
                pose_feature_channels,
                pose_feature_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(pose_feature_channels),
            nn.ReLU(inplace=True),
        )

        if freeze_pretrained:
            self.freeze_pretrained_model()

    @staticmethod
    def _build_model(
        pretrained: bool,
        progress: bool,
    ) -> nn.Module:
        """Construct LR-ASPP across modern and older Torchvision APIs."""
        if LRASPP_MobileNet_V3_Large_Weights is not None:
            weights = (
                LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                if pretrained
                else None
            )

            # When full segmentation weights are absent, explicitly avoid
            # downloading separate backbone weights. This is useful for
            # offline unit tests and random-initialization experiments.
            weights_backbone = None

            return lraspp_mobilenet_v3_large(
                weights=weights,
                weights_backbone=weights_backbone,
                progress=progress,
            )

        # Legacy Torchvision API.
        return lraspp_mobilenet_v3_large(
            pretrained=pretrained,
            progress=progress,
        )

    def _validate_torchvision_structure(self) -> None:
        """Fail early if Torchvision changes the expected LR-ASPP layout."""
        if not hasattr(self.model, "backbone"):
            raise RuntimeError(
                "The installed Torchvision LR-ASPP model does not expose "
                "a 'backbone' module."
            )

        if not hasattr(self.model, "classifier"):
            raise RuntimeError(
                "The installed Torchvision LR-ASPP model does not expose "
                "a 'classifier' module."
            )

        required_classifier_modules = (
            "cbr",
            "scale",
        )

        for name in required_classifier_modules:
            if not hasattr(self.model.classifier, name):
                raise RuntimeError(
                    "The installed Torchvision LR-ASPP classifier does not "
                    f"expose the expected '{name}' module."
                )

    def freeze_pretrained_model(self) -> None:
        """Freeze the pretrained backbone and LR-ASPP classifier."""
        self.freeze_pretrained = True

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def unfreeze_pretrained_model(self) -> None:
        """Enable fine-tuning of the backbone and segmentation classifier."""
        self.freeze_pretrained = False

        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "LRASPPSemanticBranch":
        """Keep frozen pretrained BatchNorm layers in evaluation mode."""
        super().train(mode)

        if self.freeze_pretrained:
            self.model.eval()

        return self

    def forward_backbone(self, x: Tensor) -> Dict[str, Tensor]:
        """Return Torchvision's native low- and high-level feature maps."""
        self._validate_input(x)

        features = self.model.backbone(x)

        if not isinstance(features, (dict, OrderedDict)):
            raise TypeError(
                "LR-ASPP backbone must return a mapping containing "
                "'low' and 'high' feature tensors."
            )

        if "low" not in features or "high" not in features:
            raise KeyError(
                "LR-ASPP backbone output must contain 'low' and 'high'. "
                f"Received keys: {list(features.keys())}."
            )

        low = features["low"]
        high = features["high"]

        self._validate_feature_channels(
            low=low,
            high=high,
        )

        return {
            "low": low,
            "high": high,
        }

    def forward_context(self, high: Tensor) -> Tensor:
        """Apply LR-ASPP's pretrained semantic context transformation.

        This reproduces the high-level branch inside Torchvision's
        LR-ASPP classifier before its final class projection:

            context = cbr(high) * scale(high)
        """
        context = self.model.classifier.cbr(high)
        scale = self.model.classifier.scale(high)

        return context * scale

    def forward_features(self, x: Tensor) -> Tensor:
        """Return a pose-ready semantic feature map.

        The returned map combines:

        - spatial detail from the low-level MobileNetV3 features;
        - semantic context from LR-ASPP's pretrained high-level branch.
        """
        features = self.forward_backbone(x)

        low = features["low"]
        high = features["high"]

        context = self.forward_context(high)

        context = F.interpolate(
            context,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low = self.low_projection(low)
        context = self.context_projection(context)

        fused = torch.cat(
            [low, context],
            dim=1,
        )

        return self.pose_feature_fusion(fused)

    def forward_all(self, x: Tensor) -> Dict[str, Tensor]:
        """Return backbone, pose, and segmentation outputs in one pass.

        This method avoids evaluating the MobileNetV3 backbone twice when
        both semantic logits and pose features are needed during training.
        """
        self._validate_input(x)

        input_size = x.shape[-2:]
        features = self.forward_backbone(x)

        low = features["low"]
        high = features["high"]

        context = self.forward_context(high)
        context_at_low_resolution = F.interpolate(
            context,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low_pose = self.low_projection(low)
        context_pose = self.context_projection(
            context_at_low_resolution
        )

        pose_features = self.pose_feature_fusion(
            torch.cat(
                [low_pose, context_pose],
                dim=1,
            )
        )

        # Use the complete pretrained LR-ASPP classifier for segmentation.
        segmentation_logits = self.model.classifier(
            {
                "low": low,
                "high": high,
            }
        )

        segmentation_logits = F.interpolate(
            segmentation_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "low": low,
            "high": high,
            "context": context,
            "pose_features": pose_features,
            "logits": segmentation_logits,
        }

    def forward(self, x: Tensor) -> Tensor:
        """Return full-resolution LR-ASPP semantic logits."""
        self._validate_input(x)

        output = self.model(x)

        if not isinstance(output, (dict, OrderedDict)):
            raise TypeError(
                "Torchvision LR-ASPP must return a mapping containing "
                "the 'out' segmentation tensor."
            )

        if "out" not in output:
            raise KeyError(
                "Torchvision LR-ASPP output does not contain 'out'. "
                f"Received keys: {list(output.keys())}."
            )

        return output["out"]

    @classmethod
    def _validate_feature_channels(
        cls,
        low: Tensor,
        high: Tensor,
    ) -> None:
        if low.ndim != 4 or high.ndim != 4:
            raise ValueError(
                "LR-ASPP backbone features must be four-dimensional. "
                f"Received low={tuple(low.shape)}, "
                f"high={tuple(high.shape)}."
            )

        if low.shape[1] != cls.LOW_CHANNELS:
            raise RuntimeError(
                "Unexpected LR-ASPP low-level channel count. "
                f"Expected {cls.LOW_CHANNELS}, "
                f"received {low.shape[1]}."
            )

        if high.shape[1] != cls.HIGH_CHANNELS:
            raise RuntimeError(
                "Unexpected LR-ASPP high-level channel count. "
                f"Expected {cls.HIGH_CHANNELS}, "
                f"received {high.shape[1]}."
            )

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError(
                "LRASPPSemanticBranch expects a torch.Tensor, "
                f"but received {type(x).__name__}."
            )

        if x.ndim != 4:
            raise ValueError(
                "LRASPPSemanticBranch expects input shaped "
                f"[B, 3, H, W], but received {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                "LRASPPSemanticBranch expects three RGB channels, "
                f"but received {x.shape[1]} channels."
            )

        if not x.is_floating_point():
            raise TypeError(
                "LRASPPSemanticBranch expects floating-point image "
                f"tensors, but received dtype {x.dtype}."
            )