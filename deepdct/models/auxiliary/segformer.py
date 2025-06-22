"""Frozen SegFormer semantic cue provider for DeepDCT-VO.

The branch accepts RGB tensors in ``[0, 1]`` and returns a single-channel
foreground-probability map at the input resolution.  It deliberately exposes
the same ``Tensor -> Tensor`` contract as ``LRASPPSemanticBranch``.
"""

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from transformers import AutoImageProcessor
    from transformers import SegformerForSemanticSegmentation
except ImportError:  # pragma: no cover - exercised in preflight
    AutoImageProcessor = None
    SegformerForSemanticSegmentation = None


DEFAULT_FOREGROUND_LABELS: Tuple[str, ...] = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "motorbike",
    "bicycle",
    "bike",
)


def _normalize_label(label: object) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


class SegFormerSemanticBranch(nn.Module):
    """Convert SegFormer logits into one foreground-probability channel.

    Args:
        model_name_or_path: Hugging Face model ID or local model directory.
        foreground_class_ids: Explicit semantic class IDs to sum. When omitted,
            IDs are inferred from ``foreground_labels`` and ``config.id2label``.
        foreground_labels: Case-insensitive label names treated as foreground.
        feed_size: Optional ``(height, width)`` inference size. If omitted, the
            image processor's configured size is used when available.
        freeze_pretrained: Disable gradients and keep SegFormer in eval mode.
        local_files_only: Refuse network downloads and use the local cache/path.
    """

    def __init__(
        self,
        model_name_or_path: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        foreground_class_ids: Optional[Sequence[int]] = None,
        foreground_labels: Sequence[str] = DEFAULT_FOREGROUND_LABELS,
        feed_size: Optional[Tuple[int, int]] = None,
        freeze_pretrained: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if AutoImageProcessor is None or SegformerForSemanticSegmentation is None:
            raise ImportError(
                "SegFormer support requires transformers. Install it with "
                "`python -m pip install transformers`."
            )
        self.model_name_or_path = str(model_name_or_path)
        self.freeze_pretrained = bool(freeze_pretrained)
        self.local_files_only = bool(local_files_only)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )

        self.feed_size = self._resolve_feed_size(feed_size)
        self.foreground_labels = tuple(_normalize_label(x) for x in foreground_labels)
        resolved_ids = (
            tuple(int(x) for x in foreground_class_ids)
            if foreground_class_ids is not None
            else self._infer_foreground_class_ids(self.foreground_labels)
        )
        if not resolved_ids:
            raise ValueError(
                "No SegFormer foreground classes were resolved. Supply "
                "foreground_class_ids explicitly or use matching label names."
            )
        num_labels = int(self.model.config.num_labels)
        if min(resolved_ids) < 0 or max(resolved_ids) >= num_labels:
            raise ValueError(
                f"foreground_class_ids must be in [0, {num_labels - 1}], "
                f"but received {resolved_ids}."
            )
        self.foreground_class_ids = tuple(sorted(set(resolved_ids)))
        self.register_buffer(
            "foreground_index",
            torch.tensor(self.foreground_class_ids, dtype=torch.long),
            persistent=True,
        )

        image_mean = getattr(self.image_processor, "image_mean", None)
        image_std = getattr(self.image_processor, "image_std", None)
        if image_mean is None or image_std is None:
            raise ValueError("The SegFormer image processor lacks mean/std values.")
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.freeze_pretrained:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def _resolve_feed_size(
        self,
        feed_size: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        if feed_size is not None:
            height, width = (int(feed_size[0]), int(feed_size[1]))
        else:
            size = getattr(self.image_processor, "size", {})
            if "height" in size and "width" in size:
                height, width = int(size["height"]), int(size["width"])
            elif "shortest_edge" in size:
                height = width = int(size["shortest_edge"])
            else:
                height = width = 512
        if height <= 0 or width <= 0:
            raise ValueError(f"feed_size must be positive, received {(height, width)}.")
        return height, width

    def _infer_foreground_class_ids(
        self,
        labels: Iterable[str],
    ) -> Tuple[int, ...]:
        requested = set(labels)
        matches = []
        for raw_id, raw_label in self.model.config.id2label.items():
            normalized = _normalize_label(raw_label)
            tokens = set(normalized.replace(",", " ").split())
            if normalized in requested or tokens.intersection(requested):
                matches.append(int(raw_id))
        return tuple(matches)

    def train(self, mode: bool = True) -> "SegFormerSemanticBranch":
        super().train(mode)
        if self.freeze_pretrained:
            self.model.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "SegFormer input must have shape [B, 3, H, W], "
                f"received {tuple(image.shape)}."
            )
        if not torch.is_floating_point(image):
            raise TypeError("SegFormer input must be a floating-point tensor.")

        output_size = image.shape[-2:]
        pixel_values = F.interpolate(
            image,
            size=self.feed_size,
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        logits = self.model(pixel_values=pixel_values).logits
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.index_select(1, self.foreground_index).sum(
            dim=1,
            keepdim=True,
        ).clamp_(0.0, 1.0)
