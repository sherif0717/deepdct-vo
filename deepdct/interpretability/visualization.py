from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_image(image: Tensor) -> Tensor:
    image = image.detach().float().cpu()

    image_min = image.min()
    image_max = image.max()

    return (image - image_min) / (image_max - image_min).clamp_min(1e-8)


def tensor_to_rgb(image: Tensor) -> np.ndarray:
    """
    Convert CHW or BCHW tensor into an HWC RGB NumPy image.
    """
    if image.ndim == 4:
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected CHW/BCHW image; got {tuple(image.shape)}.")

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.shape[0] > 3:
        image = image[:3]

    image = normalize_image(image)
    return image.permute(1, 2, 0).numpy()


def reduce_feature_map(
    feature_map: Tensor,
    reduction: str = "mean_abs",
) -> Tensor:
    """
    Reduce BCHW or CHW feature tensors into one 2D visualization.
    """
    feature_map = feature_map.detach().float().cpu()

    if feature_map.ndim == 4:
        feature_map = feature_map[0]

    if feature_map.ndim == 2:
        return feature_map

    if feature_map.ndim != 3:
        raise ValueError(
            f"Expected HW/CHW/BCHW tensor; got {tuple(feature_map.shape)}."
        )

    if reduction == "mean":
        reduced = feature_map.mean(dim=0)
    elif reduction == "mean_abs":
        reduced = feature_map.abs().mean(dim=0)
    elif reduction == "max_abs":
        reduced = feature_map.abs().amax(dim=0)
    elif reduction == "l2":
        reduced = torch.sqrt((feature_map ** 2).sum(dim=0))
    else:
        raise ValueError(f"Unsupported reduction '{reduction}'.")

    return reduced


def save_heatmap(
    tensor: Tensor,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    ensure_directory(output_path.parent)

    heatmap = reduce_feature_map(tensor)
    heatmap = normalize_image(heatmap)

    figure, axis = plt.subplots(figsize=(6, 5))
    image_handle = axis.imshow(heatmap.numpy(), cmap="viridis")
    axis.axis("off")

    if title:
        axis.set_title(title)

    figure.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_channel_grid(
    feature_map: Tensor,
    output_path: Path,
    max_channels: int = 16,
    title: Optional[str] = None,
) -> None:
    ensure_directory(output_path.parent)

    feature_map = feature_map.detach().float().cpu()

    if feature_map.ndim == 4:
        feature_map = feature_map[0]

    if feature_map.ndim != 3:
        raise ValueError(
            f"Expected CHW/BCHW feature tensor; got {tuple(feature_map.shape)}."
        )

    channel_count = min(feature_map.shape[0], max_channels)
    columns = min(4, channel_count)
    rows = math.ceil(channel_count / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
    )

    for channel_index in range(rows * columns):
        axis = axes[channel_index // columns][channel_index % columns]
        axis.axis("off")

        if channel_index >= channel_count:
            continue

        channel = normalize_image(feature_map[channel_index])
        axis.imshow(channel.numpy(), cmap="viridis")
        axis.set_title(f"Channel {channel_index}")

    if title:
        figure.suptitle(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_overlay(
    image: Tensor,
    heatmap: Tensor,
    output_path: Path,
    title: Optional[str] = None,
    alpha: float = 0.45,
) -> None:
    ensure_directory(output_path.parent)

    rgb = tensor_to_rgb(image)

    if heatmap.ndim == 4:
        heatmap = heatmap[0, 0]
    elif heatmap.ndim == 3:
        heatmap = heatmap[0]

    heatmap = heatmap.detach().float().cpu()

    if heatmap.shape != rgb.shape[:2]:
        heatmap = F.interpolate(
            heatmap[None, None],
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    heatmap = normalize_image(heatmap).numpy()

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(rgb)
    axis.imshow(heatmap, cmap="jet", alpha=alpha)
    axis.axis("off")

    if title:
        axis.set_title(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)