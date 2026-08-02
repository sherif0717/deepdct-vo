"""
Compute quantitative internal-attention statistics for DeepDCT-VO.

This script analyzes the attention maps retained by AttentionDownBlock
and AttentionUpBlock during inference.

Per-layer, per-frame measurements:
    - attention minimum
    - attention maximum
    - attention mean
    - attention standard deviation
    - mean Bernoulli entropy
    - fraction below a configurable threshold
    - fraction above a configurable threshold
    - spatial coefficient of variation
    - temporal cosine similarity to the previous analyzed frame
    - temporal mean absolute difference

Generated outputs
-----------------
<output-dir>/sequence_<sequence>/
├── frame_attention_statistics.csv
├── layer_attention_summary.csv
├── layer_attention_summary.json
├── attention_mean_by_layer.png
├── attention_std_by_layer.png
├── attention_entropy_by_layer.png
├── attention_low_fraction_by_layer.png
├── attention_high_fraction_by_layer.png
└── temporal_similarity_by_layer.png

Example
-------
python scripts/interpretability/analyze_attention_statistics.py \\
    --checkpoint checkpoints/best_validation.pt \\
    --sequence 10 \\
    --batch-size 1 \\
    --num-workers 0 \\
    --output-dir experiments/interpretability/attention_statistics

Notes
-----
The script imports model-construction and dataset helpers from the adjacent
inspect_deepdct_vo.py script. Keep both files in:

    scripts/interpretability/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Make imports reliable when this file is executed directly from the
# repository root:
#
#     python scripts/interpretability/analyze_attention_statistics.py
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIRECTORY),
    )


from inspect_deepdct_vo import (  # noqa: E402
    build_dataloader,
    build_dataset,
    build_model,
    load_checkpoint,
    metadata_value,
    resolve_device,
    resolve_evaluation_configuration,
)

from deepdct.interpretability import (  # noqa: E402
    collect_internal_attention_maps,
)


@dataclass
class FrameAttentionStatistics:
    """Statistics for one attention layer on one frame pair."""

    sequence: str
    dataset_sample_index: int
    frame_prev: int
    frame_curr: int
    module_name: str
    branch: str
    path: str
    stage: str
    attention_type: str

    channels: int
    height: int
    width: int
    num_values: int

    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    coefficient_of_variation: float

    entropy_bits: float

    low_fraction: float
    high_fraction: float

    temporal_cosine_similarity: Optional[float]
    temporal_mean_absolute_difference: Optional[float]


@dataclass
class LayerAttentionSummary:
    """Aggregate statistics for one attention module."""

    module_name: str
    branch: str
    path: str
    stage: str
    attention_type: str

    frames_analyzed: int
    temporal_comparisons: int

    mean_attention: float
    std_attention_across_frames: float

    mean_within_map_std: float
    mean_entropy_bits: float

    mean_low_fraction: float
    mean_high_fraction: float

    mean_temporal_cosine_similarity: Optional[float]
    std_temporal_cosine_similarity: Optional[float]

    mean_temporal_absolute_difference: Optional[float]
    std_temporal_absolute_difference: Optional[float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute sequence-level statistics for DeepDCT-VO "
            "internal attention maps."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/best_validation.pt"
        ),
        help="Checkpoint to analyze.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=(
            "Root containing sequences/, out_csv/, "
            "and poses/."
        ),
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default="10",
        help="KITTI sequence to analyze.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/interpretability/"
            "attention_statistics"
        ),
        help="Root output directory.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "DataLoader batch size. This analysis currently "
            "requires batch size 1."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Inference device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "First dataset sample index eligible for "
            "analysis."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of analyzed samples. "
            "Omit to analyze the remainder of the sequence."
        ),
    )

    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help=(
            "Analyze every Nth sample after --start-index."
        ),
    )

    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.25,
        help=(
            "Attention values below this threshold count "
            "toward low_fraction."
        ),
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.75,
        help=(
            "Attention values above this threshold count "
            "toward high_fraction."
        ),
    )

    parser.add_argument(
        "--entropy-epsilon",
        type=float,
        default=1e-6,
        help=(
            "Numerical clamp used for Bernoulli entropy."
        ),
    )

    parser.add_argument(
        "--temporal-reduction",
        choices=[
            "mean",
            "mean_abs",
            "max",
        ],
        default="mean",
        help=(
            "Channel-reduction method used before comparing "
            "consecutive attention maps."
        ),
    )

    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Condition Model T using ground-truth rotation."
        ),
    )

    parser.add_argument(
        "--include-layers",
        nargs="+",
        default=None,
        help=(
            "Optional exact module names to include. "
            "When omitted, all retained attention maps "
            "are analyzed."
        ),
    )

    parser.add_argument(
        "--exclude-rotation",
        action="store_true",
        help="Exclude rotation-branch attention maps.",
    )

    parser.add_argument(
        "--exclude-translation",
        action="store_true",
        help="Exclude translation-branch attention maps.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help=(
            "Print progress after this many analyzed samples."
        ),
    )

    parser.add_argument(
        "--rotation-loss-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--translation-loss-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate command-line options."""

    if args.batch_size != 1:
        raise ValueError(
            "Attention statistics currently require "
            "--batch-size 1 because each module retains only "
            "the most recent forward-pass attention tensor."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if args.start_index < 0:
        raise ValueError(
            "--start-index cannot be negative."
        )

    if (
        args.max_samples is not None
        and args.max_samples <= 0
    ):
        raise ValueError(
            "--max-samples must be positive."
        )

    if args.sample_stride <= 0:
        raise ValueError(
            "--sample-stride must be positive."
        )

    if not 0.0 <= args.low_threshold <= 1.0:
        raise ValueError(
            "--low-threshold must be in [0, 1]."
        )

    if not 0.0 <= args.high_threshold <= 1.0:
        raise ValueError(
            "--high-threshold must be in [0, 1]."
        )

    if (
        args.low_threshold
        >= args.high_threshold
    ):
        raise ValueError(
            "--low-threshold must be smaller than "
            "--high-threshold."
        )

    if args.entropy_epsilon <= 0.0:
        raise ValueError(
            "--entropy-epsilon must be positive."
        )

    if args.entropy_epsilon >= 0.5:
        raise ValueError(
            "--entropy-epsilon must be smaller than 0.5."
        )

    if args.log_interval <= 0:
        raise ValueError(
            "--log-interval must be positive."
        )

    if (
        args.exclude_rotation
        and args.exclude_translation
    ):
        raise ValueError(
            "Cannot exclude both rotation and translation "
            "attention branches."
        )


def seed_everything(
    seed: int,
) -> None:
    """Seed random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_tensor(
    batch: Mapping[str, object],
    key: str,
    device: torch.device,
) -> Tensor:
    """Select and move a required batch tensor."""

    value = batch.get(key)

    if not torch.is_tensor(value):
        raise TypeError(
            f"batch[{key!r}] must be a torch.Tensor."
        )

    return value.to(
        device=device,
        non_blocking=True,
    )


def classify_attention_module(
    module_name: str,
    channels: int,
) -> Tuple[str, str, str, str]:
    """
    Derive branch, path, stage, and attention type from a module name.

    Example:
        rotation_aresunet.encoder.enc2.attn

    Returns:
        branch = rotation
        path = encoder
        stage = enc2
        attention_type = channel_spatial
    """

    if module_name.startswith(
        "rotation_aresunet."
    ):
        branch = "rotation"
    elif module_name.startswith(
        "translation_aresunet."
    ):
        branch = "translation"
    else:
        branch = "other"

    if ".encoder." in module_name:
        path = "encoder"
    elif ".decoder." in module_name:
        path = "decoder"
    else:
        path = "other"

    stage = "unknown"

    for component in module_name.split("."):
        if (
            component.startswith("enc")
            or component.startswith("dec")
        ):
            stage = component
            break

    attention_type = (
        "spatial"
        if channels == 1
        else "channel_spatial"
    )

    return (
        branch,
        path,
        stage,
        attention_type,
    )


def reduce_attention_for_temporal_comparison(
    attention_map: Tensor,
    reduction: str,
) -> Tensor:
    """
    Convert BCHW attention into one normalized spatial map.

    Returned shape:
        [H, W]
    """

    tensor = (
        attention_map
        .detach()
        .float()
        .cpu()
    )

    if tensor.ndim != 4:
        raise ValueError(
            "Expected attention map with shape BCHW, "
            f"received {tuple(tensor.shape)}."
        )

    if tensor.shape[0] != 1:
        raise ValueError(
            "Temporal comparison expects batch size 1, "
            f"received {tensor.shape[0]}."
        )

    tensor = tensor[0]

    if reduction == "mean":
        reduced = tensor.mean(dim=0)
    elif reduction == "mean_abs":
        reduced = tensor.abs().mean(dim=0)
    elif reduction == "max":
        reduced = tensor.amax(dim=0)
    else:
        raise ValueError(
            f"Unsupported reduction: {reduction}"
        )

    return reduced


def bernoulli_entropy_bits(
    attention_map: Tensor,
    epsilon: float,
) -> float:
    """
    Compute average Bernoulli entropy of gate coefficients.

    For one coefficient p:

        H(p) = -p log2(p) - (1-p) log2(1-p)

    Interpretation:
        near 0 bits:
            highly saturated coefficient near 0 or 1

        near 1 bit:
            uncertain/pass-through coefficient near 0.5
    """

    probability = (
        attention_map
        .detach()
        .float()
        .clamp(
            min=epsilon,
            max=1.0 - epsilon,
        )
    )

    entropy = -(
        probability
        * torch.log2(probability)
        + (1.0 - probability)
        * torch.log2(1.0 - probability)
    )

    return float(
        entropy.mean().item()
    )


def cosine_similarity(
    current: Tensor,
    previous: Tensor,
) -> float:
    """Compute cosine similarity between flattened maps."""

    if current.shape != previous.shape:
        previous = F.interpolate(
            previous[
                None,
                None,
            ],
            size=current.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    current_vector = current.reshape(-1)
    previous_vector = previous.reshape(-1)

    similarity = F.cosine_similarity(
        current_vector.unsqueeze(0),
        previous_vector.unsqueeze(0),
        dim=1,
        eps=1e-8,
    )

    return float(
        similarity.item()
    )


def mean_absolute_difference(
    current: Tensor,
    previous: Tensor,
) -> float:
    """Compute mean absolute difference between two maps."""

    if current.shape != previous.shape:
        previous = F.interpolate(
            previous[
                None,
                None,
            ],
            size=current.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    return float(
        torch.mean(
            torch.abs(
                current - previous
            )
        ).item()
    )


def module_is_included(
    module_name: str,
    args: argparse.Namespace,
) -> bool:
    """Apply branch and exact-layer filters."""

    if (
        args.include_layers is not None
        and module_name
        not in args.include_layers
    ):
        return False

    if (
        args.exclude_rotation
        and module_name.startswith(
            "rotation_aresunet."
        )
    ):
        return False

    if (
        args.exclude_translation
        and module_name.startswith(
            "translation_aresunet."
        )
    ):
        return False

    return True


def analyze_attention_map(
    attention_map: Tensor,
    module_name: str,
    sequence: str,
    dataset_sample_index: int,
    frame_prev: int,
    frame_curr: int,
    previous_temporal_map: Optional[Tensor],
    args: argparse.Namespace,
) -> Tuple[
    FrameAttentionStatistics,
    Tensor,
]:
    """Compute statistics for one retained attention map."""

    tensor = (
        attention_map
        .detach()
        .float()
        .cpu()
    )

    if tensor.ndim != 4:
        raise ValueError(
            f"{module_name} returned an attention map with "
            f"shape {tuple(tensor.shape)}; expected BCHW."
        )

    batch_size, channels, height, width = (
        tensor.shape
    )

    if batch_size != 1:
        raise ValueError(
            f"{module_name} has batch size {batch_size}; "
            "the statistics script requires batch size 1."
        )

    minimum = float(
        tensor.min().item()
    )
    maximum = float(
        tensor.max().item()
    )
    mean = float(
        tensor.mean().item()
    )
    standard_deviation = float(
        tensor.std(
            unbiased=False
        ).item()
    )

    coefficient_of_variation = (
        standard_deviation
        / max(abs(mean), 1e-8)
    )

    entropy_bits = bernoulli_entropy_bits(
        tensor,
        epsilon=args.entropy_epsilon,
    )

    low_fraction = float(
        (
            tensor
            < args.low_threshold
        )
        .float()
        .mean()
        .item()
    )

    high_fraction = float(
        (
            tensor
            > args.high_threshold
        )
        .float()
        .mean()
        .item()
    )

    temporal_map = (
        reduce_attention_for_temporal_comparison(
            tensor,
            reduction=args.temporal_reduction,
        )
    )

    temporal_cosine: Optional[float]
    temporal_difference: Optional[float]

    if previous_temporal_map is None:
        temporal_cosine = None
        temporal_difference = None
    else:
        temporal_cosine = cosine_similarity(
            temporal_map,
            previous_temporal_map,
        )

        temporal_difference = (
            mean_absolute_difference(
                temporal_map,
                previous_temporal_map,
            )
        )

    (
        branch,
        path,
        stage,
        attention_type,
    ) = classify_attention_module(
        module_name=module_name,
        channels=channels,
    )

    statistics = FrameAttentionStatistics(
        sequence=sequence,
        dataset_sample_index=(
            dataset_sample_index
        ),
        frame_prev=frame_prev,
        frame_curr=frame_curr,
        module_name=module_name,
        branch=branch,
        path=path,
        stage=stage,
        attention_type=attention_type,
        channels=channels,
        height=height,
        width=width,
        num_values=(
            channels
            * height
            * width
        ),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=(
            standard_deviation
        ),
        coefficient_of_variation=(
            coefficient_of_variation
        ),
        entropy_bits=entropy_bits,
        low_fraction=low_fraction,
        high_fraction=high_fraction,
        temporal_cosine_similarity=(
            temporal_cosine
        ),
        temporal_mean_absolute_difference=(
            temporal_difference
        ),
    )

    return (
        statistics,
        temporal_map,
    )


def write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
) -> None:
    """Write dataclass instances to CSV."""

    if not rows:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionaries = [
        asdict(row)
        for row in rows
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                dictionaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            dictionaries
        )


def optional_mean(
    values: Sequence[Optional[float]],
) -> Optional[float]:
    """Return the mean of non-None values."""

    numeric_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not numeric_values:
        return None

    return float(
        np.mean(numeric_values)
    )


def optional_std(
    values: Sequence[Optional[float]],
) -> Optional[float]:
    """Return population standard deviation of non-None values."""

    numeric_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not numeric_values:
        return None

    return float(
        np.std(
            numeric_values,
            ddof=0,
        )
    )


def summarize_layers(
    frame_rows: Sequence[
        FrameAttentionStatistics
    ],
) -> List[LayerAttentionSummary]:
    """Aggregate frame-level rows by attention layer."""

    grouped: DefaultDict[
        str,
        List[FrameAttentionStatistics],
    ] = defaultdict(list)

    for row in frame_rows:
        grouped[row.module_name].append(
            row
        )

    summaries: List[
        LayerAttentionSummary
    ] = []

    for module_name in sorted(grouped):
        rows = grouped[module_name]
        first = rows[0]

        temporal_cosines = [
            row.temporal_cosine_similarity
            for row in rows
        ]

        temporal_differences = [
            row.temporal_mean_absolute_difference
            for row in rows
        ]

        temporal_comparisons = sum(
            value is not None
            for value in temporal_cosines
        )

        summaries.append(
            LayerAttentionSummary(
                module_name=module_name,
                branch=first.branch,
                path=first.path,
                stage=first.stage,
                attention_type=(
                    first.attention_type
                ),
                frames_analyzed=len(rows),
                temporal_comparisons=(
                    temporal_comparisons
                ),
                mean_attention=float(
                    np.mean(
                        [
                            row.mean
                            for row in rows
                        ]
                    )
                ),
                std_attention_across_frames=float(
                    np.std(
                        [
                            row.mean
                            for row in rows
                        ],
                        ddof=0,
                    )
                ),
                mean_within_map_std=float(
                    np.mean(
                        [
                            row.standard_deviation
                            for row in rows
                        ]
                    )
                ),
                mean_entropy_bits=float(
                    np.mean(
                        [
                            row.entropy_bits
                            for row in rows
                        ]
                    )
                ),
                mean_low_fraction=float(
                    np.mean(
                        [
                            row.low_fraction
                            for row in rows
                        ]
                    )
                ),
                mean_high_fraction=float(
                    np.mean(
                        [
                            row.high_fraction
                            for row in rows
                        ]
                    )
                ),
                mean_temporal_cosine_similarity=(
                    optional_mean(
                        temporal_cosines
                    )
                ),
                std_temporal_cosine_similarity=(
                    optional_std(
                        temporal_cosines
                    )
                ),
                mean_temporal_absolute_difference=(
                    optional_mean(
                        temporal_differences
                    )
                ),
                std_temporal_absolute_difference=(
                    optional_std(
                        temporal_differences
                    )
                ),
            )
        )

    return summaries


def shortened_layer_name(
    module_name: str,
) -> str:
    """Create compact plot labels."""

    return (
        module_name
        .replace(
            "rotation_aresunet.",
            "R.",
        )
        .replace(
            "translation_aresunet.",
            "T.",
        )
        .replace(
            ".encoder.",
            ".enc.",
        )
        .replace(
            ".decoder.",
            ".dec.",
        )
        .replace(
            ".attn",
            "",
        )
    )


def plot_layer_metric(
    summaries: Sequence[
        LayerAttentionSummary
    ],
    value_name: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Plot one aggregate metric by attention layer."""

    labels = [
        shortened_layer_name(
            summary.module_name
        )
        for summary in summaries
    ]

    values = [
        getattr(
            summary,
            value_name,
        )
        for summary in summaries
    ]

    numeric_values = [
        (
            float(value)
            if value is not None
            else float("nan")
        )
        for value in values
    ]

    figure = plt.figure(
        figsize=(14, 7)
    )
    axes = figure.add_subplot(111)

    positions = np.arange(
        len(labels)
    )

    axes.bar(
        positions,
        numeric_values,
    )

    axes.set_xticks(
        positions
    )
    axes.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
    )

    axes.set_ylabel(
        y_label
    )
    axes.set_title(
        title
    )
    axes.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def run_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sequence: str,
    use_internal_depth: bool,
    args: argparse.Namespace,
) -> List[FrameAttentionStatistics]:
    """Run attention analysis over selected sequence samples."""

    frame_rows: List[
        FrameAttentionStatistics
    ] = []

    previous_maps: Dict[
        str,
        Tensor,
    ] = {}

    analyzed_samples = 0
    encountered_samples = 0

    start_time = time.perf_counter()

    model.eval()

    with torch.inference_mode():
        for dataset_index, batch in enumerate(
            dataloader
        ):
            encountered_samples += 1

            if dataset_index < args.start_index:
                continue

            relative_index = (
                dataset_index
                - args.start_index
            )

            if (
                relative_index
                % args.sample_stride
                != 0
            ):
                continue

            if (
                args.max_samples is not None
                and analyzed_samples
                >= args.max_samples
            ):
                break

            image_prev = select_tensor(
                batch,
                "image_prev",
                device,
            )

            image_curr = select_tensor(
                batch,
                "image_curr",
                device,
            )

            rotation_gt = select_tensor(
                batch,
                "rotation_gt",
                device,
            )

            if use_internal_depth:
                depth_curr: Optional[
                    Tensor
                ] = None
            else:
                depth_curr = select_tensor(
                    batch,
                    "depth_curr",
                    device,
                )

            model_inputs = {
                "image_prev": image_prev,
                "image_curr": image_curr,
                "depth_curr": depth_curr,
                "rotation_for_translation": (
                    rotation_gt
                    if args.use_ground_truth_rotation
                    else None
                ),
                "use_ground_truth_rotation": (
                    args.use_ground_truth_rotation
                ),
            }

            model(**model_inputs)

            attention_maps = (
                collect_internal_attention_maps(
                    model
                )
            )

            if not attention_maps:
                raise RuntimeError(
                    "No attention maps were collected. "
                    "Verify that AttentionDownBlock and "
                    "AttentionUpBlock assign "
                    "`self.last_attention_map`."
                )

            frame_prev = int(
                metadata_value(
                    batch,
                    "frame_prev",
                    0,
                )
            )

            frame_curr = int(
                metadata_value(
                    batch,
                    "frame_curr",
                    0,
                )
            )

            selected_layer_count = 0

            for (
                module_name,
                attention_map,
            ) in attention_maps.items():
                if not module_is_included(
                    module_name,
                    args,
                ):
                    continue

                selected_layer_count += 1

                previous_map = (
                    previous_maps.get(
                        module_name
                    )
                )

                (
                    statistics,
                    temporal_map,
                ) = analyze_attention_map(
                    attention_map=attention_map,
                    module_name=module_name,
                    sequence=sequence,
                    dataset_sample_index=(
                        dataset_index
                    ),
                    frame_prev=frame_prev,
                    frame_curr=frame_curr,
                    previous_temporal_map=(
                        previous_map
                    ),
                    args=args,
                )

                frame_rows.append(
                    statistics
                )

                previous_maps[
                    module_name
                ] = temporal_map

            if selected_layer_count == 0:
                raise RuntimeError(
                    "Attention maps were collected, but none "
                    "matched the requested layer or branch "
                    "filters."
                )

            analyzed_samples += 1

            if (
                analyzed_samples
                % args.log_interval
                == 0
            ):
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"analyzed_samples="
                    f"{analyzed_samples} "
                    f"dataset_index="
                    f"{dataset_index} "
                    f"frame={frame_prev}"
                    f"->{frame_curr} "
                    f"layers="
                    f"{selected_layer_count} "
                    f"elapsed="
                    f"{elapsed:.2f}s "
                    f"throughput="
                    f"{analyzed_samples / elapsed:.2f}"
                    f" samples/s"
                )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if analyzed_samples == 0:
        raise RuntimeError(
            "No samples were analyzed. Check "
            "--start-index, --max-samples, and "
            "--sample-stride."
        )

    print()
    print("=" * 72)
    print("Attention analysis complete")
    print("=" * 72)
    print(
        f"Sequence:              "
        f"{sequence}"
    )
    print(
        f"Samples analyzed:      "
        f"{analyzed_samples}"
    )
    print(
        f"Frame-layer records:   "
        f"{len(frame_rows)}"
    )
    print(
        f"Elapsed time:          "
        f"{elapsed:.2f} s"
    )
    print(
        f"Throughput:            "
        f"{analyzed_samples / elapsed:.2f} "
        f"samples/s"
    )
    print("=" * 72)

    return frame_rows


def main() -> None:
    """Run sequence-level attention analysis."""

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = resolve_device(
        args.device
    )

    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    evaluation_configuration = (
        resolve_evaluation_configuration(
            args=args,
            checkpoint=checkpoint,
        )
    )

    dataset = build_dataset(
        args=args,
        evaluation_configuration=(
            evaluation_configuration
        ),
    )

    dataloader = build_dataloader(
        dataset=dataset,
        args=args,
        device=device,
    )

    model = build_model(
        checkpoint=checkpoint,
        evaluation_configuration=(
            evaluation_configuration
        ),
        device=device,
    )

    output_directory = (
        args.output_dir
        / f"sequence_{args.sequence}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("DeepDCT-VO attention statistics")
    print("=" * 72)
    print(
        f"Checkpoint:            "
        f"{args.checkpoint.resolve()}"
    )
    print(
        f"Checkpoint epoch:      "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Sequence:              "
        f"{args.sequence}"
    )
    print(
        f"Dataset samples:       "
        f"{len(dataset)}"
    )
    print(
        f"Device:                "
        f"{device}"
    )
    print(
        f"Start index:           "
        f"{args.start_index}"
    )
    print(
        f"Maximum samples:       "
        f"{args.max_samples}"
    )
    print(
        f"Sample stride:         "
        f"{args.sample_stride}"
    )
    print(
        f"Low threshold:         "
        f"{args.low_threshold}"
    )
    print(
        f"High threshold:        "
        f"{args.high_threshold}"
    )
    print(
        f"Semantic cues:         "
        f"{evaluation_configuration['use_semantic_cues']}"
    )
    print(
        f"Depth cues:            "
        f"{evaluation_configuration['use_depth_cues']}"
    )
    print(
        f"Output directory:      "
        f"{output_directory.resolve()}"
    )
    print("=" * 72)

    frame_rows = run_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        sequence=args.sequence,
        use_internal_depth=bool(
            evaluation_configuration[
                "use_depth_cues"
            ]
        ),
        args=args,
    )

    summaries = summarize_layers(
        frame_rows
    )

    write_dataclass_csv(
        output_directory
        / "frame_attention_statistics.csv",
        frame_rows,
    )

    write_dataclass_csv(
        output_directory
        / "layer_attention_summary.csv",
        summaries,
    )

    json_payload = {
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "sequence": args.sequence,
        "configuration": {
            "start_index": (
                args.start_index
            ),
            "max_samples": (
                args.max_samples
            ),
            "sample_stride": (
                args.sample_stride
            ),
            "low_threshold": (
                args.low_threshold
            ),
            "high_threshold": (
                args.high_threshold
            ),
            "temporal_reduction": (
                args.temporal_reduction
            ),
            "use_ground_truth_rotation": (
                args.use_ground_truth_rotation
            ),
            "use_semantic_cues": bool(
                evaluation_configuration[
                    "use_semantic_cues"
                ]
            ),
            "use_depth_cues": bool(
                evaluation_configuration[
                    "use_depth_cues"
                ]
            ),
        },
        "layers": [
            asdict(summary)
            for summary in summaries
        ],
    }

    with (
        output_directory
        / "layer_attention_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_payload,
            file,
            indent=2,
            sort_keys=True,
        )

    plot_layer_metric(
        summaries=summaries,
        value_name="mean_attention",
        title=(
            f"DeepDCT-VO mean attention — "
            f"sequence {args.sequence}"
        ),
        y_label="Mean attention coefficient",
        output_path=(
            output_directory
            / "attention_mean_by_layer.png"
        ),
    )

    plot_layer_metric(
        summaries=summaries,
        value_name="mean_within_map_std",
        title=(
            f"DeepDCT-VO attention variation — "
            f"sequence {args.sequence}"
        ),
        y_label=(
            "Mean within-map standard deviation"
        ),
        output_path=(
            output_directory
            / "attention_std_by_layer.png"
        ),
    )

    plot_layer_metric(
        summaries=summaries,
        value_name="mean_entropy_bits",
        title=(
            f"DeepDCT-VO attention entropy — "
            f"sequence {args.sequence}"
        ),
        y_label="Mean Bernoulli entropy (bits)",
        output_path=(
            output_directory
            / "attention_entropy_by_layer.png"
        ),
    )

    plot_layer_metric(
        summaries=summaries,
        value_name="mean_low_fraction",
        title=(
            f"Fraction of attention below "
            f"{args.low_threshold:.2f} — "
            f"sequence {args.sequence}"
        ),
        y_label="Low-attention fraction",
        output_path=(
            output_directory
            / "attention_low_fraction_by_layer.png"
        ),
    )

    plot_layer_metric(
        summaries=summaries,
        value_name="mean_high_fraction",
        title=(
            f"Fraction of attention above "
            f"{args.high_threshold:.2f} — "
            f"sequence {args.sequence}"
        ),
        y_label="High-attention fraction",
        output_path=(
            output_directory
            / "attention_high_fraction_by_layer.png"
        ),
    )

    plot_layer_metric(
        summaries=summaries,
        value_name=(
            "mean_temporal_cosine_similarity"
        ),
        title=(
            f"Attention temporal similarity — "
            f"sequence {args.sequence}"
        ),
        y_label=(
            "Mean consecutive-map cosine similarity"
        ),
        output_path=(
            output_directory
            / "temporal_similarity_by_layer.png"
        ),
    )

    print()
    print("Layer summary")
    print("-" * 100)

    for summary in summaries:
        temporal_similarity = (
            "n/a"
            if (
                summary
                .mean_temporal_cosine_similarity
                is None
            )
            else (
                f"{summary.mean_temporal_cosine_similarity:.6f}"
            )
        )

        print(
            f"{summary.module_name:<55} "
            f"mean={summary.mean_attention:.6f} "
            f"std={summary.mean_within_map_std:.6f} "
            f"entropy={summary.mean_entropy_bits:.6f} "
            f"low={summary.mean_low_fraction:.6f} "
            f"high={summary.mean_high_fraction:.6f} "
            f"temporal_cos={temporal_similarity}"
        )

    print("-" * 100)
    print(
        f"Outputs saved to: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()