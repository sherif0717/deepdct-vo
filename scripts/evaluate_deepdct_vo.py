"""Evaluate the selected DeepDCT-VO checkpoint on held-out KITTI sequence 10.

Default evaluation:

    checkpoint: checkpoints/best_validation.pt
    test sequence: 10

The script performs five stages:

1. Freeze model selection by loading best_validation.pt.
2. Evaluate the checkpoint on sequence 10 without parameter updates.
3. Compare test metrics with validation metrics stored in the checkpoint.
4. Save frame-wise predictions, targets, errors, and worst-frame reports.
5. Reconstruct trajectories and calculate trajectory-level metrics.

Generated outputs
-----------------

evaluation/sequence_10/
├── summary.json
├── frame_predictions.csv
├── worst_rotation_frames.csv
├── worst_translation_frames.csv
├── axis_metrics.csv
├── predicted_trajectory.txt
├── ground_truth_trajectory.txt
├── trajectory_xy.png
├── trajectory_xz.png
├── rotation_error_histogram.png
├── translation_error_histogram.png
└── checkpoint_comparison.txt

Important trajectory assumption
-------------------------------

Trajectory integration assumes that:

- rotation_gt and rotation predictions are relative Euler rotations;
- translation_gt and directional_translation predictions are relative
  translations expressed in the previous camera/local frame;
- the six label components can therefore be composed as relative SE(3)
  transformations.

If directional_translation represents a transformed DCT coordinate that
requires an inverse DCT operation before SE(3) composition, replace
``relative_pose_to_transform`` with the corresponding inverse mapping from
the project's DCT geometry implementation. Frame-wise evaluation remains
valid independently of this trajectory assumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from deepdct.data.training_dataset import DeepDCTTrainingDataset
from deepdct.models.deepdct_vo import DeepDCTVO


@dataclass
class AggregateMetrics:
    """Aggregate frame-level pose-regression metrics."""

    total_mse: float
    rotation_mse: float
    translation_mse: float

    rotation_rmse: float
    translation_rmse: float

    rotation_mae: float
    translation_mae: float

    rotation_axis_mae_x: float
    rotation_axis_mae_y: float
    rotation_axis_mae_z: float

    rotation_axis_rmse_x: float
    rotation_axis_rmse_y: float
    rotation_axis_rmse_z: float

    translation_axis_mae_x: float
    translation_axis_mae_y: float
    translation_axis_mae_z: float

    translation_axis_rmse_x: float
    translation_axis_rmse_y: float
    translation_axis_rmse_z: float

    num_samples: int
    num_batches: int
    elapsed_seconds: float
    samples_per_second: float


@dataclass
class TrajectoryMetrics:
    """Trajectory-level metrics from integrated relative poses."""

    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_max: float

    rpe_translation_rmse: float
    rpe_translation_mean: float

    rpe_rotation_rmse_degrees: float
    rpe_rotation_mean_degrees: float

    path_length_ground_truth: float
    path_length_predicted: float

    endpoint_error: float
    endpoint_error_percent: float

    translational_drift_percent: float
    rotational_drift_degrees_per_100m: float


@dataclass
class FramePrediction:
    """Frame-level prediction, target, and error values."""

    sequence: str
    frame_prev: int
    frame_curr: int
    image_prev_path: str
    image_curr_path: str

    rotation_gt_x: float
    rotation_gt_y: float
    rotation_gt_z: float

    rotation_pred_x: float
    rotation_pred_y: float
    rotation_pred_z: float

    rotation_error_x: float
    rotation_error_y: float
    rotation_error_z: float

    rotation_l2_error: float
    rotation_squared_error: float

    translation_gt_x: float
    translation_gt_y: float
    translation_gt_z: float

    translation_pred_x: float
    translation_pred_y: float
    translation_pred_z: float

    translation_error_x: float
    translation_error_y: float
    translation_error_z: float

    translation_l2_error: float
    translation_squared_error: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate best_validation.pt on held-out KITTI sequence 10."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_validation.pt"),
        help="Selected checkpoint. Do not select it using sequence 10.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing sequences/, out_csv/, and poses/.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default="10",
        help="Held-out KITTI sequence.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation output directory. Defaults to "
            "evaluation/sequence_<sequence>."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Evaluation device.",
    )

    parser.add_argument(
        "--rotation-loss-weight",
        type=float,
        default=None,
        help=(
            "Rotation-loss weight. When omitted, use checkpoint "
            "configuration or 1.0."
        ),
    )

    parser.add_argument(
        "--translation-loss-weight",
        type=float,
        default=None,
        help=(
            "Translation-loss weight. When omitted, use checkpoint "
            "configuration or 1.0."
        ),
    )

    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Condition Model T on ground-truth rotation. By default, "
            "Model T uses predicted rotation, matching normal inference."
        ),
    )

    parser.add_argument(
        "--worst-frame-count",
        type=int,
        default=50,
        help="Number of worst frames saved for each prediction head.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print evaluation progress every N batches.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip trajectory reconstruction and trajectory metrics.",
    )

    parser.add_argument(
        "--euler-order",
        choices=["xyz", "zyx"],
        default="xyz",
        help=(
            "Euler composition order used for relative-pose integration. "
            "Use the order matching the DCT label-generation pipeline."
        ),
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help="Interpret rotation labels and predictions as degrees.",
    )


    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            Path("evaluation")
            / f"sequence_{args.sequence}"
        )

    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate evaluation options."""

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.worst_frame_count <= 0:
        raise ValueError("--worst-frame-count must be positive.")

    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")

    for name in (
        "rotation_loss_weight",
        "translation_loss_weight",
    ):
        value = getattr(args, name)

        if value is not None and value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} cannot be negative."
            )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested evaluation device."""

    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable."
        )

    return torch.device(device_name)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    """Load and validate the selected checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
    }

    missing = required_keys.difference(checkpoint.keys())

    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing)}."
        )

    return checkpoint


def get_checkpoint_configuration(
    checkpoint: Mapping[str, object],
) -> Mapping[str, object]:
    """Return checkpoint configuration, or an empty mapping."""

    configuration = checkpoint.get("configuration", {})

    if not isinstance(configuration, Mapping):
        raise TypeError(
            "Checkpoint configuration must be a mapping."
        )

    return configuration


def resolve_evaluation_configuration(
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
) -> Dict[str, object]:
    """Resolve model and evaluation settings from checkpoint metadata."""

    configuration = get_checkpoint_configuration(checkpoint)

    height = int(configuration.get("height", 120))
    width = int(configuration.get("width", 120))
    camera = str(configuration.get("camera", "left"))

    pretrained_semantic = bool(
        configuration.get("pretrained_semantic", True)
    )
    freeze_semantic = bool(
        configuration.get("freeze_semantic", True)
    )

    share_aresunet = bool(
        configuration.get(
            "share_aresunet_between_models",
            False,
        )
    )

    if args.rotation_loss_weight is None:
        rotation_loss_weight = float(
            configuration.get("rotation_loss_weight", 1.0)
        )
    else:
        rotation_loss_weight = args.rotation_loss_weight

    if args.translation_loss_weight is None:
        translation_loss_weight = float(
            configuration.get("translation_loss_weight", 1.0)
        )
    else:
        translation_loss_weight = (
            args.translation_loss_weight
        )

    return {
        "height": height,
        "width": width,
        "camera": camera,
        "pretrained_semantic": pretrained_semantic,
        "freeze_semantic": freeze_semantic,
        "share_aresunet_between_models": share_aresunet,
        "rotation_loss_weight": rotation_loss_weight,
        "translation_loss_weight": translation_loss_weight,
    }


def build_dataset(
    args: argparse.Namespace,
    evaluation_configuration: Mapping[str, object],
) -> DeepDCTTrainingDataset:
    """Build the held-out sequence-10 dataset."""

    return DeepDCTTrainingDataset(
        data_root=args.data_root,
        sequences=(args.sequence,),
        camera=str(evaluation_configuration["camera"]),
        image_size=(
            int(evaluation_configuration["height"]),
            int(evaluation_configuration["width"]),
        ),
        # Match the currently selected baseline experiment:
        # depth_curr is supplied as an all-zero placeholder.
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=True,
    )


def build_dataloader(
    dataset: DeepDCTTrainingDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    """Build deterministic held-out evaluation DataLoader."""

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def build_model(
    checkpoint: Mapping[str, object],
    evaluation_configuration: Mapping[str, object],
    device: torch.device,
) -> DeepDCTVO:
    """Rebuild and restore the selected model."""

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(
            int(evaluation_configuration["height"]),
            int(evaluation_configuration["width"]),
        ),
        pretrained_semantic=bool(
            evaluation_configuration["pretrained_semantic"]
        ),
        freeze_semantic=bool(
            evaluation_configuration["freeze_semantic"]
        ),
        normalize_semantic_input=True,
        normalize_semantic_map=True,
        share_aresunet_between_models=bool(
            evaluation_configuration[
                "share_aresunet_between_models"
            ]
        ),
        # Evaluation reproduces the zero-depth baseline.
        depth_checkpoint_dir=None,
        freeze_depth=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def move_tensor(
    batch: Mapping[str, object],
    key: str,
    device: torch.device,
) -> Tensor:
    """Move a required batch tensor to the evaluation device."""

    value = batch.get(key)

    if not torch.is_tensor(value):
        raise TypeError(
            f"batch[{key!r}] must be a torch.Tensor."
        )

    return value.to(
        device=device,
        non_blocking=True,
    )


def metadata_value(
    batch: Mapping[str, object],
    key: str,
    index: int,
) -> object:
    """Extract one metadata element from a collated DataLoader batch."""

    value = batch[key]

    if torch.is_tensor(value):
        return value[index].item()

    if isinstance(value, (list, tuple)):
        return value[index]

    if isinstance(value, str):
        return value

    raise TypeError(
        f"Unsupported metadata type for {key}: "
        f"{type(value).__name__}."
    )


def evaluate_model(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    device: torch.device,
    rotation_loss_weight: float,
    translation_loss_weight: float,
    use_ground_truth_rotation: bool,
    log_interval: int,
) -> Tuple[
    AggregateMetrics,
    List[FramePrediction],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate all frames and retain predictions for error analysis."""

    rotation_criterion = nn.MSELoss(reduction="sum")
    translation_criterion = nn.MSELoss(reduction="sum")

    frame_predictions: List[FramePrediction] = []

    all_rotation_gt: List[np.ndarray] = []
    all_rotation_pred: List[np.ndarray] = []
    all_translation_gt: List[np.ndarray] = []
    all_translation_pred: List[np.ndarray] = []

    total_rotation_squared_error = 0.0
    total_translation_squared_error = 0.0

    num_samples = 0
    num_batches = 0

    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            image_prev = move_tensor(
                batch,
                "image_prev",
                device,
            )
            image_curr = move_tensor(
                batch,
                "image_curr",
                device,
            )
            depth_curr = move_tensor(
                batch,
                "depth_curr",
                device,
            )
            rotation_gt = move_tensor(
                batch,
                "rotation_gt",
                device,
            )
            translation_gt = move_tensor(
                batch,
                "translation_gt",
                device,
            )

            outputs = model(
                image_prev=image_prev,
                image_curr=image_curr,
                depth_curr=depth_curr,
                rotation_for_translation=(
                    rotation_gt
                    if use_ground_truth_rotation
                    else None
                ),
                use_ground_truth_rotation=(
                    use_ground_truth_rotation
                ),
            )

            rotation_pred = outputs["rotation"]
            translation_pred = outputs[
                "directional_translation"
            ]

            expected_shape = (
                image_prev.shape[0],
                3,
            )

            if rotation_pred.shape != expected_shape:
                raise ValueError(
                    "rotation prediction has unexpected shape: "
                    f"{tuple(rotation_pred.shape)}."
                )

            if translation_pred.shape != expected_shape:
                raise ValueError(
                    "translation prediction has unexpected shape: "
                    f"{tuple(translation_pred.shape)}."
                )

            if not torch.isfinite(rotation_pred).all():
                raise FloatingPointError(
                    "Non-finite rotation prediction encountered."
                )

            if not torch.isfinite(translation_pred).all():
                raise FloatingPointError(
                    "Non-finite translation prediction encountered."
                )

            rotation_sum_squared_error = rotation_criterion(
                rotation_pred,
                rotation_gt,
            )
            translation_sum_squared_error = (
                translation_criterion(
                    translation_pred,
                    translation_gt,
                )
            )

            total_rotation_squared_error += float(
                rotation_sum_squared_error.item()
            )
            total_translation_squared_error += float(
                translation_sum_squared_error.item()
            )

            rotation_gt_np = (
                rotation_gt.detach().cpu().numpy()
            )
            rotation_pred_np = (
                rotation_pred.detach().cpu().numpy()
            )
            translation_gt_np = (
                translation_gt.detach().cpu().numpy()
            )
            translation_pred_np = (
                translation_pred.detach().cpu().numpy()
            )

            all_rotation_gt.append(rotation_gt_np)
            all_rotation_pred.append(rotation_pred_np)
            all_translation_gt.append(translation_gt_np)
            all_translation_pred.append(translation_pred_np)

            current_batch_size = image_prev.shape[0]

            for sample_index in range(current_batch_size):
                r_gt = rotation_gt_np[sample_index]
                r_pred = rotation_pred_np[sample_index]
                t_gt = translation_gt_np[sample_index]
                t_pred = translation_pred_np[sample_index]

                r_error = r_pred - r_gt
                t_error = t_pred - t_gt

                frame_predictions.append(
                    FramePrediction(
                        sequence=str(
                            metadata_value(
                                batch,
                                "sequence",
                                sample_index,
                            )
                        ),
                        frame_prev=int(
                            metadata_value(
                                batch,
                                "frame_prev",
                                sample_index,
                            )
                        ),
                        frame_curr=int(
                            metadata_value(
                                batch,
                                "frame_curr",
                                sample_index,
                            )
                        ),
                        image_prev_path=str(
                            metadata_value(
                                batch,
                                "image_prev_path",
                                sample_index,
                            )
                        ),
                        image_curr_path=str(
                            metadata_value(
                                batch,
                                "image_curr_path",
                                sample_index,
                            )
                        ),
                        rotation_gt_x=float(r_gt[0]),
                        rotation_gt_y=float(r_gt[1]),
                        rotation_gt_z=float(r_gt[2]),
                        rotation_pred_x=float(r_pred[0]),
                        rotation_pred_y=float(r_pred[1]),
                        rotation_pred_z=float(r_pred[2]),
                        rotation_error_x=float(r_error[0]),
                        rotation_error_y=float(r_error[1]),
                        rotation_error_z=float(r_error[2]),
                        rotation_l2_error=float(
                            np.linalg.norm(r_error)
                        ),
                        rotation_squared_error=float(
                            np.sum(r_error ** 2)
                        ),
                        translation_gt_x=float(t_gt[0]),
                        translation_gt_y=float(t_gt[1]),
                        translation_gt_z=float(t_gt[2]),
                        translation_pred_x=float(t_pred[0]),
                        translation_pred_y=float(t_pred[1]),
                        translation_pred_z=float(t_pred[2]),
                        translation_error_x=float(t_error[0]),
                        translation_error_y=float(t_error[1]),
                        translation_error_z=float(t_error[2]),
                        translation_l2_error=float(
                            np.linalg.norm(t_error)
                        ),
                        translation_squared_error=float(
                            np.sum(t_error ** 2)
                        ),
                    )
                )

            num_samples += current_batch_size
            num_batches += 1

            if num_batches % log_interval == 0:
                running_rotation_mse = (
                    total_rotation_squared_error
                    / (num_samples * 3)
                )
                running_translation_mse = (
                    total_translation_squared_error
                    / (num_samples * 3)
                )

                running_total = (
                    rotation_loss_weight
                    * running_rotation_mse
                    + translation_loss_weight
                    * running_translation_mse
                )

                print(
                    f"evaluation "
                    f"batch={num_batches} "
                    f"samples={num_samples} "
                    f"loss={running_total:.6f} "
                    f"rotation_loss="
                    f"{running_rotation_mse:.6f} "
                    f"translation_loss="
                    f"{running_translation_mse:.6f}"
                )

    elapsed_seconds = time.perf_counter() - start_time

    if num_samples == 0:
        raise RuntimeError(
            "The evaluation DataLoader produced no samples."
        )

    rotation_gt_array = np.concatenate(
        all_rotation_gt,
        axis=0,
    )
    rotation_pred_array = np.concatenate(
        all_rotation_pred,
        axis=0,
    )
    translation_gt_array = np.concatenate(
        all_translation_gt,
        axis=0,
    )
    translation_pred_array = np.concatenate(
        all_translation_pred,
        axis=0,
    )

    rotation_error = (
        rotation_pred_array - rotation_gt_array
    )
    translation_error = (
        translation_pred_array - translation_gt_array
    )

    rotation_mse = float(
        np.mean(rotation_error ** 2)
    )
    translation_mse = float(
        np.mean(translation_error ** 2)
    )

    total_mse = (
        rotation_loss_weight * rotation_mse
        + translation_loss_weight * translation_mse
    )

    rotation_axis_mae = np.mean(
        np.abs(rotation_error),
        axis=0,
    )
    translation_axis_mae = np.mean(
        np.abs(translation_error),
        axis=0,
    )

    rotation_axis_rmse = np.sqrt(
        np.mean(rotation_error ** 2, axis=0)
    )
    translation_axis_rmse = np.sqrt(
        np.mean(translation_error ** 2, axis=0)
    )

    metrics = AggregateMetrics(
        total_mse=float(total_mse),
        rotation_mse=rotation_mse,
        translation_mse=translation_mse,
        rotation_rmse=float(math.sqrt(rotation_mse)),
        translation_rmse=float(
            math.sqrt(translation_mse)
        ),
        rotation_mae=float(
            np.mean(np.abs(rotation_error))
        ),
        translation_mae=float(
            np.mean(np.abs(translation_error))
        ),
        rotation_axis_mae_x=float(
            rotation_axis_mae[0]
        ),
        rotation_axis_mae_y=float(
            rotation_axis_mae[1]
        ),
        rotation_axis_mae_z=float(
            rotation_axis_mae[2]
        ),
        rotation_axis_rmse_x=float(
            rotation_axis_rmse[0]
        ),
        rotation_axis_rmse_y=float(
            rotation_axis_rmse[1]
        ),
        rotation_axis_rmse_z=float(
            rotation_axis_rmse[2]
        ),
        translation_axis_mae_x=float(
            translation_axis_mae[0]
        ),
        translation_axis_mae_y=float(
            translation_axis_mae[1]
        ),
        translation_axis_mae_z=float(
            translation_axis_mae[2]
        ),
        translation_axis_rmse_x=float(
            translation_axis_rmse[0]
        ),
        translation_axis_rmse_y=float(
            translation_axis_rmse[1]
        ),
        translation_axis_rmse_z=float(
            translation_axis_rmse[2]
        ),
        num_samples=num_samples,
        num_batches=num_batches,
        elapsed_seconds=float(elapsed_seconds),
        samples_per_second=float(
            num_samples / elapsed_seconds
        ),
    )

    return (
        metrics,
        frame_predictions,
        rotation_gt_array,
        rotation_pred_array,
        translation_gt_array,
        translation_pred_array,
    )


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Return an X-axis rotation matrix."""

    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Return a Y-axis rotation matrix."""

    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Return a Z-axis rotation matrix."""

    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def euler_to_rotation_matrix(
    euler: Sequence[float],
    order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    """Convert a three-element Euler vector into a rotation matrix."""

    x, y, z = (
        float(euler[0]),
        float(euler[1]),
        float(euler[2]),
    )

    if angles_in_degrees:
        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

    rx = rotation_matrix_x(x)
    ry = rotation_matrix_y(y)
    rz = rotation_matrix_z(z)

    if order == "xyz":
        return rz @ ry @ rx

    if order == "zyx":
        return rx @ ry @ rz

    raise ValueError(
        f"Unsupported Euler order: {order}"
    )


def relative_pose_to_transform(
    rotation: Sequence[float],
    translation: Sequence[float],
    euler_order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    """Build a 4x4 relative SE(3) transformation."""

    transform = np.eye(4, dtype=np.float64)

    transform[:3, :3] = euler_to_rotation_matrix(
        rotation,
        order=euler_order,
        angles_in_degrees=angles_in_degrees,
    )

    transform[:3, 3] = np.asarray(
        translation,
        dtype=np.float64,
    )

    return transform


def integrate_relative_poses(
    rotations: np.ndarray,
    translations: np.ndarray,
    euler_order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    """Integrate relative poses from identity into global transforms."""

    if rotations.shape != translations.shape:
        raise ValueError(
            "Rotation and translation arrays must have matching "
            f"shapes, but received {rotations.shape} and "
            f"{translations.shape}."
        )

    if rotations.ndim != 2 or rotations.shape[1] != 3:
        raise ValueError(
            "Relative pose arrays must have shape [N, 3]."
        )

    trajectory = np.zeros(
        (rotations.shape[0] + 1, 4, 4),
        dtype=np.float64,
    )

    trajectory[0] = np.eye(4, dtype=np.float64)

    for index in range(rotations.shape[0]):
        relative_transform = relative_pose_to_transform(
            rotation=rotations[index],
            translation=translations[index],
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        )

        trajectory[index + 1] = (
            trajectory[index] @ relative_transform
        )

    return trajectory


def rotation_angle_degrees(
    rotation_matrix: np.ndarray,
) -> float:
    """Return the principal angle of a rotation matrix in degrees."""

    cosine = (
        np.trace(rotation_matrix) - 1.0
    ) / 2.0

    cosine = float(np.clip(cosine, -1.0, 1.0))

    return math.degrees(math.acos(cosine))


def compute_trajectory_metrics(
    ground_truth_trajectory: np.ndarray,
    predicted_trajectory: np.ndarray,
) -> TrajectoryMetrics:
    """Compute aligned-origin trajectory metrics."""

    if ground_truth_trajectory.shape != predicted_trajectory.shape:
        raise ValueError(
            "Ground-truth and predicted trajectories must have "
            "identical shapes."
        )

    gt_positions = ground_truth_trajectory[:, :3, 3]
    pred_positions = predicted_trajectory[:, :3, 3]

    # Both trajectories begin at identity. No scale or similarity
    # alignment is applied because translation magnitude is part of the
    # model output being evaluated.
    position_errors = np.linalg.norm(
        pred_positions - gt_positions,
        axis=1,
    )

    ate_rmse = float(
        np.sqrt(np.mean(position_errors ** 2))
    )

    gt_steps = np.linalg.norm(
        np.diff(gt_positions, axis=0),
        axis=1,
    )
    pred_steps = np.linalg.norm(
        np.diff(pred_positions, axis=0),
        axis=1,
    )

    gt_path_length = float(np.sum(gt_steps))
    pred_path_length = float(np.sum(pred_steps))

    endpoint_error = float(
        np.linalg.norm(
            pred_positions[-1] - gt_positions[-1]
        )
    )

    endpoint_error_percent = (
        100.0 * endpoint_error / gt_path_length
        if gt_path_length > 0.0
        else float("nan")
    )

    relative_translation_errors: List[float] = []
    relative_rotation_errors: List[float] = []

    for index in range(
        ground_truth_trajectory.shape[0] - 1
    ):
        gt_relative = (
            np.linalg.inv(ground_truth_trajectory[index])
            @ ground_truth_trajectory[index + 1]
        )

        pred_relative = (
            np.linalg.inv(predicted_trajectory[index])
            @ predicted_trajectory[index + 1]
        )

        relative_error = (
            np.linalg.inv(gt_relative)
            @ pred_relative
        )

        relative_translation_errors.append(
            float(
                np.linalg.norm(
                    relative_error[:3, 3]
                )
            )
        )

        relative_rotation_errors.append(
            rotation_angle_degrees(
                relative_error[:3, :3]
            )
        )

    translation_error_array = np.asarray(
        relative_translation_errors,
        dtype=np.float64,
    )

    rotation_error_array = np.asarray(
        relative_rotation_errors,
        dtype=np.float64,
    )

    rpe_translation_rmse = float(
        np.sqrt(
            np.mean(translation_error_array ** 2)
        )
    )

    rpe_rotation_rmse = float(
        np.sqrt(
            np.mean(rotation_error_array ** 2)
        )
    )

    translational_drift_percent = (
        100.0
        * float(np.sum(translation_error_array))
        / gt_path_length
        if gt_path_length > 0.0
        else float("nan")
    )

    rotational_drift_degrees_per_100m = (
        100.0
        * float(np.sum(rotation_error_array))
        / gt_path_length
        if gt_path_length > 0.0
        else float("nan")
    )

    return TrajectoryMetrics(
        ate_rmse=ate_rmse,
        ate_mean=float(np.mean(position_errors)),
        ate_median=float(
            np.median(position_errors)
        ),
        ate_max=float(np.max(position_errors)),
        rpe_translation_rmse=rpe_translation_rmse,
        rpe_translation_mean=float(
            np.mean(translation_error_array)
        ),
        rpe_rotation_rmse_degrees=(
            rpe_rotation_rmse
        ),
        rpe_rotation_mean_degrees=float(
            np.mean(rotation_error_array)
        ),
        path_length_ground_truth=gt_path_length,
        path_length_predicted=pred_path_length,
        endpoint_error=endpoint_error,
        endpoint_error_percent=float(
            endpoint_error_percent
        ),
        translational_drift_percent=float(
            translational_drift_percent
        ),
        rotational_drift_degrees_per_100m=float(
            rotational_drift_degrees_per_100m
        ),
    )


def write_dataclass_rows(
    path: Path,
    rows: Sequence[object],
) -> None:
    """Write dataclass instances to CSV."""

    if not rows:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    dictionaries = [
        asdict(row)
        for row in rows
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(dictionaries[0].keys()),
        )

        writer.writeheader()
        writer.writerows(dictionaries)


def write_axis_metrics(
    path: Path,
    metrics: AggregateMetrics,
) -> None:
    """Write per-axis error metrics to CSV."""

    rows = [
        {
            "target": "rotation",
            "axis": "x",
            "mae": metrics.rotation_axis_mae_x,
            "rmse": metrics.rotation_axis_rmse_x,
        },
        {
            "target": "rotation",
            "axis": "y",
            "mae": metrics.rotation_axis_mae_y,
            "rmse": metrics.rotation_axis_rmse_y,
        },
        {
            "target": "rotation",
            "axis": "z",
            "mae": metrics.rotation_axis_mae_z,
            "rmse": metrics.rotation_axis_rmse_z,
        },
        {
            "target": "translation",
            "axis": "x",
            "mae": metrics.translation_axis_mae_x,
            "rmse": metrics.translation_axis_rmse_x,
        },
        {
            "target": "translation",
            "axis": "y",
            "mae": metrics.translation_axis_mae_y,
            "rmse": metrics.translation_axis_rmse_y,
        },
        {
            "target": "translation",
            "axis": "z",
            "mae": metrics.translation_axis_mae_z,
            "rmse": metrics.translation_axis_rmse_z,
        },
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target",
                "axis",
                "mae",
                "rmse",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_kitti_trajectory(
    path: Path,
    trajectory: np.ndarray,
) -> None:
    """Save trajectory using KITTI's flattened 3x4 matrix format."""

    matrices = trajectory[:, :3, :4].reshape(
        trajectory.shape[0],
        12,
    )

    np.savetxt(
        path,
        matrices,
        fmt="%.12e",
    )


def plot_trajectory(
    ground_truth_trajectory: np.ndarray,
    predicted_trajectory: np.ndarray,
    axis_a: int,
    axis_b: int,
    axis_a_label: str,
    axis_b_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot ground-truth and predicted trajectory projections."""

    gt_positions = ground_truth_trajectory[:, :3, 3]
    pred_positions = predicted_trajectory[:, :3, 3]

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111)

    axes.plot(
        gt_positions[:, axis_a],
        gt_positions[:, axis_b],
        label="Ground truth",
    )

    axes.plot(
        pred_positions[:, axis_a],
        pred_positions[:, axis_b],
        label="Prediction",
    )

    axes.scatter(
        [gt_positions[0, axis_a]],
        [gt_positions[0, axis_b]],
        marker="o",
        label="Start",
    )

    axes.scatter(
        [gt_positions[-1, axis_a]],
        [gt_positions[-1, axis_b]],
        marker="x",
        label="GT end",
    )

    axes.scatter(
        [pred_positions[-1, axis_a]],
        [pred_positions[-1, axis_b]],
        marker="+",
        label="Predicted end",
    )

    axes.set_xlabel(axis_a_label)
    axes.set_ylabel(axis_b_label)
    axes.set_title(title)
    axes.axis("equal")
    axes.grid(True)
    axes.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
    )
    plt.close(figure)


def plot_error_histogram(
    errors: np.ndarray,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    """Plot an L2 frame-error histogram."""

    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111)

    axes.hist(errors, bins=50)

    axes.set_xlabel(x_label)
    axes.set_ylabel("Frame count")
    axes.set_title(title)
    axes.grid(True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
    )
    plt.close(figure)


def write_checkpoint_comparison(
    output_path: Path,
    checkpoint: Mapping[str, object],
    test_metrics: AggregateMetrics,
) -> Dict[str, Optional[float]]:
    """Compare held-out test losses with stored validation losses."""

    validation_metrics = checkpoint.get(
        "validation_metrics",
        {},
    )

    if not isinstance(validation_metrics, Mapping):
        validation_metrics = {}

    validation_total = validation_metrics.get(
        "total_loss"
    )
    validation_rotation = validation_metrics.get(
        "rotation_loss"
    )
    validation_translation = validation_metrics.get(
        "translation_loss"
    )

    comparison = {
        "validation_total_loss": (
            float(validation_total)
            if validation_total is not None
            else None
        ),
        "validation_rotation_loss": (
            float(validation_rotation)
            if validation_rotation is not None
            else None
        ),
        "validation_translation_loss": (
            float(validation_translation)
            if validation_translation is not None
            else None
        ),
        "test_total_loss": test_metrics.total_mse,
        "test_rotation_loss": test_metrics.rotation_mse,
        "test_translation_loss": (
            test_metrics.translation_mse
        ),
        "test_to_validation_total_ratio": None,
        "test_to_validation_rotation_ratio": None,
        "test_to_validation_translation_ratio": None,
    }

    if validation_total not in (None, 0):
        comparison[
            "test_to_validation_total_ratio"
        ] = (
            test_metrics.total_mse
            / float(validation_total)
        )

    if validation_rotation not in (None, 0):
        comparison[
            "test_to_validation_rotation_ratio"
        ] = (
            test_metrics.rotation_mse
            / float(validation_rotation)
        )

    if validation_translation not in (None, 0):
        comparison[
            "test_to_validation_translation_ratio"
        ] = (
            test_metrics.translation_mse
            / float(validation_translation)
        )

    lines = [
        "DeepDCT-VO checkpoint comparison",
        "=" * 52,
        f"Checkpoint epoch: {checkpoint.get('epoch')}",
        "",
        f"Validation total loss: "
        f"{comparison['validation_total_loss']}",
        f"Test total loss:       "
        f"{comparison['test_total_loss']:.9f}",
        f"Test/validation ratio: "
        f"{comparison['test_to_validation_total_ratio']}",
        "",
        f"Validation rotation:   "
        f"{comparison['validation_rotation_loss']}",
        f"Test rotation:         "
        f"{comparison['test_rotation_loss']:.9f}",
        f"Test/validation ratio: "
        f"{comparison['test_to_validation_rotation_ratio']}",
        "",
        f"Validation translation:"
        f" {comparison['validation_translation_loss']}",
        f"Test translation:      "
        f"{comparison['test_translation_loss']:.9f}",
        f"Test/validation ratio: "
        f"{comparison['test_to_validation_translation_ratio']}",
        "",
        "Interpretation:",
        "  ratio near 1.0: test performance is close to validation",
        "  moderately above 1.0: sequence/domain shift is present",
        "  substantially above 1.0: strong generalization gap",
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return comparison


def print_summary(
    checkpoint: Mapping[str, object],
    metrics: AggregateMetrics,
    trajectory_metrics: Optional[TrajectoryMetrics],
    output_dir: Path,
) -> None:
    """Print final evaluation results."""

    print()
    print("=" * 72)
    print("DeepDCT-VO held-out test evaluation")
    print("=" * 72)
    print(f"Checkpoint epoch:       {checkpoint.get('epoch')}")
    print(f"Test samples:           {metrics.num_samples}")
    print(f"Test batches:           {metrics.num_batches}")
    print(f"Total MSE:              {metrics.total_mse:.9f}")
    print(f"Rotation MSE:           {metrics.rotation_mse:.9f}")
    print(
        f"Translation MSE:        "
        f"{metrics.translation_mse:.9f}"
    )
    print(f"Rotation RMSE:          {metrics.rotation_rmse:.9f}")
    print(
        f"Translation RMSE:       "
        f"{metrics.translation_rmse:.9f}"
    )
    print(f"Rotation MAE:           {metrics.rotation_mae:.9f}")
    print(
        f"Translation MAE:        "
        f"{metrics.translation_mae:.9f}"
    )
    print(
        f"Elapsed time:           "
        f"{metrics.elapsed_seconds:.2f} s"
    )
    print(
        f"Throughput:             "
        f"{metrics.samples_per_second:.2f} samples/s"
    )

    if trajectory_metrics is not None:
        print("-" * 72)
        print(
            f"ATE RMSE:               "
            f"{trajectory_metrics.ate_rmse:.6f}"
        )
        print(
            f"RPE translation RMSE:   "
            f"{trajectory_metrics.rpe_translation_rmse:.6f}"
        )
        print(
            f"RPE rotation RMSE:      "
            f"{trajectory_metrics.rpe_rotation_rmse_degrees:.6f} deg"
        )
        print(
            f"Endpoint error:         "
            f"{trajectory_metrics.endpoint_error:.6f}"
        )
        print(
            f"Endpoint error:         "
            f"{trajectory_metrics.endpoint_error_percent:.3f}%"
        )
        print(
            f"Approx. translation drift: "
            f"{trajectory_metrics.translational_drift_percent:.3f}%"
        )
        print(
            "Approx. rotation drift:    "
            f"{trajectory_metrics.rotational_drift_degrees_per_100m:.3f} "
            "deg/100m"
        )

    print("-" * 72)
    print(f"Outputs saved to:       {output_dir.resolve()}")
    print("=" * 72)


def main() -> None:
    """Run held-out evaluation."""

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = resolve_device(args.device)

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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    print("=" * 72)
    print("DeepDCT-VO evaluation")
    print("=" * 72)
    print(f"Checkpoint:        {args.checkpoint.resolve()}")
    print(f"Checkpoint epoch:  {checkpoint['epoch']}")
    print(f"Test sequence:     {args.sequence}")
    print(f"Test samples:      {len(dataset)}")
    print(f"Device:            {device}")
    print(
        f"Input size:        "
        f"{evaluation_configuration['height']} x "
        f"{evaluation_configuration['width']}"
    )
    print(
        f"Camera:            "
        f"{evaluation_configuration['camera']}"
    )
    print(
        "Depth source:      zero placeholder, matching selected baseline"
    )
    print(
        f"GT rotation conditioning: "
        f"{args.use_ground_truth_rotation}"
    )
    print("=" * 72)

    (
        aggregate_metrics,
        frame_predictions,
        rotation_gt,
        rotation_pred,
        translation_gt,
        translation_pred,
    ) = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        rotation_loss_weight=float(
            evaluation_configuration[
                "rotation_loss_weight"
            ]
        ),
        translation_loss_weight=float(
            evaluation_configuration[
                "translation_loss_weight"
            ]
        ),
        use_ground_truth_rotation=(
            args.use_ground_truth_rotation
        ),
        log_interval=args.log_interval,
    )

    write_dataclass_rows(
        args.output_dir / "frame_predictions.csv",
        frame_predictions,
    )

    rotation_sorted = sorted(
        frame_predictions,
        key=lambda row: row.rotation_l2_error,
        reverse=True,
    )

    translation_sorted = sorted(
        frame_predictions,
        key=lambda row: row.translation_l2_error,
        reverse=True,
    )

    write_dataclass_rows(
        args.output_dir / "worst_rotation_frames.csv",
        rotation_sorted[: args.worst_frame_count],
    )

    write_dataclass_rows(
        args.output_dir / "worst_translation_frames.csv",
        translation_sorted[: args.worst_frame_count],
    )

    write_axis_metrics(
        args.output_dir / "axis_metrics.csv",
        aggregate_metrics,
    )

    rotation_l2_errors = np.linalg.norm(
        rotation_pred - rotation_gt,
        axis=1,
    )

    translation_l2_errors = np.linalg.norm(
        translation_pred - translation_gt,
        axis=1,
    )

    plot_error_histogram(
        errors=rotation_l2_errors,
        title="Sequence 10 rotation error distribution",
        x_label="Rotation L2 error",
        output_path=(
            args.output_dir
            / "rotation_error_histogram.png"
        ),
    )

    plot_error_histogram(
        errors=translation_l2_errors,
        title="Sequence 10 translation error distribution",
        x_label="Translation L2 error",
        output_path=(
            args.output_dir
            / "translation_error_histogram.png"
        ),
    )

    trajectory_metrics: Optional[TrajectoryMetrics] = None

    if not args.skip_trajectory:
        ground_truth_trajectory = integrate_relative_poses(
            rotations=rotation_gt,
            translations=translation_gt,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )

        predicted_trajectory = integrate_relative_poses(
            rotations=rotation_pred,
            translations=translation_pred,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )

        trajectory_metrics = compute_trajectory_metrics(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=predicted_trajectory,
        )

        save_kitti_trajectory(
            args.output_dir
            / "ground_truth_trajectory.txt",
            ground_truth_trajectory,
        )

        save_kitti_trajectory(
            args.output_dir
            / "predicted_trajectory.txt",
            predicted_trajectory,
        )

        plot_trajectory(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=predicted_trajectory,
            axis_a=0,
            axis_b=1,
            axis_a_label="X",
            axis_b_label="Y",
            title="Sequence 10 trajectory: XY projection",
            output_path=(
                args.output_dir
                / "trajectory_xy.png"
            ),
        )

        plot_trajectory(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=predicted_trajectory,
            axis_a=0,
            axis_b=2,
            axis_a_label="X",
            axis_b_label="Z",
            title="Sequence 10 trajectory: XZ projection",
            output_path=(
                args.output_dir
                / "trajectory_xz.png"
            ),
        )

    checkpoint_comparison = write_checkpoint_comparison(
        output_path=(
            args.output_dir
            / "checkpoint_comparison.txt"
        ),
        checkpoint=checkpoint,
        test_metrics=aggregate_metrics,
    )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_sequence": args.sequence,
        "device": str(device),
        "evaluation_configuration": dict(
            evaluation_configuration
        ),
        "frame_metrics": asdict(aggregate_metrics),
        "trajectory_metrics": (
            asdict(trajectory_metrics)
            if trajectory_metrics is not None
            else None
        ),
        "validation_test_comparison": (
            checkpoint_comparison
        ),
        "trajectory_assumption": (
            "Predicted and target directional translations were "
            "treated as directly composable previous-frame/local "
            "translations. Apply inverse DCT conversion first if "
            "the stored translation labels use another coordinate "
            "representation."
        ),
    }

    with (
        args.output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    print_summary(
        checkpoint=checkpoint,
        metrics=aggregate_metrics,
        trajectory_metrics=trajectory_metrics,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()