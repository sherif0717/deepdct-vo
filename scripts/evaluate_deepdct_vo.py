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

    # ----------------------------------------------------------
    # Training-objective-compatible evaluation loss.
    #
    # A1:
    #     MSE(raw rotation, raw GT)
    #     + MSE(translation, translation GT)
    #
    # A2:
    #     MAE(normalized rotation, normalized rotation GT)
    #     + MAE(translation, translation GT)
    #
    # These fields are used when comparing the held-out test
    # result against the validation loss stored in the checkpoint.
    # ----------------------------------------------------------
    objective_name: str
    total_objective_loss: float
    rotation_objective_loss: float
    translation_objective_loss: float

    # Physical-unit diagnostic metrics.
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

    # Gated-expert conditioning diagnostics.
    # NaN for non-gated translation decoders.
    translation_gate_0: float
    translation_gate_1: float
    translation_gate_2: float

    translation_gate_entropy: float
    translation_dominant_expert: int

    translation_expert_0_x: float
    translation_expert_0_y: float
    translation_expert_0_z: float

    translation_expert_1_x: float
    translation_expert_1_y: float
    translation_expert_1_z: float

    translation_expert_2_x: float
    translation_expert_2_y: float
    translation_expert_2_z: float


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
        "--require-a6-protocol",
        action="store_true",
        help=(
            "Require checkpoint metadata to satisfy Track-A A6: "
            "training sequences exactly 00-08, validation disabled, "
            "and checkpoint selection by final fixed epoch."
        ),
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

    # ----------------------------------------------------------
    # Track-A A5: paper translation scaling / post-processing.
    #
    # The network predictions and frame-level metrics remain
    # untouched. This factor is applied only when reconstructing
    # the predicted trajectory.
    #
    # Examples:
    #     1.000 -> unscaled control
    #     0.975 -> paper sequence-09 scaling
    #     1.007 -> paper sequence-10 scaling
    # ----------------------------------------------------------
    parser.add_argument(
        "--translation-scale-factor",
        type=float,
        default=1.0,
        help=(
            "A5 multiplicative scale applied to predicted directional "
            "translation only during trajectory reconstruction. "
            "Frame-level predictions and losses remain unscaled."
        ),
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

    parser.add_argument(
        "--translation-num-experts",
        type=int,
        default=3,
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
    
    if not math.isfinite(args.translation_scale_factor):
        raise ValueError(
            "--translation-scale-factor must be finite."
        )

    if args.translation_scale_factor <= 0.0:
        raise ValueError(
            "--translation-scale-factor must be greater than zero."
        )

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

def validate_a6_checkpoint_protocol(
    checkpoint: Mapping[str, object],
) -> None:
    """Verify that a checkpoint satisfies Track-A A6."""

    configuration = get_checkpoint_configuration(
        checkpoint
    )

    expected_training_sequences = {
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
    }

    target_sequences = {
        "09",
        "10",
    }

    train_sequences = {
        str(sequence).zfill(2)
        for sequence in configuration.get(
            "train_sequences",
            [],
        )
    }

    validation_sequences = {
        str(sequence).zfill(2)
        for sequence in configuration.get(
            "validation_sequences",
            [],
        )
    }

    if train_sequences != expected_training_sequences:
        raise ValueError(
            "A6 protocol check failed: checkpoint must "
            "have been trained on exactly sequences 00-08. "
            f"Checkpoint records: {sorted(train_sequences)}."
        )

    target_leakage = (
        train_sequences
        | validation_sequences
    ) & target_sequences

    if target_leakage:
        raise ValueError(
            "A6 protocol check failed: target sequence "
            "leakage detected. Sequence(s) "
            f"{sorted(target_leakage)} appear in training "
            "or validation metadata."
        )

    validation_enabled = bool(
        configuration.get(
            "validation_enabled",
            True,
        )
    )

    if validation_enabled:
        raise ValueError(
            "A6 protocol check failed: checkpoint records "
            "validation_enabled=True."
        )

    checkpoint_selection = str(
        configuration.get(
            "checkpoint_selection",
            "",
        )
    )

    if checkpoint_selection != "final_epoch":
        raise ValueError(
            "A6 protocol check failed: expected "
            "checkpoint_selection='final_epoch', but received "
            f"{checkpoint_selection!r}."
        )

    track_a_protocol = str(
        configuration.get(
            "track_a_protocol",
            "",
        )
    )

    if (
        track_a_protocol
        != "A6_unseen_00_08_to_09_10"
    ):
        raise ValueError(
            "A6 protocol check failed: checkpoint does not "
            "identify itself as "
            "'A6_unseen_00_08_to_09_10'."
        )

    print("=" * 72)
    print("Track-A A6 checkpoint protocol: PASS")
    print("=" * 72)
    print(
        "Training sequences:    "
        f"{sorted(train_sequences)}"
    )
    print("Validation sequences:  NONE")
    print("Unseen targets:        09, 10")
    print("Checkpoint selection:  final fixed epoch")
    print("=" * 72)

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

    # --------------------------------------------------------------
    # Pose-loss / rotation-normalization configuration.
    #
    # Backward compatibility:
    #     old/A1 checkpoints -> scale 1.0, MSE
    #
    # A2 checkpoints should record:
    #     rotation_normalization_scale = 0.175
    #     pose_loss_type = "mae"
    # --------------------------------------------------------------
    rotation_normalization_scale = float(
        configuration.get(
            "rotation_normalization_scale",
            1.0,
        )
    )

    if rotation_normalization_scale <= 0.0:
        raise ValueError(
            "Checkpoint rotation_normalization_scale must be "
            "greater than zero, but received "
            f"{rotation_normalization_scale}."
        )

    pose_loss_type = str(
        configuration.get(
            "pose_loss_type",
            "mse",
        )
    ).lower()

    if pose_loss_type not in {"mse", "mae"}:
        raise ValueError(
            "Unsupported checkpoint pose_loss_type: "
            f"{pose_loss_type!r}. Expected 'mse' or 'mae'."
        )

    rotation_geometry_weight = float(
        configuration.get(
            "rotation_geometry_weight",
            0.0,
        )
    )

    rotation_geometry_bank_size = configuration.get(
        "rotation_geometry_bank_size"
    )

    rotation_geometry_temperature = configuration.get(
        "rotation_geometry_temperature"
    )

    rotation_geometry_supervision = configuration.get(
        "rotation_geometry_supervision"
    )

    semantic_map_mode = str(
        configuration.get(
            "semantic_map_mode",
            "foreground_probability",
        )
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

    use_semantic_cues = bool(
        configuration.get(
            "use_semantic_cues",
            False,
        )
    )

    use_depth_cues = bool(
        configuration.get(
            "use_depth_cues",
            False,
        )
    )

    checkpoint_uses_gt_rotation = bool(
        configuration.get(
            "use_ground_truth_rotation",
            False,
        )
    )

    is_track_a_a4 = (
        checkpoint_uses_gt_rotation
        and use_semantic_cues
        and use_depth_cues
    )

    if is_track_a_a4:
        if pose_loss_type != "mae":
            raise ValueError(
                "A4 checkpoint invariant failed: "
                "pose_loss_type must be 'mae'."
            )

        if not math.isclose(
            rotation_normalization_scale,
            0.175,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "A4 checkpoint invariant failed: "
                "rotation_normalization_scale must be 0.175."
            )

        if semantic_map_mode != "foreground_probability":
            raise ValueError(
                "A4 checkpoint invariant failed: "
                "semantic_map_mode must be "
                "'foreground_probability'."
            )

    return {
        "height": height,
        "width": width,
        "camera": camera,
        "rotation_normalization_scale": (
            rotation_normalization_scale
        ),
        "pose_loss_type": pose_loss_type,
        "use_ground_truth_rotation": bool(
            configuration.get(
                "use_ground_truth_rotation",
                False,
            )
        ),
        "pretrained_semantic": pretrained_semantic,
        "freeze_semantic": freeze_semantic,
        "semantic_map_mode": semantic_map_mode,
        "share_aresunet_between_models": share_aresunet,
        "rotation_loss_weight": rotation_loss_weight,
        "translation_loss_weight": translation_loss_weight,
        "use_semantic_cues": bool(
            configuration.get(
                "use_semantic_cues",
                False,
            )
        ),
        "use_depth_cues": bool(
            configuration.get(
                "use_depth_cues",
                False,
            )
        ),
        "depth_checkpoint_dir": configuration.get(
            "depth_checkpoint_dir"
        ),
        "depth_model_name": configuration.get(
            "depth_model_name",
            "lite-mono-tiny",
        ),
        "depth_output_mode": configuration.get(
            "depth_output_mode",
            "normalized_depth",
        ),
        "depth_normalization_meters": float(
            configuration.get(
                "depth_normalization_meters",
                80.0,
            )
        ),
        "translation_decoder": str(
            configuration.get(
                "translation_decoder",
                "dense",
            )
        ),
        "translation_mlp_hidden_dims": tuple(
            configuration.get(
                "translation_mlp_hidden_dims",
                (256, 64),
            )
        ),
        "translation_projection_channels": int(
            configuration.get(
                "translation_projection_channels",
                8,
            )
        ),
        "translation_pool_size": tuple(
            configuration.get(
                "translation_pool_size",
                (4, 4),
            )
        ),
        "translation_aggregation_hidden_dim": int(
            configuration.get(
                "translation_aggregation_hidden_dim",
                64,
            )
        ),
        "translation_num_experts": int(
            configuration.get(
                "translation_num_experts",
                3,
            )
        ),
        "rotation_pool_size": tuple(
            configuration.get(
                "rotation_pool_size",
                (120, 120),
            )
        ),
        "rotation_representation_dimension": int(
            configuration.get(
                "rotation_representation_dimension",
                14400,
            )
        ),
        "rotation_geometry_weight": (
            rotation_geometry_weight
        ),

        "rotation_geometry_bank_size": (
            rotation_geometry_bank_size
        ),

        "rotation_geometry_temperature": (
            rotation_geometry_temperature
        ),

        "rotation_geometry_supervision": (
            rotation_geometry_supervision
        ),
        "semantic_model": str(
            configuration.get(
                "semantic_model",
                "lraspp",
            )
        ),

        "depth_model": str(
            configuration.get(
                "depth_model",
                "lite_mono",
            )
        ),
        "use_ground_truth_rotation": (
            checkpoint_uses_gt_rotation
        ),
        "use_semantic_cues": (
            use_semantic_cues
        ),
        "use_depth_cues": (
            use_depth_cues
        ),
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
        rotation_pool_size=tuple(
            evaluation_configuration[
                "rotation_pool_size"
            ]
        ),
        rotation_normalization_scale=float(
            evaluation_configuration[
                "rotation_normalization_scale"
            ]
        ),
        translation_decoder_type=str(
            evaluation_configuration[
                "translation_decoder"
            ]
        ),
        translation_mlp_hidden_dims=tuple(
            evaluation_configuration[
                "translation_mlp_hidden_dims"
            ]
        ),
        translation_projection_channels=int(
            evaluation_configuration[
                "translation_projection_channels"
            ]
        ),
        translation_pool_size=tuple(
            evaluation_configuration[
                "translation_pool_size"
            ]
        ),
        translation_aggregation_hidden_dim=int(
            evaluation_configuration[
                "translation_aggregation_hidden_dim"
            ]
        ),
        translation_num_experts=int(
            evaluation_configuration[
                "translation_num_experts"
            ]
        ),
        use_semantic_cues=bool(
            evaluation_configuration["use_semantic_cues"]
        ),
        semantic_map_mode=evaluation_configuration.get(
            "semantic_map_mode",
            "foreground_probability",
        ),
        use_depth_cues=bool(
            evaluation_configuration["use_depth_cues"]
        ),
        depth_checkpoint_dir=(
            evaluation_configuration["depth_checkpoint_dir"]
            if evaluation_configuration["use_depth_cues"]
            else None
        ),
        depth_model_name=str(
            evaluation_configuration["depth_model_name"]
        ),
        depth_output_mode=str(
            evaluation_configuration[
                "depth_output_mode"
            ]
        ),
        depth_normalization_meters=float(
            evaluation_configuration[
                "depth_normalization_meters"
            ]
        ),
        freeze_depth=True,
        freeze_semantic_model=True,
        freeze_depth_model=True,
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
    pose_loss_type: str,
    rotation_normalization_scale: float,
    use_ground_truth_rotation: bool,
    use_internal_depth: bool,
    log_interval: int,
) -> Tuple[
    AggregateMetrics,
    List[FramePrediction],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:    
    """Evaluate all frames and retain predictions for error analysis."""

    all_rotation_gt: List[np.ndarray] = []
    all_rotation_pred: List[np.ndarray] = []
    all_translation_gt: List[np.ndarray] = []
    all_translation_pred: List[np.ndarray] = []

    all_rotation_representations: List[np.ndarray] = []
    rotation_rep_buffer: Dict[str, Tensor] = {}

    all_translation_representations: List[np.ndarray] = []
    translation_rep_buffer: Dict[str, Tensor] = {}

    # --------------------------------------------------------------
    # Physical-unit diagnostic criteria.
    #
    # These remain MSE regardless of the training objective because
    # MSE/RMSE are useful physical evaluation metrics and preserve
    # comparability with A1.
    # --------------------------------------------------------------
    rotation_mse_criterion = nn.MSELoss(
        reduction="sum"
    )
    translation_mse_criterion = nn.MSELoss(
        reduction="sum"
    )

    # --------------------------------------------------------------
    # Checkpoint/training-objective-compatible criterion.
    #
    # A1 -> MSE
    # A2 -> MAE/L1
    # --------------------------------------------------------------
    if pose_loss_type == "mse":
        objective_criterion = nn.MSELoss(
            reduction="sum"
        )
    elif pose_loss_type == "mae":
        objective_criterion = nn.L1Loss(
            reduction="sum"
        )
    else:
        raise ValueError(
            "pose_loss_type must be 'mse' or 'mae', "
            f"but received {pose_loss_type!r}."
        )

    if rotation_normalization_scale <= 0.0:
        raise ValueError(
            "rotation_normalization_scale must be greater than zero, "
            f"but received {rotation_normalization_scale}."
        )

    frame_predictions: List[FramePrediction] = []

    # Physical MSE accumulators.
    total_rotation_squared_error = 0.0
    total_translation_squared_error = 0.0

    # Training-objective-compatible accumulators.
    total_rotation_objective_error = 0.0
    total_translation_objective_error = 0.0

    num_samples = 0
    num_batches = 0

    start_time = time.perf_counter()

    # ----------------------------------------------------------
    # Rotation representation audit hook.
    #
    # RotationHead inherits RegressionHead's
    # representation_input_module(), which returns the final
    # dense regression layer.
    #
    # A forward-pre-hook therefore captures the flattened
    # representation immediately before Linear(14400, 3).
    # ----------------------------------------------------------
    rotation_head_module = (
        model.rotation_head.representation_input_module()
    )

    rotation_dimension = int(
        model.rotation_head.representation_dim
    )

    rotation_head_name = (
        "rotation_head.dense "
        f"(compact {rotation_dimension}-D "
        "task-relevant rotation representation)"
    )

    def capture_rotation_representation(
        module: nn.Module,
        inputs: Tuple[Tensor, ...],
    ) -> None:
        if not inputs:
            raise RuntimeError(
                "Rotation head received no positional input."
            )

        representation = inputs[0]

        if not torch.is_tensor(representation):
            raise TypeError(
                "Expected rotation-head representation to be a tensor, "
                f"but received {type(representation).__name__}."
            )

        if representation.ndim != 2:
            raise ValueError(
                "Expected flattened rotation representation [B, D], "
                f"but received {tuple(representation.shape)}."
            )

        rotation_rep_buffer["value"] = (
            representation.detach().cpu()
        )

    rotation_rep_hook = (
        rotation_head_module.register_forward_pre_hook(
            capture_rotation_representation
        )
    )

    print(
        "Capturing rotation representation from: "
        f"{rotation_head_name}"
    )

    translation_head_module = (
        model.translation_head.representation_input_module()
    )

    if model.translation_head.decoder_type == "dense":
        translation_head_name = (
            "translation_head.dense"
        )

    elif model.translation_head.decoder_type == "mlp":
        translation_head_name = (
            "translation_head.dense.0 "
            "(flattened 14400-D representation)"
        )

    elif model.translation_head.decoder_type == "pooled_mlp":
        translation_head_name = (
            "translation_head.dense.0 "
            "(compact pooled representation)"
        )

    elif model.translation_head.decoder_type == "pooled_linear":
        translation_head_name = (
            "translation_head.dense "
            "(frozen compact pooled representation)"
        )

    elif model.translation_head.decoder_type == "gated_expert":
        translation_head_name = (
            "translation_head.gate "
            "(shared compact conditioning representation)"
        )

    else:
        raise RuntimeError(
            "Unsupported translation decoder type: "
            f"{model.translation_head.decoder_type}"
        )

    def capture_representation(
        module: nn.Module,
        inputs: Tuple[Tensor, ...],
    ) -> None:
        if not inputs:
            raise RuntimeError(
                "Translation head received no positional input."
            )

        representation = inputs[0]

        if not torch.is_tensor(representation):
            raise TypeError(
                "Expected translation-head input to be a tensor, "
                f"but received {type(representation).__name__}."
            )

        translation_rep_buffer["value"] = (
            representation.detach().cpu()
        )


    translation_rep_hook = (
        translation_head_module.register_forward_pre_hook(
            capture_representation
        )
    )

    print(
        "Capturing translation representation from: "
        f"{translation_head_name}"
    )

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
            if use_internal_depth:
                depth_curr = None
            else:
                depth_curr = move_tensor(
                    batch,
                    "depth_curr",
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
                return_intermediates=True,
            )

            # --------------------------------------------------
            # A3 runtime audit
            # --------------------------------------------------
            if use_ground_truth_rotation:
                rotation_used = outputs.get(
                    "rotation_used_for_translation"
                )

                if rotation_used is None:
                    raise KeyError(
                        "A3 evaluation requires "
                        "'rotation_used_for_translation'."
                    )

                if not torch.equal(
                    rotation_used,
                    rotation_gt,
                ):
                    raise RuntimeError(
                        "A3 evaluation invariant failed: "
                        "Model T did not receive the exact "
                        "ground-truth physical rotation."
                    )

            # --------------------------------------------------
            # Retrieve rotation representation captured by the
            # forward-pre-hook.
            # --------------------------------------------------
            if "value" not in rotation_rep_buffer:
                raise RuntimeError(
                    "Rotation representation hook did not fire. "
                    f"Hook target: {rotation_head_name}"
                )

            batch_rotation_representation = (
                rotation_rep_buffer.pop("value")
            )

            if (
                batch_rotation_representation.shape[0]
                != image_prev.shape[0]
            ):
                raise ValueError(
                    "Rotation representation batch size does not "
                    "match evaluator batch size: "
                    f"{tuple(batch_rotation_representation.shape)} "
                    f"vs batch={image_prev.shape[0]}."
                )

            all_rotation_representations.append(
                batch_rotation_representation.numpy()
            )

            if (
                model.translation_head.decoder_type
                == "gated_expert"
            ):
                # Representation explicitly exposed by DeepDCTVO.forward().
                batch_translation_representation = (
                    outputs[
                        "translation_representation"
                    ]
                    .detach()
                    .cpu()
                )

                batch_gate_logits = (
                    outputs[
                        "translation_gate_logits"
                    ]
                    .detach()
                    .cpu()
                )

                batch_gate_weights = (
                    outputs[
                        "translation_gate_weights"
                    ]
                    .detach()
                    .cpu()
                )

                batch_expert_predictions = (
                    outputs[
                        "translation_expert_predictions"
                    ]
                    .detach()
                    .cpu()
                )

                # The representation hook also fires on translation_head.gate.
                # Its value is redundant for gated_expert, so discard it.
                translation_rep_buffer.pop(
                    "value",
                    None,
                )

            else:
                # Preserve hook-based extraction for older decoder experiments.
                if "value" not in translation_rep_buffer:
                    raise RuntimeError(
                        "Translation representation hook did not fire. "
                        f"Hook target: {translation_head_name}"
                    )

                batch_translation_representation = (
                    translation_rep_buffer.pop("value")
                )

            if (
                batch_translation_representation.shape[0]
                != image_prev.shape[0]
            ):
                raise ValueError(
                    "Translation representation batch size does not "
                    "match evaluator batch size: "
                    f"{tuple(batch_translation_representation.shape)} "
                    f"vs batch={image_prev.shape[0]}."
                )

            all_translation_representations.append(
                batch_translation_representation.numpy()
            )

            # Physical Euler prediction in radians.
            #
            # DeepDCTVO guarantees that outputs["rotation"] has already been
            # de-normalized using rotation_normalization_scale.
            rotation_pred = outputs["rotation"]

            # Raw regression-space rotation prediction.
            #
            # A2 uses this tensor for its normalized MAE objective.
            rotation_pred_normalized = outputs[
                "rotation_normalized"
            ]

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
                        
            if (
                rotation_pred_normalized.shape
                != expected_shape
            ):
                raise ValueError(
                    "normalized rotation prediction has unexpected shape: "
                    f"{tuple(rotation_pred_normalized.shape)}."
                )

            if not torch.isfinite(
                rotation_pred_normalized
            ).all():
                raise FloatingPointError(
                    "Non-finite normalized rotation prediction encountered."
                )

            if not torch.isfinite(rotation_pred).all():
                raise FloatingPointError(
                    "Non-finite rotation prediction encountered."
                )

            if not torch.isfinite(translation_pred).all():
                raise FloatingPointError(
                    "Non-finite translation prediction encountered."
                )
            
            # --------------------------------------------------------------
            # Rotation target in the same regression space used by Model R.
            #
            # A1:
            #     scale = 1.0
            #
            # A2:
            #     scale = 0.175
            #
            # Dataset labels remain physical radians.
            # --------------------------------------------------------------
            rotation_gt_normalized = (
                rotation_gt
                / rotation_normalization_scale
            )

            if not torch.isfinite(
                rotation_gt_normalized
            ).all():
                raise FloatingPointError(
                    "Non-finite normalized rotation target encountered."
                )

            # --------------------------------------------------------------
            # Physical-unit MSE diagnostics
            # --------------------------------------------------------------
            rotation_sum_squared_error = (
                rotation_mse_criterion(
                    rotation_pred,
                    rotation_gt,
                )
            )

            translation_sum_squared_error = (
                translation_mse_criterion(
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

            # --------------------------------------------------------------
            # Training-objective-compatible held-out loss
            #
            # Rotation is evaluated in normalized regression space so that
            # this value is directly comparable with A2 validation loss.
            #
            # Translation remains in its existing physical/directional space.
            # --------------------------------------------------------------
            rotation_objective_error = (
                objective_criterion(
                    rotation_pred_normalized,
                    rotation_gt_normalized,
                )
            )

            translation_objective_error = (
                objective_criterion(
                    translation_pred,
                    translation_gt,
                )
            )

            total_rotation_objective_error += float(
                rotation_objective_error.item()
            )

            total_translation_objective_error += float(
                translation_objective_error.item()
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

            if (
                model.translation_head.decoder_type
                == "gated_expert"
            ):
                gate_weights_np = (
                    batch_gate_weights.numpy()
                )

                expert_predictions_np = (
                    batch_expert_predictions.numpy()
                )

            else:
                gate_weights_np = None
                expert_predictions_np = None

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


                if gate_weights_np is not None:
                    sample_gate_weights = (
                        gate_weights_np[sample_index]
                    )

                    sample_expert_predictions = (
                        expert_predictions_np[
                            sample_index
                        ]
                    )

                    gate_entropy = -float(
                        np.sum(
                            sample_gate_weights
                            * np.log(
                                sample_gate_weights
                                + 1.0e-12
                            )
                        )
                    )

                    dominant_expert = int(
                        np.argmax(
                            sample_gate_weights
                        )
                    )

                else:
                    sample_gate_weights = np.full(
                        3,
                        np.nan,
                        dtype=np.float32,
                    )

                    sample_expert_predictions = np.full(
                        (3, 3),
                        np.nan,
                        dtype=np.float32,
                    )

                    gate_entropy = float("nan")
                    dominant_expert = -1

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
                        translation_gate_0=float(
                            sample_gate_weights[0]
                        ),
                        translation_gate_1=float(
                            sample_gate_weights[1]
                        ),
                        translation_gate_2=float(
                            sample_gate_weights[2]
                        ),

                        translation_gate_entropy=float(
                            gate_entropy
                        ),

                        translation_dominant_expert=int(
                            dominant_expert
                        ),

                        translation_expert_0_x=float(
                            sample_expert_predictions[0, 0]
                        ),
                        translation_expert_0_y=float(
                            sample_expert_predictions[0, 1]
                        ),
                        translation_expert_0_z=float(
                            sample_expert_predictions[0, 2]
                        ),

                        translation_expert_1_x=float(
                            sample_expert_predictions[1, 0]
                        ),
                        translation_expert_1_y=float(
                            sample_expert_predictions[1, 1]
                        ),
                        translation_expert_1_z=float(
                            sample_expert_predictions[1, 2]
                        ),

                        translation_expert_2_x=float(
                            sample_expert_predictions[2, 0]
                        ),
                        translation_expert_2_y=float(
                            sample_expert_predictions[2, 1]
                        ),
                        translation_expert_2_z=float(
                            sample_expert_predictions[2, 2]
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

                running_rotation_objective = (
                    total_rotation_objective_error
                    / (num_samples * 3)
                )

                running_translation_objective = (
                    total_translation_objective_error
                    / (num_samples * 3)
                )

                running_objective = (
                    rotation_loss_weight
                    * running_rotation_objective
                    + translation_loss_weight
                    * running_translation_objective
                )

                print(
                    f"evaluation "
                    f"batch={num_batches} "
                    f"samples={num_samples} "
                    f"objective={pose_loss_type} "
                    f"loss={running_objective:.6f} "
                    f"rotation_loss="
                    f"{running_rotation_objective:.6f} "
                    f"translation_loss="
                    f"{running_translation_objective:.6f} "
                    f"physical_rotation_mse="
                    f"{running_rotation_mse:.6f} "
                    f"physical_translation_mse="
                    f"{running_translation_mse:.6f}"
                )

    rotation_rep_hook.remove()
    translation_rep_hook.remove()

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

    rotation_representation_array = np.concatenate(
        all_rotation_representations,
        axis=0,
    )

    if rotation_representation_array.shape[0] != num_samples:
        raise ValueError(
            "Rotation representation count does not match "
            "evaluated sample count: "
            f"{rotation_representation_array.shape[0]} "
            f"vs {num_samples}."
        )

    translation_representation_array = np.concatenate(
        all_translation_representations,
        axis=0,
    )

    if translation_representation_array.shape[0] != num_samples:
        raise ValueError(
            "Translation representation count does not match "
            "evaluated sample count: "
            f"{translation_representation_array.shape[0]} "
            f"vs {num_samples}."
        )

    if rotation_representation_array.ndim != 2:
        raise ValueError(
            "Rotation representation must have shape [N, D], "
            f"but received "
            f"{rotation_representation_array.shape}."
        )

    if (
        rotation_representation_array.shape[0]
        != len(frame_predictions)
    ):
        raise ValueError(
            "Rotation representation count does not match "
            "frame prediction count: "
            f"{rotation_representation_array.shape[0]} "
            f"vs {len(frame_predictions)}."
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

    # --------------------------------------------------------------
    # Held-out objective in the same space and with the same loss
    # definition used during training/validation.
    # --------------------------------------------------------------
    rotation_objective_loss = (
        total_rotation_objective_error
        / (num_samples * 3)
    )

    translation_objective_loss = (
        total_translation_objective_error
        / (num_samples * 3)
    )

    total_objective_loss = (
        rotation_loss_weight
        * rotation_objective_loss
        + translation_loss_weight
        * translation_objective_loss
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
        objective_name=pose_loss_type,

        total_objective_loss=float(
            total_objective_loss
        ),

        rotation_objective_loss=float(
            rotation_objective_loss
        ),

        translation_objective_loss=float(
            translation_objective_loss
        ),

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
        rotation_representation_array,
        translation_representation_array,
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
    """
    Integrate directional relative-pose predictions from identity.

    Parameters
    ----------
    rotations:
        Relative Euler rotations with shape [N, 3].

        For transition i -> j:

            R_relative = R_i.T @ R_j

    translations:
        Directional translation values t_c with shape [N, 3].

        These are not directly the translation components of the relative
        SE(3) transforms. The DeepDCT label generator defines:

            R_relative = R_i.T @ R_j
            R_half = sqrt(R_relative)
            t_c = R_j.T @ R_half @ (t_j - t_i)

    euler_order:
        Euler order used by the labels and predictions, normally "xyz".

    angles_in_degrees:
        Whether Euler values are expressed in degrees.

    Returns
    -------
    np.ndarray
        Absolute trajectory with shape [N + 1, 4, 4], beginning at identity.

    Notes
    -----
    Directional translation is decoded recursively. At each step:

        R_j = R_i @ R_relative

        delta_t_world = R_half.T @ R_j @ t_c

        t_relative_Ci = R_i.T @ delta_t_world

        T_j = T_i @ T_relative
    """

    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)

    if rotations.shape != translations.shape:
        raise ValueError(
            "Rotation and directional-translation arrays must have "
            f"matching shapes, but received {rotations.shape} and "
            f"{translations.shape}."
        )

    if rotations.ndim != 2 or rotations.shape[1] != 3:
        raise ValueError(
            "Relative pose arrays must have shape [N, 3], but received "
            f"{rotations.shape}."
        )

    if not np.all(np.isfinite(rotations)):
        raise ValueError(
            "Rotation predictions contain non-finite values."
        )

    if not np.all(np.isfinite(translations)):
        raise ValueError(
            "Directional-translation predictions contain non-finite values."
        )

    def project_rotation_to_so3(
        rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Project a nearly rotational matrix onto the closest valid SO(3)
        rotation matrix.
        """
        u, _, vt = np.linalg.svd(rotation)
        projected = u @ vt

        if np.linalg.det(projected) < 0.0:
            u[:, -1] *= -1.0
            projected = u @ vt

        return projected

    def rotation_square_root(
        rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the principal square root of an SO(3) rotation.

        The implementation converts the rotation to axis-angle form and
        divides its principal rotation angle by two.
        """
        rotation = project_rotation_to_so3(rotation)

        cosine = float(
            np.clip(
                (np.trace(rotation) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
        angle = math.acos(cosine)

        # Identity or an extremely small rotation.
        if angle < 1e-12:
            return np.eye(3, dtype=np.float64)

        # General case away from pi.
        if abs(math.pi - angle) > 1e-7:
            axis = np.asarray(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ],
                dtype=np.float64,
            )

            axis /= 2.0 * math.sin(angle)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-12:
                raise ValueError(
                    "Could not recover a valid rotation axis."
                )

            axis /= axis_norm

        else:
            # Stable axis recovery near a 180-degree rotation.
            diagonal = np.diag(rotation)
            axis = np.sqrt(
                np.maximum(
                    (diagonal + 1.0) / 2.0,
                    0.0,
                )
            )

            largest = int(np.argmax(axis))

            if axis[largest] < 1e-12:
                raise ValueError(
                    "Could not recover the axis of a pi rotation."
                )

            if largest == 0:
                axis[1] = math.copysign(
                    axis[1],
                    rotation[0, 1] + rotation[1, 0],
                )
                axis[2] = math.copysign(
                    axis[2],
                    rotation[0, 2] + rotation[2, 0],
                )
            elif largest == 1:
                axis[0] = math.copysign(
                    axis[0],
                    rotation[0, 1] + rotation[1, 0],
                )
                axis[2] = math.copysign(
                    axis[2],
                    rotation[1, 2] + rotation[2, 1],
                )
            else:
                axis[0] = math.copysign(
                    axis[0],
                    rotation[0, 2] + rotation[2, 0],
                )
                axis[1] = math.copysign(
                    axis[1],
                    rotation[1, 2] + rotation[2, 1],
                )

            axis /= np.linalg.norm(axis)

        half_angle = 0.5 * angle

        axis_skew = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )

        half_rotation = (
            np.eye(3, dtype=np.float64)
            + math.sin(half_angle) * axis_skew
            + (1.0 - math.cos(half_angle))
            * (axis_skew @ axis_skew)
        )

        return project_rotation_to_so3(half_rotation)

    trajectory = np.zeros(
        (rotations.shape[0] + 1, 4, 4),
        dtype=np.float64,
    )
    trajectory[0] = np.eye(4, dtype=np.float64)

    for index in range(rotations.shape[0]):
        # Use the evaluator's existing Euler conversion implementation.
        rotation_only_transform = relative_pose_to_transform(
            rotation=rotations[index],
            translation=np.zeros(3, dtype=np.float64),
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        )

        relative_rotation = project_rotation_to_so3(
            rotation_only_transform[:3, :3]
        )

        current_rotation = project_rotation_to_so3(
            trajectory[index, :3, :3]
        )

        next_rotation = project_rotation_to_so3(
            current_rotation @ relative_rotation
        )

        relative_rotation_half = rotation_square_root(
            relative_rotation
        )

        directional_translation = translations[index]

        # Invert:
        #
        #     t_c = R_j.T @ R_half @ delta_t_world
        #
        # to recover:
        #
        #     delta_t_world = R_half.T @ R_j @ t_c
        delta_translation_world = (
            relative_rotation_half.T
            @ next_rotation
            @ directional_translation
        )

        # Translation component required by:
        #
        #     T_Ci_Cj = inv(T_W_Ci) @ T_W_Cj
        relative_translation_current_frame = (
            current_rotation.T
            @ delta_translation_world
        )

        relative_transform = np.eye(
            4,
            dtype=np.float64,
        )
        relative_transform[:3, :3] = relative_rotation
        relative_transform[:3, 3] = (
            relative_translation_current_frame
        )

        trajectory[index + 1] = (
            trajectory[index] @ relative_transform
        )

        # Suppress small numerical departures from SO(3).
        trajectory[index + 1, :3, :3] = (
            project_rotation_to_so3(
                trajectory[index + 1, :3, :3]
            )
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
        "objective_name": (
            test_metrics.objective_name
        ),

        "test_total_loss": (
            test_metrics.total_objective_loss
        ),

        "test_rotation_loss": (
            test_metrics.rotation_objective_loss
        ),

        "test_translation_loss": (
            test_metrics.translation_objective_loss
        ),
        "test_to_validation_total_ratio": None,
        "test_to_validation_rotation_ratio": None,
        "test_to_validation_translation_ratio": None,
    }

    if validation_total not in (None, 0):
        comparison[
            "test_to_validation_total_ratio"
        ] = (
            test_metrics.total_objective_loss
            / float(validation_total)
        )

    if validation_rotation not in (None, 0):
        comparison[
            "test_to_validation_rotation_ratio"
        ] = (
            test_metrics.rotation_objective_loss
            / float(validation_rotation)
        )

    if validation_translation not in (None, 0):
        comparison[
            "test_to_validation_translation_ratio"
        ] = (
            test_metrics.translation_objective_loss
            / float(validation_translation)
        )

    lines = [
        "DeepDCT-VO checkpoint comparison",
        "=" * 52,
        f"Checkpoint epoch: {checkpoint.get('epoch')}",
        f"Objective:        {test_metrics.objective_name.upper()}",
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

def find_translation_head_module(
    model: nn.Module,
) -> Tuple[str, nn.Module]:
    """
    Return the DeepDCT-VO translation PoseHead.

    Its forward-pre-hook captures the representation supplied by
    Model T to the translation pose head.
    """

    if not hasattr(model, "translation_head"):
        raise AttributeError(
            "DeepDCTVO model has no 'translation_head' attribute."
        )

    translation_head = model.translation_head

    print(
        "Translation representation hook target: "
        "translation_head"
    )
    print(
        "Translation head module: "
        f"{translation_head.__class__.__name__}"
    )

    return "translation_head", translation_head

def capture_translation_head_input(module, inputs):
    """
    Save the tensor presented to the translation head.

    This is the representation we want to probe:
        h_t -> existing translation head -> [tx, ty, tz]
    """
    if not inputs:
        raise RuntimeError(
            "Translation-head hook received no positional input."
        )

    representation = inputs[0]

    if isinstance(representation, (tuple, list)):
        if len(representation) != 1:
            raise RuntimeError(
                "Translation head receives multiple tensors. "
                "Inspect the module before choosing the probe representation."
            )
        representation = representation[0]

    if not torch.is_tensor(representation):
        raise TypeError(
            "Expected translation-head input to be a tensor, "
            f"got {type(representation)}"
        )

    _translation_rep_buffer["value"] = (
        representation.detach().cpu()
    )




def print_summary(
    checkpoint: Mapping[str, object],
    metrics: AggregateMetrics,
    trajectory_metrics: Optional[TrajectoryMetrics],
    unscaled_trajectory_metrics: Optional[TrajectoryMetrics],
    evaluation_configuration: Mapping[str, object],
    translation_scale_factor: float,
    output_dir: Path,
) -> None:
    """Print final evaluation results."""

    print()
    print("=" * 72)
    print("DeepDCT-VO held-out test evaluation")
    print("=" * 72)

    if bool(evaluation_configuration["use_depth_cues"]):
        print("Depth source:           internal Lite-Mono")
    else:
        print("Depth source:           dataset placeholder")

    print(
        f"Semantic cues:          "
        f"{bool(evaluation_configuration['use_semantic_cues'])}"
    )
    print(
        f"Depth cues:             "
        f"{bool(evaluation_configuration['use_depth_cues'])}"
    )
    
    if bool(
        evaluation_configuration[
            "use_semantic_cues"
        ]
    ):
        print(
            "Semantic auxiliary: LR-ASPP"
        )

        print(
            "Semantic map mode: "
            f"{evaluation_configuration['semantic_map_mode']}"
        )

    if bool(
        evaluation_configuration[
            "use_depth_cues"
        ]
    ):
        print(
            "Depth auxiliary:    Lite-Mono"
        )

        print(
            "Depth model:        "
            f"{evaluation_configuration['depth_model_name']}"
        )

        print(
            "Depth output:       "
            f"{evaluation_configuration['depth_output_mode']}"
        )

        print(
            "Depth normalization:"
            f" {float(evaluation_configuration['depth_normalization_meters']):.1f} m"
        )

    print(
        f"Translation decoder:    "
        f"{evaluation_configuration['translation_decoder']}"
    )

    if (
        evaluation_configuration[
            "translation_decoder"
        ]
        in {
            "pooled_mlp",
            "pooled_linear",
        }
    ):
        projection_channels = int(
            evaluation_configuration[
                "translation_projection_channels"
            ]
        )

        pool_size = tuple(
            evaluation_configuration[
                "translation_pool_size"
            ]
        )

        compact_dimension = (
            projection_channels
            * pool_size[0]
            * pool_size[1]
        )

        print(
            f"Projection channels:    "
            f"{projection_channels}"
        )

        print(
            f"Pooling size:           "
            f"{pool_size}"
        )

        print(
            f"Compact representation: "
            f"{compact_dimension}"
        )

        if (
            evaluation_configuration[
                "translation_decoder"
            ]
            == "pooled_linear"
        ):
            print(
                f"Translation readout:    "
                f"Linear({compact_dimension}, 3)"
            )

    print(f"Checkpoint epoch:       {checkpoint.get('epoch')}")
    print(f"Test samples:           {metrics.num_samples}")
    print(f"Test batches:           {metrics.num_batches}")

    print(
        f"Pose objective:         "
        f"{metrics.objective_name.upper()}"
    )

    print(
        f"Rotation norm scale:    "
        f"{float(evaluation_configuration['rotation_normalization_scale']):.6f}"
    )

    print(
        f"A5 translation scale:   "
        f"{translation_scale_factor:.6f}"
    )

    print(
        "A5 scaling scope:     "
        "trajectory post-processing only"
    )

    print(
        f"Objective total loss:   "
        f"{metrics.total_objective_loss:.9f}"
    )

    print(
        f"Objective rotation:     "
        f"{metrics.rotation_objective_loss:.9f}"
    )

    print(
        f"Objective translation:  "
        f"{metrics.translation_objective_loss:.9f}"
    )

    print("-" * 72)

    print(f"Physical Total MSE:     {metrics.total_mse:.9f}")
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

        if unscaled_trajectory_metrics is not None:
            print("Trajectory results: UNscaled control")
            print(
                f"ATE RMSE:               "
                f"{unscaled_trajectory_metrics.ate_rmse:.6f}"
            )
            print(
                f"RPE translation RMSE:   "
                f"{unscaled_trajectory_metrics.rpe_translation_rmse:.6f}"
            )
            print(
                f"RPE rotation RMSE:      "
                f"{unscaled_trajectory_metrics.rpe_rotation_rmse_degrees:.6f} deg"
            )
            print(
                f"Endpoint error:         "
                f"{unscaled_trajectory_metrics.endpoint_error:.6f}"
            )
            print(
                f"Endpoint error:         "
                f"{unscaled_trajectory_metrics.endpoint_error_percent:.3f}%"
            )
            print(
                f"Approx. translation drift: "
                f"{unscaled_trajectory_metrics.translational_drift_percent:.3f}%"
            )
            print(
                "Approx. rotation drift:    "
                f"{unscaled_trajectory_metrics.rotational_drift_degrees_per_100m:.3f} "
                "deg/100m"
            )

            print("-" * 72)

        print(
            "Trajectory results: A5 post-processed "
            f"(translation scale={translation_scale_factor:.6f})"
        )

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

def main() -> None:
    """Run held-out evaluation."""

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = resolve_device(args.device)

    if (
        args.require_a6_protocol
        and args.sequence not in {
            "09",
            "10",
        }
    ):
        raise ValueError(
            "Track-A A6 evaluation targets must be "
            "sequence 09 or sequence 10."
        )
    

    if args.require_a6_protocol:
        expected_translation_scales = {
            "09": 0.975,
            "10": 1.007,
        }

        expected_scale = (
            expected_translation_scales[
                args.sequence
            ]
        )

        if not math.isclose(
            args.translation_scale_factor,
            expected_scale,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "A6 inherited A5 scaling mismatch for "
                f"sequence {args.sequence}: expected "
                f"{expected_scale:.3f}, received "
                f"{args.translation_scale_factor:.6f}."
            )

    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if args.require_a6_protocol:
        validate_a6_checkpoint_protocol(
            checkpoint
        )

    evaluation_configuration = (
        resolve_evaluation_configuration(
            args=args,
            checkpoint=checkpoint,
        )
    )

    # --------------------------------------------------------------
    # Model-T rotation conditioning
    #
    # A3 checkpoints record use_ground_truth_rotation=True.
    # Restore that automatically so evaluation cannot accidentally
    # turn an A3 checkpoint back into predicted-rotation inference.
    #
    # The CLI flag remains useful for explicitly running a GT-rotation
    # diagnostic on an older/A1/A2 checkpoint.
    # --------------------------------------------------------------
    checkpoint_uses_gt_rotation = bool(
        evaluation_configuration[
            "use_ground_truth_rotation"
        ]
    )

    effective_use_ground_truth_rotation = (
        checkpoint_uses_gt_rotation
        or args.use_ground_truth_rotation
    )

    # --------------------------------------------------------------
    # A2 rotation normalization is defined for radian-valued labels.
    # --------------------------------------------------------------
    rotation_normalization_scale = float(
        evaluation_configuration[
            "rotation_normalization_scale"
        ]
    )

    if (
        rotation_normalization_scale != 1.0
        and args.angles_in_degrees
    ):
        raise ValueError(
            "A normalized-rotation checkpoint cannot be evaluated "
            "with --angles-in-degrees. "
            "A2 rotation normalization uses physical radians."
        )

    rotation_geometry_enabled = (
        float(
            evaluation_configuration[
                "rotation_geometry_weight"
            ]
        )
        > 0.0
    )

    if (
        rotation_geometry_enabled
        and checkpoint.get(
            "rotation_geometry_state"
        ) is None
    ):
        print(
            "WARNING: continuous SO(3) rotation geometry "
            "is enabled in checkpoint configuration, but "
            "'rotation_geometry_state' is absent. "
            "Test inference remains valid, but the checkpoint "
            "does not contain resumable geometry-bank state."
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

    checkpoint_representation_dim = (
        evaluation_configuration.get(
            "rotation_representation_dimension"
        )
    )

    actual_rotation_representation_dim = int(
        model.rotation_head.representation_dim
    )

    if checkpoint_representation_dim is not None:
        if (
            actual_rotation_representation_dim
            != int(checkpoint_representation_dim)
        ):
            raise RuntimeError(
                "Rotation representation dimension mismatch: "
                f"checkpoint records "
                f"{checkpoint_representation_dim}, "
                f"but rebuilt model exposes "
                f"{actual_rotation_representation_dim}."
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
    if bool(evaluation_configuration["use_depth_cues"]):
        print("Depth source:      internal Lite-Mono")
    else:
        print("Depth source:      dataset placeholder")

    print(
        f"Semantic cues:     "
        f"{bool(evaluation_configuration['use_semantic_cues'])}"
    )

    print(
        f"Depth cues:        "
        f"{bool(evaluation_configuration['use_depth_cues'])}"
    )

    rotation_geometry_enabled = (
        float(
            evaluation_configuration[
                "rotation_geometry_weight"
            ]
        )
        > 0.0
    )

    print(
        f"Rotation geometry: "
        f"{rotation_geometry_enabled}"
    )

    print(
        f"Rotation rep dim:  "
        f"{actual_rotation_representation_dim}"
    )

    print(
        f"Pose objective:     "
        f"{str(evaluation_configuration['pose_loss_type']).upper()}"
    )

    print(
        f"Rotation scale:     "
        f"{float(evaluation_configuration['rotation_normalization_scale']):.6f}"
    )

    if rotation_geometry_enabled:
        print(
            f"Geometry weight:   "
            f"{evaluation_configuration['rotation_geometry_weight']}"
        )

        print(
            f"Geometry bank size:"
            f" {evaluation_configuration['rotation_geometry_bank_size']}"
        )

        print(
            f"Geometry temp:     "
            f"{evaluation_configuration['rotation_geometry_temperature']}"
        )

        print(
            f"Geometry sup:      "
            f"{evaluation_configuration['rotation_geometry_supervision']}"
        )

    print(
        f"GT rotation conditioning: "
        f"{effective_use_ground_truth_rotation}"
    )

    print(
        f"Checkpoint GT rotation:   "
        f"{checkpoint_uses_gt_rotation}"
    )

    if effective_use_ground_truth_rotation:
        print(
            "Model-T rotation source:   GROUND TRUTH"
        )
    else:
        print(
            "Model-T rotation source:   PREDICTED"
        )
    print("=" * 72)

    (
        aggregate_metrics,
        frame_predictions,
        rotation_gt,
        rotation_pred,
        translation_gt,
        translation_pred,
        rotation_representation,
        translation_representation,
    ) = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        rotation_loss_weight=float(
            evaluation_configuration["rotation_loss_weight"]
        ),
        translation_loss_weight=float(
            evaluation_configuration["translation_loss_weight"]
        ),
        pose_loss_type=str(
            evaluation_configuration[
                "pose_loss_type"
            ]
        ),

        rotation_normalization_scale=float(
            evaluation_configuration[
                "rotation_normalization_scale"
            ]
        ),
        use_ground_truth_rotation=(
            effective_use_ground_truth_rotation
        ),
        use_internal_depth=bool(
            evaluation_configuration["use_depth_cues"]
        ),
        log_interval=args.log_interval,
    )

    write_dataclass_rows(
        args.output_dir / "frame_predictions.csv",
        frame_predictions,
    )

    regime_configuration = checkpoint.get(
        "translation_loss_configuration",
        {}
    )

    regime_balancing = regime_configuration.get(
        "regime_balancing"
    )

    motion_regime_array = None

    if regime_balancing is not None:
        low_threshold = float(
            regime_balancing["low_threshold"]
        )

        high_threshold = float(
            regime_balancing["high_threshold"]
        )

        forward_translation = (
            translation_gt[:, 2]
        )

        motion_regime_array = np.full(
            shape=forward_translation.shape,
            fill_value=1,
            dtype=np.int64,
        )

        motion_regime_array[
            forward_translation <= low_threshold
        ] = 0

        motion_regime_array[
            forward_translation > high_threshold
        ] = 2

    # ----------------------------------------------------------
    # Save rotation representation for offline representation
    # and cross-sequence alignment audits.
    # ----------------------------------------------------------
    rotation_representation_path = (
        args.output_dir
        / "rotation_representations.npz"
    )

    np.savez_compressed(
        rotation_representation_path,
        rotation_representation=rotation_representation,
        rotation_gt=rotation_gt,
        translation_gt=translation_gt,
        motion_regime=motion_regime_array,
    )

    print(
        "Saved rotation representations: "
        f"{rotation_representation_path}"
    )

    print(
        "Rotation representation shape: "
        f"{rotation_representation.shape}"
    )

    translation_representation_path = (
        args.output_dir
        / "translation_representations.npz"
    )

    np.savez_compressed(
        translation_representation_path,
        translation_rep=translation_representation,
    )

    print(
        "Saved translation representations: "
        f"{translation_representation_path}"
    )

    print(
        "Translation representation shape: "
        f"{translation_representation.shape}"
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
        title=(
            f"Sequence {args.sequence} rotation error distribution"
        ),
        x_label="Rotation L2 error",
        output_path=(
            args.output_dir
            / "rotation_error_histogram.png"
        ),
    )

    plot_error_histogram(
        errors=translation_l2_errors,
        title=(
            f"Sequence {args.sequence} translation error distribution"
        ),
        x_label="Translation L2 error",
        output_path=(
            args.output_dir
            / "translation_error_histogram.png"
        ),
    )

    # ----------------------------------------------------------
    # Track-A A5: translation-scale post-processing
    #
    # IMPORTANT:
    #   - translation_pred remains the raw network prediction.
    #   - frame-level metrics remain identical to A4.
    #   - scaling is applied only to the copy used for predicted
    #     trajectory reconstruction.
    #
    # Both unscaled and post-processed trajectories are retained
    # so A5 can quantify exactly what the paper scaling changes.
    # ----------------------------------------------------------
    trajectory_metrics: Optional[TrajectoryMetrics] = None
    unscaled_trajectory_metrics: Optional[TrajectoryMetrics] = None

    if not args.skip_trajectory:
        ground_truth_trajectory = integrate_relative_poses(
            rotations=rotation_gt,
            translations=translation_gt,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )

        # ------------------------------------------------------
        # A5 control trajectory:
        # exact raw network output, identical to A4 behavior.
        # ------------------------------------------------------
        unscaled_predicted_trajectory = integrate_relative_poses(
            rotations=rotation_pred,
            translations=translation_pred,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )

        unscaled_trajectory_metrics = compute_trajectory_metrics(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=unscaled_predicted_trajectory,
        )

        # ------------------------------------------------------
        # A5 paper post-processing.
        #
        # Scale directional translation uniformly in all three
        # components before the evaluator's existing directional-
        # translation -> SE(3) reconstruction.
        #
        # Do NOT modify translation_pred in place.
        # ------------------------------------------------------
        postprocessed_translation_pred = (
            translation_pred
            * float(args.translation_scale_factor)
        )

        if not np.all(
            np.isfinite(postprocessed_translation_pred)
        ):
            raise FloatingPointError(
                "A5 post-processed translation contains "
                "non-finite values."
            )

        predicted_trajectory = integrate_relative_poses(
            rotations=rotation_pred,
            translations=postprocessed_translation_pred,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )

        trajectory_metrics = compute_trajectory_metrics(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=predicted_trajectory,
        )

        # ------------------------------------------------------
        # Save GT, raw prediction, and A5 post-processed result.
        #
        # predicted_trajectory.txt remains the primary result so
        # downstream plotting/evaluation tools automatically use
        # the A5 trajectory.
        # ------------------------------------------------------
        save_kitti_trajectory(
            args.output_dir
            / "ground_truth_trajectory.txt",
            ground_truth_trajectory,
        )

        save_kitti_trajectory(
            args.output_dir
            / "predicted_trajectory_unscaled.txt",
            unscaled_predicted_trajectory,
        )

        save_kitti_trajectory(
            args.output_dir
            / "predicted_trajectory.txt",
            predicted_trajectory,
        )

        # ------------------------------------------------------
        # Unscaled control plots
        # ------------------------------------------------------
        plot_trajectory(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=unscaled_predicted_trajectory,
            axis_a=0,
            axis_b=1,
            axis_a_label="X",
            axis_b_label="Y",
            title=(
                f"Sequence {args.sequence} trajectory: "
                "XY projection (unscaled)"
            ),
            output_path=(
                args.output_dir
                / "trajectory_xy_unscaled.png"
            ),
        )

        plot_trajectory(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=unscaled_predicted_trajectory,
            axis_a=0,
            axis_b=2,
            axis_a_label="X",
            axis_b_label="Z",
            title=(
                f"Sequence {args.sequence} trajectory: "
                "XZ projection (unscaled)"
            ),
            output_path=(
                args.output_dir
                / "trajectory_xz_unscaled.png"
            ),
        )

        # ------------------------------------------------------
        # A5 paper-scaled trajectory plots
        # ------------------------------------------------------
        plot_trajectory(
            ground_truth_trajectory=ground_truth_trajectory,
            predicted_trajectory=predicted_trajectory,
            axis_a=0,
            axis_b=1,
            axis_a_label="X",
            axis_b_label="Y",
            title=(
                f"Sequence {args.sequence} trajectory: "
                "XY projection "
                f"(translation scale "
                f"{args.translation_scale_factor:.6f})"
            ),
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
            title=(
                f"Sequence {args.sequence} trajectory: "
                "XZ projection "
                f"(translation scale "
                f"{args.translation_scale_factor:.6f})"
            ),
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
        "rotation_geometry": {
            "enabled": rotation_geometry_enabled,
            "weight": float(
                evaluation_configuration[
                    "rotation_geometry_weight"
                ]
            ),
            "bank_size": (
                evaluation_configuration[
                    "rotation_geometry_bank_size"
                ]
            ),
            "temperature": (
                evaluation_configuration[
                    "rotation_geometry_temperature"
                ]
            ),
            "supervision": (
                evaluation_configuration[
                    "rotation_geometry_supervision"
                ]
            ),
            "bank_used_during_test": False,
        },
        "rotation_representation": {
            "path": str(
                rotation_representation_path.resolve()
            ),
            "samples": int(
                rotation_representation.shape[0]
            ),
            "dimension": int(
                rotation_representation.shape[1]
            ),
        },
        "frame_metrics": asdict(aggregate_metrics),
        # ------------------------------------------------------
        # Track-A A6 evaluation protocol
        # ------------------------------------------------------
        "evaluation_protocol": {
        "track_a_stage": (
            "A6"
            if args.require_a6_protocol
            else None
        ),

        "protocol_name": (
            "unseen_00_08_to_09_10"
            if args.require_a6_protocol
            else None
        ),

        "train_sequences": (
            [
                "00",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
            ]
            if args.require_a6_protocol
            else None
        ),

        "validation_enabled": (
            False
            if args.require_a6_protocol
            else None
        ),

        "target_sequence": args.sequence,

        "checkpoint_selection": (
            "final_epoch"
            if args.require_a6_protocol
            else None
        ),
    },

        # ------------------------------------------------------
        # Track-A A5 post-processing configuration.
        # ------------------------------------------------------
        "post_processing": {
            "track_a_stage": "A5",
            "translation_scale_enabled": (
                not math.isclose(
                    args.translation_scale_factor,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            "translation_scale_factor": float(
                args.translation_scale_factor
            ),
            "translation_scale_scope": (
                "predicted directional translation during "
                "trajectory reconstruction only"
            ),
            "frame_predictions_scaled": False,
            "rotation_scaled": False,
        },

        # Primary A5 result.
        "trajectory_metrics": (
            asdict(trajectory_metrics)
            if trajectory_metrics is not None
            else None
        ),

        # Raw A4-equivalent trajectory reconstructed during
        # the same evaluator invocation.
        "trajectory_metrics_unscaled": (
            asdict(unscaled_trajectory_metrics)
            if unscaled_trajectory_metrics is not None
            else None
        ),

        "validation_test_comparison": (
            checkpoint_comparison
        ),
        "trajectory_assumption": (
            "Predicted and target translations use the DeepDCT "
            "directional-translation representation. Trajectory "
            "integration reconstructs each relative SE(3) translation "
            "using the evaluator's R_half directional-translation "
            "inverse mapping before composition."
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
        unscaled_trajectory_metrics=(
            unscaled_trajectory_metrics
        ),
        evaluation_configuration=evaluation_configuration,
        translation_scale_factor=float(
            args.translation_scale_factor
        ),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()