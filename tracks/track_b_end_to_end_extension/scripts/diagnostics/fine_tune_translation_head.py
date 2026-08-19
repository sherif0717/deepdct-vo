#!/usr/bin/env python3
"""
Fine-tune only the final DeepDCT-VO translation dense layer.

Purpose
-------
This is a controlled diagnostic experiment for determining whether the final
translation-head linear mapping is responsible for poor translation prediction.

Only:

    translation_head.dense.weight
    translation_head.dense.bias

are trainable.

Everything else is frozen, including model parameters and BatchNorm/statistical
buffers. The frozen model is therefore kept in eval() mode even during the
fine-tuning pass.

Typical baseline experiment
---------------------------

python scripts/fine_tune_translation_head.py \
    --checkpoint experiments/baseline_identity_output/best_validation.pt \
    --data-root data \
    --train-sequences 00 01 02 03 04 05 06 07 08 \
    --validation-sequences 09 \
    --epochs 10 \
    --batch-size 1 \
    --num-workers 0 \
    --learning-rate 1e-4 \
    --output-dir experiments/baseline_identity_output_translation_head_ft

The held-out KITTI sequence 10 must NOT be used during fine-tuning or model
selection.

Outputs
-------
<output-dir>/
    latest.pt
    best_validation.pt
    translation_head_ft_epoch_001.pt
    ...
    history.csv
    summary.json

The resulting best_validation.pt remains compatible with the normal
DeepDCT-VO evaluation/audit scripts because it contains a complete
model_state_dict.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepdct.data.training_dataset import DeepDCTTrainingDataset
from deepdct.models.deepdct_vo import DeepDCTVO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAINABLE_PARAMETER_NAMES = {
    "translation_head.dense.weight",
    "translation_head.dense.bias",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune only translation_head.dense in a trained "
            "DeepDCT-VO model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Source DeepDCT-VO checkpoint. Normally use "
            "best_validation.pt from the original experiment."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing sequences/, out_csv/, and poses/.",
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=[
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
        ],
        help="KITTI sequences used for translation-head fine-tuning.",
    )

    parser.add_argument(
        "--validation-sequences",
        nargs="+",
        default=["09"],
        help="KITTI sequences used for model selection.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which fine-tuned checkpoints are stored.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help=(
            "Gradient clipping applied to the translation dense layer. "
            "Use <= 0 to disable."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Condition translation on ground-truth rotation. "
            "Leave disabled for the primary experiment if the original "
            "model uses predicted-rotation conditioning."
        ),
    )

    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Store a numbered complete checkpoint after every epoch.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")

    if args.scheduler_patience < 0:
        raise ValueError("--scheduler-patience cannot be negative.")

    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError(
            "--scheduler-factor must be strictly between 0 and 1."
        )

    train_sequences = set(args.train_sequences)
    validation_sequences = set(args.validation_sequences)

    overlap = train_sequences.intersection(validation_sequences)

    if overlap:
        raise ValueError(
            "Training and validation sequences overlap: "
            f"{sorted(overlap)}"
        )

    # Sequence 10 is the held-out test sequence in the current workflow.
    if "10" in train_sequences or "10" in validation_sequences:
        raise ValueError(
            "Sequence 10 must remain held out. Do not use it for "
            "translation-head fine-tuning or validation."
        )


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------


def load_source_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Expected checkpoint to be a mapping, received "
            f"{type(checkpoint).__name__}."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    return dict(checkpoint)


def get_configuration(
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    configuration = checkpoint.get("configuration", {})

    if not isinstance(configuration, Mapping):
        configuration = {}

    return dict(configuration)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> Tuple[DeepDCTVO, Dict[str, Any]]:
    """
    Reconstruct the same architecture used by the source checkpoint.

    Defaults correspond to the baseline RGB-only 120x120 experiment.
    """

    configuration = get_configuration(checkpoint)

    height = int(configuration.get("height", 120))
    width = int(configuration.get("width", 120))

    use_semantic_cues = bool(
        configuration.get("use_semantic_cues", False)
    )

    use_depth_cues = bool(
        configuration.get("use_depth_cues", False)
    )

    pretrained_semantic = bool(
        configuration.get("pretrained_semantic", True)
    )

    freeze_semantic = bool(
        configuration.get("freeze_semantic", True)
    )

    freeze_depth = bool(
        configuration.get("freeze_depth", True)
    )

    share_aresunet = bool(
        configuration.get(
            "share_aresunet_between_models",
            False,
        )
    )

    depth_checkpoint_dir = configuration.get(
        "depth_checkpoint_dir",
        None,
    )

    depth_model_name = str(
        configuration.get(
            "depth_model_name",
            "lite-mono-tiny",
        )
    )

    depth_output_mode = str(
        configuration.get(
            "depth_output_mode",
            "normalized_depth",
        )
    )

    if not use_depth_cues:
        depth_checkpoint_dir = None

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(height, width),
        pretrained_semantic=pretrained_semantic,
        freeze_semantic=freeze_semantic,
        normalize_semantic_input=True,
        normalize_semantic_map=True,
        share_aresunet_between_models=share_aresunet,
        depth_checkpoint_dir=depth_checkpoint_dir,
        depth_model_name=depth_model_name,
        depth_output_mode=depth_output_mode,
        freeze_depth=freeze_depth,
        use_semantic_cues=use_semantic_cues,
        use_depth_cues=use_depth_cues,
    )

    state_dict = checkpoint["model_state_dict"]

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if incompatible.missing_keys or incompatible.unexpected_keys:
        message = [
            "Source checkpoint does not exactly match the reconstructed model."
        ]

        if incompatible.missing_keys:
            message.append(
                "Missing keys:\n  "
                + "\n  ".join(incompatible.missing_keys)
            )

        if incompatible.unexpected_keys:
            message.append(
                "Unexpected keys:\n  "
                + "\n  ".join(incompatible.unexpected_keys)
            )

        raise RuntimeError("\n".join(message))

    # Strict load after the compatibility check.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model.to(device), configuration


# ---------------------------------------------------------------------------
# Freeze policy
# ---------------------------------------------------------------------------


def configure_translation_dense_only(
    model: nn.Module,
) -> List[nn.Parameter]:
    """
    Freeze every parameter except translation_head.dense.{weight,bias}.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    named_parameters = dict(model.named_parameters())

    missing = (
        TRAINABLE_PARAMETER_NAMES
        - set(named_parameters.keys())
    )

    if missing:
        raise RuntimeError(
            "The expected translation dense parameters were not found:\n  "
            + "\n  ".join(sorted(missing))
        )

    trainable_parameters: List[nn.Parameter] = []

    for name in sorted(TRAINABLE_PARAMETER_NAMES):
        parameter = named_parameters[name]
        parameter.requires_grad = True
        trainable_parameters.append(parameter)

    actual_trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if actual_trainable != TRAINABLE_PARAMETER_NAMES:
        raise RuntimeError(
            "Trainable-parameter audit failed.\n"
            f"Expected: {sorted(TRAINABLE_PARAMETER_NAMES)}\n"
            f"Actual:   {sorted(actual_trainable)}"
        )

    return trainable_parameters


def print_trainable_parameter_audit(
    model: nn.Module,
) -> None:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 80)
    print("Translation-head-only parameter audit")
    print("=" * 80)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(
                f"TRAINABLE  {name:<50} "
                f"{tuple(parameter.shape)}"
            )

    print("-" * 80)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(
        "Trainable fraction:   "
        f"{100.0 * trainable / total:.8f}%"
    )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------


def build_dataset(
    *,
    data_root: Path,
    sequences: List[str],
    configuration: Mapping[str, Any],
) -> DeepDCTTrainingDataset:
    height = int(configuration.get("height", 120))
    width = int(configuration.get("width", 120))
    camera = str(configuration.get("camera", "left"))

    return DeepDCTTrainingDataset(
        data_root=data_root,
        sequences=sequences,
        camera=camera,
        image_size=(height, width),
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=False,
    )


def build_dataloader(
    dataset: DeepDCTTrainingDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Batch handling
# ---------------------------------------------------------------------------


def move_tensor(
    tensor: Tensor,
    device: torch.device,
) -> Tensor:
    return tensor.to(
        device=device,
        non_blocking=device.type == "cuda",
    )


def prepare_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Tensor]:
    required = {
        "image_prev",
        "image_curr",
        "rotation_gt",
        "translation_gt",
    }

    missing = required.difference(batch.keys())

    if missing:
        raise KeyError(
            f"Batch missing required keys: {sorted(missing)}"
        )

    tensors: Dict[str, Tensor] = {}

    for key in required:
        value = batch[key]

        if not torch.is_tensor(value):
            raise TypeError(
                f"batch[{key!r}] is not a tensor."
            )

        tensors[key] = move_tensor(
            value,
            device,
        )

    if "depth_curr" in batch:
        depth = batch["depth_curr"]

        if torch.is_tensor(depth):
            tensors["depth_curr"] = move_tensor(
                depth,
                device,
            )

    return tensors


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def predict_translation(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
    *,
    use_ground_truth_rotation: bool,
    use_internal_depth: bool,
) -> Tensor:
    rotation_for_translation: Optional[Tensor]

    if use_ground_truth_rotation:
        rotation_for_translation = tensors["rotation_gt"]
    else:
        rotation_for_translation = None

    # For cue-enabled models, do not inject the zero placeholder from the
    # training dataset. This allows the model's internal Lite-Mono branch
    # to reproduce the source model's depth-cue behavior.
    if use_internal_depth:
        depth_curr = None
    else:
        depth_curr = tensors.get("depth_curr")

    outputs = model(
        image_prev=tensors["image_prev"],
        image_curr=tensors["image_curr"],
        depth_curr=depth_curr,
        rotation_for_translation=rotation_for_translation,
        use_ground_truth_rotation=use_ground_truth_rotation,
        return_intermediates=False,
    )

    if not isinstance(outputs, Mapping):
        raise TypeError(
            "DeepDCTVO forward output must be a mapping."
        )

    if "directional_translation" not in outputs:
        raise KeyError(
            "Model output does not contain "
            "'directional_translation'."
        )

    prediction = outputs["directional_translation"]

    if tuple(prediction.shape) != tuple(
        tensors["translation_gt"].shape
    ):
        raise ValueError(
            "Translation prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} versus "
            f"{tuple(tensors['translation_gt'].shape)}."
        )

    return prediction


# ---------------------------------------------------------------------------
# Epoch processing
# ---------------------------------------------------------------------------


def train_one_epoch_head_only(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_ground_truth_rotation: bool,
    use_internal_depth: bool,
    max_grad_norm: Optional[float],
    log_interval: int,
) -> Dict[str, float]:
    """
    Train the final dense translation layer only.

    IMPORTANT:
    model.eval() is intentional. Gradients still flow to the trainable dense
    weight/bias, but BatchNorm/dropout/frozen-module behavior remains fixed.
    """

    model.eval()

    loss_sum = 0.0
    element_count = 0
    sample_count = 0

    axis_squared_error = np.zeros(3, dtype=np.float64)
    axis_element_count = 0

    start_time = time.perf_counter()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        tensors = prepare_batch(
            batch,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        prediction = predict_translation(
            model,
            tensors,
            use_ground_truth_rotation=use_ground_truth_rotation,
            use_internal_depth=use_internal_depth,
        )

        target = tensors["translation_gt"]

        loss = criterion(
            prediction,
            target,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at training batch {batch_index}."
            )

        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_grad_norm,
            )

        optimizer.step()

        error = prediction.detach() - target

        batch_elements = target.numel()
        loss_sum += (
            float(loss.detach().item())
            * batch_elements
        )
        element_count += batch_elements
        sample_count += int(target.shape[0])

        squared_error = (
            error.pow(2)
            .sum(dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        axis_squared_error += squared_error
        axis_element_count += int(target.shape[0])

        if batch_index % log_interval == 0:
            running_mse = loss_sum / element_count

            print(
                f"  train batch {batch_index:6d}/"
                f"{len(dataloader):6d} | "
                f"translation MSE {running_mse:.8f}"
            )

    elapsed = time.perf_counter() - start_time

    mse = loss_sum / max(element_count, 1)

    axis_rmse = np.sqrt(
        axis_squared_error
        / max(axis_element_count, 1)
    )

    return {
        "loss": float(mse),
        "rmse": float(np.sqrt(mse)),
        "rmse_x": float(axis_rmse[0]),
        "rmse_y": float(axis_rmse[1]),
        "rmse_z": float(axis_rmse[2]),
        "samples": int(sample_count),
        "elapsed_seconds": float(elapsed),
    }


@torch.no_grad()
def evaluate_translation(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_ground_truth_rotation: bool,
    use_internal_depth: bool,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    element_count = 0
    sample_count = 0

    axis_squared_error = np.zeros(3, dtype=np.float64)
    axis_error_sum = np.zeros(3, dtype=np.float64)
    axis_element_count = 0

    start_time = time.perf_counter()

    for batch in dataloader:
        tensors = prepare_batch(
            batch,
            device,
        )

        prediction = predict_translation(
            model,
            tensors,
            use_ground_truth_rotation=use_ground_truth_rotation,
            use_internal_depth=use_internal_depth,
        )

        target = tensors["translation_gt"]

        loss = criterion(
            prediction,
            target,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Validation produced a non-finite loss."
            )

        error = prediction - target

        batch_elements = target.numel()

        loss_sum += (
            float(loss.item())
            * batch_elements
        )
        element_count += batch_elements
        sample_count += int(target.shape[0])

        axis_squared_error += (
            error.pow(2)
            .sum(dim=0)
            .cpu()
            .numpy()
        )

        axis_error_sum += (
            error.sum(dim=0)
            .cpu()
            .numpy()
        )

        axis_element_count += int(target.shape[0])

    elapsed = time.perf_counter() - start_time

    mse = loss_sum / max(element_count, 1)

    axis_rmse = np.sqrt(
        axis_squared_error
        / max(axis_element_count, 1)
    )

    axis_bias = (
        axis_error_sum
        / max(axis_element_count, 1)
    )

    return {
        "loss": float(mse),
        "rmse": float(np.sqrt(mse)),
        "rmse_x": float(axis_rmse[0]),
        "rmse_y": float(axis_rmse[1]),
        "rmse_z": float(axis_rmse[2]),
        "bias_x": float(axis_bias[0]),
        "bias_y": float(axis_bias[1]),
        "bias_z": float(axis_bias[2]),
        "samples": int(sample_count),
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Frozen-state audit
# ---------------------------------------------------------------------------


def clone_frozen_state(
    model: nn.Module,
) -> Dict[str, Tensor]:
    """
    Save every state_dict tensor except the two trainable dense tensors.

    This includes buffers, making the final audit sensitive to accidental
    BatchNorm-statistics changes.
    """

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name not in TRAINABLE_PARAMETER_NAMES
    }


def audit_frozen_state(
    model: nn.Module,
    reference: Mapping[str, Tensor],
) -> Tuple[bool, float, Optional[str]]:
    maximum_difference = 0.0
    worst_key: Optional[str] = None

    current_state = model.state_dict()

    for name, expected in reference.items():
        actual = current_state[name].detach().cpu()

        if actual.dtype.is_floating_point:
            difference = float(
                (actual - expected).abs().max().item()
            )
        else:
            difference = (
                0.0
                if torch.equal(actual, expected)
                else float("inf")
            )

        if difference > maximum_difference:
            maximum_difference = difference
            worst_key = name

    passed = maximum_difference == 0.0

    return passed, maximum_difference, worst_key


# ---------------------------------------------------------------------------
# Checkpoint / report writing
# ---------------------------------------------------------------------------


def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    source_checkpoint_path: Path,
    source_checkpoint: Mapping[str, Any],
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    best_validation_loss: float,
    args: argparse.Namespace,
    source_configuration: Mapping[str, Any],
) -> None:
    configuration = dict(source_configuration)

    # Keep source model architecture information intact while recording
    # the new optimization configuration.
    configuration.update(
        {
            "data_root": str(args.data_root.resolve()),
            "train_sequences": list(args.train_sequences),
            "validation_sequences": list(
                args.validation_sequences
            ),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "use_ground_truth_rotation": (
                args.use_ground_truth_rotation
            ),
        }
    )

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "configuration": configuration,
        "experiment": {
            "name": args.output_dir.name,
            "experiment_type": (
                "translation_head_dense_only_finetune"
            ),
            "source_checkpoint": str(
                source_checkpoint_path.resolve()
            ),
            "source_epoch": source_checkpoint.get(
                "epoch"
            ),
            "source_best_validation_loss": (
                source_checkpoint.get(
                    "best_validation_loss"
                )
            ),
            "trainable_parameters": sorted(
                TRAINABLE_PARAMETER_NAMES
            ),
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(path)


def save_history(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_metrics(
    title: str,
    metrics: Mapping[str, float],
) -> None:
    print(title)
    print(
        f"  loss/MSE: {metrics['loss']:.8f}"
    )
    print(
        f"  RMSE:     {metrics['rmse']:.8f}"
    )
    print(
        "  axis RMSE:"
        f" x={metrics['rmse_x']:.8f}"
        f" y={metrics['rmse_y']:.8f}"
        f" z={metrics['rmse_z']:.8f}"
    )

    if "bias_x" in metrics:
        print(
            "  axis bias:"
            f" x={metrics['bias_x']:+.8f}"
            f" y={metrics['bias_y']:+.8f}"
            f" z={metrics['bias_z']:+.8f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = resolve_device(
        args.device
    )

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("DeepDCT-VO Translation-Head-Only Fine-Tuning")
    print("=" * 80)
    print(f"Source checkpoint:   {checkpoint_path}")
    print(f"Output directory:    {output_dir}")
    print(f"Device:              {device}")
    print(f"Training sequences:  {args.train_sequences}")
    print(
        f"Validation sequences:{args.validation_sequences}"
    )
    print(f"Epochs:              {args.epochs}")
    print(f"Learning rate:       {args.learning_rate:.8g}")
    print(
        "GT rotation cond.:   "
        f"{args.use_ground_truth_rotation}"
    )
    print("=" * 80)

    source_checkpoint = load_source_checkpoint(
        checkpoint_path,
        device,
    )

    print(
        "Source epoch:         "
        f"{source_checkpoint.get('epoch')}"
    )
    print(
        "Source best val loss: "
        f"{source_checkpoint.get('best_validation_loss')}"
    )

    model, source_configuration = (
        build_model_from_checkpoint(
            source_checkpoint,
            device,
        )
    )

    trainable_parameters = (
        configure_translation_dense_only(
            model
        )
    )

    print_trainable_parameter_audit(
        model
    )

    # Snapshot all frozen parameters AND buffers before any training.
    frozen_reference = clone_frozen_state(
        model
    )

    training_dataset = build_dataset(
        data_root=args.data_root,
        sequences=args.train_sequences,
        configuration=source_configuration,
    )

    validation_dataset = build_dataset(
        data_root=args.data_root,
        sequences=args.validation_sequences,
        configuration=source_configuration,
    )

    training_loader = build_dataloader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
        seed=args.seed,
    )

    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
        seed=args.seed,
    )

    print(
        f"Training samples:     {len(training_dataset)}"
    )
    print(
        f"Validation samples:   {len(validation_dataset)}"
    )

    optimizer = Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    criterion = nn.MSELoss(
        reduction="mean"
    )

    use_internal_depth = bool(
        source_configuration.get(
            "use_depth_cues",
            False,
        )
    )

    # ------------------------------------------------------------------
    # Evaluate the untouched source checkpoint on validation sequence 09.
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Pre-fine-tuning validation baseline")
    print("=" * 80)

    initial_validation = evaluate_translation(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
        use_ground_truth_rotation=(
            args.use_ground_truth_rotation
        ),
        use_internal_depth=use_internal_depth,
    )

    print_metrics(
        "Source checkpoint:",
        initial_validation,
    )

    # Best model is selected only from post-fine-tuning epochs.
    best_validation_loss = float("inf")
    best_epoch: Optional[int] = None

    history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print()
        print("=" * 80)
        print(
            f"Fine-tuning epoch {epoch}/{args.epochs}"
        )
        print("=" * 80)

        maximum_gradient_norm: Optional[float]

        if args.max_grad_norm > 0:
            maximum_gradient_norm = (
                args.max_grad_norm
            )
        else:
            maximum_gradient_norm = None

        train_metrics = (
            train_one_epoch_head_only(
                model=model,
                dataloader=training_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                use_ground_truth_rotation=(
                    args.use_ground_truth_rotation
                ),
                use_internal_depth=use_internal_depth,
                max_grad_norm=maximum_gradient_norm,
                log_interval=args.log_interval,
            )
        )

        validation_metrics = evaluate_translation(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            use_ground_truth_rotation=(
                args.use_ground_truth_rotation
            ),
            use_internal_depth=use_internal_depth,
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        is_best = (
            validation_metrics["loss"]
            < best_validation_loss
        )

        if is_best:
            best_validation_loss = float(
                validation_metrics["loss"]
            )
            best_epoch = epoch

        print_metrics(
            "Training:",
            train_metrics,
        )

        print_metrics(
            "Validation:",
            validation_metrics,
        )

        print(
            f"Learning rate: {current_lr:.8g}"
        )

        if is_best:
            print("Status:        NEW BEST")

        history_row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_rmse_x": train_metrics["rmse_x"],
            "train_rmse_y": train_metrics["rmse_y"],
            "train_rmse_z": train_metrics["rmse_z"],
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_rmse": (
                validation_metrics["rmse"]
            ),
            "validation_rmse_x": (
                validation_metrics["rmse_x"]
            ),
            "validation_rmse_y": (
                validation_metrics["rmse_y"]
            ),
            "validation_rmse_z": (
                validation_metrics["rmse_z"]
            ),
            "validation_bias_x": (
                validation_metrics["bias_x"]
            ),
            "validation_bias_y": (
                validation_metrics["bias_y"]
            ),
            "validation_bias_z": (
                validation_metrics["bias_z"]
            ),
            "is_best": bool(is_best),
        }

        history.append(
            history_row
        )

        latest_path = (
            output_dir / "latest.pt"
        )

        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            source_checkpoint_path=checkpoint_path,
            source_checkpoint=source_checkpoint,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_validation_loss=best_validation_loss,
            args=args,
            source_configuration=source_configuration,
        )

        if args.save_every_epoch:
            epoch_path = (
                output_dir
                / (
                    "translation_head_ft_epoch_"
                    f"{epoch:03d}.pt"
                )
            )

            shutil.copy2(
                latest_path,
                epoch_path,
            )

        if is_best:
            best_path = (
                output_dir
                / "best_validation.pt"
            )

            shutil.copy2(
                latest_path,
                best_path,
            )

        save_history(
            output_dir / "history.csv",
            history,
        )

    # ------------------------------------------------------------------
    # Strict frozen-model audit
    # ------------------------------------------------------------------

    (
        frozen_audit_passed,
        maximum_frozen_difference,
        worst_frozen_key,
    ) = audit_frozen_state(
        model,
        frozen_reference,
    )

    print()
    print("=" * 80)
    print("Frozen-state audit")
    print("=" * 80)
    print(
        "Status:                 "
        f"{'PASS' if frozen_audit_passed else 'FAIL'}"
    )
    print(
        "Maximum frozen diff:    "
        f"{maximum_frozen_difference:.12e}"
    )
    print(
        "Worst frozen key:       "
        f"{worst_frozen_key}"
    )
    print("=" * 80)

    if not frozen_audit_passed:
        raise RuntimeError(
            "Frozen-state audit FAILED. At least one parameter or "
            "buffer outside translation_head.dense changed."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    final_best_checkpoint = (
        output_dir / "best_validation.pt"
    )

    summary = {
        "source_checkpoint": str(
            checkpoint_path
        ),
        "source_epoch": source_checkpoint.get(
            "epoch"
        ),
        "source_best_validation_loss": (
            source_checkpoint.get(
                "best_validation_loss"
            )
        ),
        "experiment_type": (
            "translation_head_dense_only_finetune"
        ),
        "trainable_parameters": sorted(
            TRAINABLE_PARAMETER_NAMES
        ),
        "training_sequences": list(
            args.train_sequences
        ),
        "validation_sequences": list(
            args.validation_sequences
        ),
        "held_out_test_sequence": "10",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "use_ground_truth_rotation": (
            args.use_ground_truth_rotation
        ),
        "initial_validation": (
            initial_validation
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_checkpoint": str(
            final_best_checkpoint
        ),
        "frozen_state_audit": {
            "status": (
                "PASS"
                if frozen_audit_passed
                else "FAIL"
            ),
            "maximum_absolute_difference": (
                maximum_frozen_difference
            ),
            "worst_key": worst_frozen_key,
        },
    }

    save_json(
        output_dir / "summary.json",
        summary,
    )

    print()
    print("=" * 80)
    print("Fine-tuning complete")
    print("=" * 80)
    print(
        f"Best epoch:             {best_epoch}"
    )
    print(
        "Initial validation MSE: "
        f"{initial_validation['loss']:.8f}"
    )
    print(
        "Best validation MSE:    "
        f"{best_validation_loss:.8f}"
    )
    print(
        f"Best checkpoint:        {final_best_checkpoint}"
    )
    print(
        "Frozen-state audit:     PASS"
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())