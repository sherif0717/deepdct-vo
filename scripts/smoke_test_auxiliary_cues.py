#!/usr/bin/env python3
"""Smoke-test DeepDCT-VO semantic and depth cues on real KITTI samples.

This script performs a no-gradient diagnostic forward pass through the
complete DeepDCT-VO model.

It implements the following checks:

1. Construct a real DeepDCTTrainingDataset.
2. Retrieve one or more samples through a DataLoader.
3. Construct DeepDCTVO with selected cue switches.
4. Load pretrained LR-ASPP and/or Lite-Mono weights.
5. Place the model and frozen auxiliary branches in evaluation mode.
6. Run the model under torch.no_grad().
7. Request and inspect intermediate tensors.
8. Print shapes, statistics, timing, and device-memory information.
9. Optionally save RGB, semantic, and depth visualizations.

Examples
--------
Semantic cues only:

    python3 scripts/smoke_test_auxiliary_cues.py \
        --sequence 00 \
        --num-samples 8 \
        --use-semantic-cues \
        --save-visualizations

Depth cues only:

    python3 scripts/smoke_test_auxiliary_cues.py \
        --sequence 00 \
        --num-samples 8 \
        --use-depth-cues \
        --depth-checkpoint-dir weights/lite-mono-tiny-640x192 \
        --save-visualizations

Semantic and depth cues:

    python3 scripts/smoke_test_auxiliary_cues.py \
        --sequence 00 \
        --num-samples 8 \
        --use-semantic-cues \
        --use-depth-cues \
        --depth-checkpoint-dir weights/lite-mono-tiny-640x192 \
        --save-visualizations

Ground-truth rotation conditioning:

    python3 scripts/smoke_test_auxiliary_cues.py \
        --sequence 00 \
        --num-samples 8 \
        --use-semantic-cues \
        --use-depth-cues \
        --use-ground-truth-rotation
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from deepdct.data.training_dataset import DeepDCTTrainingDataset
from deepdct.models.deepdct_vo import DeepDCTVO


# ---------------------------------------------------------------------------
# Command-line configuration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse smoke-test command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run no-gradient semantic/depth cue diagnostics through "
            "DeepDCT-VO."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=(
            "Root containing KITTI sequences/, out_csv/, and poses/."
        ),
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default="00",
        help="Single KITTI odometry sequence to inspect.",
    )

    parser.add_argument(
        "--camera",
        choices=[
            "left",
            "right",
            "image_2",
            "image_3",
        ],
        default="left",
        help="KITTI camera stream.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="DeepDCT-VO input image height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="DeepDCT-VO input image width.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Smoke-test DataLoader batch size.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help=(
            "Maximum number of individual samples to process. "
            "The final batch may be truncated."
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
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="PyTorch device, for example cpu, cuda, or cuda:0.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--use-semantic-cues",
        action="store_true",
        help=(
            "Run the LR-ASPP semantic branch. When omitted, the model "
            "should use zero semantic maps."
        ),
    )

    parser.add_argument(
        "--use-depth-cues",
        action="store_true",
        help=(
            "Run the internal Lite-Mono branch. When omitted, the model "
            "should use a zero depth map."
        ),
    )

    parser.add_argument(
        "--pretrained-semantic",
        dest="pretrained_semantic",
        action="store_true",
        help="Load pretrained LR-ASPP weights.",
    )

    parser.add_argument(
        "--no-pretrained-semantic",
        dest="pretrained_semantic",
        action="store_false",
        help="Do not load pretrained LR-ASPP weights.",
    )

    parser.set_defaults(
        pretrained_semantic=True,
    )

    parser.add_argument(
        "--freeze-semantic",
        dest="freeze_semantic",
        action="store_true",
        help="Freeze the LR-ASPP semantic branch.",
    )

    parser.add_argument(
        "--no-freeze-semantic",
        dest="freeze_semantic",
        action="store_false",
        help="Leave the LR-ASPP branch trainable.",
    )

    parser.set_defaults(
        freeze_semantic=True,
    )

    parser.add_argument(
        "--depth-checkpoint-dir",
        type=Path,
        default=Path(
            "weights/lite-mono-tiny-640x192"
        ),
        help=(
            "Directory containing Lite-Mono encoder.pth and depth.pth."
        ),
    )

    parser.add_argument(
        "--depth-model-name",
        choices=[
            "lite-mono",
            "lite-mono-small",
            "lite-mono-tiny",
            "lite-mono-8m",
        ],
        default="lite-mono-tiny",
        help="Lite-Mono model variant.",
    )

    parser.add_argument(
        "--depth-output-mode",
        choices=[
            "normalized_depth",
            "depth",
            "disparity",
            "scaled_disparity",
        ],
        default="normalized_depth",
        help="Representation returned by the depth branch.",
    )

    parser.add_argument(
        "--freeze-depth",
        dest="freeze_depth",
        action="store_true",
        help="Freeze the Lite-Mono depth branch.",
    )

    parser.add_argument(
        "--no-freeze-depth",
        dest="freeze_depth",
        action="store_false",
        help="Leave the Lite-Mono depth branch trainable.",
    )

    parser.set_defaults(
        freeze_depth=True,
    )

    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Condition Model T on the dataset rotation target instead "
            "of Model R's prediction."
        ),
    )

    parser.add_argument(
        "--share-aresunet-between-models",
        action="store_true",
        help=(
            "Use one shared A-ResUNet for Models R and T. The default "
            "uses separate branches."
        ),
    )

    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save RGB, semantic, and depth PNG images.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/auxiliary_cue_smoke_test"
        ),
        help="Directory for diagnostic images.",
    )

    parser.add_argument(
        "--fail-on-constant-cue",
        action="store_true",
        help=(
            "Fail if an enabled semantic or depth cue has zero spatial "
            "standard deviation."
        ),
    )

    parser.add_argument(
        "--minimum-cue-std",
        type=float,
        default=1.0e-8,
        help=(
            "Minimum accepted spatial standard deviation for an "
            "enabled cue when --fail-on-constant-cue is used."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sequence_id(sequence: str) -> str:
    """Convert a sequence value such as '0' to KITTI form '00'."""

    stripped = sequence.strip()

    if not stripped:
        raise ValueError("Sequence ID cannot be empty.")

    try:
        sequence_number = int(stripped)
    except ValueError as error:
        raise ValueError(
            "Sequence must be an integer-like KITTI ID, "
            f"but received {sequence!r}."
        ) from error

    if sequence_number < 0:
        raise ValueError(
            "Sequence ID cannot be negative."
        )

    return f"{sequence_number:02d}"


def resolve_device(device_name: str) -> torch.device:
    """Resolve and validate the requested PyTorch device."""

    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise ValueError(
            f"Invalid device specification: {device_name!r}"
        ) from error

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was requested, but CUDA is not "
                "available to PyTorch."
            )

        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device index {device.index} does not exist. "
                    f"Available CUDA devices: "
                    f"{torch.cuda.device_count()}."
                )

    return device


def format_bytes(num_bytes: int) -> str:
    """Format a byte count using binary units."""

    value = float(num_bytes)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PiB"


def synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA when required for accurate timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_tensor_to_device(
    value: Tensor,
    device: torch.device,
) -> Tensor:
    """Move a tensor to the selected device."""

    return value.to(
        device=device,
        non_blocking=device.type == "cuda",
    )


def find_tensor(
    batch: Dict[str, object],
    candidate_keys: Sequence[str],
    *,
    required: bool,
) -> Optional[Tensor]:
    """Find a tensor in a batch using a list of possible key names."""

    for key in candidate_keys:
        value = batch.get(key)

        if torch.is_tensor(value):
            return value

    if required:
        raise KeyError(
            "Could not find a required tensor. Tried batch keys: "
            f"{list(candidate_keys)}. Available keys: "
            f"{sorted(batch.keys())}."
        )

    return None


# ---------------------------------------------------------------------------
# Dataset and model construction
# ---------------------------------------------------------------------------


def validate_paths(args: argparse.Namespace) -> None:
    """Validate important data and checkpoint paths."""

    if not args.data_root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {args.data_root}"
        )

    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            "--height and --width must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.num_samples <= 0:
        raise ValueError(
            "--num-samples must be positive."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if args.minimum_cue_std < 0.0:
        raise ValueError(
            "--minimum-cue-std cannot be negative."
        )

    if (
        args.use_semantic_cues
        and args.freeze_semantic
        and not args.pretrained_semantic
    ):
        raise ValueError(
            "A randomly initialized semantic model should not be "
            "frozen for a meaningful cue test. Use either:\n"
            "  --pretrained-semantic --freeze-semantic\n"
            "or:\n"
            "  --no-pretrained-semantic --no-freeze-semantic"
        )

    if args.use_depth_cues:
        if not args.depth_checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "Lite-Mono checkpoint directory does not exist: "
                f"{args.depth_checkpoint_dir}"
            )

        encoder_path = (
            args.depth_checkpoint_dir / "encoder.pth"
        )
        decoder_path = (
            args.depth_checkpoint_dir / "depth.pth"
        )

        missing_paths = [
            path
            for path in (encoder_path, decoder_path)
            if not path.is_file()
        ]

        if missing_paths:
            formatted = "\n".join(
                f"  - {path}"
                for path in missing_paths
            )
            raise FileNotFoundError(
                "Required Lite-Mono checkpoint files are missing:\n"
                f"{formatted}"
            )


def build_dataset(
    args: argparse.Namespace,
    sequence: str,
) -> DeepDCTTrainingDataset:
    """Build the real KITTI training dataset for one sequence."""

    return DeepDCTTrainingDataset(
        data_root=args.data_root,
        sequences=[sequence],
        camera=args.camera,
        image_size=(args.height, args.width),
        # The dataset may return a zero depth placeholder, but this
        # script deliberately does not pass it to the model when the
        # internal depth branch is being tested.
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=True,
    )


def build_dataloader(
    dataset: DeepDCTTrainingDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    """Build a deterministic smoke-test DataLoader."""

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
    args: argparse.Namespace,
    device: torch.device,
) -> DeepDCTVO:
    """Construct DeepDCTVO for the selected cue experiment."""

    depth_checkpoint_dir: Optional[str]

    if args.use_depth_cues:
        depth_checkpoint_dir = str(
            args.depth_checkpoint_dir
        )
    else:
        # Avoid loading Lite-Mono checkpoints during depth-disabled
        # ablation runs.
        depth_checkpoint_dir = None

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(args.height, args.width),
        pretrained_semantic=args.pretrained_semantic,
        freeze_semantic=args.freeze_semantic,
        normalize_semantic_input=True,
        normalize_semantic_map=True,
        depth_checkpoint_dir=depth_checkpoint_dir,
        depth_model_name=args.depth_model_name,
        depth_output_mode=args.depth_output_mode,
        freeze_depth=args.freeze_depth,
        share_aresunet_between_models=(
            args.share_aresunet_between_models
        ),
        # These arguments assume the explicit cue switches have been
        # added to DeepDCTVO.__init__ as previously recommended.
        use_semantic_cues=args.use_semantic_cues,
        use_depth_cues=args.use_depth_cues,
    )

    model = model.to(device)
    model.eval()

    # Calling model.eval() should cover the complete tree. These
    # explicit calls also document and enforce the intended state.
    if hasattr(model, "semantic_model"):
        model.semantic_model.eval()

    if hasattr(model, "depth_model"):
        model.depth_model.eval()

    return model


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------


def tensor_statistics(tensor: Tensor) -> Dict[str, object]:
    """Return shape and numerical statistics for a tensor."""

    detached = tensor.detach()

    finite_mask = torch.isfinite(detached)
    all_finite = bool(finite_mask.all().item())

    statistics: Dict[str, object] = {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "requires_grad": detached.requires_grad,
        "all_finite": all_finite,
        "numel": detached.numel(),
    }

    if detached.numel() == 0:
        statistics.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }
        )
        return statistics

    if all_finite:
        float_tensor = detached.float()

        statistics.update(
            {
                "min": float(
                    float_tensor.min().item()
                ),
                "max": float(
                    float_tensor.max().item()
                ),
                "mean": float(
                    float_tensor.mean().item()
                ),
                # unbiased=False avoids NaN for one-element tensors.
                "std": float(
                    float_tensor.std(
                        unbiased=False
                    ).item()
                ),
            }
        )
    else:
        statistics.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }
        )

    return statistics


def print_tensor_statistics(
    name: str,
    tensor: Tensor,
) -> None:
    """Print one tensor's shape and numerical statistics."""

    stats = tensor_statistics(tensor)

    print(
        f"{name:30s} "
        f"shape={str(stats['shape']):18s} "
        f"dtype={stats['dtype']:16s} "
        f"min={stats['min']: .6f} "
        f"max={stats['max']: .6f} "
        f"mean={stats['mean']: .6f} "
        f"std={stats['std']: .6f} "
        f"finite={stats['all_finite']} "
        f"grad={stats['requires_grad']}"
    )


def require_tensor(
    outputs: Dict[str, object],
    key: str,
) -> Tensor:
    """Retrieve a required tensor from model outputs."""

    value = outputs.get(key)

    if not torch.is_tensor(value):
        raise KeyError(
            f"Model output {key!r} is missing or is not a tensor. "
            f"Available output keys: {sorted(outputs.keys())}."
        )

    return value


def validate_output_shapes(
    outputs: Dict[str, object],
    image_prev: Tensor,
    image_curr: Tensor,
) -> None:
    """Check the expected DeepDCT-VO intermediate shapes."""

    batch_size = image_prev.shape[0]
    height, width = image_curr.shape[-2:]

    semantic_prev = require_tensor(
        outputs,
        "semantic_prev",
    )
    semantic_curr = require_tensor(
        outputs,
        "semantic_curr",
    )
    depth_curr = require_tensor(
        outputs,
        "depth_curr",
    )
    ci_prev = require_tensor(
        outputs,
        "ci_prev",
    )
    ci_curr = require_tensor(
        outputs,
        "ci_curr",
    )
    rotation = require_tensor(
        outputs,
        "rotation",
    )
    translation = require_tensor(
        outputs,
        "directional_translation",
    )
    rotation_used = require_tensor(
        outputs,
        "rotation_used_for_translation",
    )
    rotation_map = require_tensor(
        outputs,
        "rotation_map",
    )

    expected_single_channel = (
        batch_size,
        1,
        height,
        width,
    )

    expected_ci = (
        batch_size,
        4,
        height,
        width,
    )

    expected_motion = (
        batch_size,
        3,
    )

    expected_rotation_map = (
        batch_size,
        3,
        height,
        width,
    )

    expected_shapes = {
        "semantic_prev": (
            semantic_prev,
            expected_single_channel,
        ),
        "semantic_curr": (
            semantic_curr,
            expected_single_channel,
        ),
        "depth_curr": (
            depth_curr,
            expected_single_channel,
        ),
        "ci_prev": (
            ci_prev,
            expected_ci,
        ),
        "ci_curr": (
            ci_curr,
            expected_ci,
        ),
        "rotation": (
            rotation,
            expected_motion,
        ),
        "directional_translation": (
            translation,
            expected_motion,
        ),
        "rotation_used_for_translation": (
            rotation_used,
            expected_motion,
        ),
        "rotation_map": (
            rotation_map,
            expected_rotation_map,
        ),
    }

    for name, (
        tensor,
        expected_shape,
    ) in expected_shapes.items():
        actual_shape = tuple(tensor.shape)

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name} has shape {actual_shape}; "
                f"expected {expected_shape}."
            )


def validate_finite_outputs(
    outputs: Dict[str, object],
    tensor_keys: Iterable[str],
) -> None:
    """Fail if any requested output tensor contains NaN or infinity."""

    for key in tensor_keys:
        tensor = require_tensor(outputs, key)

        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"Model output {key!r} contains non-finite values."
            )


def validate_no_gradient_graph(
    outputs: Dict[str, object],
    tensor_keys: Iterable[str],
) -> None:
    """Verify that no output has an attached autograd graph."""

    for key in tensor_keys:
        tensor = require_tensor(outputs, key)

        if tensor.requires_grad:
            raise RuntimeError(
                f"Output {key!r} unexpectedly requires gradients. "
                "The smoke forward must run inside torch.no_grad()."
            )

        if tensor.grad_fn is not None:
            raise RuntimeError(
                f"Output {key!r} has grad_fn={tensor.grad_fn}. "
                "The no-gradient smoke test is not isolated correctly."
            )


def validate_cue_behavior(
    args: argparse.Namespace,
    outputs: Dict[str, object],
) -> None:
    """Validate enabled and disabled cue behavior."""

    semantic_prev = require_tensor(
        outputs,
        "semantic_prev",
    )
    semantic_curr = require_tensor(
        outputs,
        "semantic_curr",
    )
    depth_curr = require_tensor(
        outputs,
        "depth_curr",
    )

    if args.use_semantic_cues:
        if not torch.isfinite(semantic_prev).all():
            raise FloatingPointError(
                "semantic_prev contains non-finite values."
            )

        if not torch.isfinite(semantic_curr).all():
            raise FloatingPointError(
                "semantic_curr contains non-finite values."
            )

        # Do not fail merely because one frame has a spatially constant
        # semantic map. Hard argmax maps can legitimately be constant
        # when the model predicts one class over the full image.
    else:
        if torch.count_nonzero(semantic_prev).item() != 0:
            raise RuntimeError(
                "Semantic cues are disabled, but semantic_prev "
                "is not a zero tensor."
            )

        if torch.count_nonzero(semantic_curr).item() != 0:
            raise RuntimeError(
                "Semantic cues are disabled, but semantic_curr "
                "is not a zero tensor."
            )

    if args.use_depth_cues:
        depth_std = float(
            depth_curr.float().std(
                unbiased=False
            ).item()
        )

        if (
            args.fail_on_constant_cue
            and depth_std <= args.minimum_cue_std
        ):
            raise RuntimeError(
                "The enabled depth cue is spatially constant. "
                f"Depth std={depth_std:.8e}."
            )
    else:
        if torch.count_nonzero(depth_curr).item() != 0:
            raise RuntimeError(
                "Depth cues are disabled, but depth_curr is not "
                "a zero tensor."
            )


def validate_frozen_auxiliary_parameters(
    args: argparse.Namespace,
    model: DeepDCTVO,
) -> None:
    """Confirm frozen auxiliary branches have no trainable parameters."""

    if args.freeze_semantic:
        trainable_semantic = [
            name
            for name, parameter
            in model.semantic_model.named_parameters()
            if parameter.requires_grad
        ]

        if trainable_semantic:
            raise RuntimeError(
                "The semantic branch was requested as frozen, but "
                "these parameters remain trainable:\n  "
                + "\n  ".join(
                    trainable_semantic[:20]
                )
            )

    if args.freeze_depth:
        trainable_depth = [
            name
            for name, parameter
            in model.depth_model.named_parameters()
            if parameter.requires_grad
        ]

        if trainable_depth:
            raise RuntimeError(
                "The depth branch was requested as frozen, but "
                "these parameters remain trainable:\n  "
                + "\n  ".join(
                    trainable_depth[:20]
                )
            )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def tensor_to_rgb_uint8(
    tensor: Tensor,
) -> np.ndarray:
    """Convert one CHW RGB tensor to an HWC uint8 image."""

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            "Expected one RGB tensor shaped [3, H, W], "
            f"but received {tuple(tensor.shape)}."
        )

    image = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )

    return np.round(image * 255.0).astype(
        np.uint8
    )


def tensor_to_grayscale_uint8(
    tensor: Tensor,
) -> np.ndarray:
    """Min-max normalize one one-channel tensor for PNG output."""

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.ndim != 2:
        raise ValueError(
            "Expected grayscale tensor shaped [1, H, W] or "
            f"[H, W], but received {tuple(tensor.shape)}."
        )

    image = tensor.detach().float().cpu()

    finite_mask = torch.isfinite(image)

    if not finite_mask.all():
        image = torch.where(
            finite_mask,
            image,
            torch.zeros_like(image),
        )

    minimum = image.min()
    maximum = image.max()
    denominator = maximum - minimum

    if float(denominator.item()) <= 1.0e-12:
        normalized = torch.zeros_like(image)
    else:
        normalized = (
            image - minimum
        ) / denominator

    array = normalized.numpy()

    return np.round(array * 255.0).astype(
        np.uint8
    )


def save_visualizations(
    output_dir: Path,
    global_sample_index: int,
    image_prev: Tensor,
    image_curr: Tensor,
    outputs: Dict[str, object],
) -> None:
    """Save input RGB, semantic, and depth maps for one sample."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = f"sample_{global_sample_index:06d}"

    semantic_prev = require_tensor(
        outputs,
        "semantic_prev",
    )
    semantic_curr = require_tensor(
        outputs,
        "semantic_curr",
    )
    depth_curr = require_tensor(
        outputs,
        "depth_curr",
    )

    files = {
        f"{stem}_rgb_prev.png": (
            "RGB",
            tensor_to_rgb_uint8(
                image_prev[0]
            ),
        ),
        f"{stem}_rgb_curr.png": (
            "RGB",
            tensor_to_rgb_uint8(
                image_curr[0]
            ),
        ),
        f"{stem}_semantic_prev.png": (
            "L",
            tensor_to_grayscale_uint8(
                semantic_prev[0]
            ),
        ),
        f"{stem}_semantic_curr.png": (
            "L",
            tensor_to_grayscale_uint8(
                semantic_curr[0]
            ),
        ),
        f"{stem}_depth_curr.png": (
            "L",
            tensor_to_grayscale_uint8(
                depth_curr[0]
            ),
        ),
    }

    for filename, (
        mode,
        array,
    ) in files.items():
        image = Image.fromarray(
            array,
            mode=mode,
        )
        image.save(
            output_dir / filename
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_model_summary(
    args: argparse.Namespace,
    sequence: str,
    device: torch.device,
    dataset: DeepDCTTrainingDataset,
    model: DeepDCTVO,
) -> None:
    """Print the resolved smoke-test configuration."""

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    semantic_parameters = sum(
        parameter.numel()
        for parameter
        in model.semantic_model.parameters()
    )

    depth_parameters = sum(
        parameter.numel()
        for parameter
        in model.depth_model.parameters()
    )

    print("=" * 88)
    print("DeepDCT-VO auxiliary-cue smoke test")
    print("=" * 88)
    print(
        f"Device:                         {device}"
    )
    print(
        f"Data root:                      "
        f"{args.data_root.resolve()}"
    )
    print(
        f"Sequence:                       {sequence}"
    )
    print(
        f"Camera:                         {args.camera}"
    )
    print(
        f"Dataset samples:                {len(dataset)}"
    )
    print(
        f"Samples requested:              {args.num_samples}"
    )
    print(
        f"Batch size:                     {args.batch_size}"
    )
    print(
        f"Input size:                     "
        f"{args.height} x {args.width}"
    )
    print(
        f"Semantic cues enabled:          "
        f"{args.use_semantic_cues}"
    )
    print(
        f"Semantic pretrained:            "
        f"{args.pretrained_semantic}"
    )
    print(
        f"Semantic frozen:                "
        f"{args.freeze_semantic}"
    )
    print(
        f"Depth cues enabled:             "
        f"{args.use_depth_cues}"
    )
    print(
        f"Depth checkpoint directory:     "
        f"{args.depth_checkpoint_dir}"
    )
    print(
        f"Depth output mode:              "
        f"{args.depth_output_mode}"
    )
    print(
        f"Depth frozen:                   "
        f"{args.freeze_depth}"
    )
    print(
        f"Ground-truth rotation for T:    "
        f"{args.use_ground_truth_rotation}"
    )
    print(
        f"Shared A-ResUNet:               "
        f"{args.share_aresunet_between_models}"
    )
    print(
        f"Total parameters:               "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters:           "
        f"{trainable_parameters:,}"
    )
    print(
        f"Semantic parameters:            "
        f"{semantic_parameters:,}"
    )
    print(
        f"Depth parameters:               "
        f"{depth_parameters:,}"
    )
    print(
        f"Save visualizations:            "
        f"{args.save_visualizations}"
    )

    if args.save_visualizations:
        print(
            f"Visualization directory:        "
            f"{args.output_dir.resolve()}"
        )

    print("=" * 88)


def print_batch_metadata(
    batch: Dict[str, object],
) -> None:
    """Print metadata fields without dumping full tensors."""

    metadata_items: List[str] = []

    excluded_keys = {
        "image_prev",
        "image_curr",
        "previous_image",
        "current_image",
        "rotation",
        "rotation_gt",
        "directional_translation",
        "translation",
        "depth_curr",
    }

    for key, value in batch.items():
        if key in excluded_keys:
            continue

        if torch.is_tensor(value):
            if value.numel() <= 8:
                metadata_items.append(
                    f"{key}={value.detach().cpu().tolist()}"
                )
        elif isinstance(
            value,
            (str, int, float, bool),
        ):
            metadata_items.append(
                f"{key}={value}"
            )
        elif isinstance(value, (list, tuple)):
            if len(value) <= 8:
                metadata_items.append(
                    f"{key}={value}"
                )

    if metadata_items:
        print(
            "Metadata: "
            + ", ".join(metadata_items)
        )


def print_cuda_memory(
    device: torch.device,
) -> None:
    """Print CUDA memory statistics when applicable."""

    if device.type != "cuda":
        print("CUDA memory:                    not applicable")
        return

    allocated = torch.cuda.memory_allocated(
        device
    )
    reserved = torch.cuda.memory_reserved(
        device
    )
    peak_allocated = (
        torch.cuda.max_memory_allocated(
            device
        )
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(
            device
        )
    )

    print(
        f"CUDA allocated:                 "
        f"{format_bytes(allocated)}"
    )
    print(
        f"CUDA reserved:                  "
        f"{format_bytes(reserved)}"
    )
    print(
        f"CUDA peak allocated:            "
        f"{format_bytes(peak_allocated)}"
    )
    print(
        f"CUDA peak reserved:             "
        f"{format_bytes(peak_reserved)}"
    )


# ---------------------------------------------------------------------------
# Smoke-test execution
# ---------------------------------------------------------------------------


def run_smoke_test(
    args: argparse.Namespace,
    sequence: str,
    device: torch.device,
    dataloader: DataLoader,
    model: DeepDCTVO,
) -> Tuple[int, float]:
    """Run no-gradient cue diagnostics over selected samples."""

    tensor_keys = (
        "semantic_prev",
        "semantic_curr",
        "depth_curr",
        "ci_prev",
        "ci_curr",
        "rotation_c_prev",
        "rotation_c_curr",
        "translation_c_prev",
        "translation_c_curr",
        "rotation_map",
        "rotation_features",
        "translation_features",
        "rotation",
        "directional_translation",
        "rotation_used_for_translation",
    )

    processed_samples = 0
    total_forward_seconds = 0.0
    semantic_maps_checked = 0
    semantic_maps_nonconstant = 0
    semantic_maps_nonzero = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    with torch.no_grad():
        for batch_index, batch in enumerate(
            dataloader
        ):
            remaining = (
                args.num_samples
                - processed_samples
            )

            if remaining <= 0:
                break

            image_prev = find_tensor(
                batch,
                (
                    "image_prev",
                    "previous_image",
                    "rgb_prev",
                ),
                required=True,
            )
            image_curr = find_tensor(
                batch,
                (
                    "image_curr",
                    "current_image",
                    "rgb_curr",
                ),
                required=True,
            )

            assert image_prev is not None
            assert image_curr is not None

            current_batch_size = min(
                image_prev.shape[0],
                remaining,
            )

            image_prev = image_prev[
                :current_batch_size
            ]
            image_curr = image_curr[
                :current_batch_size
            ]

            image_prev = move_tensor_to_device(
                image_prev,
                device,
            )
            image_curr = move_tensor_to_device(
                image_curr,
                device,
            )

            rotation_for_translation: Optional[
                Tensor
            ] = None

            if args.use_ground_truth_rotation:
                rotation_for_translation = find_tensor(
                    batch,
                    (
                        "rotation",
                        "rotation_gt",
                        "target_rotation",
                    ),
                    required=True,
                )

                assert (
                    rotation_for_translation
                    is not None
                )

                rotation_for_translation = (
                    rotation_for_translation[
                        :current_batch_size
                    ]
                )

                rotation_for_translation = (
                    move_tensor_to_device(
                        rotation_for_translation,
                        device,
                    )
                )

            # Deliberately pass depth_curr=None. This ensures:
            #
            # - --use-depth-cues runs the internal Lite-Mono model.
            # - depth-disabled mode exercises the model's zero-depth
            #   ablation behavior.
            synchronize_device(device)
            start_time = time.perf_counter()

            outputs = model(
                image_prev=image_prev,
                image_curr=image_curr,
                depth_curr=None,
                rotation_for_translation=(
                    rotation_for_translation
                ),
                use_ground_truth_rotation=(
                    args.use_ground_truth_rotation
                ),
                return_intermediates=True,
            )

            synchronize_device(device)
            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            total_forward_seconds += (
                elapsed_seconds
            )

            validate_output_shapes(
                outputs=outputs,
                image_prev=image_prev,
                image_curr=image_curr,
            )

            validate_finite_outputs(
                outputs=outputs,
                tensor_keys=tensor_keys,
            )

            validate_no_gradient_graph(
                outputs=outputs,
                tensor_keys=tensor_keys,
            )

            validate_cue_behavior(
                args=args,
                outputs=outputs,
            )

            if args.use_semantic_cues:
                semantic_prev = require_tensor(
                    outputs,
                    "semantic_prev",
                )
                semantic_curr = require_tensor(
                    outputs,
                    "semantic_curr",
                )

                for semantic_map in (
                    semantic_prev,
                    semantic_curr,
                ):
                    # Evaluate each sample independently.
                    for sample_map in semantic_map:
                        semantic_maps_checked += 1

                        sample_std = float(
                            sample_map.float().std(
                                unbiased=False
                            ).item()
                        )

                        if sample_std > args.minimum_cue_std:
                            semantic_maps_nonconstant += 1

                        if torch.count_nonzero(
                            sample_map
                        ).item() > 0:
                            semantic_maps_nonzero += 1

            print()
            print("-" * 88)
            print(
                f"Batch {batch_index:04d} | "
                f"sequence={sequence} | "
                f"samples={current_batch_size} | "
                f"forward={elapsed_seconds:.4f} s | "
                f"throughput="
                f"{current_batch_size / elapsed_seconds:.2f} samples/s"
            )
            print("-" * 88)

            print_batch_metadata(batch)

            print_tensor_statistics(
                "image_prev",
                image_prev,
            )
            print_tensor_statistics(
                "image_curr",
                image_curr,
            )
            print_tensor_statistics(
                "semantic_prev",
                require_tensor(
                    outputs,
                    "semantic_prev",
                ),
            )
            print_tensor_statistics(
                "semantic_curr",
                require_tensor(
                    outputs,
                    "semantic_curr",
                ),
            )
            print_tensor_statistics(
                "depth_curr",
                require_tensor(
                    outputs,
                    "depth_curr",
                ),
            )
            print_tensor_statistics(
                "ci_prev",
                require_tensor(
                    outputs,
                    "ci_prev",
                ),
            )
            print_tensor_statistics(
                "ci_curr",
                require_tensor(
                    outputs,
                    "ci_curr",
                ),
            )
            print_tensor_statistics(
                "rotation_c_prev",
                require_tensor(
                    outputs,
                    "rotation_c_prev",
                ),
            )
            print_tensor_statistics(
                "rotation_c_curr",
                require_tensor(
                    outputs,
                    "rotation_c_curr",
                ),
            )
            print_tensor_statistics(
                "translation_c_prev",
                require_tensor(
                    outputs,
                    "translation_c_prev",
                ),
            )
            print_tensor_statistics(
                "translation_c_curr",
                require_tensor(
                    outputs,
                    "translation_c_curr",
                ),
            )
            print_tensor_statistics(
                "rotation_features",
                require_tensor(
                    outputs,
                    "rotation_features",
                ),
            )
            print_tensor_statistics(
                "translation_features",
                require_tensor(
                    outputs,
                    "translation_features",
                ),
            )
            print_tensor_statistics(
                "rotation",
                require_tensor(
                    outputs,
                    "rotation",
                ),
            )
            print_tensor_statistics(
                "directional_translation",
                require_tensor(
                    outputs,
                    "directional_translation",
                ),
            )
            print_tensor_statistics(
                "rotation_used_for_translation",
                require_tensor(
                    outputs,
                    "rotation_used_for_translation",
                ),
            )

            if args.save_visualizations:
                # Save each individual sample in the batch.
                for local_index in range(
                    current_batch_size
                ):
                    sample_outputs: Dict[
                        str,
                        object
                    ] = {}

                    for key, value in outputs.items():
                        if (
                            torch.is_tensor(value)
                            and value.ndim > 0
                            and value.shape[0]
                            == current_batch_size
                        ):
                            sample_outputs[key] = (
                                value[
                                    local_index:
                                    local_index + 1
                                ]
                            )
                        else:
                            sample_outputs[key] = value

                    save_visualizations(
                        output_dir=args.output_dir,
                        global_sample_index=(
                            processed_samples
                            + local_index
                        ),
                        image_prev=(
                            image_prev[
                                local_index:
                                local_index + 1
                            ]
                        ),
                        image_curr=(
                            image_curr[
                                local_index:
                                local_index + 1
                            ]
                        ),
                        outputs=sample_outputs,
                    )

            processed_samples += current_batch_size

    return (
        processed_samples,
        total_forward_seconds,
        semantic_maps_checked,
        semantic_maps_nonconstant,
        semantic_maps_nonzero,
    )


def main() -> None:
    """Run the complete auxiliary-cue smoke test."""

    args = parse_args()

    sequence = normalize_sequence_id(
        args.sequence
    )
    device = resolve_device(
        args.device
    )

    seed_everything(
        args.seed
    )
    validate_paths(
        args
    )

    dataset = build_dataset(
        args=args,
        sequence=sequence,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Sequence {sequence} produced an empty dataset."
        )

    dataloader = build_dataloader(
        dataset=dataset,
        args=args,
        device=device,
    )

    model = build_model(
        args=args,
        device=device,
    )

    validate_frozen_auxiliary_parameters(
        args=args,
        model=model,
    )

    print_model_summary(
        args=args,
        sequence=sequence,
        device=device,
        dataset=dataset,
        model=model,
    )

    (
        processed_samples,
        total_seconds,
        semantic_maps_checked,
        semantic_maps_nonconstant,
        semantic_maps_nonzero,
    ) = (
        run_smoke_test(
            args=args,
            sequence=sequence,
            device=device,
            dataloader=dataloader,
            model=model,
        )
    )

    if (
        args.use_semantic_cues
        and args.fail_on_constant_cue
    ):
        if semantic_maps_checked == 0:
            raise RuntimeError(
                "Semantic cues were enabled, but no semantic maps "
                "were inspected."
            )

        if semantic_maps_nonconstant == 0:
            raise RuntimeError(
                "Every inspected semantic map was spatially constant."
            )

        if semantic_maps_nonzero == 0:
            raise RuntimeError(
                "Every inspected semantic map was identically zero."
            )

        print()
        print(
            "Semantic maps checked:          "
            f"{semantic_maps_checked}"
        )
        print(
            "Nonconstant semantic maps:      "
            f"{semantic_maps_nonconstant}"
        )
        print(
            "Nonzero semantic maps:           "
            f"{semantic_maps_nonzero}"
        )
    

    if processed_samples == 0:
        raise RuntimeError(
            "The DataLoader yielded no samples."
        )

    average_seconds = (
        total_seconds / processed_samples
    )
    throughput = (
        processed_samples / total_seconds
        if total_seconds > 0.0
        else float("inf")
    )

    print()
    print("=" * 88)
    print("Smoke test completed successfully")
    print("=" * 88)
    print(
        f"Samples processed:              "
        f"{processed_samples}"
    )
    print(
        f"Total forward time:             "
        f"{total_seconds:.4f} s"
    )
    print(
        f"Average time per sample:        "
        f"{average_seconds:.4f} s"
    )
    print(
        f"Average throughput:             "
        f"{throughput:.2f} samples/s"
    )
    print_cuda_memory(
        device
    )

    if args.save_visualizations:
        print(
            f"Visualizations saved to:        "
            f"{args.output_dir.resolve()}"
        )

    print(
        "Gradient graph attached:        False"
    )
    print(
        "Finite-value checks:            Passed"
    )
    print(
        "Intermediate-shape checks:      Passed"
    )
    print(
        "Cue enable/disable checks:      Passed"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()