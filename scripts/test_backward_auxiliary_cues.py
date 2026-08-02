#!/usr/bin/env python3
"""Test backward propagation with frozen DeepDCT-VO auxiliary models.

This diagnostic performs one real training step and verifies that:

1. Semantic and depth cues are generated successfully.
2. Frozen LR-ASPP and Lite-Mono parameters remain non-trainable.
3. Frozen auxiliary parameters receive no gradients.
4. Trainable pose-network parameters receive finite gradients.
5. At least one trainable pose-network gradient is nonzero.
6. The optimizer changes trainable pose parameters.
7. The optimizer does not change frozen auxiliary parameters.
8. Frozen auxiliary modules remain in evaluation mode after model.train().

Run this script from the repository root.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from deepdct.data.training_dataset import DeepDCTTrainingDataset
from deepdct.models.deepdct_vo import DeepDCTVO


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one DeepDCT-VO backward/optimizer step while verifying "
            "that frozen semantic and depth branches remain unchanged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing sequences/, poses/, and out_csv/.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default="00",
        help="KITTI sequence used for the diagnostic batch.",
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
        help="KITTI image stream.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Input image height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="Input image width.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training diagnostic batch size.",
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
        help="PyTorch device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-4,
        help="Learning rate for the single optimizer step.",
    )

    parser.add_argument(
        "--rotation-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to rotation MSE.",
    )

    parser.add_argument(
        "--translation-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to directional-translation MSE.",
    )

    parser.add_argument(
        "--use-semantic-cues",
        action="store_true",
        help="Enable the pretrained LR-ASPP semantic branch.",
    )

    parser.add_argument(
        "--use-depth-cues",
        action="store_true",
        help="Enable the pretrained Lite-Mono depth branch.",
    )

    parser.add_argument(
        "--depth-checkpoint-dir",
        "--depth-weights-dir",
        dest="depth_checkpoint_dir",
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
        help="Lite-Mono variant.",
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
        help="Depth representation supplied to Model R and Model T.",
    )

    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Condition Model T on target rotation rather than Model R's "
            "prediction. Omit this flag for the full end-to-end dependency."
        ),
    )

    parser.add_argument(
        "--share-aresunet-between-models",
        action="store_true",
        help="Share one A-ResUNet between Models R and T.",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed all relevant random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sequence_id(sequence: str) -> str:
    """Normalize a KITTI sequence ID to two digits."""

    value = sequence.strip()

    if not value:
        raise ValueError("Sequence cannot be empty.")

    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid KITTI sequence: {sequence!r}."
        ) from error

    if number < 0:
        raise ValueError("Sequence cannot be negative.")

    return f"{number:02d}"


def resolve_device(device_name: str) -> torch.device:
    """Resolve and validate a PyTorch device."""

    device = torch.device(device_name)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access CUDA."
            )

        if (
            device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"CUDA device {device.index} does not exist."
            )

    return device


def validate_paths(args: argparse.Namespace) -> None:
    """Validate dataset and checkpoint paths."""

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

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if args.learning_rate <= 0.0:
        raise ValueError(
            "--learning-rate must be positive."
        )

    if args.rotation_loss_weight < 0.0:
        raise ValueError(
            "--rotation-loss-weight cannot be negative."
        )

    if args.translation_loss_weight < 0.0:
        raise ValueError(
            "--translation-loss-weight cannot be negative."
        )

    if (
        args.rotation_loss_weight == 0.0
        and args.translation_loss_weight == 0.0
    ):
        raise ValueError(
            "At least one loss weight must be nonzero."
        )

    if args.use_depth_cues:
        if not args.depth_checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "Depth checkpoint directory does not exist: "
                f"{args.depth_checkpoint_dir}"
            )

        required_files = (
            args.depth_checkpoint_dir / "encoder.pth",
            args.depth_checkpoint_dir / "depth.pth",
        )

        missing = [
            path
            for path in required_files
            if not path.is_file()
        ]

        if missing:
            formatted = "\n".join(
                f"  - {path}"
                for path in missing
            )
            raise FileNotFoundError(
                "Missing Lite-Mono checkpoint files:\n"
                f"{formatted}"
            )


def find_tensor(
    mapping: Dict[str, object],
    candidate_keys: Sequence[str],
    *,
    required: bool = True,
) -> Optional[Tensor]:
    """Find the first tensor associated with a candidate key."""

    for key in candidate_keys:
        value = mapping.get(key)

        if torch.is_tensor(value):
            return value

    if required:
        raise KeyError(
            "Could not find a required tensor. Tried keys "
            f"{list(candidate_keys)}. Available keys: "
            f"{sorted(mapping.keys())}."
        )

    return None


def require_output_tensor(
    outputs: Dict[str, object],
    key: str,
) -> Tensor:
    """Retrieve a required model-output tensor."""

    value = outputs.get(key)

    if not torch.is_tensor(value):
        raise KeyError(
            f"Output {key!r} is missing or is not a tensor. "
            f"Available keys: {sorted(outputs.keys())}."
        )

    return value


def build_dataset(
    args: argparse.Namespace,
    sequence: str,
) -> DeepDCTTrainingDataset:
    """Construct the real KITTI training dataset."""

    return DeepDCTTrainingDataset(
        data_root=args.data_root,
        sequences=[sequence],
        camera=args.camera,
        image_size=(args.height, args.width),
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=True,
    )


def build_dataloader(
    dataset: DeepDCTTrainingDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    """Construct a deterministic one-batch DataLoader."""

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
    """Construct DeepDCT-VO with frozen auxiliary branches."""

    depth_checkpoint_dir: Optional[str]

    if args.use_depth_cues:
        depth_checkpoint_dir = str(
            args.depth_checkpoint_dir
        )
    else:
        depth_checkpoint_dir = None

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(args.height, args.width),
        pretrained_semantic=True,
        freeze_semantic=True,
        normalize_semantic_input=True,
        normalize_semantic_map=True,
        depth_checkpoint_dir=depth_checkpoint_dir,
        depth_model_name=args.depth_model_name,
        depth_output_mode=args.depth_output_mode,
        freeze_depth=True,
        share_aresunet_between_models=(
            args.share_aresunet_between_models
        ),
        use_semantic_cues=args.use_semantic_cues,
        use_depth_cues=args.use_depth_cues,
    )

    model = model.to(device)

    # Activate trainable pose modules. The DeepDCTVO.train()
    # override must keep frozen auxiliary models in eval mode.
    model.train()

    return model


def named_trainable_parameters(
    model: DeepDCTVO,
) -> List[Tuple[str, torch.nn.Parameter]]:
    """Return all trainable model parameters."""

    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def clone_named_parameters(
    named_parameters: Iterable[
        Tuple[str, torch.nn.Parameter]
    ],
) -> Dict[str, Tensor]:
    """Clone parameter values for before/after comparison."""

    return {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
    }


def validate_auxiliary_modes(
    model: DeepDCTVO,
) -> None:
    """Verify frozen auxiliaries remain in evaluation mode."""

    if model.semantic_model.training:
        raise RuntimeError(
            "semantic_model entered training mode after model.train(). "
            "Add or correct the DeepDCTVO.train() override."
        )

    if model.depth_model.training:
        raise RuntimeError(
            "depth_model entered training mode after model.train(). "
            "Add or correct the DeepDCTVO.train() override."
        )


def validate_auxiliary_frozen(
    model: DeepDCTVO,
) -> None:
    """Verify auxiliary parameters have requires_grad=False."""

    trainable_semantic = [
        name
        for name, parameter
        in model.semantic_model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_depth = [
        name
        for name, parameter
        in model.depth_model.named_parameters()
        if parameter.requires_grad
    ]

    if trainable_semantic:
        raise RuntimeError(
            "Semantic parameters remain trainable:\n  "
            + "\n  ".join(trainable_semantic[:20])
        )

    if trainable_depth:
        raise RuntimeError(
            "Depth parameters remain trainable:\n  "
            + "\n  ".join(trainable_depth[:20])
        )


def validate_loss(
    loss: Tensor,
    name: str,
) -> None:
    """Validate one scalar loss."""

    if loss.ndim != 0:
        raise RuntimeError(
            f"{name} must be scalar, but has shape "
            f"{tuple(loss.shape)}."
        )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"{name} is NaN or infinity."
        )


def validate_trainable_gradients(
    named_parameters: Iterable[
        Tuple[str, torch.nn.Parameter]
    ],
) -> Tuple[int, int, float]:
    """Validate gradients on trainable pose-network parameters."""

    parameters_with_gradient = 0
    parameters_with_nonzero_gradient = 0
    total_gradient_norm_squared = 0.0

    missing_gradient_names: List[str] = []
    nonfinite_gradient_names: List[str] = []

    for name, parameter in named_parameters:
        gradient = parameter.grad

        if gradient is None:
            missing_gradient_names.append(name)
            continue

        parameters_with_gradient += 1

        if not torch.isfinite(gradient).all():
            nonfinite_gradient_names.append(name)
            continue

        gradient_norm = float(
            gradient.detach().float().norm().item()
        )

        total_gradient_norm_squared += (
            gradient_norm * gradient_norm
        )

        if torch.count_nonzero(gradient).item() > 0:
            parameters_with_nonzero_gradient += 1

    if nonfinite_gradient_names:
        raise FloatingPointError(
            "Non-finite gradients were found in trainable parameters:\n  "
            + "\n  ".join(nonfinite_gradient_names[:20])
        )

    if parameters_with_gradient == 0:
        raise RuntimeError(
            "No trainable parameter received a gradient."
        )

    if parameters_with_nonzero_gradient == 0:
        raise RuntimeError(
            "All trainable gradients are zero."
        )

    # Some conditional branches may legitimately be unused, so missing
    # gradients are reported but do not automatically fail the test.
    if missing_gradient_names:
        print(
            "Trainable parameters without gradients: "
            f"{len(missing_gradient_names)}"
        )

        for name in missing_gradient_names[:10]:
            print(f"  - {name}")

    total_gradient_norm = (
        total_gradient_norm_squared ** 0.5
    )

    return (
        parameters_with_gradient,
        parameters_with_nonzero_gradient,
        total_gradient_norm,
    )


def validate_no_auxiliary_gradients(
    model: DeepDCTVO,
) -> None:
    """Verify frozen auxiliary parameters have no attached gradients."""

    unexpected_semantic_gradients = [
        name
        for name, parameter
        in model.semantic_model.named_parameters()
        if parameter.grad is not None
    ]

    unexpected_depth_gradients = [
        name
        for name, parameter
        in model.depth_model.named_parameters()
        if parameter.grad is not None
    ]

    if unexpected_semantic_gradients:
        raise RuntimeError(
            "Frozen semantic parameters received gradients:\n  "
            + "\n  ".join(
                unexpected_semantic_gradients[:20]
            )
        )

    if unexpected_depth_gradients:
        raise RuntimeError(
            "Frozen depth parameters received gradients:\n  "
            + "\n  ".join(
                unexpected_depth_gradients[:20]
            )
        )
    
def maximum_absolute_difference(
    first: Tensor,
    second: Tensor,
) -> float:
    """Return the maximum absolute elementwise difference."""

    return float(
        (
            first.detach()
            - second.detach()
        ).abs().max().item()
    )

def validate_rotation_conditioning_dependency(
    model: DeepDCTVO,
    image_prev: Tensor,
    image_curr: Tensor,
    rotation_gt: Tensor,
) -> Tuple[float, float]:
    """Verify that Model T responds to its rotation-conditioning input."""

    model.eval()

    with torch.no_grad():
        predicted_conditioning_outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            rotation_for_translation=None,
            use_ground_truth_rotation=False,
            return_intermediates=True,
        )

        ground_truth_conditioning_outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            rotation_for_translation=rotation_gt,
            use_ground_truth_rotation=True,
            return_intermediates=True,
        )

    predicted_conditioning_translation = require_output_tensor(
        predicted_conditioning_outputs,
        "directional_translation",
    )

    ground_truth_conditioning_translation = require_output_tensor(
        ground_truth_conditioning_outputs,
        "directional_translation",
    )

    predicted_rotation_used = require_output_tensor(
        predicted_conditioning_outputs,
        "rotation_used_for_translation",
    )

    ground_truth_rotation_used = require_output_tensor(
        ground_truth_conditioning_outputs,
        "rotation_used_for_translation",
    )

    rotation_condition_difference = maximum_absolute_difference(
        predicted_rotation_used,
        ground_truth_rotation_used,
    )

    translation_condition_difference = maximum_absolute_difference(
        predicted_conditioning_translation,
        ground_truth_conditioning_translation,
    )

    if rotation_condition_difference <= 0.0:
        raise RuntimeError(
            "Predicted-rotation and GT-rotation conditioning supplied "
            "identical rotation values."
        )

    if translation_condition_difference <= 1.0e-12:
        raise RuntimeError(
            "Changing the rotation used by Model T did not change the "
            "translation prediction. Check rotation-map construction "
            "and translation-feature concatenation."
        )

    print(
        "Rotation-conditioning input difference: "
        f"{rotation_condition_difference:.16e}"
    )
    print(
        "Translation-output difference:           "
        f"{translation_condition_difference:.16e}"
    )

    # Restore training mode. Frozen auxiliaries should remain in eval.
    model.train()
    validate_auxiliary_modes(model)

    return (
        rotation_condition_difference,
        translation_condition_difference,
    )


def count_changed_parameters(
    before: Dict[str, Tensor],
    named_parameters: Iterable[
        Tuple[str, torch.nn.Parameter]
    ],
) -> Tuple[int, float]:
    """Count parameters changed by the optimizer step."""

    changed_count = 0
    largest_change = 0.0

    for name, parameter in named_parameters:
        if name not in before:
            raise KeyError(
                f"Missing before-step snapshot for {name!r}."
            )

        difference = (
            parameter.detach() - before[name]
        ).abs()

        maximum_change = float(
            difference.max().item()
        )

        if maximum_change > 0.0:
            changed_count += 1
            largest_change = max(
                largest_change,
                maximum_change,
            )

    return changed_count, largest_change


def validate_unchanged_parameters(
    before: Dict[str, Tensor],
    named_parameters: Iterable[
        Tuple[str, torch.nn.Parameter]
    ],
    branch_name: str,
) -> None:
    """Verify frozen parameter values did not change."""

    changed_names: List[str] = []

    for name, parameter in named_parameters:
        if name not in before:
            raise KeyError(
                f"Missing snapshot for {branch_name}.{name}."
            )

        if not torch.equal(
            before[name],
            parameter.detach(),
        ):
            changed_names.append(name)

    if changed_names:
        raise RuntimeError(
            f"Frozen {branch_name} parameters changed after "
            "optimizer.step():\n  "
            + "\n  ".join(changed_names[:20])
        )
    
def find_semantically_active_batch(
    dataloader: DataLoader,
    model: DeepDCTVO,
    device: torch.device,
    *,
    maximum_batches: int = 32,
    minimum_std: float = 1.0e-8,
) -> Dict[str, object]:
    """Find a batch for which LR-ASPP produces a nonconstant map."""

    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= maximum_batches:
                break

            image_prev = find_tensor(
                batch,
                (
                    "image_prev",
                    "previous_image",
                    "rgb_prev",
                ),
            )
            image_curr = find_tensor(
                batch,
                (
                    "image_curr",
                    "current_image",
                    "rgb_curr",
                ),
            )

            assert image_prev is not None
            assert image_curr is not None

            image_prev = image_prev.to(device)
            image_curr = image_curr.to(device)

            outputs = model(
                image_prev=image_prev,
                image_curr=image_curr,
                depth_curr=None,
                rotation_for_translation=None,
                use_ground_truth_rotation=False,
                return_intermediates=True,
            )

            semantic_prev = require_output_tensor(
                outputs,
                "semantic_prev",
            )
            semantic_curr = require_output_tensor(
                outputs,
                "semantic_curr",
            )

            previous_std = float(
                semantic_prev.float().std(
                    unbiased=False
                ).item()
            )
            current_std = float(
                semantic_curr.float().std(
                    unbiased=False
                ).item()
            )

            if (
                previous_std > minimum_std
                or current_std > minimum_std
            ):
                print(
                    "Selected semantically active batch: "
                    f"{batch_index}"
                )
                print(
                    "Semantic previous std: "
                    f"{previous_std:.9f}"
                )
                print(
                    "Semantic current std:  "
                    f"{current_std:.9f}"
                )

                model.train()
                return batch

    model.train()

    raise RuntimeError(
        "No semantically active batch was found within "
        f"{maximum_batches} batches."
    )


def main() -> None:
    """Run one complete backward-pass training diagnostic."""

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

    validate_auxiliary_frozen(model)
    validate_auxiliary_modes(model)

    trainable_parameters = named_trainable_parameters(
        model
    )

    if not trainable_parameters:
        raise RuntimeError(
            "DeepDCTVO has no trainable parameters."
        )

    optimizer = torch.optim.Adam(
        [
            parameter
            for _, parameter in trainable_parameters
        ],
        lr=args.learning_rate,
    )

    if args.use_semantic_cues:
        batch = find_semantically_active_batch(
            dataloader=dataloader,
            model=model,
            device=device,
        )
    else:
        batch = next(iter(dataloader))

    image_prev = find_tensor(
        batch,
        (
            "image_prev",
            "previous_image",
            "rgb_prev",
        ),
    )
    image_curr = find_tensor(
        batch,
        (
            "image_curr",
            "current_image",
            "rgb_curr",
        ),
    )
    rotation_gt = find_tensor(
        batch,
        (
            "rotation",
            "rotation_gt",
            "target_rotation",
        ),
    )
    translation_gt = find_tensor(
        batch,
        (
            "directional_translation",
            "translation",
            "translation_gt",
            "target_translation",
        ),
    )

    assert image_prev is not None
    assert image_curr is not None
    assert rotation_gt is not None
    assert translation_gt is not None

    image_prev = image_prev.to(
        device=device,
        non_blocking=device.type == "cuda",
    )
    image_curr = image_curr.to(
        device=device,
        non_blocking=device.type == "cuda",
    )
    rotation_gt = rotation_gt.to(
        device=device,
        non_blocking=device.type == "cuda",
    )
    translation_gt = translation_gt.to(
        device=device,
        non_blocking=device.type == "cuda",
    )

    rotation_condition_difference = None
    translation_condition_difference = None

    if (
        args.use_semantic_cues
        or args.use_depth_cues
    ):
        (
            rotation_condition_difference,
            translation_condition_difference,
        ) = validate_rotation_conditioning_dependency(
            model=model,
            image_prev=image_prev,
            image_curr=image_curr,
            rotation_gt=rotation_gt,
        )

    rotation_for_translation: Optional[Tensor]

    if args.use_ground_truth_rotation:
        rotation_for_translation = rotation_gt
    else:
        rotation_for_translation = None

    pose_before = clone_named_parameters(
        trainable_parameters
    )

    semantic_before = clone_named_parameters(
        model.semantic_model.named_parameters()
    )

    depth_before = clone_named_parameters(
        model.depth_model.named_parameters()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

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

    rotation_pred = require_output_tensor(
        outputs,
        "rotation",
    )
    translation_pred = require_output_tensor(
        outputs,
        "directional_translation",
    )

    if rotation_pred.shape != rotation_gt.shape:
        raise RuntimeError(
            "Rotation prediction/target shape mismatch: "
            f"{tuple(rotation_pred.shape)} versus "
            f"{tuple(rotation_gt.shape)}."
        )

    if translation_pred.shape != translation_gt.shape:
        raise RuntimeError(
            "Translation prediction/target shape mismatch: "
            f"{tuple(translation_pred.shape)} versus "
            f"{tuple(translation_gt.shape)}."
        )

    rotation_loss = F.mse_loss(
        rotation_pred,
        rotation_gt,
    )

    translation_loss = F.mse_loss(
        translation_pred,
        translation_gt,
    )

    total_loss = (
        args.rotation_loss_weight
        * rotation_loss
        + args.translation_loss_weight
        * translation_loss
    )

    validate_loss(
        rotation_loss,
        "rotation_loss",
    )
    validate_loss(
        translation_loss,
        "translation_loss",
    )
    validate_loss(
        total_loss,
        "total_loss",
    )

    total_loss.backward()

    (
        parameters_with_gradient,
        parameters_with_nonzero_gradient,
        total_gradient_norm,
    ) = validate_trainable_gradients(
        trainable_parameters
    )

    validate_no_auxiliary_gradients(
        model
    )

    optimizer.step()

    (
        changed_pose_parameters,
        largest_pose_change,
    ) = count_changed_parameters(
        before=pose_before,
        named_parameters=trainable_parameters,
    )

    if changed_pose_parameters == 0:
        raise RuntimeError(
            "optimizer.step() did not change any trainable "
            "pose-network parameter."
        )

    validate_unchanged_parameters(
        before=semantic_before,
        named_parameters=(
            model.semantic_model.named_parameters()
        ),
        branch_name="semantic",
    )

    validate_unchanged_parameters(
        before=depth_before,
        named_parameters=(
            model.depth_model.named_parameters()
        ),
        branch_name="depth",
    )

    # Check again after the optimizer step.
    validate_auxiliary_modes(model)
    validate_auxiliary_frozen(model)
    validate_no_auxiliary_gradients(model)

    semantic_prev = require_output_tensor(
        outputs,
        "semantic_prev",
    )
    semantic_curr = require_output_tensor(
        outputs,
        "semantic_curr",
    )
    depth_curr = require_output_tensor(
        outputs,
        "depth_curr",
    )

    print("=" * 88)
    print(
        "Backward-pass test completed successfully"
    )
    print("=" * 88)
    print(
        f"Device:                         {device}"
    )
    print(
        f"Sequence:                       {sequence}"
    )
    print(
        f"Batch size:                     "
        f"{image_prev.shape[0]}"
    )
    print(
        f"Semantic cues enabled:          "
        f"{args.use_semantic_cues}"
    )
    print(
        f"Depth cues enabled:             "
        f"{args.use_depth_cues}"
    )
    print(
        f"Ground-truth rotation for T:    "
        f"{args.use_ground_truth_rotation}"
    )
    print("-" * 88)
    print(
        f"Rotation loss:                  "
        f"{rotation_loss.item():.9f}"
    )
    print(
        f"Translation loss:               "
        f"{translation_loss.item():.9f}"
    )
    print(
        f"Total loss:                     "
        f"{total_loss.item():.9f}"
    )
    print(
        f"Total gradient norm:            "
        f"{total_gradient_norm:.9f}"
    )
    print(
        f"Trainable tensors with grad:    "
        f"{parameters_with_gradient}"
    )
    print(
        f"Trainable tensors nonzero grad: "
        f"{parameters_with_nonzero_gradient}"
    )
    print(
        f"Pose parameters updated:        "
        f"{changed_pose_parameters}"
    )
    print(
        f"Largest parameter change:       "
        f"{largest_pose_change:.9e}"
    )
    print("-" * 88)
    print(
        f"Semantic previous std:          "
        f"{semantic_prev.float().std(unbiased=False).item():.9f}"
    )
    print(
        f"Semantic current std:           "
        f"{semantic_curr.float().std(unbiased=False).item():.9f}"
    )
    print(
        f"Depth current std:              "
        f"{depth_curr.float().std(unbiased=False).item():.9f}"
    )
    print("-" * 88)
    print(
        "Frozen semantic train mode:     False"
    )
    print(
        "Frozen depth train mode:        False"
    )
    print(
        "Frozen semantic gradients:      None"
    )
    print(
        "Frozen depth gradients:         None"
    )
    print(
        "Frozen semantic parameters:     Unchanged"
    )
    print(
        "Frozen depth parameters:        Unchanged"
    )
    print(
        "Trainable pose gradients:       Finite and nonzero"
    )
    print(
        "Optimizer pose update:          Passed"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()