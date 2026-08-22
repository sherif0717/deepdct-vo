"""One-epoch validation loop for DeepDCT-VO.

Validation pipeline:

    DataLoader batch
        -> move tensors to device
        -> DeepDCTVO forward
        -> rotation loss
        -> directional-translation loss
        -> weighted total loss
        -> sample-weighted epoch averages

Unlike training:

- model.eval() is enabled;
- gradients are disabled with torch.no_grad();
- no optimizer is required;
- no backward pass is performed;
- no parameters are updated.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from deepdct.training.rotation_geometry import (
    RotationGeometryBank,
)


Batch = Mapping[str, Union[Tensor, str, int]]
LossFunction = Callable[[Tensor, Tensor], Tensor]


@dataclass
class ValidationMetrics:
    """Aggregated metrics returned after one validation epoch."""

    total_loss: float
    rotation_loss: float
    translation_loss: float
    rotation_geometry_loss: float

    num_batches: int
    num_samples: int
    elapsed_seconds: float

    skipped_batches: int = 0

    def as_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "total_loss": self.total_loss,
            "rotation_loss": self.rotation_loss,
            "translation_loss": self.translation_loss,
            "rotation_geometry_loss": (
                self.rotation_geometry_loss
            ),
            "num_batches": self.num_batches,
            "num_samples": self.num_samples,
            "elapsed_seconds": self.elapsed_seconds,
            "skipped_batches": self.skipped_batches,
        }


def validate_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    device: Union[str, torch.device],
    rotation_criterion: Optional[LossFunction] = None,
    translation_criterion: Optional[LossFunction] = None,
    rotation_loss_weight: float = 1.0,
    translation_loss_weight: float = 1.0,
    rotation_geometry_weight: float = 0.0,
    rotation_geometry_loss_fn: Optional[
        RotationGeometryBank
    ] = None,
    use_ground_truth_rotation: bool = False,
    log_interval: Optional[int] = None,
    epoch_index: Optional[int] = None,
    skip_nonfinite_batches: bool = False,
    use_internal_depth: bool = False,
) -> ValidationMetrics:
    """Evaluate ``model`` for one complete pass over ``dataloader``.

    Parameters
    ----------
    model:
        DeepDCT-VO model.

    dataloader:
        Iterable yielding dictionaries with image and pose tensors.

    device:
        Validation device, such as ``"cpu"`` or ``torch.device("cuda")``.

    rotation_criterion:
        Loss function for predicted versus ground-truth normalized
        rotation. Defaults to ``torch.nn.L1Loss()`` for the A2
        paper-style MAE objective.

    translation_criterion:
        Callable loss function for predicted versus ground-truth
        directional translation. It must accept ``(prediction, target)``
        and return a scalar tensor. Defaults to ``torch.nn.L1Loss()``
        for the A2 paper-style MAE objective.

    rotation_loss_weight:
        Scalar multiplier applied to the rotation loss.

    translation_loss_weight:
        Scalar multiplier applied to the directional-translation loss.

    rotation_geometry_weight:
        Scalar multiplier applied to the continuous SO(3)-supervised
        rotation-representation geometry loss.

    rotation_geometry_loss_fn:
        RotationGeometryBank instance containing the frozen FIFO
        memory-bank state used to evaluate rotation-representation
        geometry. Validation queries this loss but must not update its
        memory bank.

    use_ground_truth_rotation:
        When true, the ground-truth rotation target is passed to Model T
        through ``rotation_for_translation``. When false, Model T uses the
        rotation predicted by Model R.

    log_interval:
        Print running validation metrics every this many processed batches.
        Set to ``None`` to disable progress output.

    epoch_index:
        Optional epoch number included in validation log messages.

    skip_nonfinite_batches:
        When false, non-finite losses raise ``FloatingPointError``.
        When true, affected batches are skipped.

    Returns
    -------
    ValidationMetrics
        Sample-weighted average validation losses and execution statistics.
    """

    device = torch.device(device)

    # ---------------------------------------------------------------
    # A2 paper-style pose objective
    #
    # The paper uses MAE/L1 rather than MSE for both rotation and
    # directional translation.
    # ---------------------------------------------------------------
    if rotation_criterion is None:
        rotation_criterion = nn.L1Loss()

    if translation_criterion is None:
        translation_criterion = nn.L1Loss()

    _validate_configuration(
        rotation_loss_weight=rotation_loss_weight,
        translation_loss_weight=translation_loss_weight,
        rotation_geometry_weight=rotation_geometry_weight,
        log_interval=log_interval,
    )

    # ---------------------------------------------------------------
    # A2 rotation-normalization configuration
    #
    # Resolve once per validation epoch rather than once per batch.
    #
    # A1:
    #     rotation_normalization_scale = 1.0
    #
    # A2:
    #     rotation_normalization_scale = 0.175
    # ---------------------------------------------------------------
    model_for_configuration = (
        model.module
        if hasattr(model, "module")
        else model
    )

    if not hasattr(
        model_for_configuration,
        "rotation_normalization_scale",
    ):
        raise AttributeError(
            "The model does not expose "
            "'rotation_normalization_scale'. "
            "The A2-compatible DeepDCTVO implementation is required."
        )

    rotation_normalization_scale = float(
        model_for_configuration.rotation_normalization_scale
    )

    if rotation_normalization_scale <= 0.0:
        raise ValueError(
            "rotation_normalization_scale must be greater than zero, "
            f"but received {rotation_normalization_scale}."
        )

    model.eval()

    running_total_loss = 0.0
    running_rotation_loss = 0.0
    running_translation_loss = 0.0
    running_rotation_geometry_loss = 0.0

    processed_batches = 0
    processed_samples = 0
    skipped_batches = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            tensors = _prepare_batch(
                batch=batch,
                device=device,
                include_depth=not use_internal_depth,
            )

            image_prev = tensors["image_prev"]
            image_curr = tensors["image_curr"]
            rotation_gt = tensors["rotation_gt"]
            translation_gt = tensors["translation_gt"]
            depth_curr = tensors.get("depth_curr")

            batch_size = image_prev.shape[0]

            outputs = model(
                image_prev=image_prev,
                image_curr=image_curr,
                depth_curr=depth_curr,
                rotation_for_translation=(
                    rotation_gt
                    if use_ground_truth_rotation
                    else None
                ),
                use_ground_truth_rotation=use_ground_truth_rotation,
                return_intermediates=True,
            )

            _validate_model_outputs(
                outputs=outputs,
                batch_size=batch_size,
                reference=image_curr,
            )

            # A2:
            # Raw RotationHead regression output in normalized rotation units.
            predicted_rotation_normalized = outputs[
                "rotation_normalized"
            ]

            predicted_translation = outputs[
                "directional_translation"
            ]

            rotation_representation = outputs[
                "rotation_representation"
            ]

            if rotation_normalization_scale <= 0.0:
                raise ValueError(
                    "rotation_normalization_scale must be greater than zero, "
                    f"but received {rotation_normalization_scale}."
                )

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

            # ---------------------------------------------------------------
            # A2 paper-style pose losses
            #
            # Rotation:
            #     compare normalized Model-R prediction against normalized
            #     physical ground-truth rotation.
            #
            # Translation:
            #     remains in its existing physical/directional representation.
            # ---------------------------------------------------------------
            rotation_loss = rotation_criterion(
                predicted_rotation_normalized,
                rotation_gt_normalized,
            )

            translation_loss = translation_criterion(
                predicted_translation,
                translation_gt,
            )

            _validate_scalar_loss(
                loss=rotation_loss,
                name="rotation_loss",
            )

            _validate_scalar_loss(
                loss=translation_loss,
                name="translation_loss",
            )

            if rotation_geometry_weight > 0.0:
                if rotation_geometry_loss_fn is None:
                    raise RuntimeError(
                        "rotation_geometry_weight > 0 requires "
                        "rotation_geometry_loss_fn."
                    )

                rotation_geometry_loss = (
                    rotation_geometry_loss_fn.geometry_loss(
                        representation=rotation_representation,
                        rotation_gt=rotation_gt,
                    )
                )

                _validate_scalar_loss(
                    loss=rotation_geometry_loss,
                    name="rotation_geometry_loss",
                )

            else:
                rotation_geometry_loss = (
                    rotation_representation.sum() * 0.0
                )

            # --------------------------------------------------------------
            # IMPORTANT:
            # total_loss must be outside the geometry if/else so it is
            # constructed for BOTH geometry-enabled and geometry-disabled
            # validation.
            # --------------------------------------------------------------
            total_loss = (
                rotation_loss_weight * rotation_loss
                + translation_loss_weight * translation_loss
                + rotation_geometry_weight
                * rotation_geometry_loss
            )

            if not torch.isfinite(total_loss):
                message = (
                    "Non-finite validation loss encountered at batch "
                    f"{batch_index}: "
                    f"rotation_loss={rotation_loss.detach().item()}, "
                    f"translation_loss="
                    f"{translation_loss.detach().item()}, "
                    f"rotation_geometry_loss="
                    f"{rotation_geometry_loss.detach().item()}."
                )

                if skip_nonfinite_batches:
                    skipped_batches += 1
                    continue

                raise FloatingPointError(message)

            rotation_loss_value = float(
                rotation_loss.detach().item()
            )
            translation_loss_value = float(
                translation_loss.detach().item()
            )
            rotation_geometry_loss_value = float(
                rotation_geometry_loss.detach().item()
            )
            total_loss_value = float(
                total_loss.detach().item()
            )


            running_rotation_loss += (
                rotation_loss_value * batch_size
            )
            running_translation_loss += (
                translation_loss_value * batch_size
            )
            running_total_loss += (
                total_loss_value * batch_size
            )
            running_rotation_geometry_loss += (
                rotation_geometry_loss_value
                * batch_size
            )

            processed_batches += 1
            processed_samples += batch_size

            if (
                log_interval is not None
                and processed_batches % log_interval == 0
            ):
                _print_progress(
                    epoch_index=epoch_index,
                    batch_index=batch_index,
                    processed_batches=processed_batches,
                    processed_samples=processed_samples,
                    running_total_loss=running_total_loss,
                    running_rotation_loss=running_rotation_loss,
                    running_translation_loss=running_translation_loss,
                    running_rotation_geometry_loss=(
                        running_rotation_geometry_loss
                    ),
                )

    elapsed_seconds = time.perf_counter() - start_time

    if processed_batches == 0 or processed_samples == 0:
        raise RuntimeError(
            "No validation batches were successfully processed. "
            f"Skipped batches: {skipped_batches}."
        )

    return ValidationMetrics(
        total_loss=(
            running_total_loss
            / processed_samples
        ),
        rotation_loss=(
            running_rotation_loss
            / processed_samples
        ),
        translation_loss=(
            running_translation_loss
            / processed_samples
        ),
        rotation_geometry_loss=(
            running_rotation_geometry_loss
            / processed_samples
        ),
        num_batches=processed_batches,
        num_samples=processed_samples,
        elapsed_seconds=elapsed_seconds,
        skipped_batches=skipped_batches,
    )


def _prepare_batch(
    batch: Batch,
    device: torch.device,
    *,
    include_depth: bool = True,
) -> Dict[str, Tensor]:
    """Validate and move one validation batch to ``device``.

    When ``include_depth`` is false, the dataset-provided depth tensor
    is intentionally ignored. Passing no depth tensor to DeepDCTVO
    allows the model's internal Lite-Mono branch to generate depth.
    """

    required_keys = {
        "image_prev",
        "image_curr",
        "rotation_gt",
        "translation_gt",
    }

    missing_keys = required_keys.difference(
        batch.keys()
    )

    if missing_keys:
        raise KeyError(
            "Validation batch is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    tensors: Dict[str, Tensor] = {}

    for key in required_keys:
        value = batch[key]

        if not torch.is_tensor(value):
            raise TypeError(
                f"batch[{key!r}] must be a torch.Tensor, "
                f"but received {type(value).__name__}."
            )

        tensors[key] = value.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

    if include_depth and "depth_curr" in batch:
        depth_value = batch["depth_curr"]

        if not torch.is_tensor(depth_value):
            raise TypeError(
                "batch['depth_curr'] must be a torch.Tensor, "
                f"but received {type(depth_value).__name__}."
            )

        tensors["depth_curr"] = depth_value.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

    _validate_batch_shapes(
        tensors
    )

    return tensors


def _validate_batch_shapes(
    tensors: Mapping[str, Tensor],
) -> None:
    """Validate image, depth, and pose-target tensor shapes."""

    image_prev = tensors["image_prev"]
    image_curr = tensors["image_curr"]
    rotation_gt = tensors["rotation_gt"]
    translation_gt = tensors["translation_gt"]

    for name, image in (
        ("image_prev", image_prev),
        ("image_curr", image_curr),
    ):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"{name} must have shape [B, 3, H, W], "
                f"but received {tuple(image.shape)}."
            )

        if not image.is_floating_point():
            raise TypeError(
                f"{name} must use a floating-point dtype, "
                f"but received {image.dtype}."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    if image_prev.shape != image_curr.shape:
        raise ValueError(
            "image_prev and image_curr must have identical "
            f"shapes, but received {tuple(image_prev.shape)} "
            f"and {tuple(image_curr.shape)}."
        )

    batch_size = image_prev.shape[0]
    expected_pose_shape = (batch_size, 3)

    for name, target in (
        ("rotation_gt", rotation_gt),
        ("translation_gt", translation_gt),
    ):
        if target.shape != expected_pose_shape:
            raise ValueError(
                f"{name} must have shape {expected_pose_shape}, "
                f"but received {tuple(target.shape)}."
            )

        if not target.is_floating_point():
            raise TypeError(
                f"{name} must use a floating-point dtype, "
                f"but received {target.dtype}."
            )

        if not torch.isfinite(target).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    depth_curr = tensors.get("depth_curr")

    if depth_curr is not None:
        expected_depth_shape = (
            batch_size,
            1,
            image_curr.shape[2],
            image_curr.shape[3],
        )

        if depth_curr.shape != expected_depth_shape:
            raise ValueError(
                "depth_curr must have shape "
                f"{expected_depth_shape}, but received "
                f"{tuple(depth_curr.shape)}."
            )

        if not depth_curr.is_floating_point():
            raise TypeError(
                "depth_curr must use a floating-point dtype, "
                f"but received {depth_curr.dtype}."
            )

        if not torch.isfinite(depth_curr).all():
            raise ValueError(
                "depth_curr contains NaN or infinite values."
            )


def _validate_model_outputs(
    outputs: Mapping[str, Tensor],
    batch_size: int,
    reference: Tensor,
) -> None:
    """Validate model outputs required for validation."""

    required_keys = {
        "rotation",
        "rotation_normalized",
        "directional_translation",
        "rotation_used_for_translation",
    }

    missing_keys = required_keys.difference(outputs.keys())

    if missing_keys:
        raise KeyError(
            "Model output is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    expected_shape = (batch_size, 3)

    for key in required_keys:
        value = outputs[key]

        if not torch.is_tensor(value):
            raise TypeError(
                f"outputs[{key!r}] must be a torch.Tensor."
            )

        if value.shape != expected_shape:
            raise ValueError(
                f"outputs[{key!r}] must have shape "
                f"{expected_shape}, but received "
                f"{tuple(value.shape)}."
            )

        if value.device != reference.device:
            raise ValueError(
                f"outputs[{key!r}] is on {value.device}, while "
                f"the validation batch is on {reference.device}."
            )

        if not value.is_floating_point():
            raise TypeError(
                f"outputs[{key!r}] must be floating point."
            )

        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"outputs[{key!r}] contains NaN or infinity."
            )
        
    if "rotation_representation" not in outputs:
        raise KeyError(
            "Model output is missing required key: "
            "'rotation_representation'."
        )

    rotation_representation = outputs[
        "rotation_representation"
    ]

    if not torch.is_tensor(rotation_representation):
        raise TypeError(
            "outputs['rotation_representation'] "
            "must be a torch.Tensor."
        )

    if rotation_representation.ndim != 2:
        raise ValueError(
            "outputs['rotation_representation'] must have "
            "shape [B, D], but received "
            f"{tuple(rotation_representation.shape)}."
        )

    if rotation_representation.shape[0] != batch_size:
        raise ValueError(
            "outputs['rotation_representation'] batch "
            "dimension must match the validation batch: "
            f"expected {batch_size}, received "
            f"{rotation_representation.shape[0]}."
        )

    if rotation_representation.device != reference.device:
        raise ValueError(
            "outputs['rotation_representation'] is on "
            f"{rotation_representation.device}, while the "
            f"validation batch is on {reference.device}."
        )

    if not rotation_representation.is_floating_point():
        raise TypeError(
            "outputs['rotation_representation'] must be "
            "floating point."
        )

    if not torch.isfinite(
        rotation_representation
    ).all():
        raise FloatingPointError(
            "outputs['rotation_representation'] contains "
            "NaN or infinity."
        )


def _validate_scalar_loss(
    loss: Tensor,
    name: str,
) -> None:
    """Require a scalar floating-point loss tensor."""

    if not torch.is_tensor(loss):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if loss.ndim != 0:
        raise ValueError(
            f"{name} must be scalar, but received shape "
            f"{tuple(loss.shape)}."
        )

    if not loss.is_floating_point():
        raise TypeError(
            f"{name} must use a floating-point dtype."
        )


def _validate_configuration(
    rotation_loss_weight: float,
    translation_loss_weight: float,
    rotation_geometry_weight: float,
    log_interval: Optional[int],
) -> None:
    """Validate validation-loop configuration."""

    for name, value in (
        ("rotation_loss_weight", rotation_loss_weight),
        (
            "translation_loss_weight",
            translation_loss_weight,
        ),
        (
            "rotation_geometry_weight",
            rotation_geometry_weight,
        ),
    ):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be numeric."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative."
            )

    if (
        rotation_loss_weight == 0
        and translation_loss_weight == 0
        and rotation_geometry_weight == 0
    ):
        raise ValueError(
            "At least one loss weight must be positive."
        )

    if log_interval is not None:
        if not isinstance(log_interval, int):
            raise TypeError(
                "log_interval must be an integer or None."
            )

        if log_interval <= 0:
            raise ValueError(
                "log_interval must be positive."
            )


def _print_progress(
    epoch_index: Optional[int],
    batch_index: int,
    processed_batches: int,
    processed_samples: int,
    running_total_loss: float,
    running_rotation_loss: float,
    running_translation_loss: float,
    running_rotation_geometry_loss: float,
) -> None:
    """Print sample-weighted running validation averages."""

    epoch_label = (
        str(epoch_index)
        if epoch_index is not None
        else "?"
    )

    average_total = (
        running_total_loss / processed_samples
    )
    average_rotation = (
        running_rotation_loss / processed_samples
    )
    average_translation = (
        running_translation_loss / processed_samples
    )
    average_rotation_geometry = (
        running_rotation_geometry_loss
        / processed_samples
    )
    print(
        f"validation "
        f"epoch={epoch_label} "
        f"batch={batch_index + 1} "
        f"processed_batches={processed_batches} "
        f"samples={processed_samples} "
        f"loss={average_total:.6f} "
        f"rotation_loss={average_rotation:.6f} "
        f"translation_loss={average_translation:.6f} "
        f"rotation_geometry_loss="
        f"{average_rotation_geometry:.6f}"
    )