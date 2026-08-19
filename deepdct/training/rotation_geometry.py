"""
Continuous SO(3)-supervised rotation-representation geometry for DeepDCT-VO.

Purpose
-------
This module provides the auxiliary representation loss used by the
continuous rotation-geometry experiment.

The objective is:

    rotations that are close on SO(3)
        -> should have nearby rotation representations

    rotations that are far apart on SO(3)
        -> should have more separated representations

Unlike the earlier motion-regime prototype alignment experiment, this
module does NOT discretize motion into straight / moderate / strong
rotation regimes.

Instead, the supervision is continuous and comes directly from the
ground-truth relative rotation.

Typical usage
-------------

    bank = RotationGeometryBank(
        bank_size=1024,
        temperature=0.10,
        supervision="so3",
    )

    geometry_loss = bank.geometry_loss(
        representation=rotation_representation,
        rotation_gt=rotation_gt,
    )

    total_loss = (
        ...
        + rotation_geometry_weight * geometry_loss
    )

    # Update only after the current loss has been constructed.
    bank.update(
        representation=rotation_representation,
        rotation_gt=rotation_gt,
    )

Rotation convention
-------------------
DeepDCT-VO rotation labels are assumed to be Euler angles:

    [x, y, z]

in radians using the project's extrinsic xyz convention.

For extrinsic xyz:

    R = Rz(z) @ Ry(y) @ Rx(x)

The SO(3) geodesic distance between rotations R_a and R_b is

    theta = acos(
        clamp(
            (trace(R_a^T R_b) - 1) / 2,
            -1,
            +1
        )
    )

and lies in [0, pi].

Geometry loss
-------------
For each query representation, reference samples are supplied by:

1. other samples in the current batch; and
2. a FIFO memory bank from previous batches.

The target neighborhood distribution is generated from ground-truth
SO(3) geodesic distances:

    p_ij proportional to exp(-d_SO3(i,j) / temperature)

The learned representation neighborhood is generated from cosine
similarity:

    q_ij proportional to exp(cos(z_i,z_j) / temperature)

The auxiliary loss minimizes:

    KL(p || q)

This gives continuous supervision rather than assigning samples to
discrete rotation bins.

The memory bank is especially important for DeepDCT-VO experiments
using batch size 1, where no useful within-batch pairwise geometry
would otherwise exist.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Euler -> SO(3)
# ============================================================================


def euler_xyz_to_rotation_matrix(
    angles: Tensor,
) -> Tensor:
    """
    Convert extrinsic xyz Euler angles to rotation matrices.

    Parameters
    ----------
    angles:
        Tensor with shape [..., 3] containing

            [x, y, z]

        Euler angles in radians.

    Returns
    -------
    Tensor
        Rotation matrices with shape [..., 3, 3].

    Notes
    -----
    For extrinsic xyz:

        R = Rz(z) @ Ry(y) @ Rx(x)

    This matches the convention used by the DeepDCT-VO KITTI
    relative-rotation pipeline.
    """

    if not torch.is_tensor(angles):
        raise TypeError(
            "angles must be a torch.Tensor."
        )

    if angles.ndim < 1 or angles.shape[-1] != 3:
        raise ValueError(
            "angles must have final dimension 3, "
            f"but received shape {tuple(angles.shape)}."
        )

    if not angles.is_floating_point():
        raise TypeError(
            "angles must use a floating-point dtype."
        )

    if not torch.isfinite(angles).all():
        raise ValueError(
            "angles contains NaN or infinite values."
        )

    x = angles[..., 0]
    y = angles[..., 1]
    z = angles[..., 2]

    cx = torch.cos(x)
    sx = torch.sin(x)

    cy = torch.cos(y)
    sy = torch.sin(y)

    cz = torch.cos(z)
    sz = torch.sin(z)

    # ------------------------------------------------------------------
    # R = Rz @ Ry @ Rx
    # ------------------------------------------------------------------

    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx

    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx

    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    row0 = torch.stack(
        (r00, r01, r02),
        dim=-1,
    )

    row1 = torch.stack(
        (r10, r11, r12),
        dim=-1,
    )

    row2 = torch.stack(
        (r20, r21, r22),
        dim=-1,
    )

    return torch.stack(
        (row0, row1, row2),
        dim=-2,
    )


# ============================================================================
# SO(3) distance
# ============================================================================


def so3_geodesic_distance(
    rotation_a: Tensor,
    rotation_b: Tensor,
    eps: float = 1.0e-7,
) -> Tensor:
    """
    Compute geodesic angular distance between corresponding rotations.

    Parameters
    ----------
    rotation_a:
        Tensor [..., 3, 3].

    rotation_b:
        Tensor broadcast-compatible with rotation_a.

    eps:
        Numerical guard used around acos.

    Returns
    -------
    Tensor
        Geodesic distance in radians.

    Formula
    -------

        R_error = R_a^T R_b

        theta = acos(
            (trace(R_error) - 1) / 2
        )
    """

    _validate_rotation_matrix_tensor(
        rotation_a,
        name="rotation_a",
    )

    _validate_rotation_matrix_tensor(
        rotation_b,
        name="rotation_b",
    )

    relative = (
        rotation_a.transpose(-1, -2)
        @ rotation_b
    )

    trace = (
        relative[..., 0, 0]
        + relative[..., 1, 1]
        + relative[..., 2, 2]
    )

    cosine = (trace - 1.0) * 0.5

    # Do not clamp to 1-eps here. Doing that would make identical
    # rotations have a small non-zero target distance.
    cosine = cosine.clamp(
        min=-1.0,
        max=1.0,
    )

    # acos itself is numerically well behaved for our detached
    # supervision targets. The optional eps argument is retained
    # for API compatibility / future use.
    del eps

    return torch.acos(cosine)


def pairwise_so3_geodesic_distance(
    query_rotation: Tensor,
    reference_rotation: Tensor,
) -> Tensor:
    """
    Compute all query-reference SO(3) geodesic distances.

    Parameters
    ----------
    query_rotation:
        [N, 3, 3].

    reference_rotation:
        [M, 3, 3].

    Returns
    -------
    Tensor
        [N, M] distances in radians.
    """

    _validate_rotation_matrix_tensor(
        query_rotation,
        name="query_rotation",
    )

    _validate_rotation_matrix_tensor(
        reference_rotation,
        name="reference_rotation",
    )

    if query_rotation.ndim != 3:
        raise ValueError(
            "query_rotation must have shape [N, 3, 3], "
            f"but received {tuple(query_rotation.shape)}."
        )

    if reference_rotation.ndim != 3:
        raise ValueError(
            "reference_rotation must have shape [M, 3, 3], "
            f"but received {tuple(reference_rotation.shape)}."
        )

    # [N, 1, 3, 3]
    query = query_rotation[:, None, :, :]

    # [1, M, 3, 3]
    reference = reference_rotation[
        None,
        :,
        :,
        :,
    ]

    return so3_geodesic_distance(
        query,
        reference,
    )


def pairwise_euler_so3_geodesic_distance(
    query_angles: Tensor,
    reference_angles: Tensor,
) -> Tensor:
    """
    Convenience wrapper for Euler xyz rotation tensors.

    Parameters
    ----------
    query_angles:
        [N, 3], radians.

    reference_angles:
        [M, 3], radians.

    Returns
    -------
    Tensor
        [N, M] geodesic SO(3) distances in radians.
    """

    query_rotation = (
        euler_xyz_to_rotation_matrix(
            query_angles
        )
    )

    reference_rotation = (
        euler_xyz_to_rotation_matrix(
            reference_angles
        )
    )

    return pairwise_so3_geodesic_distance(
        query_rotation,
        reference_rotation,
    )


# ============================================================================
# Continuous representation geometry loss
# ============================================================================


def continuous_so3_geometry_loss(
    query_representation: Tensor,
    query_rotation_gt: Tensor,
    reference_representation: Tensor,
    reference_rotation_gt: Tensor,
    temperature: float = 0.10,
) -> Tensor:
    """
    Match representation neighborhoods to continuous SO(3) neighborhoods.

    Parameters
    ----------
    query_representation:
        Compact learned representation [N, D].

    query_rotation_gt:
        Ground-truth Euler xyz rotations [N, 3], radians.

    reference_representation:
        Detached reference representations [M, D].

    reference_rotation_gt:
        Corresponding reference Euler xyz rotations [M, 3].

    temperature:
        Positive soft-neighborhood temperature.

    Returns
    -------
    Tensor
        Scalar KL-divergence geometry loss.

    Notes
    -----
    The target neighborhood uses continuous SO(3) distance:

        target_logits = -d_SO3 / temperature

    The learned neighborhood uses cosine similarity:

        representation_logits = cosine_similarity / temperature

    Ground-truth target probabilities are detached. Gradients therefore
    flow only through query_representation.
    """

    _validate_geometry_inputs(
        query_representation=query_representation,
        query_rotation_gt=query_rotation_gt,
        reference_representation=reference_representation,
        reference_rotation_gt=reference_rotation_gt,
    )

    temperature = _validate_temperature(
        temperature
    )

    if reference_representation.shape[0] == 0:
        return (
            query_representation.sum()
            * 0.0
        )

    # --------------------------------------------------------------
    # Learned latent geometry.
    # --------------------------------------------------------------

    query_normalized = F.normalize(
        query_representation,
        p=2,
        dim=1,
        eps=1.0e-12,
    )

    reference_normalized = F.normalize(
        reference_representation.detach(),
        p=2,
        dim=1,
        eps=1.0e-12,
    )

    # [N, M]
    cosine_similarity = (
        query_normalized
        @ reference_normalized.transpose(0, 1)
    )

    representation_logits = (
        cosine_similarity
        / temperature
    )

    representation_log_probability = (
        F.log_softmax(
            representation_logits,
            dim=1,
        )
    )

    # --------------------------------------------------------------
    # Continuous SO(3) target geometry.
    #
    # No gradient is required through rotation labels.
    # --------------------------------------------------------------

    with torch.no_grad():
        so3_distance = (
            pairwise_euler_so3_geodesic_distance(
                query_rotation_gt.detach(),
                reference_rotation_gt.detach(),
            )
        )

        target_logits = (
            -so3_distance
            / temperature
        )

        target_probability = F.softmax(
            target_logits,
            dim=1,
        )

    # --------------------------------------------------------------
    # KL(
    #     SO(3) target neighborhood
    #     ||
    #     representation neighborhood
    # )
    # --------------------------------------------------------------

    loss = F.kl_div(
        representation_log_probability,
        target_probability,
        reduction="batchmean",
        log_target=False,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Continuous SO(3) geometry loss became non-finite."
        )

    return loss


# ============================================================================
# FIFO reference bank
# ============================================================================


class RotationGeometryBank:
    """
    FIFO memory bank for continuous SO(3) representation supervision.

    Why the bank is required
    ------------------------
    DeepDCT-VO is commonly trained with:

        batch_size = 1

    A pairwise geometry objective cannot learn anything from a batch
    containing only one sample.

    The bank therefore retains representations and their ground-truth
    rotations from previous optimization steps.

    Current samples query this detached reference set. The current
    representation receives gradients; stored representations never do.

    Parameters
    ----------
    bank_size:
        Maximum number of stored reference samples.

    temperature:
        Temperature used by continuous_so3_geometry_loss().

    supervision:
        Supervision mode.

        Currently supported:

            "so3"
            "continuous_so3"

        The alias is accepted to make checkpoint/configuration metadata
        explicit while preserving a short CLI option.
    """

    SUPPORTED_SUPERVISION = (
        "so3",
        "continuous_so3",
    )

    def __init__(
        self,
        bank_size: int = 1024,
        temperature: float = 0.10,
        supervision: str = "so3",
    ) -> None:

        if not isinstance(bank_size, int):
            raise TypeError(
                "bank_size must be an integer."
            )

        if bank_size <= 0:
            raise ValueError(
                "bank_size must be positive."
            )

        self.bank_size = int(
            bank_size
        )

        self.temperature = (
            _validate_temperature(
                temperature
            )
        )

        supervision = str(
            supervision
        ).lower()

        if (
            supervision
            not in self.SUPPORTED_SUPERVISION
        ):
            raise ValueError(
                "Unsupported rotation geometry supervision "
                f"{supervision!r}. Supported values are "
                f"{self.SUPPORTED_SUPERVISION}."
            )

        self.supervision = supervision

        self.representations: Optional[
            Tensor
        ] = None

        self.rotations: Optional[
            Tensor
        ] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        """Number of references currently stored."""

        if self.representations is None:
            return 0

        return int(
            self.representations.shape[0]
        )

    @property
    def is_empty(self) -> bool:
        return self.num_samples == 0

    # ------------------------------------------------------------------
    # Geometry loss
    # ------------------------------------------------------------------

    def geometry_loss(
        self,
        representation: Tensor,
        rotation_gt: Tensor,
    ) -> Tensor:
        """
        Return continuous SO(3)-supervised geometry loss.

        Current-batch samples and bank samples are both used as
        references when possible.

        For batch size 1 with an empty bank, the result is exactly zero.
        Once the bank contains references, batch-size-1 training receives
        meaningful geometry supervision.
        """

        _validate_query(
            representation,
            rotation_gt,
        )

        batch_size = (
            representation.shape[0]
        )

        reference_representation_parts = []
        reference_rotation_parts = []

        # ----------------------------------------------------------
        # Previous detached bank references.
        # ----------------------------------------------------------

        if (
            self.representations
            is not None
            and self.rotations
            is not None
        ):
            bank_representation = (
                self.representations.to(
                    device=representation.device,
                    dtype=representation.dtype,
                )
            )

            bank_rotation = (
                self.rotations.to(
                    device=rotation_gt.device,
                    dtype=rotation_gt.dtype,
                )
            )

            reference_representation_parts.append(
                bank_representation
            )

            reference_rotation_parts.append(
                bank_rotation
            )

        # ----------------------------------------------------------
        # For B > 1, add current batch as detached reference data.
        #
        # Self-pairs are handled separately below.
        # ----------------------------------------------------------

        if batch_size > 1:
            reference_representation_parts.append(
                representation.detach()
            )

            reference_rotation_parts.append(
                rotation_gt.detach()
            )

        if not reference_representation_parts:
            # B == 1 and empty memory bank.
            return (
                representation.sum()
                * 0.0
            )

        reference_representation = (
            torch.cat(
                reference_representation_parts,
                dim=0,
            )
        )

        reference_rotation = torch.cat(
            reference_rotation_parts,
            dim=0,
        )

        # ----------------------------------------------------------
        # If the current batch is part of the reference set, remove
        # exact self-pairs by assigning effectively zero probability.
        #
        # Because continuous_so3_geometry_loss() does not expose masks,
        # use the specialized masked implementation here.
        # ----------------------------------------------------------

        if batch_size > 1:
            bank_count = (
                0
                if self.representations is None
                else self.representations.shape[0]
            )

            return self._geometry_loss_with_self_mask(
                representation=representation,
                rotation_gt=rotation_gt,
                reference_representation=(
                    reference_representation
                ),
                reference_rotation_gt=(
                    reference_rotation
                ),
                self_reference_offset=int(
                    bank_count
                ),
            )

        return continuous_so3_geometry_loss(
            query_representation=(
                representation
            ),
            query_rotation_gt=rotation_gt,
            reference_representation=(
                reference_representation
            ),
            reference_rotation_gt=(
                reference_rotation
            ),
            temperature=self.temperature,
        )

    def _geometry_loss_with_self_mask(
        self,
        representation: Tensor,
        rotation_gt: Tensor,
        reference_representation: Tensor,
        reference_rotation_gt: Tensor,
        self_reference_offset: int,
    ) -> Tensor:
        """Geometry loss excluding current-batch identity pairs."""

        query_normalized = F.normalize(
            representation,
            p=2,
            dim=1,
            eps=1.0e-12,
        )

        reference_normalized = F.normalize(
            reference_representation.detach(),
            p=2,
            dim=1,
            eps=1.0e-12,
        )

        representation_logits = (
            query_normalized
            @ reference_normalized.transpose(
                0,
                1,
            )
        )

        representation_logits = (
            representation_logits
            / self.temperature
        )

        with torch.no_grad():
            so3_distance = (
                pairwise_euler_so3_geodesic_distance(
                    rotation_gt.detach(),
                    reference_rotation_gt.detach(),
                )
            )

            target_logits = (
                -so3_distance
                / self.temperature
            )

        # ----------------------------------------------------------
        # Mask each sample's own detached copy.
        # ----------------------------------------------------------

        batch_size = (
            representation.shape[0]
        )

        row_index = torch.arange(
            batch_size,
            device=representation.device,
        )

        column_index = (
            row_index
            + self_reference_offset
        )

        negative_large = torch.finfo(
            representation_logits.dtype
        ).min

        representation_logits = (
            representation_logits.clone()
        )

        target_logits = (
            target_logits.clone()
        )

        representation_logits[
            row_index,
            column_index,
        ] = negative_large

        target_logits[
            row_index,
            column_index,
        ] = negative_large

        representation_log_probability = (
            F.log_softmax(
                representation_logits,
                dim=1,
            )
        )

        with torch.no_grad():
            target_probability = (
                F.softmax(
                    target_logits,
                    dim=1,
                )
            )

        loss = F.kl_div(
            representation_log_probability,
            target_probability,
            reduction="batchmean",
            log_target=False,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Continuous SO(3) geometry loss became non-finite."
            )

        return loss

    # ------------------------------------------------------------------
    # Bank update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        representation: Tensor,
        rotation_gt: Tensor,
    ) -> None:
        """
        Append the current samples to the FIFO memory bank.

        This should be called AFTER the current optimization loss has
        been formed, normally after optimizer.step().
        """

        _validate_query(
            representation,
            rotation_gt,
        )

        representation = (
            representation
            .detach()
            .clone()
        )

        rotation_gt = (
            rotation_gt
            .detach()
            .clone()
        )

        if self.representations is None:
            new_representation = (
                representation
            )

            new_rotation = rotation_gt

        else:
            # Keep bank tensors on the same device as the newly
            # supplied tensors during active training.
            old_representation = (
                self.representations.to(
                    device=representation.device,
                    dtype=representation.dtype,
                )
            )

            old_rotation = (
                self.rotations.to(
                    device=rotation_gt.device,
                    dtype=rotation_gt.dtype,
                )
            )

            new_representation = torch.cat(
                (
                    old_representation,
                    representation,
                ),
                dim=0,
            )

            new_rotation = torch.cat(
                (
                    old_rotation,
                    rotation_gt,
                ),
                dim=0,
            )

        # FIFO truncation.
        if (
            new_representation.shape[0]
            > self.bank_size
        ):
            new_representation = (
                new_representation[
                    -self.bank_size:
                ]
            )

            new_rotation = (
                new_rotation[
                    -self.bank_size:
                ]
            )

        self.representations = (
            new_representation
        )

        self.rotations = new_rotation

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state_dict(
        self,
    ) -> Dict[str, object]:
        """
        Return a checkpoint-safe bank state.

        Stored tensors are moved to CPU so checkpoint files are not tied
        to a particular CUDA device.
        """

        return {
            "bank_size": self.bank_size,
            "temperature": (
                self.temperature
            ),
            "supervision": (
                self.supervision
            ),
            "representations": (
                None
                if self.representations is None
                else self.representations
                .detach()
                .cpu()
            ),
            "rotations": (
                None
                if self.rotations is None
                else self.rotations
                .detach()
                .cpu()
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        device: Optional[
            torch.device
        ] = None,
    ) -> None:
        """
        Restore memory-bank state from a checkpoint.
        """

        self.bank_size = int(
            state.get(
                "bank_size",
                self.bank_size,
            )
        )

        self.temperature = (
            _validate_temperature(
                float(
                    state.get(
                        "temperature",
                        self.temperature,
                    )
                )
            )
        )

        supervision = str(
            state.get(
                "supervision",
                self.supervision,
            )
        )

        if (
            supervision
            not in self.SUPPORTED_SUPERVISION
        ):
            raise ValueError(
                "Checkpoint contains unsupported "
                "rotation geometry supervision "
                f"{supervision!r}."
            )

        self.supervision = supervision

        representations = state.get(
            "representations"
        )

        rotations = state.get(
            "rotations"
        )

        if representations is None:
            self.representations = None
        elif torch.is_tensor(
            representations
        ):
            self.representations = (
                representations.to(
                    device=device
                    if device is not None
                    else representations.device
                )
            )
        else:
            raise TypeError(
                "rotation geometry bank "
                "'representations' must be a tensor or None."
            )

        if rotations is None:
            self.rotations = None
        elif torch.is_tensor(rotations):
            self.rotations = rotations.to(
                device=device
                if device is not None
                else rotations.device
            )
        else:
            raise TypeError(
                "rotation geometry bank "
                "'rotations' must be a tensor or None."
            )

        if (
            self.representations is None
            != (self.rotations is None)
        ):
            raise ValueError(
                "Rotation geometry checkpoint contains only one "
                "of representations / rotations."
            )

        if (
            self.representations
            is not None
            and self.rotations
            is not None
        ):
            if (
                self.representations.shape[0]
                != self.rotations.shape[0]
            ):
                raise ValueError(
                    "Restored rotation geometry bank has "
                    "different representation and rotation counts."
                )

    def clear(self) -> None:
        """Discard all stored reference samples."""

        self.representations = None
        self.rotations = None


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_query(
    representation: Tensor,
    rotation_gt: Tensor,
) -> None:

    if not torch.is_tensor(
        representation
    ):
        raise TypeError(
            "representation must be a torch.Tensor."
        )

    if representation.ndim != 2:
        raise ValueError(
            "representation must have shape [B, D], "
            f"but received {tuple(representation.shape)}."
        )

    if representation.shape[0] == 0:
        raise ValueError(
            "representation batch cannot be empty."
        )

    if representation.shape[1] == 0:
        raise ValueError(
            "representation dimension cannot be zero."
        )

    if not representation.is_floating_point():
        raise TypeError(
            "representation must use a floating-point dtype."
        )

    if not torch.isfinite(
        representation
    ).all():
        raise ValueError(
            "representation contains NaN or infinity."
        )

    if not torch.is_tensor(rotation_gt):
        raise TypeError(
            "rotation_gt must be a torch.Tensor."
        )

    expected_rotation_shape = (
        representation.shape[0],
        3,
    )

    if (
        rotation_gt.shape
        != expected_rotation_shape
    ):
        raise ValueError(
            "rotation_gt must have shape "
            f"{expected_rotation_shape}, "
            f"but received {tuple(rotation_gt.shape)}."
        )

    if not rotation_gt.is_floating_point():
        raise TypeError(
            "rotation_gt must use a floating-point dtype."
        )

    if not torch.isfinite(
        rotation_gt
    ).all():
        raise ValueError(
            "rotation_gt contains NaN or infinity."
        )


def _validate_geometry_inputs(
    query_representation: Tensor,
    query_rotation_gt: Tensor,
    reference_representation: Tensor,
    reference_rotation_gt: Tensor,
) -> None:

    _validate_query(
        query_representation,
        query_rotation_gt,
    )

    _validate_query(
        reference_representation,
        reference_rotation_gt,
    )

    if (
        query_representation.shape[1]
        != reference_representation.shape[1]
    ):
        raise ValueError(
            "Query and reference representation dimensions "
            "must match, but received "
            f"{query_representation.shape[1]} and "
            f"{reference_representation.shape[1]}."
        )

    if (
        query_representation.device
        != reference_representation.device
    ):
        raise ValueError(
            "Query and reference representations must "
            "be on the same device."
        )


def _validate_rotation_matrix_tensor(
    rotation: Tensor,
    name: str,
) -> None:

    if not torch.is_tensor(rotation):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if (
        rotation.ndim < 2
        or rotation.shape[-2:]
        != (3, 3)
    ):
        raise ValueError(
            f"{name} must end in shape [3, 3], "
            f"but received {tuple(rotation.shape)}."
        )

    if not rotation.is_floating_point():
        raise TypeError(
            f"{name} must use a floating-point dtype."
        )

    if not torch.isfinite(rotation).all():
        raise ValueError(
            f"{name} contains NaN or infinity."
        )


def _validate_temperature(
    temperature: float,
) -> float:

    if not isinstance(
        temperature,
        (int, float),
    ):
        raise TypeError(
            "temperature must be numeric."
        )

    temperature = float(
        temperature
    )

    if (
        not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError(
            "temperature must be a positive finite value."
        )

    return temperature


__all__ = [
    "RotationGeometryBank",
    "continuous_so3_geometry_loss",
    "euler_xyz_to_rotation_matrix",
    "pairwise_euler_so3_geodesic_distance",
    "pairwise_so3_geodesic_distance",
    "so3_geodesic_distance",
]