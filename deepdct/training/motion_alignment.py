"""Motion-conditioned representation alignment utilities for DeepDCT-VO.

This module implements the representation-consistency component used by
the compact rotation-bottleneck experiment.

Motivation
----------
The rotation branch produces a compact task-relevant representation:

    z_R : Tensor[B, D]

before the final rotation regression layer.

Each sample is also assigned to one of three forward-motion regimes using
training-set-derived ground-truth t_z thresholds:

    0 = low
    1 = medium
    2 = high

The alignment objective encourages representations belonging to the same
motion regime to remain close to a shared running prototype.

Because DeepDCT-VO is commonly trained with batch_size=1, ordinary
within-mini-batch contrastive or centroid losses are unsuitable. Instead,
this module maintains exponential-moving-average (EMA) prototypes that
persist across batches.

For sample i with representation z_i and regime r_i:

    z_hat_i = normalize(z_i)

    p_hat_r = normalize(p_r)

    L_align =
        mean_i || z_hat_i - p_hat_(r_i) ||^2

Only already-initialized prototypes contribute to the loss. The first
sample observed for a regime initializes its prototype but incurs no
alignment penalty.

Prototype updates are detached from autograd and should occur after the
loss for the current batch has been formed.

Validation may call ``alignment_loss`` against the training prototypes,
but must NOT call ``update``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class RotationRegimePrototypeBank:
    """EMA prototype bank for motion-conditioned rotation representations.

    Parameters
    ----------
    num_regimes:
        Number of discrete motion regimes.

        The current experiment uses:

            0 = low forward motion
            1 = medium forward motion
            2 = high forward motion

        so ``num_regimes=3``.

    momentum:
        Exponential-moving-average momentum.

        Prototype update:

            p_new =
                momentum * p_old
                + (1 - momentum) * batch_centroid

        A value close to one produces slowly changing cross-batch
        prototypes. The recommended initial value is ``0.95``.

    Notes
    -----
    Prototypes are created lazily on the same device and with the same
    dtype as the first representation tensor received by the bank.

    The bank is intentionally not an ``nn.Module`` because the prototypes
    are not trainable parameters. They are stateful statistics updated
    explicitly with ``update()``.
    """

    def __init__(
        self,
        num_regimes: int = 3,
        momentum: float = 0.95,
    ) -> None:
        if not isinstance(num_regimes, int):
            raise TypeError(
                "num_regimes must be an integer."
            )

        if num_regimes <= 0:
            raise ValueError(
                "num_regimes must be positive."
            )

        if not isinstance(momentum, (int, float)):
            raise TypeError(
                "momentum must be numeric."
            )

        momentum = float(momentum)

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1, "
                f"but received {momentum}."
            )

        self.num_regimes = int(num_regimes)
        self.momentum = momentum

        # Lazily initialized because representation dimension and device
        # are not known when the training script constructs the bank.
        self.prototypes: Optional[Tensor] = None
        self.initialized: Optional[Tensor] = None

    @property
    def representation_dim(self) -> Optional[int]:
        """Return prototype dimensionality once the bank is initialized."""

        if self.prototypes is None:
            return None

        return int(
            self.prototypes.shape[1]
        )

    @property
    def is_initialized(self) -> bool:
        """Return whether storage for the prototype bank exists."""

        return (
            self.prototypes is not None
            and self.initialized is not None
        )

    def _validate_representation(
        self,
        representation: Tensor,
    ) -> None:
        """Validate a compact [B, D] representation tensor."""

        if not torch.is_tensor(representation):
            raise TypeError(
                "representation must be a torch.Tensor."
            )

        if representation.ndim != 2:
            raise ValueError(
                "representation must have shape [B, D], "
                f"but received {tuple(representation.shape)}."
            )

        if representation.shape[0] <= 0:
            raise ValueError(
                "representation batch dimension must be positive."
            )

        if representation.shape[1] <= 0:
            raise ValueError(
                "representation dimension must be positive."
            )

        if not representation.is_floating_point():
            raise TypeError(
                "representation must use a floating-point dtype, "
                f"but received {representation.dtype}."
            )

        if not torch.isfinite(
            representation
        ).all():
            raise ValueError(
                "representation contains NaN or infinite values."
            )

        if self.prototypes is not None:
            expected_dimension = (
                self.prototypes.shape[1]
            )

            if (
                representation.shape[1]
                != expected_dimension
            ):
                raise ValueError(
                    "Representation dimension does not match "
                    "the existing prototype bank. "
                    f"Expected {expected_dimension}, "
                    f"received {representation.shape[1]}."
                )

    def _validate_regimes(
        self,
        regimes: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Validate integer regime labels."""

        if not torch.is_tensor(regimes):
            raise TypeError(
                "regimes must be a torch.Tensor."
            )

        if regimes.ndim != 1:
            raise ValueError(
                "regimes must have shape [B], "
                f"but received {tuple(regimes.shape)}."
            )

        if regimes.shape[0] != batch_size:
            raise ValueError(
                "regime batch size does not match representation "
                f"batch size: {regimes.shape[0]} vs {batch_size}."
            )

        if regimes.device != device:
            raise ValueError(
                "regimes and representation must be on the same "
                f"device, but received {regimes.device} and {device}."
            )

        if regimes.dtype != torch.long:
            raise TypeError(
                "regimes must use torch.long dtype, "
                f"but received {regimes.dtype}."
            )

        minimum = int(
            regimes.min().item()
        )

        maximum = int(
            regimes.max().item()
        )

        if minimum < 0:
            raise ValueError(
                "Motion-regime indices cannot be negative."
            )

        if maximum >= self.num_regimes:
            raise ValueError(
                "Motion-regime index exceeds configured range. "
                f"Maximum valid index is {self.num_regimes - 1}, "
                f"but received {maximum}."
            )

    def _ensure_storage(
        self,
        representation: Tensor,
    ) -> None:
        """Create prototype storage when first needed."""

        if self.prototypes is not None:
            return

        dimension = int(
            representation.shape[1]
        )

        self.prototypes = torch.zeros(
            self.num_regimes,
            dimension,
            device=representation.device,
            dtype=representation.dtype,
        )

        self.initialized = torch.zeros(
            self.num_regimes,
            device=representation.device,
            dtype=torch.bool,
        )

    def _prepare_inputs(
        self,
        representation: Tensor,
        regimes: Tensor,
    ) -> None:
        """Perform common validation and lazy initialization."""

        self._validate_representation(
            representation
        )

        self._validate_regimes(
            regimes=regimes,
            batch_size=representation.shape[0],
            device=representation.device,
        )

        self._ensure_storage(
            representation
        )

        assert self.prototypes is not None
        assert self.initialized is not None

        if (
            self.prototypes.device
            != representation.device
        ):
            raise RuntimeError(
                "Prototype bank and representation are on "
                "different devices."
            )

        if (
            self.prototypes.dtype
            != representation.dtype
        ):
            raise RuntimeError(
                "Prototype bank and representation use "
                "different dtypes."
            )

    def alignment_loss(
        self,
        representation: Tensor,
        regimes: Tensor,
    ) -> Tensor:
        """Calculate motion-conditioned representation alignment loss.

        Parameters
        ----------
        representation:
            Compact task-relevant rotation representation shaped ``[B, D]``.

        regimes:
            Integer motion-regime labels shaped ``[B]``.

            Expected coding:

                0 = low
                1 = medium
                2 = high

        Returns
        -------
        Tensor
            Scalar differentiable alignment loss.

        Notes
        -----
        Representations and prototypes are L2-normalized before comparison.

        A regime contributes only if its prototype was initialized by an
        earlier sample. Therefore the first sample observed for a regime
        receives zero alignment loss for that regime.

        Prototype tensors are detached, so gradients propagate only through
        ``representation``.
        """

        self._prepare_inputs(
            representation=representation,
            regimes=regimes,
        )

        assert self.prototypes is not None
        assert self.initialized is not None

        normalized_representation = F.normalize(
            representation,
            p=2,
            dim=1,
            eps=1.0e-12,
        )

        regime_losses = []

        for regime_index in range(
            self.num_regimes
        ):
            mask = (
                regimes == regime_index
            )

            if not torch.any(mask):
                continue

            # The first observation initializes this regime during
            # update(), but there is no historical target yet.
            if not bool(
                self.initialized[
                    regime_index
                ].item()
            ):
                continue

            prototype = (
                self.prototypes[
                    regime_index
                ]
                .detach()
            )

            normalized_prototype = F.normalize(
                prototype,
                p=2,
                dim=0,
                eps=1.0e-12,
            )

            regime_representation = (
                normalized_representation[
                    mask
                ]
            )

            target = (
                normalized_prototype
                .unsqueeze(0)
                .expand_as(
                    regime_representation
                )
            )

            regime_loss = F.mse_loss(
                regime_representation,
                target,
                reduction="mean",
            )

            regime_losses.append(
                regime_loss
            )

        if not regime_losses:
            # Preserve connection to the representation graph while
            # producing an exact scalar zero.
            return (
                representation.sum()
                * 0.0
            )

        return torch.stack(
            regime_losses
        ).mean()

    @torch.no_grad()
    def update(
        self,
        representation: Tensor,
        regimes: Tensor,
    ) -> None:
        """Update regime prototypes using detached representations.

        This should normally be called once per successfully optimized
        training batch, after the current batch's alignment loss has
        already been calculated.

        Do not call this method during validation or test evaluation.
        """

        self._prepare_inputs(
            representation=representation,
            regimes=regimes,
        )

        assert self.prototypes is not None
        assert self.initialized is not None

        normalized_representation = F.normalize(
            representation.detach(),
            p=2,
            dim=1,
            eps=1.0e-12,
        )

        for regime_index in range(
            self.num_regimes
        ):
            mask = (
                regimes == regime_index
            )

            if not torch.any(mask):
                continue

            regime_representation = (
                normalized_representation[
                    mask
                ]
            )

            batch_centroid = (
                regime_representation.mean(
                    dim=0
                )
            )

            # Keep the stored target on the unit sphere as well.
            batch_centroid = F.normalize(
                batch_centroid,
                p=2,
                dim=0,
                eps=1.0e-12,
            )

            if not bool(
                self.initialized[
                    regime_index
                ].item()
            ):
                self.prototypes[
                    regime_index
                ].copy_(
                    batch_centroid
                )

                self.initialized[
                    regime_index
                ] = True

                continue

            self.prototypes[
                regime_index
            ].mul_(
                self.momentum
            ).add_(
                batch_centroid,
                alpha=(
                    1.0
                    - self.momentum
                ),
            )

            # EMA of unit vectors does not necessarily remain exactly
            # unit length, so renormalize after every update.
            normalized_updated = F.normalize(
                self.prototypes[
                    regime_index
                ],
                p=2,
                dim=0,
                eps=1.0e-12,
            )

            self.prototypes[
                regime_index
            ].copy_(
                normalized_updated
            )

    @torch.no_grad()
    def to(
        self,
        device: torch.device,
    ) -> "RotationRegimePrototypeBank":
        """Move initialized prototype state to ``device``.

        This method is useful after checkpoint restoration.

        It intentionally mirrors the convenience of ``nn.Module.to()``
        while keeping the prototype bank independent of model parameters.
        """

        device = torch.device(
            device
        )

        if self.prototypes is not None:
            self.prototypes = (
                self.prototypes.to(
                    device=device
                )
            )

        if self.initialized is not None:
            self.initialized = (
                self.initialized.to(
                    device=device
                )
            )

        return self

    def state_dict(
        self,
    ) -> Dict[str, object]:
        """Return checkpointable prototype-bank state."""

        return {
            "num_regimes": int(
                self.num_regimes
            ),
            "momentum": float(
                self.momentum
            ),
            "representation_dim": (
                self.representation_dim
            ),
            "prototypes": (
                None
                if self.prototypes is None
                else (
                    self.prototypes
                    .detach()
                    .cpu()
                    .clone()
                )
            ),
            "initialized": (
                None
                if self.initialized is None
                else (
                    self.initialized
                    .detach()
                    .cpu()
                    .clone()
                )
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        device: Optional[
            torch.device
        ] = None,
    ) -> None:
        """Restore checkpointed prototype-bank state.

        Parameters
        ----------
        state:
            Mapping previously returned by ``state_dict()``.

        device:
            Optional target device. When omitted, state is restored on CPU.

        Raises
        ------
        ValueError
            If the checkpointed state is incompatible with this bank.
        """

        if not isinstance(state, Mapping):
            raise TypeError(
                "Prototype-bank state must be a mapping."
            )

        state_num_regimes = int(
            state.get(
                "num_regimes",
                self.num_regimes,
            )
        )

        if (
            state_num_regimes
            != self.num_regimes
        ):
            raise ValueError(
                "Checkpoint prototype-bank regime count "
                "does not match the configured bank: "
                f"{state_num_regimes} vs "
                f"{self.num_regimes}."
            )

        state_momentum = float(
            state.get(
                "momentum",
                self.momentum,
            )
        )

        if not (
            0.0
            <= state_momentum
            < 1.0
        ):
            raise ValueError(
                "Checkpoint prototype momentum is invalid: "
                f"{state_momentum}."
            )

        prototypes = state.get(
            "prototypes"
        )

        initialized = state.get(
            "initialized"
        )

        if (
            prototypes is None
            and initialized is None
        ):
            self.prototypes = None
            self.initialized = None
            self.momentum = state_momentum
            return

        if not torch.is_tensor(
            prototypes
        ):
            raise TypeError(
                "Checkpoint 'prototypes' must be a tensor or None."
            )

        if not torch.is_tensor(
            initialized
        ):
            raise TypeError(
                "Checkpoint 'initialized' must be a tensor or None."
            )

        if prototypes.ndim != 2:
            raise ValueError(
                "Checkpoint prototypes must have shape "
                f"[R, D], received {tuple(prototypes.shape)}."
            )

        if (
            prototypes.shape[0]
            != self.num_regimes
        ):
            raise ValueError(
                "Checkpoint prototype regime dimension "
                f"is {prototypes.shape[0]}, expected "
                f"{self.num_regimes}."
            )

        if initialized.shape != (
            self.num_regimes,
        ):
            raise ValueError(
                "Checkpoint initialized mask must have shape "
                f"({self.num_regimes},), received "
                f"{tuple(initialized.shape)}."
            )

        if not prototypes.is_floating_point():
            raise TypeError(
                "Checkpoint prototypes must be floating point."
            )

        if not torch.isfinite(
            prototypes
        ).all():
            raise ValueError(
                "Checkpoint prototypes contain NaN or infinity."
            )

        if device is None:
            resolved_device = (
                torch.device("cpu")
            )
        else:
            resolved_device = (
                torch.device(device)
            )

        self.prototypes = (
            prototypes.detach()
            .clone()
            .to(
                device=resolved_device
            )
        )

        self.initialized = (
            initialized.detach()
            .clone()
            .to(
                device=resolved_device,
                dtype=torch.bool,
            )
        )

        self.momentum = (
            state_momentum
        )

    @torch.no_grad()
    def regime_counts_initialized(
        self,
    ) -> int:
        """Return the number of regimes with valid running prototypes."""

        if self.initialized is None:
            return 0

        return int(
            self.initialized.sum().item()
        )

    @torch.no_grad()
    def prototype_norms(
        self,
    ) -> Optional[Tensor]:
        """Return detached L2 norms of current prototypes."""

        if self.prototypes is None:
            return None

        return torch.linalg.vector_norm(
            self.prototypes.detach(),
            ord=2,
            dim=1,
        ).cpu()