"""Rotation and directional-translation heads for DeepDCT-VO.

Fig. 2 defines two separate regression models:

Model R:
    cat(C_prev, C_curr, S_curr, D_curr)
        -> Conv2D(1, 3x3)
        -> ReLU
        -> Dropout(0.2)
        -> Flatten
        -> Dense(3)
        -> LeakyReLU
        -> rotation

Model T:
    cat(C_prev, C_curr, S_curr, D_curr, rotation_map)
        -> Conv2D(1, 3x3)
        -> ReLU
        -> Dropout(0.2)
        -> Flatten
        -> Dense(3)
        -> LeakyReLU
        -> directional translation

The two heads use the same internal topology but consume different
numbers of input channels.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RegressionHead(nn.Module):
    """Shared paper-style regression block.

    Args:
        in_channels:
            Number of channels in the concatenated fusion tensor.

        input_size:
            Spatial size used before flattening. The paper commonly uses
            120 x 120 for its lightweight configuration.

        dropout:
            Dropout probability used after the convolution.

        negative_slope:
            Negative slope for the final LeakyReLU.

    Input:
        Tensor shaped [B, in_channels, H, W].

    Output:
        Tensor shaped [B, 3].
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int] = (120, 120),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive, "
                f"but received {in_channels}."
            )

        if len(input_size) != 2:
            raise ValueError(
                "input_size must contain height and width."
            )

        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(
                "input_size values must be positive, "
                f"but received {input_size}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1, "
                f"but received {dropout}."
            )

        if negative_slope < 0.0:
            raise ValueError(
                "negative_slope cannot be negative."
            )

        self.in_channels = in_channels
        self.input_size = tuple(input_size)

        # CF2 in Eqs. (16) and (21): reduce fused channels to one.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

        # The paper states dropout=0.2 for the Conv2D stage.
        self.dropout = nn.Dropout2d(
            p=dropout,
        )

        flattened_size = (
            self.input_size[0]
            * self.input_size[1]
        )

        self.dense = nn.Linear(
            in_features=flattened_size,
            out_features=3,
            bias=True,
        )

        self.output_activation = nn.Identity()

        # self.output_activation = nn.LeakyReLU(
        #     negative_slope=negative_slope,
        #     inplace=False,
        # )

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)

        if x.shape[-2:] != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.dense(x)
        x = self.output_activation(x)

        return x
    
    def representation_input_module(
        self,
    ) -> nn.Module:
        """Return the module receiving the pre-regression representation.

        For the paper-style RegressionHead:

            fusion tensor
                -> Conv2D(C -> 1)
                -> ReLU
                -> Dropout
                -> Flatten
                -> representation
                -> Linear(14400, 3)

        A forward-pre-hook on the returned dense layer therefore
        captures the exact flattened representation consumed by the
        final rotation regressor.

        With the default 120 x 120 input size:

            representation shape = [B, 14400]

        This method is inherited by RotationHead.

        DirectionalTranslationHead overrides this method because its
        supported decoder variants expose different representations.
        """

        return self.dense

    def _validate_input(self, x: Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError(
                "RegressionHead expects a torch.Tensor, "
                f"but received {type(x).__name__}."
            )

        if x.ndim != 4:
            raise ValueError(
                "RegressionHead expects [B, C, H, W], "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                "Unexpected fusion-channel count. "
                f"Expected {self.in_channels}, "
                f"received {x.shape[1]}."
            )

        if not x.is_floating_point():
            raise TypeError(
                "RegressionHead expects floating-point features, "
                f"but received {x.dtype}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "RegressionHead input contains NaN or infinite values."
            )


class RotationHead(RegressionHead):
    """Estimate relative roll, pitch, and yaw through a compact bottleneck.

    Architecture
    ------------
    fusion tensor
        -> Conv2D(C -> 1)
        -> ReLU
        -> Dropout
        -> AdaptiveAvgPool2d(PH, PW)
        -> Flatten
        -> compact rotation representation z_R
        -> Linear(D, 3)
        -> rotation

    The compact representation is the exact representation consumed by
    the final rotation readout, making it directly task-relevant.

    With the recommended default pool_size=(8, 8):

        representation_dim = 1 * 8 * 8 = 64

    instead of the previous 120 * 120 = 14400 dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int] = (120, 120),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
        pool_size: Tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_size=input_size,
            dropout=dropout,
            negative_slope=negative_slope,
        )

        if (
            len(pool_size) != 2
            or pool_size[0] <= 0
            or pool_size[1] <= 0
        ):
            raise ValueError(
                "pool_size must contain two positive integers, "
                f"but received {pool_size}."
            )

        self.pool_size = tuple(
            int(value)
            for value in pool_size
        )

        self.pool = nn.AdaptiveAvgPool2d(
            self.pool_size
        )

        # The inherited convolution produces one channel.
        self.representation_dim = int(
            self.pool_size[0]
            * self.pool_size[1]
        )

        # Replace inherited Linear(14400, 3).
        self.dense = nn.Linear(
            in_features=self.representation_dim,
            out_features=3,
            bias=True,
        )

    def extract_representation(
        self,
        x: Tensor,
    ) -> Tensor:
        """Return compact task-relevant rotation representation z_R."""

        self._validate_input(x)

        if x.shape[-2:] != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.pool(x)

        representation = torch.flatten(
            x,
            start_dim=1,
        )

        if representation.ndim != 2:
            raise RuntimeError(
                "Rotation representation must be [B, D], "
                f"but received {tuple(representation.shape)}."
            )

        if (
            representation.shape[1]
            != self.representation_dim
        ):
            raise RuntimeError(
                "Unexpected rotation representation dimension. "
                f"Expected {self.representation_dim}, "
                f"received {representation.shape[1]}."
            )

        return representation

    def forward_with_representation(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return both rotation prediction and compact representation."""

        representation = self.extract_representation(x)

        rotation = self.dense(
            representation
        )

        rotation = self.output_activation(
            rotation
        )

        return rotation, representation

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        rotation, _ = self.forward_with_representation(
            x
        )

        return rotation

    def representation_input_module(
        self,
    ) -> nn.Module:
        """Return final readout receiving compact z_R.

        A forward-pre-hook on this module captures the exact compact
        representation used to predict rotation.
        """

        return self.dense

class DirectionalTranslationHead(RegressionHead):
    """Estimate local directional translation.

    Supported decoder types
    -----------------------
    dense:
        Conv C->1
        -> ReLU
        -> Dropout
        -> Flatten
        -> Linear(D, 3)

    mlp:
        Conv C->1
        -> ReLU
        -> Dropout
        -> Flatten
        -> Linear(D, H1)
        -> ReLU
        -> Linear(H1, H2)
        -> ReLU
        -> Linear(H2, 3)

    pooled_mlp:
        Conv C->K
        -> ReLU
        -> Dropout
        -> AdaptiveAvgPool2d(PH, PW)
        -> Flatten
        -> Linear(K*PH*PW, H)
        -> ReLU
        -> Linear(H, 3)

    pooled_linear:
        Conv C->K
        -> ReLU
        -> Dropout
        -> AdaptiveAvgPool2d(PH, PW)
        -> Flatten
        -> Linear(K*PH*PW, 3)

        Intended for the frozen-representation linear-readout
        experiment. The compact representation extractor can be
        frozen while only the final Linear layer is optimized.

    gated_expert:
        Conv C->K
        -> ReLU
        -> Dropout
        -> AdaptiveAvgPool2d(PH, PW)
        -> Flatten
        -> compact representation
        -> gate: Linear(D, E) -> Softmax
        -> E independent Linear(D, 3) experts
        -> gate-weighted sum
        -> 3-D translation
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int] = (120, 120),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
        decoder_type: str = "dense",
        mlp_hidden_dims: Tuple[int, int] = (256, 64),
        projection_channels: int = 8,
        pool_size: Tuple[int, int] = (4, 4),
        aggregation_hidden_dim: int = 64,
        num_experts: int = 3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            input_size=input_size,
            dropout=dropout,
            negative_slope=negative_slope,
        )

        decoder_type = decoder_type.lower()

        if decoder_type not in {
            "dense",
            "mlp",
            "pooled_mlp",
            "pooled_linear",
            "gated_expert",
        }:
            raise ValueError(
                "decoder_type must be one of "
                "{'dense', 'mlp', 'pooled_mlp', 'gated_expert'}, "
                "'pooled_linear', 'gated_expert'}, "
                f"but received {decoder_type!r}."
            )

        if (
            len(mlp_hidden_dims) != 2
            or mlp_hidden_dims[0] <= 0
            or mlp_hidden_dims[1] <= 0
        ):
            raise ValueError(
                "mlp_hidden_dims must contain two positive integers, "
                f"but received {mlp_hidden_dims}."
            )

        if projection_channels <= 0:
            raise ValueError(
                "projection_channels must be positive, "
                f"but received {projection_channels}."
            )

        if (
            len(pool_size) != 2
            or pool_size[0] <= 0
            or pool_size[1] <= 0
        ):
            raise ValueError(
                "pool_size must contain two positive integers, "
                f"but received {pool_size}."
            )

        if aggregation_hidden_dim <= 0:
            raise ValueError(
                "aggregation_hidden_dim must be positive, "
                f"but received {aggregation_hidden_dim}."
            )
        if num_experts <= 0:
            raise ValueError(
                "num_experts must be positive, "
                f"but received {num_experts}."
            )

        self.decoder_type = decoder_type

        self.mlp_hidden_dims = tuple(
            int(value)
            for value in mlp_hidden_dims
        )

        self.projection_channels = int(
            projection_channels
        )

        self.pool_size = tuple(
            int(value)
            for value in pool_size
        )

        self.aggregation_hidden_dim = int(
            aggregation_hidden_dim
        )

        self.num_experts = int(
            num_experts
        )

        flattened_size = (
            self.input_size[0]
            * self.input_size[1]
        )

        # ----------------------------------------------------------
        # Existing nonlinear decoder:
        #
        # Conv C->1 -> Flatten(14400) -> MLP
        # ----------------------------------------------------------
        if self.decoder_type == "mlp":
            self.dense = nn.Sequential(
                nn.Linear(
                    flattened_size,
                    self.mlp_hidden_dims[0],
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    self.mlp_hidden_dims[0],
                    self.mlp_hidden_dims[1],
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    self.mlp_hidden_dims[1],
                    3,
                ),
            )

        # ----------------------------------------------------------
        # Structured aggregation experiment:
        #
        # Conv C->K
        # -> ReLU
        # -> Dropout
        # -> AdaptiveAvgPool
        # -> compact representation
        # -> small MLP
        # ----------------------------------------------------------
        elif self.decoder_type == "pooled_mlp":
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.projection_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )

            self.pool = nn.AdaptiveAvgPool2d(
                self.pool_size
            )

            compact_size = (
                self.projection_channels
                * self.pool_size[0]
                * self.pool_size[1]
            )

            self.dense = nn.Sequential(
                nn.Linear(
                    compact_size,
                    self.aggregation_hidden_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    self.aggregation_hidden_dim,
                    3,
                ),
            )

        # ----------------------------------------------------------
        # Frozen-representation linear-readout experiment:
        #
        # Conv C->K
        # -> ReLU
        # -> Dropout
        # -> AdaptiveAvgPool
        # -> compact representation h
        # -> Linear(D, 3)
        #
        # Default:
        #   K = 8
        #   pool = 4 x 4
        #   D = 8 * 4 * 4 = 128
        #
        # During the experiment, Conv/Pool and everything upstream
        # are frozen. Only self.dense is trainable.
        # ----------------------------------------------------------
        elif self.decoder_type == "pooled_linear":
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.projection_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )

            self.pool = nn.AdaptiveAvgPool2d(
                self.pool_size
            )

            compact_size = (
                self.projection_channels
                * self.pool_size[0]
                * self.pool_size[1]
            )

            self.representation_dim = int(
                compact_size
            )

            self.dense = nn.Linear(
                in_features=self.representation_dim,
                out_features=3,
                bias=True,
            )

        # ----------------------------------------------------------
        # Gated-expert conditioning experiment:
        #
        # Conv C->K
        # -> ReLU
        # -> Dropout
        # -> AdaptiveAvgPool
        # -> compact representation h
        #
        # h -> softmax gate over E experts
        # h -> E independent Linear(D, 3) experts
        #
        # translation = sum_k gate_k * expert_k(h)
        # ----------------------------------------------------------
        elif self.decoder_type == "gated_expert":
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.projection_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )

            self.pool = nn.AdaptiveAvgPool2d(
                self.pool_size
            )

            compact_size = (
                self.projection_channels
                * self.pool_size[0]
                * self.pool_size[1]
            )

            self.representation_dim = int(
                compact_size
            )

            # Remove the inherited dense regression layer from the
            # active architecture. It is not used by gated_expert.
            self.dense = nn.Identity()

            # Small gating branch:
            #
            # default:
            #   [B, 128] -> [B, 3]
            #
            # Softmax is applied in forward().
            self.gate = nn.Linear(
                self.representation_dim,
                self.num_experts,
                bias=True,
            )

            # Three tiny translation experts by default:
            #
            # each:
            #   [B, 128] -> [B, 3]
            self.experts = nn.ModuleList(
                [
                    nn.Linear(
                        compact_size,
                        3,
                        bias=True,
                    )
                    for _ in range(self.num_experts)
                ]
            )

            # Evaluation/analysis diagnostics.
            #
            # These are populated during forward() but are not
            # parameters and do not alter the training objective.
            self.last_representation = None
            self.last_gate_logits = None
            self.last_gate_weights = None
            self.last_expert_predictions = None

            # Evaluation/analysis diagnostics.
            #
            # These are populated during forward() but are not
            # parameters and do not alter the training objective.
            self.last_gate_logits = None
            self.last_gate_weights = None
            self.last_expert_predictions = None

        self.output_activation = nn.Identity()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Run the selected translation decoder."""

        # ----------------------------------------------------------
        # Original dense / MLP experiments.
        #
        # Preserve their existing execution path exactly.
        # ----------------------------------------------------------
        if self.decoder_type in {
            "dense",
            "mlp",
        }:
            return super().forward(x)

        # ----------------------------------------------------------
        # pooled_mlp, pooled_linear, and gated_expert share the
        # same structured representation extractor.
        # ----------------------------------------------------------
        self._validate_input(x)

        if x.shape[-2:] != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        # Learned multi-channel translation projection.
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Structured spatial aggregation.
        x = self.pool(x)

        # Compact representation.
        #
        # Default:
        #
        #   [B, 8, 4, 4]
        #       ->
        #   [B, 128]
        representation = torch.flatten(
            x,
            start_dim=1,
        )

        if (
            self.decoder_type == "gated_expert"
            and representation.shape[1]
            != self.representation_dim
        ):
            raise RuntimeError(
                "Unexpected gated-expert representation size. "
                f"Expected {self.representation_dim}, "
                f"received {representation.shape[1]}."
            )
        
        # ----------------------------------------------------------
        # Frozen-representation linear readout.
        # ----------------------------------------------------------
        if self.decoder_type == "pooled_linear":
            if (
                representation.shape[1]
                != self.representation_dim
            ):
                raise RuntimeError(
                    "Unexpected pooled-linear representation size. "
                    f"Expected {self.representation_dim}, "
                    f"received {representation.shape[1]}."
                )

            translation = self.dense(
                representation
            )

            translation = self.output_activation(
                translation
            )

            return translation

        # ----------------------------------------------------------
        # Existing structured-aggregation decoder.
        # ----------------------------------------------------------
        if self.decoder_type == "pooled_mlp":
            translation = self.dense(
                representation
            )

            translation = self.output_activation(
                translation
            )

            return translation

        # ----------------------------------------------------------
        # Gated-expert conditioning decoder.
        # ----------------------------------------------------------

        # [B, 128] -> [B, E]
        gate_logits = self.gate(
            representation
        )

        # Soft assignment across experts.
        #
        # Each row sums to 1.
        #
        # [B, E]
        gate_weights = torch.softmax(
            gate_logits,
            dim=-1,
        )

        # Each expert independently predicts a complete
        # 3-D translation vector.
        #
        # List of E tensors:
        #   each [B, 3]
        #
        # Stack:
        #   [B, E, 3]
        expert_predictions = torch.stack(
            [
                expert(representation)
                for expert in self.experts
            ],
            dim=1,
        )

        # Gate-weighted expert combination:
        #
        # gate_weights:
        #   [B, E]
        #
        # gate_weights.unsqueeze(-1):
        #   [B, E, 1]
        #
        # expert_predictions:
        #   [B, E, 3]
        #
        # result:
        #   [B, 3]
        translation = torch.sum(
            gate_weights.unsqueeze(-1)
            * expert_predictions,
            dim=1,
        )

        translation = self.output_activation(
            translation
        )

        self.last_representation = (
            representation.detach()
        )

        self.last_gate_logits = (
            gate_logits.detach()
        )

        self.last_gate_weights = (
            gate_weights.detach()
        )

        self.last_expert_predictions = (
            expert_predictions.detach()
        )

        # Save detached diagnostics so evaluation code can inspect
        # routing without retaining the autograd graph.
        self.last_gate_logits = (
            gate_logits.detach()
        )

        self.last_gate_weights = (
            gate_weights.detach()
        )

        self.last_expert_predictions = (
            expert_predictions.detach()
        )

        return translation
    
    def conditioning_diagnostics(
        self,
    ):
        """Return cached gated-expert conditioning diagnostics.

        Returns
        -------
        dict
            representation:
                Shared compact representation [B, D].

                With the default configuration:
                    projection_channels = 8
                    pool_size = (4, 4)

                D = 8 * 4 * 4 = 128.

            gate_logits:
                Unnormalized expert-routing logits [B, E].

            gate_weights:
                Softmax expert weights [B, E].

            expert_predictions:
                Per-expert translation predictions [B, E, 3].

        Raises
        ------
        RuntimeError
            If called for a decoder other than gated_expert or before
            the first gated-expert forward pass.
        """

        if self.decoder_type != "gated_expert":
            raise RuntimeError(
                "conditioning_diagnostics() is only available "
                "for decoder_type='gated_expert'."
            )

        if self.last_representation is None:
            raise RuntimeError(
                "No gated-expert diagnostics are available yet. "
                "Run a forward pass first."
            )

        return {
            "representation": (
                self.last_representation
            ),
            "gate_logits": (
                self.last_gate_logits
            ),
            "gate_weights": (
                self.last_gate_weights
            ),
            "expert_predictions": (
                self.last_expert_predictions
            ),
        }

    def representation_input_module(
        self,
    ) -> nn.Module:
        """Return module receiving the representation to be audited.

        dense:
            input to final Linear(14400, 3)

        mlp:
            input to first Linear(14400, H1)

        pooled_mlp:
            input to first Linear(K*PH*PW, H)

        gated_expert:
            input to gate Linear(K*PH*PW, E)

            The gate and all experts consume the same compact
            representation, so a forward-pre-hook on the gate
            captures the shared conditioning representation.

        pooled_linear:
            input to Linear(K*PH*PW, 3)

        A forward-pre-hook on this module therefore captures the
        representation immediately before regression/conditioning.
        """

        if self.decoder_type == "dense":
            return self.dense

        if self.decoder_type in {
            "mlp",
            "pooled_mlp",
        }:
            return self.dense[0]
        
        if self.decoder_type == "pooled_linear":
            return self.dense

        if self.decoder_type == "gated_expert":
            return self.gate

        raise RuntimeError(
            "Unsupported decoder_type: "
            f"{self.decoder_type!r}."
        )