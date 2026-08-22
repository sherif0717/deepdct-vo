"""Paper-style DeepDCT-VO orchestration consistent with Fig. 2.

The model uses:

    LR-ASPP:
        I_(k-1) -> S_(k-1)
        I_k     -> S_k

    Model R:
        CI_(k-1) = cat(I_(k-1), S_(k-1))
        CI_k     = cat(I_k, S_k)

        C_(k-1)^R = AResUNet_R(CI_(k-1))
        C_k^R     = AResUNet_R(CI_k)

        cat(C_(k-1)^R, C_k^R, S_k, D_k)
            -> RotationHead
            -> rotation

    Model T:
        C_(k-1)^T = AResUNet_T(CI_(k-1))
        C_k^T     = AResUNet_T(CI_k)

        cat(C_(k-1)^T, C_k^T, S_k, D_k, rotation_map)
            -> DirectionalTranslationHead
            -> directional_translation

The semantic branch is expected to use the updated Fig. 2-compatible
``LRASPPSemanticBranch`` whose ``forward`` method returns a normalized
single-channel semantic map directly.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .auxiliary.lite_mono import LiteMonoDepthBranch
from .auxiliary.lraspp_fig2 import LRASPPSemanticBranch
from .blocks import AResUNet
from .pose_head import (
    DirectionalTranslationHead,
    RotationHead,
)


PathLike = Union[str, Path]
ModelOutput = Dict[str, Tensor]


class DeepDCTVO(nn.Module):
    """DeepDCT-VO with separate rotation and translation models.

    Each motion model owns one A-ResUNet that is reused for the previous
    and current frame, giving Siamese weight sharing across time.

    Args:
        aresunet_output_channels:
            Number of channels returned by ``AResUNet``. With the paper-style
            output layer this is normally 1.

        input_size:
            Spatial size expected by the regression heads.

        pretrained_semantic:
            Load pretrained Torchvision LR-ASPP weights.

        freeze_semantic:
            Freeze LR-ASPP and keep it in evaluation mode.

        normalize_semantic_input:
            Apply ImageNet normalization inside ``LRASPPSemanticBranch``.
            Input RGB tensors should then contain values in [0, 1].

        normalize_semantic_map:
            Normalize LR-ASPP class IDs to [0, 1] before concatenating the
            semantic map with RGB.

        share_aresunet_between_models:
            If False, Model R and Model T use distinct A-ResUNet instances,
            which is the closer interpretation of Fig. 2. If True, the same
            A-ResUNet is shared across both models as an experimental
            parameter-reduction variant.

        rotation_normalization_scale:
            Scale used to map the RotationHead regression output back to
            physical Euler radians.

            A value of 1.0 preserves the original A1 behavior.

            For the A2 paper-reproduction experiment, use 0.175 so that:

                rotation_normalized = rotation_gt / 0.175

            during loss computation, while:

                rotation_physical = rotation_normalized * 0.175

            is supplied to Model T and downstream evaluation.
    """

    def __init__(
        self,
        aresunet_output_channels: int = 1,
        input_size: Tuple[int, int] = (120, 120),
        pretrained_semantic: bool = True,
        freeze_semantic: bool = True,
        normalize_semantic_input: bool = True,
        normalize_semantic_map: bool = True,
        semantic_map_mode: str = "foreground_probability",
        share_aresunet_between_models: bool = False,
        rotation_pool_size: Tuple[int, int] = (8, 8),

        # A2 paper-reproduction rotation normalization.
        #
        # 1.0   -> A1 behavior: RotationHead output is already physical radians.
        # 0.175 -> A2 behavior: RotationHead learns normalized rotation;
        #          physical rotation = normalized_prediction * 0.175.
        rotation_normalization_scale: float = 1.0,

        translation_decoder_type: str = "dense",
        translation_mlp_hidden_dims: Tuple[int, int] = (256, 64),
        translation_projection_channels: int = 8,
        translation_pool_size: Tuple[int, int] = (4, 4),
        translation_aggregation_hidden_dim: int = 64,
        translation_num_experts: int = 3,
        # Lite-Mono configuration
        depth_checkpoint_dir: Optional[PathLike] = (
            "weights/lite-mono-tiny-640x192"
        ),
        depth_model_name: str = "lite-mono-tiny",
        depth_feed_size: Tuple[int, int] = (192, 640),
        freeze_depth: bool = True,
        depth_normalization_meters: float = 80.0,
        depth_model: Optional[nn.Module] = None,
        use_semantic_cues: bool = False,
        use_depth_cues: bool = False,
        freeze_semantic_model: bool = True,
        freeze_depth_model: bool = True,
        depth_weights_dir: Optional[str] = None,
        depth_output_mode: str = "normalized_depth",
    ) -> None:
        super().__init__()

        self.use_semantic_cues = use_semantic_cues
        self.use_depth_cues = use_depth_cues
        self.freeze_semantic_model = freeze_semantic_model
        self.freeze_depth_model = freeze_depth_model
        if aresunet_output_channels <= 0:
            raise ValueError(
                "aresunet_output_channels must be positive, "
                f"but received {aresunet_output_channels}."
            )

        if len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError(
                "input_size must contain two positive integers, "
                f"but received {input_size}."
            )

        self.aresunet_output_channels = aresunet_output_channels
        self.input_size = tuple(input_size)

        self.rotation_pool_size = tuple(
            int(value)
            for value in rotation_pool_size
        )

        # ---------------------------------------------------------------
        # Rotation normalization
        # ---------------------------------------------------------------
        #
        # Model R's regression head operates in normalized rotation units.
        # The value is converted back to physical Euler radians before it
        # is exposed as outputs["rotation"] or supplied to Model T.
        #
        # A1:
        #     rotation_normalization_scale = 1.0
        #
        # A2:
        #     rotation_normalization_scale = 0.175
        #
        rotation_normalization_scale = float(
            rotation_normalization_scale
        )

        if not (
            rotation_normalization_scale > 0.0
        ):
            raise ValueError(
                "rotation_normalization_scale must be greater than zero, "
                f"but received {rotation_normalization_scale}."
            )

        self.rotation_normalization_scale = (
            rotation_normalization_scale
        )

        self.translation_decoder_type = (
            translation_decoder_type
        )

        self.translation_mlp_hidden_dims = tuple(
            translation_mlp_hidden_dims
        )

        self.translation_projection_channels = int(
            translation_projection_channels
        )

        self.translation_pool_size = tuple(
            translation_pool_size
        )

        self.translation_aggregation_hidden_dim = int(
            translation_aggregation_hidden_dim
        )

        self.translation_num_experts = int(
            translation_num_experts
        )

        self.semantic_model = LRASPPSemanticBranch(
            pretrained=pretrained_semantic,
            freeze_pretrained=freeze_semantic,
            normalize_input=normalize_semantic_input,
            normalize_map=normalize_semantic_map,
            semantic_map_mode=semantic_map_mode,
        )

        if depth_model is not None:
            # Dependency injection for unit tests or alternative depth models.
            self.depth_model = depth_model
        else:
            self.depth_model = LiteMonoDepthBranch(
                checkpoint_dir=depth_checkpoint_dir,
                model_name=depth_model_name,
                feed_size=depth_feed_size,
                output_mode="normalized_depth",
                normalization_depth=depth_normalization_meters,
                freeze_pretrained=freeze_depth,
            )

        # Siamese across timestamps within Model R.
        self.rotation_aresunet = AResUNet()

        # Fig. 2 depicts separate Model R and Model T paths.
        if share_aresunet_between_models:
            self.translation_aresunet = self.rotation_aresunet
        else:
            self.translation_aresunet = AResUNet()

        # Model R:
        #   C_prev + C_curr + S_curr + D_curr
        rotation_input_channels = (
            2 * aresunet_output_channels
            + 1  # S_curr
            + 1  # D_curr
        )

        # Model T:
        #   C_prev + C_curr + S_curr + D_curr + rotation_map
        translation_input_channels = (
            2 * aresunet_output_channels
            + 1  # S_curr
            + 1  # D_curr
            + 3  # broadcast rotation vector
        )

        self.rotation_head = RotationHead(
            in_channels=rotation_input_channels,
            input_size=self.input_size,
            pool_size=self.rotation_pool_size,
        )

        self.translation_head = DirectionalTranslationHead(
            in_channels=translation_input_channels,
            input_size=self.input_size,
            decoder_type=self.translation_decoder_type,
            mlp_hidden_dims=self.translation_mlp_hidden_dims,
            projection_channels=(
                self.translation_projection_channels
            ),
            pool_size=(
                self.translation_pool_size
            ),
            aggregation_hidden_dim=(
                self.translation_aggregation_hidden_dim
            ),
            num_experts=(
                self.translation_num_experts
            ),
        )

    def train(
        self,
        mode: bool = True,
    ) -> "DeepDCTVO":
        """Set training mode while keeping frozen auxiliaries in eval mode.

        Calling ``model.train()`` recursively places all child modules in
        training mode. Frozen semantic and depth models should remain in
        evaluation mode so that BatchNorm statistics and dropout behavior
        do not change during pose-network training.
        """
        super().train(mode)

        if self.freeze_semantic_model:
            self.semantic_model.eval()

        if self.freeze_depth_model:
            self.depth_model.eval()

        return self

    def forward(
        self,
        image_prev: Tensor,
        image_curr: Tensor,
        depth_curr: Optional[Tensor] = None,
        rotation_for_translation: Optional[Tensor] = None,
        use_ground_truth_rotation: bool = False,
        return_intermediates: bool = False,
    ) -> ModelOutput:
        """Estimate rotation and directional translation.

        Args:
            image_prev:
                Previous RGB frame, shaped [B, 3, H, W].

            image_curr:
                Current RGB frame, shaped [B, 3, H, W].

            depth_curr:
                Optional externally supplied one-channel depth map D_k,
                shaped [B, 1, H, W]. It should already be normalized
                according to the selected depth pipeline.

                When supplied, this tensor takes precedence over the internal
                depth model, regardless of ``self.use_depth_cues``.

            rotation_for_translation:
                Optional [B, 3] rotation vector supplied to Model T.

            use_ground_truth_rotation:
                If True, ``rotation_for_translation`` is required and is
                supplied to Model T. If False, Model T uses Model R's
                prediction.

            return_intermediates:
                Include semantic maps, A-ResUNet inputs, feature maps, cue
                state indicators, and fusion tensors in the returned
                dictionary.

        Returns:
            Dictionary containing:

                rotation:
                    Predicted physical Euler rotation [B, 3].
                    This remains in radians for downstream evaluation,
                    trajectory reconstruction, and Model-T conditioning.

                rotation_normalized:
                    RotationHead regression output [B, 3] in normalized
                    rotation units. For A2, physical rotation is obtained as:

                        rotation = rotation_normalized * 0.175

                    This tensor is intended for the paper-style normalized
                    rotation loss.

                directional_translation:
                    Predicted local directional translation [B, 3].

                rotation_used_for_translation:
                    Physical rotation actually supplied to Model T [B, 3].
        """
        self._validate_image_inputs(
            image_prev=image_prev,
            image_curr=image_curr,
            rotation_for_translation=rotation_for_translation,
            use_ground_truth_rotation=use_ground_truth_rotation,
        )

        # ---------------------------------------------------------------
        # Depth cue D_k
        # ---------------------------------------------------------------
        # An externally supplied depth tensor always takes precedence.
        depth_was_supplied = depth_curr is not None

        if depth_curr is None:
            if self.use_depth_cues:
                if self.freeze_depth_model:
                    with torch.no_grad():
                        depth_curr = self.depth_model(image_curr)
                else:
                    depth_curr = self.depth_model(image_curr)
            else:
                # Preserve the expected one-channel fusion shape during
                # depth-disabled ablation experiments.
                depth_curr = torch.zeros_like(
                    image_curr[:, :1]
                )

        self._validate_depth(
            depth_curr=depth_curr,
            reference=image_curr,
        )

        # ---------------------------------------------------------------
        # Semantic cues S_(k-1) and S_k
        # ---------------------------------------------------------------
        if self.use_semantic_cues:
            if self.freeze_semantic_model:
                with torch.no_grad():
                    semantic_prev = self.semantic_model(
                        image_prev
                    )
                    semantic_curr = self.semantic_model(
                        image_curr
                    )
            else:
                semantic_prev = self.semantic_model(
                    image_prev
                )
                semantic_curr = self.semantic_model(
                    image_curr
                )
        else:
            # Preserve the four-channel A-ResUNet input shape while
            # performing a semantic-disabled ablation.
            semantic_prev = torch.zeros_like(
                image_prev[:, :1]
            )
            semantic_curr = torch.zeros_like(
                image_curr[:, :1]
            )

        self._validate_semantic_maps(
            semantic_prev=semantic_prev,
            semantic_curr=semantic_curr,
            reference=image_curr,
        )

        # ---------------------------------------------------------------
        # RGB-semantic fusion
        #
        # Eq. (14)/(19):
        #   CI_(k-1) = concat(I_(k-1), S_(k-1))
        #   CI_k     = concat(I_k, S_k)
        # ---------------------------------------------------------------
        ci_prev = torch.cat(
            [
                image_prev,
                semantic_prev,
            ],
            dim=1,
        )

        ci_curr = torch.cat(
            [
                image_curr,
                semantic_curr,
            ],
            dim=1,
        )

        if ci_prev.shape[1] != 4:
            raise RuntimeError(
                "Expected previous A-ResUNet input to have "
                f"4 channels, but received {ci_prev.shape[1]}."
            )

        if ci_curr.shape[1] != 4:
            raise RuntimeError(
                "Expected current A-ResUNet input to have "
                f"4 channels, but received {ci_curr.shape[1]}."
            )

        # ---------------------------------------------------------------
        # Model R
        #
        # The same rotation-branch A-ResUNet weights are reused across
        # timestamps.
        # ---------------------------------------------------------------
        c_prev_r = self.rotation_aresunet(
            ci_prev
        )

        c_curr_r = self.rotation_aresunet(
            ci_curr
        )

        self._validate_aresunet_outputs(
            c_prev=c_prev_r,
            c_curr=c_curr_r,
            branch_name="rotation",
        )

        # Model R fusion:
        #   [C_(k-1), C_k, S_k, D_k]
        rotation_features = torch.cat(
            [
                c_prev_r,
                c_curr_r,
                semantic_curr,
                depth_curr,
            ],
            dim=1,
        )

        # ---------------------------------------------------------------
        # A2 rotation-output convention
        #
        # RotationHead predicts in normalized rotation units.
        #
        # A1:
        #     scale = 1.0
        #     normalized == physical radians
        #
        # A2:
        #     scale = 0.175
        #     physical radians = normalized * 0.175
        #
        # Keeping both representations is intentional:
        #
        #     predicted_rotation_normalized
        #         -> used by the A2 rotation MAE objective
        #
        #     predicted_rotation
        #         -> physical Euler radians
        #         -> Model T conditioning
        #         -> evaluation
        #         -> CSV output
        #         -> trajectory reconstruction
        # ---------------------------------------------------------------
        (
            predicted_rotation_normalized,
            rotation_representation,
        ) = self.rotation_head.forward_with_representation(
            rotation_features
        )

        predicted_rotation = (
            predicted_rotation_normalized
            * self.rotation_normalization_scale
        )

        # ---------------------------------------------------------------
        # Rotation conditioning for Model T
        #
        # Normal/A1/A2:
        #     Model T receives Model R's predicted physical rotation.
        #
        # Track-A A3:
        #     Model T receives the dataset ground-truth physical rotation.
        #
        # IMPORTANT:
        # ``rotation_for_translation`` is already in physical Euler
        # radians. It must NOT be divided by rotation_normalization_scale.
        # Rotation normalization applies only to Model R's regression loss,
        # not to the conditioning vector supplied to Model T.
        # ---------------------------------------------------------------
        if use_ground_truth_rotation:
            # Validation in _validate_image_inputs() should guarantee
            # that this is not None.
            if rotation_for_translation is None:
                raise RuntimeError(
                    "rotation_for_translation must be supplied "
                    "when use_ground_truth_rotation=True."
                )

            rotation_used_for_translation = (
                rotation_for_translation
            )
        else:
            rotation_used_for_translation = (
                predicted_rotation
            )

        # ---------------------------------------------------------------
        # Model T
        #
        # The same translation-branch A-ResUNet weights are reused across
        # timestamps.
        # ---------------------------------------------------------------
        c_prev_t = self.translation_aresunet(
            ci_prev
        )

        c_curr_t = self.translation_aresunet(
            ci_curr
        )

        self._validate_aresunet_outputs(
            c_prev=c_prev_t,
            c_curr=c_curr_t,
            branch_name="translation",
        )

        rotation_map = self._vector_to_spatial_map(
            rotation_used_for_translation,
            spatial_size=semantic_curr.shape[-2:],
        )

        # Model T fusion:
        #   [C_(k-1), C_k, S_k, D_k, R_k]
        translation_features = torch.cat(
            [
                c_prev_t,
                c_curr_t,
                semantic_curr,
                depth_curr,
                rotation_map,
            ],
            dim=1,
        )

        directional_translation = (
            self.translation_head(
                translation_features
            )
        )

        outputs: ModelOutput = {
            # Physical Euler rotation in radians.
            #
            # Keep this key backward-compatible because evaluation,
            # trajectory reconstruction, diagnostics, and downstream
            # scripts already interpret outputs["rotation"] as physical
            # rotation.
            "rotation": predicted_rotation,

            # Regression-space rotation used by the A2 paper-style
            # normalized rotation loss.
            "rotation_normalized": (
                predicted_rotation_normalized
            ),

            "directional_translation": (
                directional_translation
            ),

            # Always physical rotation units:
            # - predicted radians during normal inference/training
            # - supplied GT radians during A3-style GT conditioning
            "rotation_used_for_translation": (
                rotation_used_for_translation
            ),
        }

        if return_intermediates:
            outputs.update(
                {
                    "rotation_representation": (
                        rotation_representation
                    ),
                    "rotation_physical": (
                        predicted_rotation
                    ),
                    "semantic_prev": semantic_prev,
                    "semantic_curr": semantic_curr,
                    "depth_curr": depth_curr,
                    "depth_was_supplied": torch.tensor(
                        depth_was_supplied,
                        device=image_curr.device,
                        dtype=torch.bool,
                    ),
                    "semantic_cues_enabled": torch.tensor(
                        self.use_semantic_cues,
                        device=image_curr.device,
                        dtype=torch.bool,
                    ),
                    "depth_cues_enabled": torch.tensor(
                        self.use_depth_cues,
                        device=image_curr.device,
                        dtype=torch.bool,
                    ),
                    "semantic_model_frozen": torch.tensor(
                        self.freeze_semantic_model,
                        device=image_curr.device,
                        dtype=torch.bool,
                    ),
                    "depth_model_frozen": torch.tensor(
                        self.freeze_depth_model,
                        device=image_curr.device,
                        dtype=torch.bool,
                    ),
                    "ci_prev": ci_prev,
                    "ci_curr": ci_curr,
                    "rotation_c_prev": c_prev_r,
                    "rotation_c_curr": c_curr_r,
                    "translation_c_prev": c_prev_t,
                    "translation_c_curr": c_curr_t,
                    "rotation_map": rotation_map,
                    "rotation_features": (
                        rotation_features
                    ),
                    "translation_features": (
                        translation_features
                    ),
                }
            )

            # -----------------------------------------------------------
            # Translation conditioning diagnostics
            #
            # Only available for the gated-expert translation decoder.
            # -----------------------------------------------------------
            if (
                self.translation_head.decoder_type
                == "gated_expert"
            ):
                conditioning = (
                    self.translation_head
                    .conditioning_diagnostics()
                )

                outputs.update(
                    {
                        "translation_representation": (
                            conditioning[
                                "representation"
                            ]
                        ),
                        "translation_gate_logits": (
                            conditioning[
                                "gate_logits"
                            ]
                        ),
                        "translation_gate_weights": (
                            conditioning[
                                "gate_weights"
                            ]
                        ),
                        "translation_expert_predictions": (
                            conditioning[
                                "expert_predictions"
                            ]
                        ),
                    }
                )

        return outputs

    
    def _semantic_cue(self, image: torch.Tensor) -> torch.Tensor:
        if not self.use_semantic_cues:
            return torch.zeros_like(image[:, :1])

        if self.freeze_semantic_model:
            with torch.no_grad():
                semantic = self.semantic_model(image)
        else:
            semantic = self.semantic_model(image)

        return semantic


    def _depth_cue(self, image: torch.Tensor) -> torch.Tensor:
        if not self.use_depth_cues:
            return torch.zeros_like(image[:, :1])

        if self.freeze_depth_model:
            with torch.no_grad():
                depth = self.depth_model(image)
        else:
            depth = self.depth_model(image)

        return depth

   
    @staticmethod
    def _validate_image_inputs(
        image_prev: torch.Tensor,
        image_curr: torch.Tensor,
        rotation_for_translation: Optional[torch.Tensor],
        use_ground_truth_rotation: bool,
    ) -> None:
        images = {
            "image_prev": image_prev,
            "image_curr": image_curr,
        }

        for name, tensor in images.items():
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"but received {type(tensor).__name__}."
                )

            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have shape [B, 3, H, W], "
                    f"but received {tuple(tensor.shape)}."
                )

            if tensor.shape[1] != 3:
                raise ValueError(
                    f"{name} must contain three RGB channels, "
                    f"but received {tensor.shape[1]}."
                )

            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} must be floating point, "
                    f"but received {tensor.dtype}."
                )

            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )

        if image_prev.shape != image_curr.shape:
            raise ValueError(
                "image_prev and image_curr must have identical shapes, "
                f"but received {tuple(image_prev.shape)} and "
                f"{tuple(image_curr.shape)}."
            )

        if image_prev.device != image_curr.device:
            raise ValueError(
                "image_prev and image_curr must be on the same device."
            )

        if image_prev.dtype != image_curr.dtype:
            raise TypeError(
                "image_prev and image_curr must have the same dtype."
            )

        if use_ground_truth_rotation:
            if rotation_for_translation is None:
                raise ValueError(
                    "rotation_for_translation is required when "
                    "use_ground_truth_rotation=True."
                )

            if not torch.is_tensor(rotation_for_translation):
                raise TypeError(
                    "rotation_for_translation must be a torch.Tensor."
                )

            expected_shape = (
                image_curr.shape[0],
                3,
            )

            if rotation_for_translation.shape != expected_shape:
                raise ValueError(
                    "rotation_for_translation must have shape "
                    f"{expected_shape}, but received "
                    f"{tuple(rotation_for_translation.shape)}."
                )

            if not rotation_for_translation.is_floating_point():
                raise TypeError(
                    "rotation_for_translation must be floating point."
                )

            if rotation_for_translation.device != image_curr.device:
                raise ValueError(
                    "rotation_for_translation and image_curr must be "
                    "on the same device."
                )

            if rotation_for_translation.dtype != image_curr.dtype:
                raise TypeError(
                    "rotation_for_translation and image_curr must have "
                    "the same dtype."
                )

            if not torch.isfinite(
                rotation_for_translation
            ).all():
                raise ValueError(
                    "rotation_for_translation contains NaN or "
                    "infinite values."
                )

    @staticmethod
    def _validate_depth(
        depth_curr: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(depth_curr):
            raise TypeError(
                "depth_curr must be a torch.Tensor, "
                f"but received {type(depth_curr).__name__}."
            )

        if depth_curr.ndim != 4:
            raise ValueError(
                "depth_curr must have shape [B, 1, H, W], "
                f"but received {tuple(depth_curr.shape)}."
            )

        expected_shape = (
            reference.shape[0],
            1,
            reference.shape[2],
            reference.shape[3],
        )

        if depth_curr.shape != expected_shape:
            raise ValueError(
                "depth_curr must have shape "
                f"{expected_shape}, but received "
                f"{tuple(depth_curr.shape)}."
            )

        if not depth_curr.is_floating_point():
            raise TypeError(
                "depth_curr must be floating point, "
                f"but received {depth_curr.dtype}."
            )

        if depth_curr.dtype != reference.dtype:
            raise TypeError(
                "depth_curr and image_curr must have the same dtype."
            )

        if depth_curr.device != reference.device:
            raise ValueError(
                "depth_curr and image_curr must be on the same device."
            )

        if not torch.isfinite(depth_curr).all():
            raise ValueError(
                "depth_curr contains NaN or infinite values."
            )
    
    def semantic_outputs(
        self,
        image: Tensor,
    ) -> Dict[str, Tensor]:
        """Expose LR-ASPP diagnostics without changing the main forward path.

        Returns the full logits, integer labels, and normalized semantic map
        generated by ``lraspp_fig2.py``.
        """
        return self.semantic_model.forward_all(image)
    
    

    @staticmethod
    def _vector_to_spatial_map(
        vector: Tensor,
        spatial_size: Tuple[int, int],
    ) -> Tensor:
        """Broadcast a [B, 3] rotation vector across height and width."""
        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(
                "Rotation must have shape [B, 3], "
                f"but received {tuple(vector.shape)}."
            )

        height, width = spatial_size

        return vector[:, :, None, None].expand(
            -1,
            -1,
            height,
            width,
        )

    def _validate_aresunet_outputs(
        self,
        c_prev: Tensor,
        c_curr: Tensor,
        branch_name: str,
    ) -> None:
        if c_prev.ndim != 4 or c_curr.ndim != 4:
            raise ValueError(
                f"{branch_name} A-ResUNet outputs must be [B, C, H, W]."
            )

        if c_prev.shape != c_curr.shape:
            raise ValueError(
                f"{branch_name} A-ResUNet outputs must have identical "
                f"shapes, but received {tuple(c_prev.shape)} and "
                f"{tuple(c_curr.shape)}."
            )

        if c_prev.shape[1] != self.aresunet_output_channels:
            raise ValueError(
                f"{branch_name} A-ResUNet returned "
                f"{c_prev.shape[1]} channels, but DeepDCTVO was configured "
                f"for {self.aresunet_output_channels}."
            )

    @staticmethod
    def _validate_semantic_maps(
        semantic_prev: Tensor,
        semantic_curr: Tensor,
        reference: Tensor,
    ) -> None:
        expected_shape = (
            reference.shape[0],
            1,
            reference.shape[2],
            reference.shape[3],
        )

        if semantic_prev.shape != expected_shape:
            raise ValueError(
                "semantic_prev must have shape "
                f"{expected_shape}, but received {tuple(semantic_prev.shape)}."
            )

        if semantic_curr.shape != expected_shape:
            raise ValueError(
                "semantic_curr must have shape "
                f"{expected_shape}, but received {tuple(semantic_curr.shape)}."
            )

        if semantic_prev.dtype != reference.dtype:
            raise TypeError(
                "semantic_prev and RGB input must have the same dtype."
            )

        if semantic_curr.dtype != reference.dtype:
            raise TypeError(
                "semantic_curr and RGB input must have the same dtype."
            )


