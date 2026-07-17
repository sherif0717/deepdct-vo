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
    """

    def __init__(
        self,
        aresunet_output_channels: int = 1,
        input_size: Tuple[int, int] = (120, 120),
        pretrained_semantic: bool = True,
        freeze_semantic: bool = True,
        normalize_semantic_input: bool = True,
        normalize_semantic_map: bool = True,
        share_aresunet_between_models: bool = False,
            # Lite-Mono configuration
        depth_checkpoint_dir: Optional[PathLike] = (
            "weights/lite-mono-tiny-640x192"
        ),
        depth_model_name: str = "lite-mono-tiny",
        depth_feed_size: Tuple[int, int] = (192, 640),
        freeze_depth: bool = True,
        depth_normalization_meters: float = 80.0,
        depth_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

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

        self.semantic_model = LRASPPSemanticBranch(
            pretrained=pretrained_semantic,
            freeze_pretrained=freeze_semantic,
            normalize_input=normalize_semantic_input,
            normalize_map=normalize_semantic_map,
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
        )

        self.translation_head = DirectionalTranslationHead(
            in_channels=translation_input_channels,
            input_size=self.input_size,
        )

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
                Current one-channel depth map D_k, shaped [B, 1, H, W].
                It should already be normalized according to the selected
                depth pipeline.

            rotation_for_translation:
                Optional [B, 3] rotation vector supplied to Model T.

            use_ground_truth_rotation:
                If True, ``rotation_for_translation`` is required and is
                supplied to Model T. If False, Model T uses Model R's
                prediction.

            return_intermediates:
                Include semantic maps, A-ResUNet inputs, feature maps, and
                fusion tensors in the returned dictionary.

        Returns:
            Dictionary containing:

                rotation:
                    Predicted rotation [B, 3].

                directional_translation:
                    Predicted local directional translation [B, 3].

                rotation_used_for_translation:
                    Rotation actually supplied to Model T [B, 3].
        """
        self._validate_image_inputs(
            image_prev=image_prev,
            image_curr=image_curr,
            rotation_for_translation=rotation_for_translation,
            use_ground_truth_rotation=use_ground_truth_rotation,
        )

        depth_was_supplied = depth_curr is not None

        if depth_curr is None:
            depth_curr = self.depth_model(image_curr)

        self._validate_depth(
            depth_curr=depth_curr,
            reference=image_curr,
        )

        # Updated lraspp_fig2.py returns [B, 1, H, W] directly.
        semantic_prev = self.semantic_model(image_prev)
        semantic_curr = self.semantic_model(image_curr)

        self._validate_semantic_maps(
            semantic_prev=semantic_prev,
            semantic_curr=semantic_curr,
            reference=image_curr,
        )

        # Eq. (14)/(19): concatenate RGB and one-channel semantic map.
        ci_prev = torch.cat(
            [image_prev, semantic_prev],
            dim=1,
        )
        ci_curr = torch.cat(
            [image_curr, semantic_curr],
            dim=1,
        )

        # Model R: same A-ResUNet weights are reused across timestamps.
        c_prev_r = self.rotation_aresunet(ci_prev)
        c_curr_r = self.rotation_aresunet(ci_curr)

        self._validate_aresunet_outputs(
            c_prev=c_prev_r,
            c_curr=c_curr_r,
            branch_name="rotation",
        )

        rotation_features = torch.cat(
            [
                c_prev_r,
                c_curr_r,
                semantic_curr,
                depth_curr,
            ],
            dim=1,
        )

        predicted_rotation = self.rotation_head(
            rotation_features
        )

        if use_ground_truth_rotation:
            rotation_used_for_translation = (
                rotation_for_translation
            )
        else:
            rotation_used_for_translation = (
                predicted_rotation
            )

        # Model T: same translation A-ResUNet is reused across time.
        c_prev_t = self.translation_aresunet(ci_prev)
        c_curr_t = self.translation_aresunet(ci_curr)

        self._validate_aresunet_outputs(
            c_prev=c_prev_t,
            c_curr=c_curr_t,
            branch_name="translation",
        )

        rotation_map = self._vector_to_spatial_map(
            rotation_used_for_translation,
            spatial_size=semantic_curr.shape[-2:],
        )

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

        directional_translation = self.translation_head(
            translation_features
        )

        outputs: ModelOutput = {
            "rotation": predicted_rotation,
            "directional_translation": directional_translation,
            "rotation_used_for_translation": rotation_used_for_translation,
        }

        if return_intermediates:
            outputs.update(
                {
                    "semantic_prev": semantic_prev,
                    "semantic_curr": semantic_curr,
                    "depth_curr": depth_curr,
                    "depth_was_supplied": torch.tensor(
                        depth_was_supplied,
                        device=image_curr.device,
                    ),
                    "ci_prev": ci_prev,
                    "ci_curr": ci_curr,
                    "rotation_c_prev": c_prev_r,
                    "rotation_c_curr": c_curr_r,
                    "translation_c_prev": c_prev_t,
                    "translation_c_curr": c_curr_t,
                    "rotation_map": rotation_map,
                    "rotation_features": rotation_features,
                    "translation_features": translation_features,
                }
            )

        return outputs

    
   
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


