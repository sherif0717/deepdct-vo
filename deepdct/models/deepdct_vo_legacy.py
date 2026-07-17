"""Paper-style DeepDCT-VO orchestration."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .auxiliary.lraspp import LRASPPSemanticBranch
from .blocks import AResUNet
from .pose_head import (
    DirectionalTranslationHead,
    RotationHead,
)


class DeepDCTVO(nn.Module):
    """DeepDCT-VO with separate rotation and translation models.

    Within each model, one A-ResUNet is reused for the two timestamps.

    Model R:
        cat(C_prev, C_curr, S_curr, D_curr) -> rotation

    Model T:
        cat(C_prev, C_curr, S_curr, D_curr, rotation_map)
            -> directional translation
    """

    def __init__(
        self,
        aresunet_output_channels: int = 1,
        semantic_classes: int = 21,
        input_size=(120, 120),
        pretrained_semantic: bool = True,
        freeze_semantic: bool = True,
        share_aresunet_between_models: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.semantic_classes = semantic_classes

        self.semantic_model = LRASPPSemanticBranch(
            pretrained=pretrained_semantic,
            freeze_pretrained=freeze_semantic,
        )

        # Each is Siamese across time.
        self.rotation_aresunet = AResUNet()

        if share_aresunet_between_models:
            self.translation_aresunet = self.rotation_aresunet
        else:
            self.translation_aresunet = AResUNet()

        # C_prev + C_curr + S_curr + D_curr
        rotation_input_channels = (
            2 * aresunet_output_channels + 1 + 1
        )

        # C_prev + C_curr + S_curr + D_curr + rotation_map(3)
        translation_input_channels = (
            2 * aresunet_output_channels + 1 + 1 + 3
        )

        self.rotation_head = RotationHead(
            in_channels=rotation_input_channels,
            input_size=input_size,
        )

        self.translation_head = DirectionalTranslationHead(
            in_channels=translation_input_channels,
            input_size=input_size,
        )

    def forward(
        self,
        image_prev,
        image_curr,
        depth_curr,
        rotation_for_translation: Optional[torch.Tensor] = None,
        use_ground_truth_rotation: bool = False,
        return_intermediates: bool = False,
    ):
        self._validate_inputs(
            image_prev,
            image_curr,
            depth_curr,
        )

        semantic_prev = self._semantic_map(image_prev)
        semantic_curr = self._semantic_map(image_curr)

        ci_prev = torch.cat(
            [image_prev, semantic_prev],
            dim=1,
        )

        ci_curr = torch.cat(
            [image_curr, semantic_curr],
            dim=1,
        )

        # Model R: Siamese A-ResUNet application.
        c_prev_r = self.rotation_aresunet(ci_prev)
        c_curr_r = self.rotation_aresunet(ci_curr)

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
            if rotation_for_translation is None:
                raise ValueError(
                    "rotation_for_translation is required when "
                    "use_ground_truth_rotation=True."
                )
            translation_rotation = rotation_for_translation
        else:
            translation_rotation = predicted_rotation

        # Model T: distinct Siamese A-ResUNet in the paper-style model.
        c_prev_t = self.translation_aresunet(ci_prev)
        c_curr_t = self.translation_aresunet(ci_curr)

        rotation_map = self._vector_to_spatial_map(
            translation_rotation,
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

        outputs = {
            "rotation": predicted_rotation,
            "directional_translation": directional_translation,
        }

        if return_intermediates:
            outputs.update(
                {
                    "semantic_prev": semantic_prev,
                    "semantic_curr": semantic_curr,
                    "depth_curr": depth_curr,
                    "ci_prev": ci_prev,
                    "ci_curr": ci_curr,
                    "rotation_c_prev": c_prev_r,
                    "rotation_c_curr": c_curr_r,
                    "translation_c_prev": c_prev_t,
                    "translation_c_curr": c_curr_t,
                }
            )

        return outputs

    def _semantic_map(self, image):
        logits = self.semantic_model(image)

        class_ids = logits.argmax(
            dim=1,
            keepdim=True,
        )

        denominator = max(
            self.semantic_classes - 1,
            1,
        )

        return class_ids.to(image.dtype) / denominator

    @staticmethod
    def _vector_to_spatial_map(vector, spatial_size):
        """Broadcast [B, 3] rotation across H and W."""
        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(
                "Rotation must have shape [B, 3], "
                f"got {tuple(vector.shape)}."
            )

        return vector[:, :, None, None].expand(
            -1,
            -1,
            spatial_size[0],
            spatial_size[1],
        )

    def _validate_inputs(
        self,
        image_prev,
        image_curr,
        depth_curr,
    ):
        if image_prev.ndim != 4 or image_prev.shape[1] != 3:
            raise ValueError(
                "image_prev must have shape [B, 3, H, W]."
            )

        if image_curr.shape != image_prev.shape:
            raise ValueError(
                "image_prev and image_curr must have identical shapes."
            )

        if depth_curr.ndim != 4 or depth_curr.shape[1] != 1:
            raise ValueError(
                "depth_curr must have shape [B, 1, H, W]."
            )

        if depth_curr.shape[0] != image_curr.shape[0]:
            raise ValueError(
                "Depth and image batch sizes must match."
            )

        if depth_curr.shape[-2:] != image_curr.shape[-2:]:
            raise ValueError(
                "Depth and image spatial dimensions must match."
            )