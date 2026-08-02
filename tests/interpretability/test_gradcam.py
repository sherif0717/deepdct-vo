"""Tests for regression Grad-CAM."""

import pytest
import torch
import torch.nn as nn

from deepdct.interpretability.gradcam import (
    RegressionGradCAM,
)
from deepdct.interpretability.targets import (
    regression_target,
)


class SmallPoseRegressionNetwork(nn.Module):
    """
    Minimal two-head spatial regression network.

    The output keys match DeepDCT-VO:
        rotation
        directional_translation
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.rotation_head = nn.Linear(
            in_features=8,
            out_features=3,
        )

        self.translation_head = nn.Linear(
            in_features=8,
            out_features=3,
        )

    def forward(
        self,
        image: torch.Tensor,
    ):
        features = self.features(image)

        pooled = self.pool(
            features
        ).flatten(1)

        return {
            "rotation": self.rotation_head(
                pooled
            ),
            "directional_translation": (
                self.translation_head(pooled)
            ),
        }


def test_regression_target_selects_rotation_component() -> None:
    outputs = {
        "rotation": torch.tensor(
            [[1.0, 2.0, 3.0]]
        ),
        "directional_translation": torch.tensor(
            [[4.0, 5.0, 6.0]]
        ),
    }

    target = regression_target(
        outputs=outputs,
        target="rotation_y",
        sample_index=0,
    )

    assert target.ndim == 0
    assert target.item() == pytest.approx(2.0)


def test_regression_target_selects_translation_component() -> None:
    outputs = {
        "rotation": torch.tensor(
            [[1.0, 2.0, 3.0]]
        ),
        "directional_translation": torch.tensor(
            [[4.0, 5.0, 6.0]]
        ),
    }

    target = regression_target(
        outputs=outputs,
        target="translation_z",
        sample_index=0,
    )

    assert target.ndim == 0
    assert target.item() == pytest.approx(6.0)


def test_regression_target_computes_rotation_norm() -> None:
    outputs = {
        "rotation": torch.tensor(
            [[3.0, 4.0, 0.0]]
        ),
        "directional_translation": torch.zeros(
            1,
            3,
        ),
    }

    target = regression_target(
        outputs=outputs,
        target="rotation_norm",
        sample_index=0,
    )

    assert target.item() == pytest.approx(5.0)


def test_regression_target_computes_translation_norm() -> None:
    outputs = {
        "rotation": torch.zeros(
            1,
            3,
        ),
        "directional_translation": torch.tensor(
            [[0.0, 5.0, 12.0]]
        ),
    }

    target = regression_target(
        outputs=outputs,
        target="translation_norm",
        sample_index=0,
    )

    assert target.item() == pytest.approx(13.0)


def test_regression_target_rejects_unknown_target() -> None:
    outputs = {
        "rotation": torch.zeros(
            1,
            3,
        ),
        "directional_translation": torch.zeros(
            1,
            3,
        ),
    }

    with pytest.raises(ValueError):
        regression_target(
            outputs=outputs,
            target="unknown_target",
        )


def test_regression_gradcam_returns_normalized_spatial_map() -> None:
    torch.manual_seed(7)

    model = SmallPoseRegressionNetwork()
    model.eval()

    image = torch.randn(
        1,
        3,
        24,
        32,
        requires_grad=True,
    )

    with RegressionGradCAM(
        model=model,
        target_layer="features.2",
    ) as gradcam:
        cam, outputs = gradcam.generate(
            model_inputs={
                "image": image,
            },
            target_function=(
                lambda model_outputs: regression_target(
                    outputs=model_outputs,
                    target="translation_z",
                    sample_index=0,
                )
            ),
            output_size=(24, 32),
        )

    assert isinstance(outputs, dict)

    assert cam.shape == (
        1,
        1,
        24,
        32,
    )

    assert cam.device.type == "cpu"
    assert torch.isfinite(cam).all()

    assert cam.min().item() >= 0.0
    assert cam.max().item() <= 1.0


def test_regression_gradcam_supports_rotation_target() -> None:
    torch.manual_seed(11)

    model = SmallPoseRegressionNetwork()
    model.eval()

    image = torch.randn(
        1,
        3,
        20,
        20,
        requires_grad=True,
    )

    with RegressionGradCAM(
        model=model,
        target_layer="features.2",
    ) as gradcam:
        cam, _ = gradcam.generate(
            model_inputs={
                "image": image,
            },
            target_function=(
                lambda outputs: regression_target(
                    outputs=outputs,
                    target="rotation_norm",
                    sample_index=0,
                )
            ),
            output_size=(20, 20),
        )

    assert cam.shape == (
        1,
        1,
        20,
        20,
    )

    assert torch.isfinite(cam).all()


def test_regression_gradcam_rejects_non_spatial_target_layer() -> None:
    model = SmallPoseRegressionNetwork()
    model.eval()

    image = torch.randn(
        1,
        3,
        16,
        16,
        requires_grad=True,
    )

    with RegressionGradCAM(
        model=model,
        target_layer="rotation_head",
    ) as gradcam:
        with pytest.raises(
            ValueError,
            match="expects BCHW",
        ):
            gradcam.generate(
                model_inputs={
                    "image": image,
                },
                target_function=(
                    lambda outputs: regression_target(
                        outputs=outputs,
                        target="rotation_x",
                        sample_index=0,
                    )
                ),
                output_size=(16, 16),
            )


def test_regression_gradcam_removes_hooks() -> None:
    model = SmallPoseRegressionNetwork()

    gradcam = RegressionGradCAM(
        model=model,
        target_layer="features.2",
    )

    gradcam.close()

    assert gradcam._forward_handle is not None
    assert gradcam._backward_handle is not None