"""Unit tests for DeepDCT-VO rotation and translation heads."""

import pytest
import torch

from deepdct.models.pose_head import (
    DirectionalTranslationHead,
    RegressionHead,
    RotationHead,
)


@pytest.fixture
def rotation_head():
    head = RotationHead(
        in_channels=4,
        input_size=(120, 120),
        dropout=0.2,
        negative_slope=0.01,
    )
    head.eval()
    return head


@pytest.fixture
def translation_head():
    head = DirectionalTranslationHead(
        in_channels=7,
        input_size=(120, 120),
        dropout=0.2,
        negative_slope=0.01,
    )
    head.eval()
    return head


def test_rotation_head_returns_three_values(rotation_head):
    x = torch.randn(2, 4, 120, 120)

    with torch.no_grad():
        output = rotation_head(x)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_translation_head_returns_three_values(translation_head):
    x = torch.randn(2, 7, 120, 120)

    with torch.no_grad():
        output = translation_head(x)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_rotation_head_preserves_batch_size(rotation_head, batch_size):
    x = torch.randn(batch_size, 4, 120, 120)

    with torch.no_grad():
        output = rotation_head(x)

    assert output.shape == (batch_size, 3)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_translation_head_preserves_batch_size(
    translation_head,
    batch_size,
):
    x = torch.randn(batch_size, 7, 120, 120)

    with torch.no_grad():
        output = translation_head(x)

    assert output.shape == (batch_size, 3)


@pytest.mark.parametrize(
    "height,width",
    [(120, 120), (96, 160), (64, 64)],
)
def test_rotation_head_accepts_variable_spatial_sizes(
    rotation_head,
    height,
    width,
):
    x = torch.randn(2, 4, height, width)

    with torch.no_grad():
        output = rotation_head(x)

    assert output.shape == (2, 3)


@pytest.mark.parametrize(
    "height,width",
    [(120, 120), (96, 160), (64, 64)],
)
def test_translation_head_accepts_variable_spatial_sizes(
    translation_head,
    height,
    width,
):
    x = torch.randn(2, 7, height, width)

    with torch.no_grad():
        output = translation_head(x)

    assert output.shape == (2, 3)


def test_rotation_head_is_differentiable():
    head = RotationHead(
        in_channels=4,
        input_size=(32, 32),
        dropout=0.0,
    )
    x = torch.randn(2, 4, 32, 32, requires_grad=True)

    head(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert head.conv.weight.grad is not None
    assert head.dense.weight.grad is not None


def test_translation_head_is_differentiable():
    head = DirectionalTranslationHead(
        in_channels=7,
        input_size=(32, 32),
        dropout=0.0,
    )
    x = torch.randn(2, 7, 32, 32, requires_grad=True)

    head(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert head.conv.weight.grad is not None
    assert head.dense.weight.grad is not None


def test_rotation_head_matches_model_r_channel_count(rotation_head):
    assert isinstance(rotation_head, RegressionHead)
    assert rotation_head.conv.in_channels == 4
    assert rotation_head.conv.out_channels == 1
    assert rotation_head.conv.kernel_size == (3, 3)
    assert rotation_head.conv.padding == (1, 1)


def test_translation_head_matches_model_t_channel_count(translation_head):
    assert isinstance(translation_head, RegressionHead)
    assert translation_head.conv.in_channels == 7
    assert translation_head.conv.out_channels == 1
    assert translation_head.conv.kernel_size == (3, 3)
    assert translation_head.conv.padding == (1, 1)


def test_dense_layers_have_three_outputs(
    rotation_head,
    translation_head,
):
    assert rotation_head.dense.in_features == 120 * 120
    assert translation_head.dense.in_features == 120 * 120
    assert rotation_head.dense.out_features == 3
    assert translation_head.dense.out_features == 3


def test_heads_use_leaky_relu_output_activation(
    rotation_head,
    translation_head,
):
    assert isinstance(
        rotation_head.output_activation,
        torch.nn.LeakyReLU,
    )
    assert isinstance(
        translation_head.output_activation,
        torch.nn.LeakyReLU,
    )
    assert rotation_head.output_activation.negative_slope == pytest.approx(
        0.01
    )
    assert translation_head.output_activation.negative_slope == pytest.approx(
        0.01
    )


@pytest.mark.parametrize(
    "head_class,input_channels",
    [
        (RotationHead, 4),
        (DirectionalTranslationHead, 7),
    ],
)
def test_heads_reject_wrong_channel_count(
    head_class,
    input_channels,
):
    head = head_class(
        in_channels=input_channels,
        input_size=(32, 32),
    )
    x = torch.randn(2, input_channels + 1, 32, 32)

    with pytest.raises(ValueError, match="channel"):
        head(x)


@pytest.mark.parametrize(
    "invalid_shape",
    [(4, 120, 120), (2, 4, 120), (2, 4)],
)
def test_rotation_head_rejects_non_4d_input(
    rotation_head,
    invalid_shape,
):
    x = torch.randn(*invalid_shape)

    with pytest.raises(ValueError):
        rotation_head(x)


def test_rotation_head_rejects_integer_input(rotation_head):
    x = torch.randint(
        0,
        10,
        (2, 4, 120, 120),
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        rotation_head(x)


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_rotation_head_rejects_non_finite_input(
    rotation_head,
    invalid_value,
):
    x = torch.randn(2, 4, 120, 120)
    x[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError):
        rotation_head(x)


@pytest.mark.parametrize("invalid_channels", [0, -1])
def test_regression_head_rejects_invalid_channel_configuration(
    invalid_channels,
):
    with pytest.raises(ValueError):
        RegressionHead(in_channels=invalid_channels)


@pytest.mark.parametrize(
    "invalid_size",
    [(0, 120), (120, 0), (-1, 120), (120,)],
)
def test_regression_head_rejects_invalid_input_size(invalid_size):
    with pytest.raises(ValueError):
        RegressionHead(
            in_channels=4,
            input_size=invalid_size,
        )


@pytest.mark.parametrize("invalid_dropout", [-0.1, 1.0, 1.1])
def test_regression_head_rejects_invalid_dropout(invalid_dropout):
    with pytest.raises(ValueError):
        RegressionHead(
            in_channels=4,
            dropout=invalid_dropout,
        )


def test_regression_head_rejects_negative_leaky_relu_slope():
    with pytest.raises(ValueError):
        RegressionHead(
            in_channels=4,
            negative_slope=-0.01,
        )
