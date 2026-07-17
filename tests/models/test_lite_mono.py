"""Unit tests for the Lite-Mono auxiliary depth branch.

Run from the repository root with:

    pytest tests/models/test_lite_mono.py -v

The default tests do not require pretrained checkpoints. A checkpoint smoke
test runs automatically only when the expected weight files are present under:

    weights/lite-mono-tiny-640x192/
"""

from pathlib import Path

import pytest
import torch

from deepdct.models.auxiliary.lite_mono import (
    LiteMonoDepthBranch,
    disparity_to_depth,
)


CHECKPOINT_DIR = Path("weights/lite-mono-tiny-640x192")


@pytest.fixture(scope="module")
def random_model() -> LiteMonoDepthBranch:
    """Construct a randomly initialized Lite-Mono-Tiny model."""
    model = LiteMonoDepthBranch(
        checkpoint_dir=None,
        model_name="lite-mono-tiny",
        feed_size=(192, 640),
        min_depth=0.1,
        max_depth=100.0,
        normalization_depth=80.0,
        output_mode="normalized_depth",
        freeze_pretrained=False,
    )
    model.eval()
    return model


@pytest.fixture
def image_curr() -> torch.Tensor:
    """Return a valid current RGB frame in [0, 1]."""
    return torch.rand(
        1,
        3,
        120,
        120,
        dtype=torch.float32,
    )


def test_constructs_without_checkpoint() -> None:
    model = LiteMonoDepthBranch(
        checkpoint_dir=None,
        model_name="lite-mono-tiny",
        feed_size=(192, 640),
        freeze_pretrained=False,
    )

    assert isinstance(model, LiteMonoDepthBranch)
    assert model.feed_height == 192
    assert model.feed_width == 640
    assert model.model_name == "lite-mono-tiny"


def test_forward_returns_single_channel_depth(
    random_model: LiteMonoDepthBranch,
    image_curr: torch.Tensor,
) -> None:
    with torch.no_grad():
        depth_curr = random_model(image_curr)

    assert depth_curr.shape == (1, 1, 120, 120)
    assert depth_curr.dtype == image_curr.dtype
    assert torch.isfinite(depth_curr).all()


def test_forward_preserves_input_spatial_size(
    random_model: LiteMonoDepthBranch,
) -> None:
    image = torch.rand(1, 3, 96, 160)

    with torch.no_grad():
        depth = random_model(image)

    assert depth.shape[-2:] == image.shape[-2:]


def test_normalized_depth_is_in_unit_interval(
    random_model: LiteMonoDepthBranch,
    image_curr: torch.Tensor,
) -> None:
    with torch.no_grad():
        depth_curr = random_model(image_curr)

    assert depth_curr.min().item() >= 0.0
    assert depth_curr.max().item() <= 1.0


def test_forward_all_returns_expected_keys(
    random_model: LiteMonoDepthBranch,
    image_curr: torch.Tensor,
) -> None:
    with torch.no_grad():
        outputs = random_model.forward_all(image_curr)

    assert set(outputs) == {
        "disparity",
        "scaled_disparity",
        "depth",
        "normalized_depth",
    }

    for output in outputs.values():
        assert output.shape == (1, 1, 120, 120)
        assert torch.isfinite(output).all()


def test_forward_matches_selected_output_mode(
    random_model: LiteMonoDepthBranch,
    image_curr: torch.Tensor,
) -> None:
    with torch.no_grad():
        direct = random_model(image_curr)
        all_outputs = random_model.forward_all(image_curr)

    assert torch.allclose(
        direct,
        all_outputs["normalized_depth"],
    )


@pytest.mark.parametrize(
    "output_mode",
    [
        "disparity",
        "scaled_disparity",
        "depth",
        "normalized_depth",
    ],
)
def test_all_output_modes_return_one_channel_map(
    output_mode: str,
    image_curr: torch.Tensor,
) -> None:
    model = LiteMonoDepthBranch(
        checkpoint_dir=None,
        model_name="lite-mono-tiny",
        feed_size=(192, 640),
        output_mode=output_mode,
        freeze_pretrained=False,
    ).eval()

    with torch.no_grad():
        output = model(image_curr)

    assert output.shape == (1, 1, 120, 120)
    assert torch.isfinite(output).all()


def test_disparity_to_depth_known_values() -> None:
    disparity = torch.tensor(
        [[[[0.0, 0.5, 1.0]]]],
        dtype=torch.float32,
    )

    scaled_disparity, depth = disparity_to_depth(
        disparity=disparity,
        min_depth=0.1,
        max_depth=100.0,
    )

    min_disp = 1.0 / 100.0
    max_disp = 1.0 / 0.1

    expected_scaled = torch.tensor(
        [[[
            [
                min_disp,
                min_disp + 0.5 * (max_disp - min_disp),
                max_disp,
            ]
        ]]],
        dtype=torch.float32,
    )

    assert torch.allclose(
        scaled_disparity,
        expected_scaled,
    )
    assert torch.allclose(
        depth,
        1.0 / expected_scaled,
    )


def test_disparity_to_depth_rejects_nonpositive_minimum() -> None:
    disparity = torch.rand(1, 1, 4, 4)

    with pytest.raises(ValueError, match="min_depth"):
        disparity_to_depth(
            disparity,
            min_depth=0.0,
            max_depth=100.0,
        )


def test_disparity_to_depth_rejects_invalid_maximum() -> None:
    disparity = torch.rand(1, 1, 4, 4)

    with pytest.raises(ValueError, match="max_depth"):
        disparity_to_depth(
            disparity,
            min_depth=1.0,
            max_depth=1.0,
        )


def test_freeze_disables_parameter_gradients() -> None:
    model = LiteMonoDepthBranch(
        checkpoint_dir=None,
        model_name="lite-mono-tiny",
        freeze_pretrained=False,
    )

    model.freeze()

    assert model.freeze_pretrained is True
    assert model.encoder.training is False
    assert model.decoder.training is False
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )


def test_unfreeze_enables_parameter_gradients() -> None:
    model = LiteMonoDepthBranch(
        checkpoint_dir=None,
        model_name="lite-mono-tiny",
        freeze_pretrained=True,
    )

    model.unfreeze()

    assert model.freeze_pretrained is False
    assert all(
        parameter.requires_grad
        for parameter in model.parameters()
    )


def test_train_keeps_frozen_modules_in_eval_mode() -> None:
    model = LiteMonoDepthBranch(
        checkpoint_dir=None,
        model_name="lite-mono-tiny",
        freeze_pretrained=True,
    )

    model.train()

    assert model.training is True
    assert model.encoder.training is False
    assert model.decoder.training is False


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (1, 1, 120, 120),
        (1, 4, 120, 120),
        (1, 3, 120),
        (3, 120, 120),
    ],
)
def test_rejects_invalid_input_shape(
    random_model: LiteMonoDepthBranch,
    invalid_shape,
) -> None:
    image = torch.rand(*invalid_shape)

    with pytest.raises(ValueError):
        random_model(image)


def test_rejects_integer_input(
    random_model: LiteMonoDepthBranch,
) -> None:
    image = torch.randint(
        0,
        256,
        (1, 3, 120, 120),
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        random_model(image)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_nonfinite_input(
    random_model: LiteMonoDepthBranch,
    invalid_value: float,
) -> None:
    image = torch.rand(1, 3, 120, 120)
    image[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError):
        random_model(image)


@pytest.mark.parametrize(
    "invalid_model",
    [
        "lite-mono-micro",
        "",
    ],
)
def test_rejects_unsupported_model_name(
    invalid_model: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported Lite-Mono"):
        LiteMonoDepthBranch(
            checkpoint_dir=None,
            model_name=invalid_model,
        )


@pytest.mark.parametrize(
    "invalid_feed_size",
    [
        (0, 640),
        (192, 0),
        (-1, 640),
        (192,),
    ],
)
def test_rejects_invalid_feed_size(
    invalid_feed_size,
) -> None:
    with pytest.raises(ValueError):
        LiteMonoDepthBranch(
            checkpoint_dir=None,
            feed_size=invalid_feed_size,
        )


@pytest.mark.parametrize(
    "invalid_mode",
    [
        "metric",
        "inverse_depth",
        "",
    ],
)
def test_rejects_invalid_output_mode(
    invalid_mode: str,
) -> None:
    with pytest.raises(ValueError, match="output_mode"):
        LiteMonoDepthBranch(
            checkpoint_dir=None,
            output_mode=invalid_mode,
        )


def test_missing_checkpoint_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        LiteMonoDepthBranch(
            checkpoint_dir="weights/does-not-exist",
            model_name="lite-mono-tiny",
        )


def test_checkpoint_directory_requires_both_files(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "lite-mono"
    checkpoint_dir.mkdir()

    torch.save({}, checkpoint_dir / "encoder.pth")

    with pytest.raises(FileNotFoundError, match="depth.pth"):
        LiteMonoDepthBranch(
            checkpoint_dir=checkpoint_dir,
            model_name="lite-mono-tiny",
        )


@pytest.mark.skipif(
    not (
        (CHECKPOINT_DIR / "encoder.pth").is_file()
        and (CHECKPOINT_DIR / "depth.pth").is_file()
    ),
    reason="Pretrained Lite-Mono checkpoint is not available.",
)
def test_pretrained_checkpoint_loads_and_runs() -> None:
    model = LiteMonoDepthBranch(
        checkpoint_dir=CHECKPOINT_DIR,
        model_name="lite-mono-tiny",
        output_mode="normalized_depth",
        normalization_depth=80.0,
        freeze_pretrained=True,
    ).eval()

    image = torch.rand(1, 3, 120, 120)

    with torch.no_grad():
        depth = model(image)

    assert model.feed_height == 192
    assert model.feed_width == 640
    assert depth.shape == (1, 1, 120, 120)
    assert depth.min().item() >= 0.0
    assert depth.max().item() <= 1.0
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )
