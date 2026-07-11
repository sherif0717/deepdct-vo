import pytest
import torch

from deepdct.models.blocks import AResUNet


IN_CHANNELS = 4
EXPECTED_OUT_CHANNELS = 1


def test_aresunet_forward_pass_completes():
    """The complete encoder-decoder pipeline should execute without error."""
    model = AResUNet()
    x = torch.randn(2, IN_CHANNELS, 64, 96)

    output = model(x)

    assert isinstance(output, torch.Tensor)


def test_aresunet_output_spatial_dimensions():
    """A-ResUNet should restore the input spatial resolution."""
    model = AResUNet()
    x = torch.randn(2, IN_CHANNELS, 64, 96)

    output = model(x)

    assert output.shape[-2:] == x.shape[-2:]


def test_aresunet_output_channel_count():
    """The decoder should produce the expected number of output channels."""
    model = AResUNet()
    x = torch.randn(2, IN_CHANNELS, 64, 96)

    output = model(x)

    assert output.shape[1] == EXPECTED_OUT_CHANNELS


def test_aresunet_preserves_batch_size():
    """Encoder-decoder integration should preserve the input batch size."""
    model = AResUNet()
    x = torch.randn(3, IN_CHANNELS, 64, 96)

    output = model(x)

    assert output.shape[0] == x.shape[0]


@pytest.mark.parametrize(
    "batch_size,height,width",
    [
        (1, 32, 32),
        (2, 64, 64),
        (1, 64, 96),
        (3, 96, 128),
        (2, 128, 160),
    ],
)
def test_aresunet_variable_input_sizes(batch_size, height, width):
    """
    A-ResUNet should support different batch sizes and spatial dimensions.

    Spatial dimensions are chosen to be divisible by the encoder's total
    downsampling factor.
    """
    model = AResUNet()
    x = torch.randn(batch_size, IN_CHANNELS, height, width)

    output = model(x)

    assert output.shape == (
        batch_size,
        EXPECTED_OUT_CHANNELS,
        height,
        width,
    )