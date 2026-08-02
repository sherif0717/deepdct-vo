import torch

from deepdct.models.attention import AttentionDownBlock
from deepdct.models.attention import AttentionUpBlock


def test_attention_down_block_preserves_shape():
    block = AttentionDownBlock(channels=8)

    x = torch.randn(2, 8, 30, 30)
    y = block(x)

    assert y.shape == x.shape


def test_attention_down_block_is_differentiable():
    block = AttentionDownBlock(channels=8)

    x = torch.randn(2, 8, 30, 30, requires_grad=True)
    y = block(x)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_attention_down_block_retains_attention_map():
    channels = 8

    block = AttentionDownBlock(
        channels=channels,
    )

    block.eval()

    x = torch.randn(
        1,
        channels,
        32,
        32,
    )

    with torch.no_grad():
        output = block(x)

    assert block.last_attention_map is not None

    attention_map = block.last_attention_map

    assert attention_map.shape == (
        1,
        channels,
        32,
        32,
    )

    assert output.shape == x.shape
    assert torch.isfinite(attention_map).all()
    assert attention_map.min().item() >= 0.0
    assert attention_map.max().item() <= 1.0

    def test_attention_up_block_retains_attention_map():
        channels = 8

        block = AttentionUpBlock(
            channels=channels,
        )

        block.eval()

        x = torch.randn(
            1,
            channels,
            32,
            32,
        )

        with torch.no_grad():
            output = block(x)

        assert block.last_attention_map is not None

        attention_map = block.last_attention_map

        assert attention_map.shape == (
            1,
            1,
            32,
            32,
        )

        assert output.shape == x.shape
        assert torch.isfinite(attention_map).all()
        assert attention_map.min().item() >= 0.0
        assert attention_map.max().item() <= 1.0