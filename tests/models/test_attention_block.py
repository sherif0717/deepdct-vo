import torch

from deepdct.models.attention import AttentionDownBlock


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