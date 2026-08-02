"""Tests for attention-map retention and automatic collection."""

import torch
import torch.nn as nn

from deepdct.interpretability.hooks import (
    collect_internal_attention_maps,
)
from deepdct.models.attention import (
    AttentionDownBlock,
    AttentionUpBlock,
)


class SmallAttentionNetwork(nn.Module):
    """
    Minimal network containing encoder and decoder attention blocks.

    The module names intentionally resemble the DeepDCT-VO hierarchy so
    that collection behavior can be tested without constructing the full
    semantic, depth, rotation, and translation models.
    """

    def __init__(
        self,
        channels: int = 8,
    ) -> None:
        super().__init__()

        self.rotation_aresunet = nn.Module()
        self.rotation_aresunet.encoder = nn.Module()
        self.rotation_aresunet.encoder.enc1 = nn.Module()
        self.rotation_aresunet.encoder.enc1.attn = (
            AttentionDownBlock(
                channels=channels,
            )
        )

        self.rotation_aresunet.decoder = nn.Module()
        self.rotation_aresunet.decoder.dec1 = nn.Module()
        self.rotation_aresunet.decoder.dec1.attn = (
            AttentionUpBlock(
                channels=channels,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.rotation_aresunet
            .encoder
            .enc1
            .attn(x)
        )

        x = (
            self.rotation_aresunet
            .decoder
            .dec1
            .attn(x)
        )

        return x


def test_attention_down_block_initially_has_no_map() -> None:
    block = AttentionDownBlock(
        channels=8,
    )

    assert block.last_attention_map is None


def test_attention_up_block_initially_has_no_map() -> None:
    block = AttentionUpBlock(
        channels=8,
    )

    assert block.last_attention_map is None


def test_attention_down_block_retains_attention_map() -> None:
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

    attention_map = block.last_attention_map

    assert attention_map is not None

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


def test_attention_up_block_retains_attention_map() -> None:
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

    attention_map = block.last_attention_map

    assert attention_map is not None

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


def test_attention_collection_is_empty_before_forward() -> None:
    model = SmallAttentionNetwork(
        channels=8,
    )

    attention_maps = (
        collect_internal_attention_maps(model)
    )

    assert attention_maps == {}


def test_attention_collection_finds_encoder_and_decoder_maps() -> None:
    model = SmallAttentionNetwork(
        channels=8,
    )
    model.eval()

    x = torch.randn(
        1,
        8,
        24,
        24,
    )

    with torch.no_grad():
        output = model(x)

    attention_maps = (
        collect_internal_attention_maps(model)
    )

    expected_names = {
        (
            "rotation_aresunet."
            "encoder.enc1.attn"
        ),
        (
            "rotation_aresunet."
            "decoder.dec1.attn"
        ),
    }

    assert set(attention_maps.keys()) == (
        expected_names
    )

    encoder_map = attention_maps[
        "rotation_aresunet.encoder.enc1.attn"
    ]

    decoder_map = attention_maps[
        "rotation_aresunet.decoder.dec1.attn"
    ]

    assert encoder_map.shape == (
        1,
        8,
        24,
        24,
    )

    assert decoder_map.shape == (
        1,
        1,
        24,
        24,
    )

    assert output.shape == (
        1,
        8,
        24,
        24,
    )


def test_collected_attention_maps_are_detached_and_on_cpu() -> None:
    model = SmallAttentionNetwork(
        channels=8,
    )
    model.eval()

    x = torch.randn(
        1,
        8,
        16,
        16,
        requires_grad=True,
    )

    output = model(x)

    assert output.requires_grad

    attention_maps = (
        collect_internal_attention_maps(model)
    )

    for attention_map in attention_maps.values():
        assert attention_map.device.type == "cpu"
        assert not attention_map.requires_grad
        assert attention_map.grad_fn is None


def test_second_forward_replaces_stored_attention_maps() -> None:
    model = SmallAttentionNetwork(
        channels=8,
    )
    model.eval()

    first_input = torch.zeros(
        1,
        8,
        16,
        16,
    )

    second_input = torch.ones(
        1,
        8,
        16,
        16,
    )

    with torch.no_grad():
        model(first_input)

    first_maps = (
        collect_internal_attention_maps(model)
    )

    with torch.no_grad():
        model(second_input)

    second_maps = (
        collect_internal_attention_maps(model)
    )

    assert first_maps.keys() == second_maps.keys()

    # At least one attention block should update its stored tensor.
    changed = any(
        not torch.equal(
            first_maps[name],
            second_maps[name],
        )
        for name in first_maps
    )

    assert changed