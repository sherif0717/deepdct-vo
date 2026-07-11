import pytest
import torch

from deepdct.models.decoder import (
    DecoderBlock,
    DecoderPath,
    DecoderStage,
)


BATCH_SIZE = 2
BASE_CHANNELS = 2
INPUT_SIZE = 120


def test_decoder_block_preserves_spatial_shape():
    """DecoderBlock changes channels while preserving spatial dimensions."""
    block = DecoderBlock(
        in_channels=12,
        out_channels=4,
    )

    x = torch.randn(BATCH_SIZE, 12, 60, 60)
    output = block(x)

    assert output.shape == (BATCH_SIZE, 4, 60, 60)


def test_decoder_block_is_differentiable():
    """Gradients should propagate through DecoderBlock."""
    block = DecoderBlock(
        in_channels=12,
        out_channels=4,
    )

    x = torch.randn(
        BATCH_SIZE,
        12,
        60,
        60,
        requires_grad=True,
    )

    output = block(x)
    output.mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_decoder_stage_without_encoder_attention():
    """
    A regular decoder stage should:

        30 x 30 decoder input
            -> upsample to 60 x 60
            -> concatenate with skip
            -> attention
            -> decoder block
            -> residual addition with skip
    """
    stage = DecoderStage(
        x_channels=8,
        skip_channels=4,
        out_channels=4,
    )

    x = torch.randn(BATCH_SIZE, 8, 30, 30)
    skip = torch.randn(BATCH_SIZE, 4, 60, 60)

    output = stage(x, skip)

    assert output.shape == skip.shape
    assert output.shape == (BATCH_SIZE, 4, 60, 60)


def test_decoder_stage_with_encoder_attention():
    """
    The first decoder stage should add the cached encoder-attention tensor
    to the upsampled decoder tensor before concatenating the skip tensor.
    """
    stage = DecoderStage(
        x_channels=8,
        skip_channels=8,
        out_channels=8,
    )

    x = torch.randn(BATCH_SIZE, 8, 15, 15)
    skip = torch.randn(BATCH_SIZE, 8, 30, 30)
    encoder_attn = torch.randn(BATCH_SIZE, 8, 30, 30)

    output = stage(
        x,
        skip,
        encoder_attn=encoder_attn,
    )

    assert output.shape == skip.shape
    assert output.shape == (BATCH_SIZE, 8, 30, 30)


def test_decoder_stage_rejects_wrong_encoder_attention_shape():
    """Cached attention must match the upsampled decoder tensor."""
    stage = DecoderStage(
        x_channels=8,
        skip_channels=8,
        out_channels=8,
    )

    x = torch.randn(BATCH_SIZE, 8, 15, 15)
    skip = torch.randn(BATCH_SIZE, 8, 30, 30)

    wrong_encoder_attn = torch.randn(
        BATCH_SIZE,
        8,
        15,
        15,
    )

    with pytest.raises(
        ValueError,
        match="Upsampled decoder feature and encoder-attention feature must match",
    ):
        stage(
            x,
            skip,
            encoder_attn=wrong_encoder_attn,
        )


def test_decoder_stage_rejects_wrong_skip_spatial_shape():
    """The skip tensor must match the upsampled decoder spatial dimensions."""
    stage = DecoderStage(
        x_channels=8,
        skip_channels=4,
        out_channels=4,
    )

    x = torch.randn(BATCH_SIZE, 8, 30, 30)

    wrong_skip = torch.randn(
        BATCH_SIZE,
        4,
        30,
        30,
    )

    with pytest.raises(
        ValueError,
        match="Decoder and skip spatial dimensions must match",
    ):
        stage(x, wrong_skip)


def test_decoder_path_output_shape():
    """
    DecoderPath should reconstruct the encoder input resolution.

    Encoder hierarchy for base_channels=2:

        skip1       : [B, 2, 120, 120]
        skip2       : [B, 4,  60,  60]
        skip3       : [B, 8,  30,  30]
        encoder x   : [B, 8,  15,  15]
        cached attn : [B, 8,  30,  30]

    Decoder output:

        [B, 1, 120, 120]
    """
    decoder = DecoderPath(
        base_channels=BASE_CHANNELS,
    )

    x = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 8,
        INPUT_SIZE // 8,
    )

    skip1 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
    )

    skip2 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 2,
        INPUT_SIZE // 2,
        INPUT_SIZE // 2,
    )

    skip3 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 4,
        INPUT_SIZE // 4,
    )

    encoder_attn = torch.randn_like(skip3)

    output = decoder(
        x,
        skips=(skip1, skip2, skip3),
        encoder_attn=encoder_attn,
    )

    assert output.shape == (
        BATCH_SIZE,
        1,
        INPUT_SIZE,
        INPUT_SIZE,
    )


def test_decoder_path_output_is_finite():
    """
    The output block applies BatchNorm after Sigmoid, so its final output is
    not constrained to [0, 1]. It should nevertheless contain finite values.
    """
    decoder = DecoderPath(
        base_channels=BASE_CHANNELS,
    )

    x = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 8,
        INPUT_SIZE // 8,
    )

    skips = (
        torch.randn(
            BATCH_SIZE,
            BASE_CHANNELS,
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        torch.randn(
            BATCH_SIZE,
            BASE_CHANNELS * 2,
            INPUT_SIZE // 2,
            INPUT_SIZE // 2,
        ),
        torch.randn(
            BATCH_SIZE,
            BASE_CHANNELS * 4,
            INPUT_SIZE // 4,
            INPUT_SIZE // 4,
        ),
    )

    encoder_attn = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 4,
        INPUT_SIZE // 4,
    )

    decoder.eval()

    with torch.no_grad():
        output = decoder(
            x,
            skips=skips,
            encoder_attn=encoder_attn,
        )

    assert output.shape == (
        BATCH_SIZE,
        1,
        INPUT_SIZE,
        INPUT_SIZE,
    )

    assert torch.isfinite(output).all()


def test_decoder_path_is_differentiable():
    """The complete decoder path should support backward propagation."""
    decoder = DecoderPath(
        base_channels=BASE_CHANNELS,
    )

    x = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 8,
        INPUT_SIZE // 8,
        requires_grad=True,
    )

    skip1 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
        requires_grad=True,
    )

    skip2 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 2,
        INPUT_SIZE // 2,
        INPUT_SIZE // 2,
        requires_grad=True,
    )

    skip3 = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 4,
        INPUT_SIZE // 4,
        requires_grad=True,
    )

    encoder_attn = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 4,
        INPUT_SIZE // 4,
        requires_grad=True,
    )

    output = decoder(
        x,
        skips=(skip1, skip2, skip3),
        encoder_attn=encoder_attn,
    )

    loss = output.square().mean()
    loss.backward()

    for tensor in (
        x,
        skip1,
        skip2,
        skip3,
        encoder_attn,
    ):
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape
        assert torch.isfinite(tensor.grad).all()


def test_decoder_path_parameter_gradients():
    """Trainable decoder parameters should receive finite gradients."""
    decoder = DecoderPath(
        base_channels=BASE_CHANNELS,
    )

    x = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 8,
        INPUT_SIZE // 8,
    )

    skips = (
        torch.randn(
            BATCH_SIZE,
            BASE_CHANNELS,
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        torch.randn(
            BATCH_SIZE,
            BASE_CHANNELS * 2,
            INPUT_SIZE // 2,
            INPUT_SIZE // 2,
        ),
        torch.randn(
            BATCH_SIZE,
            BASE_CHANNELS * 4,
            INPUT_SIZE // 4,
            INPUT_SIZE // 4,
        ),
    )

    encoder_attn = torch.randn(
        BATCH_SIZE,
        BASE_CHANNELS * 4,
        INPUT_SIZE // 4,
        INPUT_SIZE // 4,
    )

    output = decoder(
        x,
        skips=skips,
        encoder_attn=encoder_attn,
    )

    output.square().mean().backward()

    parameters_with_grad = [
        parameter
        for parameter in decoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    assert parameters_with_grad

    for parameter in parameters_with_grad:
        assert torch.isfinite(parameter.grad).all()