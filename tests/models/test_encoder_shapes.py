import torch

from deepdct.models.encoder import EncoderBlock, EncoderStage, EncoderPath


def test_encoder_block_output_shape():
    block = EncoderBlock(in_channels=4, out_channels=2)

    x = torch.randn(2, 4, 120, 120)
    y = block(x)

    assert y.shape == (2, 2, 120, 120)


def test_encoder_stage_output_and_skip_shapes():
    stage = EncoderStage(in_channels=4, out_channels=2)

    x = torch.randn(2, 4, 120, 120)
    y, skip = stage(x)

    assert skip.shape == (2, 2, 120, 120)
    assert y.shape == (2, 2, 60, 60)


def test_encoder_path_shapes():
    encoder = EncoderPath(in_channels=4, base_channels=2)

    x = torch.randn(2, 4, 120, 120)
    y, skips, cache_x_attn = encoder(x)

    assert y.shape == (2, 8, 15, 15)

    skip1, skip2, skip3 = skips
    assert skip1.shape == (2, 2, 120, 120)
    assert skip2.shape == (2, 4, 60, 60)
    assert skip3.shape == (2, 8, 30, 30)

    assert cache_x_attn.shape == (2, 8, 30, 30)