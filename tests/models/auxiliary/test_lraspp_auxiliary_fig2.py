import pytest
import torch
import torch.nn as nn

from deepdct.models.auxiliary.lraspp_fig2 import (
    LRASPPSemanticBranch,
)


class DummySegmentationModel(nn.Module):
    """Deterministic segmentation model for semantic-map tests."""

    def __init__(self) -> None:
        super().__init__()

        self.backbone = nn.Identity()

        self.classifier = nn.Module()
        self.classifier.low_classifier = nn.Conv2d(
            1,
            3,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape

        logits = torch.zeros(
            batch_size,
            3,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
        )

        # Background wins, but not with probability 1.
        logits[:, 0] = 2.0
        logits[:, 1] = 1.0
        logits[:, 2] = 0.0

        return {"out": logits}


def build_test_branch(
    semantic_map_mode: str,
) -> LRASPPSemanticBranch:
    branch = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        normalize_input=False,
        normalize_map=True,
        semantic_map_mode=semantic_map_mode,
        progress=False,
    )

    branch.model = DummySegmentationModel()
    branch.num_classes = 3

    return branch


def test_foreground_probability_shape() -> None:
    branch = build_test_branch(
        semantic_map_mode="foreground_probability",
    )

    image = torch.rand(2, 3, 32, 48)
    semantic = branch(image)

    assert semantic.shape == (2, 1, 32, 48)


def test_foreground_probability_is_bounded() -> None:
    branch = build_test_branch(
        semantic_map_mode="foreground_probability",
    )

    image = torch.rand(1, 3, 16, 16)
    semantic = branch(image)

    assert torch.all(semantic >= 0.0)
    assert torch.all(semantic <= 1.0)


def test_foreground_probability_preserves_soft_evidence() -> None:
    branch = build_test_branch(
        semantic_map_mode="foreground_probability",
    )

    image = torch.rand(1, 3, 16, 16)
    semantic = branch(image)

    # Background is the argmax everywhere, but foreground probability
    # must remain positive because nonbackground logits are finite.
    assert torch.count_nonzero(semantic).item() == semantic.numel()
    assert semantic.min().item() > 0.0


def test_foreground_probability_matches_softmax() -> None:
    branch = build_test_branch(
        semantic_map_mode="foreground_probability",
    )

    image = torch.rand(1, 3, 8, 8)

    with torch.no_grad():
        logits = branch.forward_logits(image)
        semantic = branch(image)

    expected = 1.0 - torch.softmax(
        logits,
        dim=1,
    )[:, 0:1]

    torch.testing.assert_close(
        semantic,
        expected,
    )


def test_class_index_preserves_legacy_behavior() -> None:
    branch = build_test_branch(
        semantic_map_mode="class_index",
    )

    image = torch.rand(1, 3, 8, 8)
    semantic = branch(image)

    # Background class 0 wins everywhere.
    assert torch.count_nonzero(semantic).item() == 0


def test_invalid_semantic_map_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_map_mode",
    ):
        LRASPPSemanticBranch(
            pretrained=False,
            semantic_map_mode="invalid",
            progress=False,
        )