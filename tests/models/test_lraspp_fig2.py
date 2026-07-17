"""Unit tests for the Fig. 2-compatible LR-ASPP semantic branch."""

import pytest
import torch

from deepdct.models.auxiliary.lraspp_fig2 import (
    LRASPPSemanticBranch,
)


@pytest.fixture
def model() -> LRASPPSemanticBranch:
    """Create an offline-safe LR-ASPP wrapper for unit tests."""
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        normalize_input=True,
        normalize_map=True,
        progress=False,
    )

    # LR-ASPP contains BatchNorm layers and a global pooling branch.
    # Evaluation mode keeps shape-focused unit tests independent of
    # batch-statistics requirements.
    semantic_model.eval()

    return semantic_model


@pytest.fixture
def rgb_batch() -> torch.Tensor:
    """Return a valid RGB batch scaled to [0, 1]."""
    return torch.rand(
        2,
        3,
        120,
        120,
        dtype=torch.float32,
    )


def test_lraspp_constructs_without_pretrained_download() -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        progress=False,
    )

    assert isinstance(semantic_model, LRASPPSemanticBranch)
    assert semantic_model.num_classes > 1


def test_forward_returns_single_channel_semantic_map(
    model: LRASPPSemanticBranch,
    rgb_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        semantic_map = model(rgb_batch)

    assert semantic_map.shape == (2, 1, 120, 120)
    assert semantic_map.dtype == rgb_batch.dtype


def test_forward_map_values_are_normalized(
    model: LRASPPSemanticBranch,
    rgb_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        semantic_map = model.forward_map(rgb_batch)

    assert torch.isfinite(semantic_map).all()
    assert semantic_map.min().item() >= 0.0
    assert semantic_map.max().item() <= 1.0


def test_forward_logits_returns_full_resolution_class_scores(
    model: LRASPPSemanticBranch,
    rgb_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        logits = model.forward_logits(rgb_batch)

    assert logits.ndim == 4
    assert logits.shape[0] == rgb_batch.shape[0]
    assert logits.shape[1] == model.num_classes
    assert logits.shape[-2:] == rgb_batch.shape[-2:]
    assert logits.is_floating_point()


def test_forward_labels_returns_integer_class_ids(
    model: LRASPPSemanticBranch,
    rgb_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        labels = model.forward_labels(rgb_batch)

    assert labels.shape == (2, 1, 120, 120)
    assert labels.dtype == torch.int64
    assert labels.min().item() >= 0
    assert labels.max().item() < model.num_classes


def test_forward_all_returns_consistent_outputs(
    model: LRASPPSemanticBranch,
    rgb_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        outputs = model.forward_all(rgb_batch)

    assert set(outputs) == {
        "logits",
        "labels",
        "semantic_map",
    }

    expected_labels = torch.argmax(
        outputs["logits"],
        dim=1,
        keepdim=True,
    )

    expected_map = expected_labels.to(rgb_batch.dtype)
    expected_map = expected_map / float(model.num_classes - 1)

    assert torch.equal(outputs["labels"], expected_labels)
    assert torch.allclose(outputs["semantic_map"], expected_map)


def test_forward_is_alias_of_forward_map(
    model: LRASPPSemanticBranch,
    rgb_batch: torch.Tensor,
) -> None:
    with torch.no_grad():
        direct_output = model(rgb_batch)
        explicit_output = model.forward_map(rgb_batch)

    assert torch.equal(direct_output, explicit_output)


def test_non_normalized_map_returns_float_class_ids(
    rgb_batch: torch.Tensor,
) -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        normalize_map=False,
        progress=False,
    )
    semantic_model.eval()

    with torch.no_grad():
        labels = semantic_model.forward_labels(rgb_batch)
        semantic_map = semantic_model.forward_map(rgb_batch)

    assert semantic_map.dtype == rgb_batch.dtype
    assert torch.equal(
        semantic_map,
        labels.to(rgb_batch.dtype),
    )


def test_freeze_pretrained_model_disables_gradients() -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        progress=False,
    )

    semantic_model.freeze_pretrained_model()

    assert semantic_model.freeze_pretrained is True
    assert semantic_model.model.training is False
    assert all(
        not parameter.requires_grad
        for parameter in semantic_model.model.parameters()
    )


def test_unfreeze_pretrained_model_enables_gradients() -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=True,
        progress=False,
    )

    semantic_model.unfreeze_pretrained_model()

    assert semantic_model.freeze_pretrained is False
    assert all(
        parameter.requires_grad
        for parameter in semantic_model.model.parameters()
    )


def test_train_keeps_frozen_pretrained_model_in_eval_mode() -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=True,
        progress=False,
    )

    semantic_model.train()

    assert semantic_model.training is True
    assert semantic_model.model.training is False


def test_prepare_input_applies_imagenet_normalization() -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        normalize_input=True,
        progress=False,
    )

    x = torch.zeros(
        1,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    prepared = semantic_model.prepare_input(x)

    expected = (
        x
        - semantic_model.image_mean
    ) / semantic_model.image_std

    assert torch.allclose(prepared, expected)


def test_prepare_input_can_skip_normalization() -> None:
    semantic_model = LRASPPSemanticBranch(
        pretrained=False,
        freeze_pretrained=False,
        normalize_input=False,
        progress=False,
    )

    x = torch.rand(
        1,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    prepared = semantic_model.prepare_input(x)

    assert prepared is x


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (2, 1, 120, 120),
        (2, 4, 120, 120),
        (2, 3, 120),
        (3, 120, 120),
    ],
)
def test_rejects_invalid_input_shape(
    model: LRASPPSemanticBranch,
    invalid_shape,
) -> None:
    x = torch.rand(*invalid_shape)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_integer_input(
    model: LRASPPSemanticBranch,
) -> None:
    x = torch.randint(
        0,
        256,
        (2, 3, 120, 120),
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(x)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_input(
    model: LRASPPSemanticBranch,
    invalid_value: float,
) -> None:
    x = torch.rand(
        2,
        3,
        120,
        120,
        dtype=torch.float32,
    )
    x[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError):
        model(x)
