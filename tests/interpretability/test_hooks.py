"""Tests for forward-hook feature collection and module resolution."""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from deepdct.interpretability.hooks import (
    FeatureMapCollector,
    resolve_module,
    tensor_from_output,
)


class SmallFeatureNetwork(nn.Module):
    """Minimal convolutional network for testing feature hooks."""

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)


class DictionaryOutputModule(nn.Module):
    """Module whose forward result is a tensor dictionary."""

    def forward(self, x: torch.Tensor):
        return {
            "features": x * 2.0,
        }


class TupleOutputModule(nn.Module):
    """Module whose forward result is a tensor tuple."""

    def forward(self, x: torch.Tensor):
        return (
            x,
            x + 1.0,
        )


def test_resolve_module_finds_named_child() -> None:
    model = SmallFeatureNetwork()

    module = resolve_module(
        model,
        "encoder",
    )

    assert isinstance(module, nn.Sequential)


def test_resolve_module_finds_indexed_child() -> None:
    model = SmallFeatureNetwork()

    module = resolve_module(
        model,
        "encoder.0",
    )

    assert isinstance(module, nn.Conv2d)


def test_resolve_module_rejects_unknown_child() -> None:
    model = SmallFeatureNetwork()

    with pytest.raises(
        AttributeError,
        match="has no child",
    ):
        resolve_module(
            model,
            "encoder.unknown",
        )


def test_tensor_from_output_accepts_tensor() -> None:
    output = torch.randn(
        1,
        3,
        8,
        8,
    )

    extracted = tensor_from_output(output)

    assert extracted is output


def test_tensor_from_output_accepts_tuple() -> None:
    first = torch.randn(
        1,
        3,
        8,
        8,
    )
    second = torch.randn(
        1,
        3,
        8,
        8,
    )

    extracted = tensor_from_output(
        (first, second)
    )

    assert extracted is first


def test_tensor_from_output_accepts_mapping() -> None:
    feature = torch.randn(
        1,
        3,
        8,
        8,
    )

    extracted = tensor_from_output(
        {
            "feature": feature,
        }
    )

    assert extracted is feature


def test_tensor_from_output_rejects_unsupported_output() -> None:
    with pytest.raises(TypeError):
        tensor_from_output(
            "not a tensor"
        )


def test_feature_collector_captures_requested_layers() -> None:
    model = SmallFeatureNetwork()
    model.eval()

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    with FeatureMapCollector(
        model=model,
        module_paths=[
            "encoder.0",
            "decoder.0",
        ],
        detach=True,
        move_to_cpu=True,
    ) as collector:
        output = model(x)

    assert output.shape == (
        1,
        2,
        16,
        16,
    )

    assert isinstance(
        collector.activations,
        OrderedDict,
    )

    assert list(
        collector.activations.keys()
    ) == [
        "encoder.0",
        "decoder.0",
    ]

    assert collector.activations[
        "encoder.0"
    ].shape == (
        1,
        4,
        16,
        16,
    )

    assert collector.activations[
        "decoder.0"
    ].shape == (
        1,
        2,
        16,
        16,
    )


def test_feature_collector_detaches_and_moves_to_cpu() -> None:
    model = SmallFeatureNetwork()
    model.eval()

    x = torch.randn(
        1,
        3,
        16,
        16,
        requires_grad=True,
    )

    with FeatureMapCollector(
        model=model,
        module_paths=["encoder.0"],
        detach=True,
        move_to_cpu=True,
    ) as collector:
        model(x)

    activation = collector.activations[
        "encoder.0"
    ]

    assert activation.device.type == "cpu"
    assert not activation.requires_grad
    assert activation.grad_fn is None


def test_feature_collector_removes_hooks_after_context() -> None:
    model = SmallFeatureNetwork()
    model.eval()

    collector = FeatureMapCollector(
        model=model,
        module_paths=["encoder.0"],
    )

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    with collector:
        model(x)

    assert collector._handles == []

    previous_activation = collector.activations[
        "encoder.0"
    ].clone()

    # The collector has exited its context, so this forward pass must
    # not replace the already collected activation.
    model(x + 1.0)

    assert torch.equal(
        collector.activations["encoder.0"],
        previous_activation,
    )


def test_feature_collector_clear_removes_activations() -> None:
    model = SmallFeatureNetwork()

    x = torch.randn(
        1,
        3,
        16,
        16,
    )

    collector = FeatureMapCollector(
        model=model,
        module_paths=["encoder.0"],
    )

    with collector:
        model(x)

    assert collector.activations

    collector.clear()

    assert not collector.activations