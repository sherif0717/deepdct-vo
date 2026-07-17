"""Forward-pass tests for the Fig. 2 DeepDCT-VO model.

Run directly with:

    pytest tests/models/deepdct_vo_forward.py -v

Pytest normally auto-discovers files named ``test_*.py``. Rename this file to
``test_deepdct_vo_forward.py`` if you want it included automatically by
``pytest tests``.
"""

import pytest
import torch
import torch.nn as nn

from deepdct.models.deepdct_vo import DeepDCTVO


class DummySemanticBranch(nn.Module):
    """Fast deterministic substitute for LR-ASPP."""

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return image.mean(
            dim=1,
            keepdim=True,
        ).clamp(0.0, 1.0)

    def forward_all(
        self,
        image: torch.Tensor,
    ):
        semantic_map = self.forward(image)

        labels = torch.zeros(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            dtype=torch.int64,
            device=image.device,
        )

        logits = torch.zeros(
            image.shape[0],
            2,
            image.shape[2],
            image.shape[3],
            dtype=image.dtype,
            device=image.device,
        )

        return {
            "logits": logits,
            "labels": labels,
            "semantic_map": semantic_map,
        }


class DummyDepthBranch(nn.Module):
    """Deterministic one-channel depth generator."""

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return image.mean(
            dim=1,
            keepdim=True,
        ).clamp(0.0, 1.0)


class CountingDepthBranch(nn.Module):
    """Depth branch that records how many times it is called."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1

        return image.mean(
            dim=1,
            keepdim=True,
        )


@pytest.fixture
def model() -> DeepDCTVO:
    network = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(120, 120),
        pretrained_semantic=False,
        freeze_semantic=True,
        depth_model=DummyDepthBranch(),
        share_aresunet_between_models=False,
    )

    network.semantic_model = DummySemanticBranch()
    network.eval()

    return network


@pytest.fixture
def inputs():
    image_prev = torch.rand(
        2,
        3,
        120,
        120,
        dtype=torch.float32,
    )

    image_curr = torch.rand(
        2,
        3,
        120,
        120,
        dtype=torch.float32,
    )

    depth_curr = torch.rand(
        2,
        1,
        120,
        120,
        dtype=torch.float32,
    )

    return image_prev, image_curr, depth_curr


def test_deepdct_vo_forward_returns_motion_vectors(model, inputs):
    image_prev, image_curr, depth_curr = inputs

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
        )

    assert set(outputs) == {
        "rotation",
        "directional_translation",
        "rotation_used_for_translation",
    }
    assert outputs["rotation"].shape == (2, 3)
    assert outputs["directional_translation"].shape == (2, 3)
    assert outputs["rotation_used_for_translation"].shape == (2, 3)

    for value in outputs.values():
        assert torch.isfinite(value).all()


def test_predicted_rotation_is_used_by_default(model, inputs):
    image_prev, image_curr, depth_curr = inputs

    with torch.no_grad():
        outputs = model(image_prev, image_curr, depth_curr)

    assert torch.equal(
        outputs["rotation_used_for_translation"],
        outputs["rotation"],
    )


def test_ground_truth_rotation_can_condition_model_t(model, inputs):
    image_prev, image_curr, depth_curr = inputs
    rotation_gt = torch.tensor(
        [[0.01, -0.02, 0.03], [-0.04, 0.05, -0.06]],
        dtype=image_curr.dtype,
    )

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
            rotation_for_translation=rotation_gt,
            use_ground_truth_rotation=True,
        )

    assert torch.equal(outputs["rotation_used_for_translation"], rotation_gt)
    assert outputs["directional_translation"].shape == (2, 3)


def test_ground_truth_mode_requires_rotation_tensor(model, inputs):
    image_prev, image_curr, depth_curr = inputs

    with pytest.raises(ValueError, match="rotation_for_translation"):
        model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
            use_ground_truth_rotation=True,
        )


def test_intermediate_shapes_follow_fig2(model, inputs):
    image_prev, image_curr, depth_curr = inputs

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
            return_intermediates=True,
        )

    assert outputs["semantic_prev"].shape == (2, 1, 120, 120)
    assert outputs["semantic_curr"].shape == (2, 1, 120, 120)
    assert outputs["ci_prev"].shape == (2, 4, 120, 120)
    assert outputs["ci_curr"].shape == (2, 4, 120, 120)

    assert outputs["rotation_c_prev"].shape == (2, 1, 120, 120)
    assert outputs["rotation_c_curr"].shape == (2, 1, 120, 120)
    assert outputs["translation_c_prev"].shape == (2, 1, 120, 120)
    assert outputs["translation_c_curr"].shape == (2, 1, 120, 120)

    assert outputs["rotation_features"].shape == (2, 4, 120, 120)
    assert outputs["rotation_map"].shape == (2, 3, 120, 120)
    assert outputs["translation_features"].shape == (2, 7, 120, 120)


def test_rotation_map_contains_broadcast_rotation_values(model, inputs):
    image_prev, image_curr, depth_curr = inputs
    rotation_gt = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
        dtype=image_curr.dtype,
    )

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
            rotation_for_translation=rotation_gt,
            use_ground_truth_rotation=True,
            return_intermediates=True,
        )

    rotation_map = outputs["rotation_map"]
    assert torch.equal(rotation_map[:, :, 0, 0], rotation_gt)
    assert torch.equal(rotation_map[:, :, -1, -1], rotation_gt)


def test_complete_model_is_differentiable():
    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(32, 32),
        pretrained_semantic=False,
        freeze_semantic=True,
        share_aresunet_between_models=False,
    )
    model.semantic_model = DummySemanticBranch()
    model.train()

    image_prev = torch.rand(2, 3, 32, 32, requires_grad=True)
    image_curr = torch.rand(2, 3, 32, 32, requires_grad=True)
    depth_curr = torch.rand(2, 1, 32, 32, requires_grad=True)

    outputs = model(image_prev, image_curr, depth_curr)
    loss = outputs["rotation"].sum() + outputs["directional_translation"].sum()
    loss.backward()

    assert image_prev.grad is not None
    assert image_curr.grad is not None
    assert depth_curr.grad is not None
    assert torch.isfinite(image_prev.grad).all()
    assert torch.isfinite(image_curr.grad).all()
    assert torch.isfinite(depth_curr.grad).all()

    assert model.rotation_head.conv.weight.grad is not None
    assert model.rotation_head.dense.weight.grad is not None
    assert model.translation_head.conv.weight.grad is not None
    assert model.translation_head.dense.weight.grad is not None

    assert any(
        p.grad is not None
        for p in model.rotation_aresunet.parameters()
        if p.requires_grad
    )
    assert any(
        p.grad is not None
        for p in model.translation_aresunet.parameters()
        if p.requires_grad
    )


def test_model_r_and_model_t_use_distinct_aresunets_by_default(model):
    assert model.rotation_aresunet is not model.translation_aresunet


def test_aresunet_can_be_shared_as_explicit_ablation():
    model = DeepDCTVO(
        aresunet_output_channels=1,
        pretrained_semantic=False,
        freeze_semantic=True,
        share_aresunet_between_models=True,
    )
    assert model.rotation_aresunet is model.translation_aresunet


def test_rejects_wrong_depth_channel_count(model, inputs):
    image_prev, image_curr, _ = inputs
    invalid_depth = torch.rand(2, 2, 120, 120)

    with pytest.raises(ValueError, match="depth_curr"):
        model(image_prev, image_curr, invalid_depth)


def test_rejects_mismatched_frame_shapes(model, inputs):
    image_prev, _, depth_curr = inputs
    invalid_current = torch.rand(2, 3, 96, 120)

    with pytest.raises(ValueError, match="identical shapes"):
        model(image_prev, invalid_current, depth_curr)


def test_real_lraspp_branch_completes_smoke_forward():
    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(32, 32),
        pretrained_semantic=False,
        freeze_semantic=True,
        share_aresunet_between_models=False,
    )
    model.eval()

    image_prev = torch.rand(1, 3, 64, 64)
    image_curr = torch.rand(1, 3, 64, 64)
    depth_curr = torch.rand(1, 1, 64, 64)

    with torch.no_grad():
        outputs = model(image_prev, image_curr, depth_curr)

    assert outputs["rotation"].shape == (1, 3)
    assert outputs["directional_translation"].shape == (1, 3)
    assert torch.isfinite(outputs["rotation"]).all()
    assert torch.isfinite(outputs["directional_translation"]).all()

    
def test_external_depth_bypasses_internal_depth_model():
    depth_model = CountingDepthBranch()

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(120, 120),
        pretrained_semantic=False,
        freeze_semantic=True,
        depth_model=depth_model,
    )
    model.semantic_model = DummySemanticBranch()
    model.eval()

    image_prev = torch.rand(1, 3, 120, 120)
    image_curr = torch.rand(1, 3, 120, 120)
    external_depth = torch.rand(1, 1, 120, 120)

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=external_depth,
            return_intermediates=True,
        )

    assert depth_model.calls == 0
    assert torch.equal(
        outputs["depth_curr"],
        external_depth,
    )
    assert outputs["depth_was_supplied"].item() is True


def test_deepdct_vo_with_real_lite_mono_depth():
    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(32, 32),
        pretrained_semantic=False,
        freeze_semantic=True,
        depth_checkpoint_dir=(
            "weights/lite-mono-tiny-640x192"
        ),
        depth_model_name="lite-mono-tiny",
        freeze_depth=True,
        depth_model=DummyDepthBranch(),
        share_aresunet_between_models=False,
    )

    model.semantic_model = DummySemanticBranch()
    model.eval()

    image_prev = torch.rand(1, 3, 64, 64)
    image_curr = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            return_intermediates=True,
        )

    assert outputs["depth_curr"].shape == (1, 1, 64, 64)
    assert outputs["depth_curr"].min().item() >= 0.0
    assert outputs["depth_curr"].max().item() <= 1.0
    assert outputs["rotation"].shape == (1, 3)
    assert outputs["directional_translation"].shape == (1, 3)


def test_internal_depth_model_is_used_when_depth_not_supplied(
    model,
    inputs,
):
    image_prev, image_curr, _ = inputs

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            return_intermediates=True,
        )

    expected_depth = image_curr.mean(
        dim=1,
        keepdim=True,
    ).clamp(0.0, 1.0)

    assert torch.allclose(
        outputs["depth_curr"],
        expected_depth,
    )

    assert outputs["depth_was_supplied"].item() is False