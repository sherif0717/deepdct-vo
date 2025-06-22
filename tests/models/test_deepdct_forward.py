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

from deepdct.models.pose_head import (
    DirectionalTranslationHead,
)


class DummySemanticBranch(nn.Module):
    """Fast deterministic substitute for LR-ASPP."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return (image.mean(
            dim=1,
            keepdim=True,
        ) * self.scale).clamp(0.0, 1.0)

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
        normalize_semantic_input=True,
        normalize_semantic_map=True,
        use_semantic_cues=True,
        use_depth_cues=True,
        depth_checkpoint_dir=None,
        freeze_depth=True,
    )

    network.semantic_model = DummySemanticBranch()
    network.depth_model = DummyDepthBranch()
    
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
        "rotation_normalized",
        "directional_translation",
        "rotation_used_for_translation",
    }
    assert outputs["rotation"].shape == (2, 3)
    assert outputs["directional_translation"].shape == (2, 3)
    assert outputs["rotation_used_for_translation"].shape == (2, 3)
    assert outputs["rotation_normalized"].shape == (
        image_prev.shape[0],
        3,
    )

    expected_rotation = (
        outputs["rotation_normalized"]
        * model.rotation_normalization_scale
    )

    assert torch.allclose(
        outputs["rotation"],
        expected_rotation,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

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

def test_a3_gt_rotation_is_not_normalized_before_model_t(
    model,
    inputs,
):
    """A3 must condition Model T with physical GT radians.

    Rotation normalization belongs only to Model-R supervision.
    """

    image_prev, image_curr, depth_curr = inputs

    model.rotation_normalization_scale = 0.175

    rotation_gt = torch.tensor(
        [
            [0.010, -0.020, 0.030],
            [-0.040, 0.050, -0.060],
        ],
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

    # A3 must use the exact PHYSICAL GT vector.
    assert torch.equal(
        outputs["rotation_used_for_translation"],
        rotation_gt,
    )

    assert torch.equal(
        outputs["rotation_map"][:, :, 0, 0],
        rotation_gt,
    )

    # Explicitly guard against accidentally feeding the normalized
    # supervision target into Model T.
    rotation_gt_normalized = (
        rotation_gt
        / model.rotation_normalization_scale
    )

    assert not torch.allclose(
        outputs["rotation_used_for_translation"],
        rotation_gt_normalized,
    )

def test_a3_does_not_replace_model_r_prediction(
    model,
    inputs,
):
    """GT rotation alters Model-T conditioning, not Model-R output."""

    image_prev, image_curr, depth_curr = inputs

    model.rotation_normalization_scale = 0.175

    rotation_gt = torch.tensor(
        [
            [0.12, -0.08, 0.04],
            [-0.10, 0.07, -0.03],
        ],
        dtype=image_curr.dtype,
    )

    with torch.no_grad():
        predicted_conditioning = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
            use_ground_truth_rotation=False,
        )

        gt_conditioning = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=depth_curr,
            rotation_for_translation=rotation_gt,
            use_ground_truth_rotation=True,
        )

    # Model R must still run normally in A3.
    assert torch.allclose(
        gt_conditioning["rotation"],
        predicted_conditioning["rotation"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert torch.allclose(
        gt_conditioning["rotation_normalized"],
        predicted_conditioning["rotation_normalized"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # Only the Model-T conditioning source changes.
    assert torch.equal(
        gt_conditioning["rotation_used_for_translation"],
        rotation_gt,
    )

    assert torch.equal(
        predicted_conditioning[
            "rotation_used_for_translation"
        ],
        predicted_conditioning["rotation"],
    )

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


def test_deepdct_vo_internal_depth_path():
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

    model.use_depth_cues = True
    model.depth_model = DummyDepthBranch()
    model.eval()

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

def test_pooled_mlp_translation_head_shape():
    model = DeepDCTVO(
        pretrained_semantic=False,
        freeze_semantic=True,
        depth_checkpoint_dir=None,
        depth_model=DummyDepthBranch(),
        use_semantic_cues=False,
        use_depth_cues=False,
        translation_decoder_type="pooled_mlp",
        translation_projection_channels=8,
        translation_pool_size=(4, 4),
        translation_aggregation_hidden_dim=64,
    )

    model.semantic_model = DummySemanticBranch()
    model.eval()

    image_prev = torch.rand(
        2,
        3,
        120,
        120,
    )
    image_curr = torch.rand(
        2,
        3,
        120,
        120,
    )

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            return_intermediates=True,
        )

    assert outputs[
        "directional_translation"
    ].shape == (2, 3)

    assert torch.isfinite(
        outputs["directional_translation"]
    ).all()

    assert (
        model.translation_head.decoder_type
        == "pooled_mlp"
    )

    assert (
        model.translation_head.representation_dim
        == 128
    )

    assert (
        model.translation_head.conv.out_channels
        == 8
    )

    assert (
        model.translation_head.pool.output_size
        == (4, 4)
    )

    assert (
        model.translation_head.dense[0].in_features
        == 128
    )

    assert (
        model.translation_head.dense[0].out_features
        == 64
    )

    assert (
        model.translation_head.dense[2].out_features
        == 3
    )

def test_pooled_mlp_compact_representation_shape():
    head = DirectionalTranslationHead(
        in_channels=7,
        input_size=(120, 120),
        decoder_type="pooled_mlp",
        projection_channels=8,
        pool_size=(4, 4),
        aggregation_hidden_dim=64,
    )

    head.eval()

    features = torch.rand(
        2,
        7,
        120,
        120,
    )

    captured = {}

    def capture_representation(
        module,
        inputs,
    ):
        captured["value"] = inputs[0].detach()

    hook = (
        head.representation_input_module()
        .register_forward_pre_hook(
            capture_representation
        )
    )

    try:
        with torch.no_grad():
            prediction = head(features)
    finally:
        hook.remove()

    assert prediction.shape == (2, 3)
    assert captured["value"].shape == (
        2,
        128,
    )

def test_a4_uses_internal_semantic_and_depth_cues(
    model,
    inputs,
):
    image_prev, image_curr, _ = inputs

    model.use_semantic_cues = True
    model.use_depth_cues = True

    model.semantic_model = DummySemanticBranch()
    model.depth_model = DummyDepthBranch()

    model.rotation_normalization_scale = 0.175

    rotation_gt = torch.tensor(
        [
            [0.010, -0.020, 0.030],
            [-0.040, 0.050, -0.060],
        ],
        dtype=image_curr.dtype,
    )

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            rotation_for_translation=rotation_gt,
            use_ground_truth_rotation=True,
            return_intermediates=True,
        )

    expected_semantic_prev = (
        image_prev
        .mean(dim=1, keepdim=True)
        .clamp(0.0, 1.0)
    )

    expected_semantic_curr = (
        image_curr
        .mean(dim=1, keepdim=True)
        .clamp(0.0, 1.0)
    )

    expected_depth_curr = (
        image_curr
        .mean(dim=1, keepdim=True)
        .clamp(0.0, 1.0)
    )

    assert torch.allclose(
        outputs["semantic_prev"],
        expected_semantic_prev,
    )

    assert torch.allclose(
        outputs["semantic_curr"],
        expected_semantic_curr,
    )

    assert torch.allclose(
        outputs["depth_curr"],
        expected_depth_curr,
    )

    assert (
        outputs["semantic_cues_enabled"].item()
        is True
    )

    assert (
        outputs["depth_cues_enabled"].item()
        is True
    )

    assert (
        outputs["depth_was_supplied"].item()
        is False
    )

    # A4 inherits A3:
    # Model T must still receive exact physical GT rotation.
    assert torch.equal(
        outputs[
            "rotation_used_for_translation"
        ],
        rotation_gt,
    )

    assert torch.equal(
        outputs["rotation_map"][:, :, 0, 0],
        rotation_gt,
    )

def test_a4_fusion_contains_semantic_depth_and_gt_rotation(
    model,
    inputs,
):
    image_prev, image_curr, _ = inputs

    model.use_semantic_cues = True
    model.use_depth_cues = True
    model.semantic_model = DummySemanticBranch()
    model.depth_model = DummyDepthBranch()

    rotation_gt = torch.tensor(
        [
            [0.01, 0.02, 0.03],
            [-0.01, -0.02, -0.03],
        ],
        dtype=image_curr.dtype,
    )

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            rotation_for_translation=rotation_gt,
            use_ground_truth_rotation=True,
            return_intermediates=True,
        )

    # RGB + 1 semantic channel.
    assert outputs["ci_prev"].shape == (
        2,
        4,
        120,
        120,
    )

    assert outputs["ci_curr"].shape == (
        2,
        4,
        120,
        120,
    )

    # Model R:
    # C_prev + C_curr + S_curr + D_curr
    assert outputs["rotation_features"].shape == (
        2,
        4,
        120,
        120,
    )

    # Model T:
    # C_prev + C_curr + S_curr + D_curr + 3-D rotation map
    assert outputs["translation_features"].shape == (
        2,
        7,
        120,
        120,
    )

    assert torch.equal(
        outputs["rotation_map"][:, :, 0, 0],
        rotation_gt,
    )

def test_a4_frozen_auxiliaries_remain_in_eval_mode(
    model,
):
    model.freeze_semantic_model = True
    model.freeze_depth_model = True

    model.train()

    assert model.training is True

    assert (
        model.semantic_model.training
        is False
    )

    assert (
        model.depth_model.training
        is False
    )

def test_depth_placeholder_is_zero_when_depth_cues_disabled(
    model,
    inputs,
):
    image_prev, image_curr, _ = inputs

    model.use_depth_cues = False

    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            depth_curr=None,
            return_intermediates=True,
        )

    assert torch.count_nonzero(
        outputs["depth_curr"]
    ).item() == 0


def test_segformer_provider_injection_shape_freezing_and_configuration(inputs):
    image_prev, image_curr, _ = inputs
    semantic = DummySemanticBranch()
    depth = DummyDepthBranch()
    model = DeepDCTVO(
        input_size=(120, 120),
        pretrained_semantic=False,
        semantic_provider="segformer",
        semantic_model_name="unit-test-segformer",
        semantic_foreground_class_ids=(1, 3),
        semantic_foreground_labels=("person", "car"),
        semantic_feed_size=(64, 96),
        semantic_model=semantic,
        freeze_semantic=True,
        depth_model=depth,
        freeze_depth=True,
        use_semantic_cues=True,
        use_depth_cues=True,
    )

    assert model.semantic_provider == "segformer"
    assert model.semantic_model_name == "unit-test-segformer"
    assert model.semantic_foreground_class_ids == (1, 3)
    assert model.semantic_feed_size == (64, 96)
    assert all(not parameter.requires_grad for parameter in semantic.parameters())

    model.train()
    assert semantic.training is False
    with torch.no_grad():
        outputs = model(
            image_prev=image_prev,
            image_curr=image_curr,
            return_intermediates=True,
        )
    assert outputs["semantic_prev"].shape == (2, 1, 120, 120)
    assert outputs["semantic_curr"].shape == (2, 1, 120, 120)


def test_unknown_semantic_provider_is_rejected():
    with pytest.raises(ValueError, match="semantic_provider"):
        DeepDCTVO(
            semantic_provider="unknown",
            semantic_model=DummySemanticBranch(),
            depth_model=DummyDepthBranch(),
        )
