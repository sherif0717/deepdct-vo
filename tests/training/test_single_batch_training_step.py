"""One-batch DeepDCT-VO training integration test.

Pipeline under test:

    real KITTI batch
    -> DeepDCTVO forward
    -> rotation loss
    -> directional-translation loss
    -> backward
    -> optimizer step

Heavy auxiliary inference is isolated using deterministic semantic and depth
stubs. The actual LR-ASPP and Lite-Mono branches are covered by their dedicated
unit tests.
"""

from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from deepdct.data.training_dataset import DeepDCTTrainingDataset
from deepdct.models.deepdct_vo import DeepDCTVO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

SEQUENCE = "00"
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
BATCH_SIZE = 1
LEARNING_RATE = 1.0e-4


class DummySemanticBranch(nn.Module):
    """Produce a deterministic single-channel semantic map."""

    def forward(self, image: Tensor) -> Tensor:
        return image.mean(dim=1, keepdim=True)


class DummyDepthBranch(nn.Module):
    """Produce a deterministic single-channel normalized depth map."""

    def forward(self, image: Tensor) -> Tensor:
        return image.mean(dim=1, keepdim=True)


def _local_training_data_exists() -> bool:
    image_directory = (
        DATA_ROOT
        / "sequences"
        / SEQUENCE
        / "image_2"
    )

    label_path = (
        DATA_ROOT
        / "out_csv"
        / f"{SEQUENCE}_dct.txt"
    )

    return image_directory.is_dir() and label_path.is_file()


pytestmark = pytest.mark.skipif(
    not _local_training_data_exists(),
    reason="KITTI sequence 00 or its DCT label file is unavailable.",
)


@pytest.fixture(scope="module")
def batch() -> Dict[str, Tensor]:
    """Load one deterministic batch from the real KITTI dataset."""

    dataset = DeepDCTTrainingDataset(
        data_root=DATA_ROOT,
        sequences=(SEQUENCE,),
        camera="left",
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return next(iter(loader))


def _construct_lightweight_integration_model() -> DeepDCTVO:
    """Construct the core model without auxiliary weight loading."""

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        pretrained_semantic=False,
        freeze_semantic=True,
        normalize_semantic_input=False,
        normalize_semantic_map=True,
        share_aresunet_between_models=False,
        depth_checkpoint_dir=None,
        freeze_depth=True,
        depth_model=DummyDepthBranch(),
    )

    model.semantic_model = DummySemanticBranch()

    return model


def test_single_batch_training_step(
    batch: Dict[str, Tensor],
) -> None:
    """One real batch should complete forward, losses, backward, and update."""

    torch.manual_seed(0)
    torch.set_num_threads(1)

    device = torch.device("cpu")

    model = _construct_lightweight_integration_model().to(device)
    model.train()

    image_prev = batch["image_prev"].to(device)
    image_curr = batch["image_curr"].to(device)
    depth_curr = batch["depth_curr"].to(device)

    rotation_gt = batch["rotation_gt"].to(device)
    translation_gt = batch["translation_gt"].to(device)

    assert image_prev.shape == (
        BATCH_SIZE,
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    assert image_curr.shape == image_prev.shape
    assert depth_curr.shape == (
        BATCH_SIZE,
        1,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    assert rotation_gt.shape == (BATCH_SIZE, 3)
    assert translation_gt.shape == (BATCH_SIZE, 3)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    tracked_parameter = trainable_parameters[0]
    parameter_before = tracked_parameter.detach().clone()

    optimizer = Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        image_prev=image_prev,
        image_curr=image_curr,
        depth_curr=depth_curr,
        use_ground_truth_rotation=False,
    )

    assert {
        "rotation",
        "directional_translation",
        "rotation_used_for_translation",
    }.issubset(outputs)

    predicted_rotation = outputs["rotation"]
    predicted_translation = outputs["directional_translation"]

    assert predicted_rotation.shape == (BATCH_SIZE, 3)
    assert predicted_translation.shape == (BATCH_SIZE, 3)

    assert torch.isfinite(predicted_rotation).all()
    assert torch.isfinite(predicted_translation).all()

    assert torch.equal(
        outputs["rotation_used_for_translation"],
        predicted_rotation,
    )

    rotation_loss = criterion(
        predicted_rotation,
        rotation_gt,
    )

    translation_loss = criterion(
        predicted_translation,
        translation_gt,
    )

    total_loss = rotation_loss + translation_loss

    assert rotation_loss.ndim == 0
    assert translation_loss.ndim == 0
    assert total_loss.ndim == 0

    assert torch.isfinite(rotation_loss)
    assert torch.isfinite(translation_loss)
    assert torch.isfinite(total_loss)

    total_loss.backward()

    parameters_with_gradients = [
        parameter
        for parameter in trainable_parameters
        if parameter.grad is not None
    ]

    assert parameters_with_gradients

    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in parameters_with_gradients
    )

    optimizer.step()

    parameter_after = tracked_parameter.detach()

    assert not torch.equal(
        parameter_before,
        parameter_after,
    )