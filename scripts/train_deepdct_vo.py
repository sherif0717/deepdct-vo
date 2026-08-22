"""Train and validate DeepDCT-VO on KITTI odometry sequences.

Example:

    python3 scripts/train_deepdct_vo.py \
        --train-sequences 00 01 02 03 04 05 06 07 08 \
        --validation-sequences 09 \
        --epochs 5 \
        --batch-size 1 \
        --num-workers 0

This script:

1. Builds independent training and validation datasets.
2. Trains for one epoch.
3. Runs validation after every epoch.
4. Saves:
       - one checkpoint per epoch;
       - latest.pt;
       - best_validation.pt.
5. Supports resuming from a checkpoint.

Current scaffold behavior
-------------------------

By default, ``allow_zero_auxiliary=True`` is used in the dataset. Therefore,
the dataset supplies zero depth maps to the model, which bypasses Lite-Mono.

The DeepDCTVO semantic branch is still executed internally. For meaningful
semantic features, use pretrained LR-ASPP weights rather than freezing a
randomly initialized semantic model.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler

from deepdct.data.training_dataset import DeepDCTTrainingDataset
from deepdct.models.deepdct_vo import DeepDCTVO
from deepdct.training.samplers import (
    build_sequence_balanced_sampler,
    summarize_sequence_distribution,
)

from deepdct.training.train_one_epoch import (
    EpochMetrics,
    train_one_epoch,
)
from deepdct.training.validate_one_epoch import (
    ValidationMetrics,
    validate_one_epoch,
)
from deepdct.training.rotation_geometry import (
    RotationGeometryBank,
)


CheckpointValue = Union[
    int,
    float,
    str,
    Dict[str, object],
]


@dataclass
class TranslationRegimeConfiguration:
    """Resolved training-set forward-motion regime configuration."""

    low_threshold: float
    high_threshold: float

    low_count: int
    medium_count: int
    high_count: int

    low_weight: float
    medium_weight: float
    high_weight: float

    low_quantile: float
    high_quantile: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "counts": {
                "low": self.low_count,
                "medium": self.medium_count,
                "high": self.high_count,
            },
            "weights": {
                "low": self.low_weight,
                "medium": self.medium_weight,
                "high": self.high_weight,
            },
            "quantiles": {
                "low": self.low_quantile,
                "high": self.high_quantile,
            },
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""

    parser = argparse.ArgumentParser(
        description=(
            "Train DeepDCT-VO with sequence-level KITTI validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing sequences/, out_csv/, and poses/.",
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=[
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
        ],
        help="KITTI sequences used for training.",
    )

    parser.add_argument(
        "--validation-sequences",
        nargs="+",
        default=["09"],
        help="KITTI sequences used for validation.",
    )

    parser.add_argument(
        "--camera",
        choices=[
            "left",
            "right",
            "image_2",
            "image_3",
        ],
        default="left",
        help="KITTI camera stream.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total number of epochs, including resumed epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training and validation batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-4,
        help="Initial Adam learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Input image height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="Input image width.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )

    parser.add_argument(
        "--rotation-pool-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(8, 8),
        help=(
            "Adaptive pooling size for the compact rotation "
            "representation. Default 8x8 gives a 64-D bottleneck."
        ),
    )

    parser.add_argument(
        "--rotation-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to rotation loss.",
    )

    # ------------------------------------------------------------------
    # Track-A pose objective / rotation normalization
    # ------------------------------------------------------------------

    parser.add_argument(
        "--pose-loss-type",
        choices=[
            "mse",
            "mae",
        ],
        default="mse",
        help=(
            "Pose regression objective. "
            "'mse' preserves the A1 baseline. "
            "'mae' enables the paper-style A2 MAE objective for "
            "both rotation and directional translation."
        ),
    )

    parser.add_argument(
        "--rotation-normalization-scale",
        type=float,
        default=1.0,
        help=(
            "Scale mapping the RotationHead regression output back to "
            "physical Euler radians. "
            "Use 1.0 for A1 and 0.175 for Track-A A2."
        ),
    )

    parser.add_argument(
        "--rotation-geometry-weight",
        type=float,
        default=0.0,
        help=(
            "Weight applied to continuous SO(3)-supervised "
            "rotation-representation geometry loss. "
            "Zero disables the geometry objective."
        ),
    )

    parser.add_argument(
        "--rotation-geometry-bank-size",
        type=int,
        default=4096,
        help=(
            "Maximum number of detached training samples retained "
            "in the FIFO memory bank used by the continuous "
            "rotation-geometry objective."
        ),
    )

    parser.add_argument(
        "--rotation-geometry-temperature",
        type=float,
        default=0.1,
        help=(
            "Temperature used by the continuous SO(3)-supervised "
            "rotation-geometry objective."
        ),
    )

    parser.add_argument(
        "--translation-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to directional-translation loss.",
    )

    parser.add_argument(
        "--translation-loss",
        choices=[
            "mse",
            "regime_balanced_mse",
        ],
        default="mse",
        help=(
            "Translation objective. "
            "'mse' preserves the ordinary unweighted MSE baseline. "
            "'regime_balanced_mse' applies fixed inverse-frequency "
            "weights based on training-set forward-displacement regimes."
        ),
    )

    parser.add_argument(
        "--translation-regime-quantiles",
        type=float,
        nargs=2,
        metavar=("LOW_Q", "HIGH_Q"),
        default=(0.25, 0.75),
        help=(
            "Training-set t_z quantiles used to define low, medium, "
            "and high forward-motion regimes when "
            "--translation-loss regime_balanced_mse is selected."
        ),
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help=(
            "Maximum gradient norm. Use a non-positive value to "
            "disable clipping."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print running metrics every N batches.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory used for training checkpoints.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint from which training should resume.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Condition Model T on ground-truth rotation rather than "
            "Model R's prediction."
        ),
    )

    parser.add_argument(
        "--pretrained-semantic",
        dest="pretrained_semantic",
        action="store_true",
        help="Use pretrained LR-ASPP weights.",
    )

    parser.add_argument(
        "--no-pretrained-semantic",
        dest="pretrained_semantic",
        action="store_false",
        help="Do not use pretrained LR-ASPP weights.",
    )

    parser.set_defaults(
        pretrained_semantic=True,
    )

    parser.add_argument(
        "--freeze-semantic",
        dest="freeze_semantic",
        action="store_true",
        help="Freeze the LR-ASPP semantic branch.",
    )

    parser.set_defaults(
        freeze_semantic=True,
    )

    parser.add_argument(
        "--share-aresunet-between-models",
        action="store_true",
        help="Share one A-ResUNet between Model R and Model T.",
    )

    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="Validation epochs before reducing the learning rate.",
    )

    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Learning-rate reduction factor.",
    )

    parser.add_argument(
        "--skip-nonfinite-batches",
        action="store_true",
        help="Skip batches with non-finite loss rather than failing.",
    )

    parser.add_argument(
        "--save-every-epoch",
        dest="save_every_epoch",
        action="store_true",
        help="Save a separately numbered checkpoint after each epoch.",
    )

    parser.add_argument(
        "--no-save-every-epoch",
        dest="save_every_epoch",
        action="store_false",
        help="Do not save separately numbered epoch checkpoints.",
    )

    parser.set_defaults(
        save_every_epoch=True,
    )

    parser.add_argument(
        "--sampling-strategy",
        choices=[
            "transition_uniform",
            "sequence_balanced",
        ],
        default="transition_uniform",
        help=(
            "Training sampling strategy. transition_uniform gives every "
            "transition equal probability; sequence_balanced gives every "
            "training sequence equal aggregate probability."
        ),
    )

    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=1.0,
        help=(
            "Inverse-frequency sequence-weighting exponent used when "
            "--sampling-strategy sequence_balanced is selected. "
            "Use 0.5 for tempered balancing and 1.0 for full balancing."
        ),
    )

    parser.add_argument(
        "--train-semantic-model",
        action="store_true",
        help="Allow semantic-model parameters to be updated.",
    )

    parser.add_argument(
        "--train-depth-model",
        action="store_true",
        help="Allow depth-model parameters to be updated.",
    )

    parser.add_argument(
        "--depth-weights-dir",
        type=str,
        default="weights/lite-mono-tiny-640x192",
    )

    parser.add_argument(
        "--depth-checkpoint-dir",
        type=Path,
        default=Path(
            "weights/lite-mono-tiny-640x192"
        ),
        help=(
            "Directory containing the Lite-Mono encoder and decoder "
            "checkpoint files."
        ),
    )

    parser.add_argument(
        "--depth-model-name",
        type=str,
        default="lite-mono-tiny",
        choices=[
            "lite-mono",
            "lite-mono-small",
            "lite-mono-tiny",
            "lite-mono-8m",
        ],
        help="Lite-Mono architecture corresponding to the checkpoint.",
    )

    parser.add_argument(
        "--depth-output-mode",
        choices=(
            "disparity",
            "scaled_disparity",
            "depth",
            "normalized_depth",
        ),
        default="normalized_depth",
    )

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load model weights from this checkpoint before training. "
            "Optimizer, scheduler, epoch counter, and best-validation state "
            "are not restored. Intended for warm-start experiments."
        ),
    )

    parser.add_argument(
        "--translation-head-only",
        action="store_true",
        help=(
            "Freeze the model except for parameters selected by the "
            "translation decoder's controlled fine-tuning policy. "
            "For pooled_linear, only translation_head.dense.weight "
            "and translation_head.dense.bias are trainable."
        ),
    )

    parser.add_argument(
        "--rotation-readout-only",
        action="store_true",
        help=(
            "Freeze the complete model except rotation_head.dense. "
            "This preserves the learned rotation representation and "
            "trains only a fresh Linear(D, 3) rotation readout."
        ),
    )

    parser.add_argument(
        "--translation-decoder",
        type=str,
        default="dense",
        choices=[
            "dense",
            "mlp",
            "pooled_mlp",
            "pooled_linear",
            "gated_expert",
        ],
        help=(
            "Translation decoder architecture. "
            "'pooled_linear' maps the compact pooled representation "
            "directly through Linear(D, 3). "
            "'gated_expert' uses the same compact representation "
            "with a learned soft gate over tiny linear experts."
        ),
    )

    parser.add_argument(
        "--translation-mlp-hidden-dims",
        type=int,
        nargs=2,
        metavar=("H1", "H2"),
        default=(256, 64),
        help=(
            "Hidden dimensions used when "
            "--translation-decoder mlp is selected."
        ),
    )

    parser.add_argument(
        "--translation-projection-channels",
        type=int,
        default=8,
        help=(
            "Number of learned translation feature maps K used by "
            "--translation-decoder pooled_mlp."
        ),
    )

    parser.add_argument(
        "--translation-pool-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(4, 4),
        help=(
            "Adaptive average-pooling output size used by "
            "--translation-decoder pooled_mlp."
        ),
    )

    parser.add_argument(
        "--translation-aggregation-hidden-dim",
        type=int,
        default=64,
        help=(
            "Hidden dimension of the small MLP following "
            "structured translation aggregation."
        ),
    )

    parser.add_argument(
        "--translation-num-experts",
        type=int,
        default=3,
        help=(
            "Number of translation experts used when "
            "--translation-decoder gated_expert. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="deepdct_vo",
        help="Name recorded in checkpoints and training summaries.",
    )

    parser.add_argument(
        "--source-experiment",
        type=str,
        default=None,
        help=(
            "Optional description of the experiment supplying the "
            "warm-start checkpoint."
        ),
    )

    # ------------------------------------------------------------------
    # Warm-start behavior
    # ------------------------------------------------------------------

    parser.add_argument(
        "--strict-init-checkpoint",
        dest="strict_init_checkpoint",
        action="store_true",
        help=(
            "Require all checkpoint model parameters to match when using "
            "--init-checkpoint."
        ),
    )

    parser.add_argument(
        "--no-strict-init-checkpoint",
        dest="strict_init_checkpoint",
        action="store_false",
        help=(
            "Allow missing or unexpected model parameters when using "
            "--init-checkpoint."
        ),
    )

    parser.add_argument(
        "--reset-translation-head-on-init",
        action="store_true",
        help=(
            "When warm-starting with --init-checkpoint, do not load any "
            "translation_head.* parameters from the source checkpoint. "
            "The translation head is initialized from the currently "
            "selected architecture instead. Intended for controlled "
            "translation-head architecture replacement experiments."
        ),
    )

    parser.add_argument(
        "--reset-rotation-readout-on-init",
        action="store_true",
        help=(
            "When warm-starting from --init-checkpoint, exclude only "
            "rotation_head.dense.weight and rotation_head.dense.bias. "
            "The upstream rotation representation extractor, including "
            "rotation_head.conv, remains loaded from the checkpoint."
        ),
    )

    parser.set_defaults(
        strict_init_checkpoint=True,
    )

    # ------------------------------------------------------------------
    # Semantic cues
    # ------------------------------------------------------------------

    parser.add_argument(
        "--use-semantic-cues",
        dest="use_semantic_cues",
        action="store_true",
        help="Enable semantic cues in DeepDCT-VO.",
    )

    parser.add_argument(
        "--no-use-semantic-cues",
        dest="use_semantic_cues",
        action="store_false",
        help="Disable semantic cues in DeepDCT-VO.",
    )

    parser.set_defaults(
        use_semantic_cues=False,
    )

    parser.set_defaults(
        pretrained_semantic=True,
    )

    parser.add_argument(
        "--no-freeze-semantic",
        dest="freeze_semantic",
        action="store_false",
        help="Allow the semantic model to train.",
    )

    parser.set_defaults(
        freeze_semantic=True,
    )

    # ------------------------------------------------------------------
    # Depth cues
    # ------------------------------------------------------------------

    parser.add_argument(
        "--use-depth-cues",
        dest="use_depth_cues",
        action="store_true",
        help="Enable depth cues in DeepDCT-VO.",
    )

    parser.add_argument(
        "--no-use-depth-cues",
        dest="use_depth_cues",
        action="store_false",
        help="Disable depth cues in DeepDCT-VO.",
    )

    parser.set_defaults(
        use_depth_cues=False,
    )

    parser.add_argument(
        "--freeze-depth",
        dest="freeze_depth",
        action="store_true",
        help="Freeze the depth model.",
    )

    parser.add_argument(
        "--no-freeze-depth",
        dest="freeze_depth",
        action="store_false",
        help="Allow the depth model to train.",
    )

    parser.set_defaults(
        freeze_depth=True,
    )

    parser.add_argument(
        "--semantic-map-mode",
        choices=(
            "foreground_probability",
            "class_index",
        ),
        default="foreground_probability",
        help=(
            "One-channel representation generated from LR-ASPP logits. "
            "'foreground_probability' uses 1 - P(background); "
            "'class_index' preserves the legacy normalized argmax map."
        ),
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "Stop training after this many consecutive epochs "
            "without a meaningful validation-loss improvement. "
            "Use 0 to disable early stopping."
        ),
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1.0e-4,
        help=(
            "Minimum validation-loss decrease required to count "
            "as an improvement for early stopping."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line configuration."""

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")

    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            "--height and --width must both be positive."
        )

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")

    if args.rotation_loss_weight < 0:
        raise ValueError(
            "--rotation-loss-weight cannot be negative."
        )

    if args.rotation_normalization_scale <= 0.0:
        raise ValueError(
            "--rotation-normalization-scale must be greater than zero."
        )

    if (
        args.pose_loss_type == "mae"
        and args.translation_loss != "mse"
    ):
        raise ValueError(
            "--pose-loss-type mae is incompatible with "
            "--translation-loss regime_balanced_mse. "
            "Track-A A2 requires ordinary MAE for translation."
        )

    if args.pose_loss_type == "mae":
        if args.rotation_geometry_weight != 0.0:
            raise ValueError(
                "Track-A A2 must not enable rotation geometry supervision."
            )

        if args.use_ground_truth_rotation:
            raise ValueError(
                "Track-A A2 must use predicted rotation for Model T. "
                "Ground-truth rotation conditioning belongs to A3."
            )

        if args.translation_head_only:
            raise ValueError(
                "Track-A A2 must train the complete A1 pose model."
            )

        if args.rotation_readout_only:
            raise ValueError(
                "Track-A A2 must not use rotation-readout-only training."
            )

    if args.translation_loss_weight < 0:
        raise ValueError(
            "--translation-loss-weight cannot be negative."
        )

    if (
        args.rotation_loss_weight == 0
        and args.translation_loss_weight == 0
        and args.rotation_geometry_weight == 0
    ):
        raise ValueError(
            "At least one loss weight must be positive."
        )
        
    if args.rotation_geometry_weight < 0.0:
        raise ValueError(
            "--rotation-geometry-weight cannot be negative."
        )

    if args.rotation_geometry_bank_size <= 0:
        raise ValueError(
            "--rotation-geometry-bank-size must be positive."
        )

    if args.rotation_geometry_temperature <= 0.0:
        raise ValueError(
            "--rotation-geometry-temperature must be positive."
        )

    if (
        len(args.rotation_pool_size) != 2
        or min(args.rotation_pool_size) <= 0
    ):
        raise ValueError(
            "--rotation-pool-size requires two positive integers."
        )

    if args.scheduler_patience < 0:
        raise ValueError(
            "--scheduler-patience cannot be negative."
        )

    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError(
            "--scheduler-factor must lie between zero and one."
        )

    training_sequences = set(args.train_sequences)
    validation_sequences = set(args.validation_sequences)

    overlap = training_sequences.intersection(
        validation_sequences
    )

    if overlap:
        raise ValueError(
            "Training and validation sequences must be disjoint. "
            f"Overlap: {sorted(overlap)}."
        )

    if (
        not args.pretrained_semantic
        and args.freeze_semantic
    ):
        raise ValueError(
            "The semantic branch cannot be frozen while using randomly "
            "initialized weights. Use either:\n"
            "  --pretrained-semantic --freeze-semantic\n"
            "or:\n"
            "  --no-pretrained-semantic --no-freeze-semantic"
        )
    
    if not 0.0 <= args.sampling_alpha <= 1.0:
        raise ValueError(
            "--sampling-alpha must lie between 0.0 and 1.0."
        )
    
    if (
        args.init_checkpoint is not None
        and args.resume is not None
    ):
        raise ValueError(
            "--init-checkpoint and --resume-checkpoint cannot be used together."
        )
    
    if (
        args.translation_head_only
        and args.init_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "--translation-head-only requires either:\n"
            "  --init-checkpoint <baseline checkpoint>\n"
            "or:\n"
            "  --resume-checkpoint <existing translation-head-only checkpoint>."
        )
    
    if any(
        dimension <= 0
        for dimension in args.translation_mlp_hidden_dims
    ):
        raise ValueError(
            "--translation-mlp-hidden-dims values "
            "must both be positive."
        )
    
    if args.translation_projection_channels <= 0:
        raise ValueError(
            "--translation-projection-channels must be positive."
        )

    if any(
        dimension <= 0
        for dimension in args.translation_pool_size
    ):
        raise ValueError(
            "--translation-pool-size values must both be positive."
        )

    if args.translation_aggregation_hidden_dim <= 0:
        raise ValueError(
            "--translation-aggregation-hidden-dim must be positive."
        )

    if args.translation_num_experts <= 0:
        raise ValueError(
            "--translation-num-experts must be positive."
        ) 

    low_q, high_q = args.translation_regime_quantiles

    if not (
        0.0 < low_q < high_q < 1.0
    ):
        raise ValueError(
            "--translation-regime-quantiles must satisfy "
            "0 < LOW_Q < HIGH_Q < 1."
        )

    if (
        args.translation_loss
        == "regime_balanced_mse"
        and args.translation_decoder != "pooled_linear"
    ):
        raise ValueError(
            "Trial-2 regime-balanced MSE is intended for "
            "--translation-decoder pooled_linear so that the "
            "architecture remains identical to Trial 1."
        )

    if (
        args.translation_loss
        == "regime_balanced_mse"
        and not args.translation_head_only
    ):
        raise ValueError(
            "Trial-2 regime-balanced MSE requires "
            "--translation-head-only to preserve the frozen "
            "128-D upstream representation."
        )
    
    if (
        args.rotation_readout_only
        and args.init_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "--rotation-readout-only requires either:\n"
            "  --init-checkpoint <source checkpoint>\n"
            "or:\n"
            "  --resume <existing rotation-readout-only checkpoint>."
        )
    
    if (
        args.translation_head_only
        and args.rotation_readout_only
    ):
        raise ValueError(
            "--translation-head-only and --rotation-readout-only "
            "cannot be enabled together."
        )
    
    if (
        args.rotation_geometry_weight > 0.0
        and args.translation_head_only
    ):
        raise ValueError(
            "--rotation-geometry-weight cannot be used with "
            "--translation-head-only because the rotation "
            "representation is frozen."
        )

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode may reduce performance, but it makes runs easier
    # to reproduce during development.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(
    args: argparse.Namespace,
    sequences: List[str],
) -> DeepDCTTrainingDataset:
    """Construct one supervised KITTI dataset."""

    return DeepDCTTrainingDataset(
        data_root=args.data_root,
        sequences=sequences,
        camera=args.camera,
        image_size=(args.height, args.width),
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=False,
    )

def build_translation_regime_configuration(
    dataset: DeepDCTTrainingDataset,
    quantiles: Tuple[float, float],
) -> TranslationRegimeConfiguration:
    """Derive fixed t_z regimes from the training dataset only."""

    if not hasattr(dataset, "records"):
        raise AttributeError(
            "DeepDCTTrainingDataset does not expose 'records'."
        )

    if not dataset.records:
        raise RuntimeError(
            "Cannot derive translation regimes from an empty dataset."
        )

    tz_values = np.asarray(
        [
            float(record.translation_gt[2])
            for record in dataset.records
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(tz_values)):
        raise ValueError(
            "Training-set t_z values contain NaN or infinity."
        )

    low_q, high_q = quantiles

    low_threshold = float(
        np.quantile(tz_values, low_q)
    )

    high_threshold = float(
        np.quantile(tz_values, high_q)
    )

    if not low_threshold < high_threshold:
        raise RuntimeError(
            "Training-set regime thresholds are not strictly ordered: "
            f"{low_threshold} >= {high_threshold}."
        )

    low_mask = (
        tz_values <= low_threshold
    )

    medium_mask = (
        (tz_values > low_threshold)
        & (tz_values <= high_threshold)
    )

    high_mask = (
        tz_values > high_threshold
    )

    low_count = int(
        np.count_nonzero(low_mask)
    )
    medium_count = int(
        np.count_nonzero(medium_mask)
    )
    high_count = int(
        np.count_nonzero(high_mask)
    )

    counts = np.asarray(
        [
            low_count,
            medium_count,
            high_count,
        ],
        dtype=np.float64,
    )

    if np.any(counts <= 0):
        raise RuntimeError(
            "Every translation regime must contain at least "
            f"one training sample; counts={counts.tolist()}."
        )

    # Raw inverse-frequency weights.
    raw_weights = 1.0 / counts

    # Normalize globally so the average sample weight over the
    # entire training set is exactly 1.
    #
    # mean_weight =
    #     sum_r N_r * (1/N_r) / N
    #
    # This is intentionally GLOBAL normalization. Do not
    # renormalize within an individual batch, because batch_size=1
    # would cancel the regime weight completely.
    total_samples = float(
        len(tz_values)
    )

    global_mean_weight = float(
        np.sum(
            counts * raw_weights
        )
        / total_samples
    )

    normalized_weights = (
        raw_weights / global_mean_weight
    )

    return TranslationRegimeConfiguration(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        low_weight=float(normalized_weights[0]),
        medium_weight=float(normalized_weights[1]),
        high_weight=float(normalized_weights[2]),
        low_quantile=float(low_q),
        high_quantile=float(high_q),
    )

class RegimeBalancedTranslationMSE(nn.Module):
    """Globally normalized regime-balanced translation MSE.

    Regime assignment is based only on ground-truth t_z.

    For each sample:

        per_sample_mse =
            mean((prediction - target)^2 over x,y,z)

        loss =
            mean(regime_weight * per_sample_mse)

    The regime weights are normalized over the full training set
    to have mean 1. They are NOT normalized within each batch.
    """

    def __init__(
        self,
        configuration: TranslationRegimeConfiguration,
    ) -> None:
        super().__init__()

        self.low_threshold = float(
            configuration.low_threshold
        )
        self.high_threshold = float(
            configuration.high_threshold
        )

        self.low_weight = float(
            configuration.low_weight
        )
        self.medium_weight = float(
            configuration.medium_weight
        )
        self.high_weight = float(
            configuration.high_weight
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have identical shapes, "
                f"but received {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        if prediction.ndim != 2 or prediction.shape[1] != 3:
            raise ValueError(
                "Translation prediction and target must have "
                f"shape [B, 3], received {tuple(prediction.shape)}."
            )

        tz = target[:, 2]

        weights = torch.full_like(
            tz,
            fill_value=self.medium_weight,
        )

        weights = torch.where(
            tz <= self.low_threshold,
            torch.full_like(
                weights,
                self.low_weight,
            ),
            weights,
        )

        weights = torch.where(
            tz > self.high_threshold,
            torch.full_like(
                weights,
                self.high_weight,
            ),
            weights,
        )

        per_sample_mse = torch.mean(
            (prediction - target) ** 2,
            dim=1,
        )

        # Deliberately NOT:
        #
        #   sum(weights * error) / sum(weights)
        #
        # because with batch_size=1 that would cancel the weight.
        loss = torch.mean(
            weights * per_sample_mse
        )

        return loss


def build_dataloader(
    dataset: DeepDCTTrainingDataset,
    args: argparse.Namespace,
    device: torch.device,
    *,
    shuffle: bool,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    """Construct a training or validation DataLoader."""

    if sampler is not None and shuffle:
        raise ValueError(
            "DataLoader cannot use both sampler and shuffle=True."
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def build_model(
    args: argparse.Namespace,
    device: torch.device,
) -> DeepDCTVO:
    """Construct DeepDCTVO using the selected cue configuration."""

    depth_checkpoint_dir = (
        str(args.depth_checkpoint_dir)
        if args.use_depth_cues
        else None
    )

    model = DeepDCTVO(
        aresunet_output_channels=1,
        input_size=(args.height, args.width),
        pretrained_semantic=args.pretrained_semantic,
        freeze_semantic=args.freeze_semantic,
        normalize_semantic_input=True,
        normalize_semantic_map=True,
        semantic_map_mode=args.semantic_map_mode,
        share_aresunet_between_models=(
            args.share_aresunet_between_models
        ),
        rotation_pool_size=tuple(
            args.rotation_pool_size
        ),
        rotation_normalization_scale=(
            args.rotation_normalization_scale
        ),
        translation_decoder_type=(
            args.translation_decoder
        ),
        translation_mlp_hidden_dims=tuple(
            args.translation_mlp_hidden_dims
        ),
        translation_projection_channels=(
            args.translation_projection_channels
        ),
        translation_pool_size=tuple(
            args.translation_pool_size
        ),
        translation_aggregation_hidden_dim=(
            args.translation_aggregation_hidden_dim
        ),
        translation_num_experts=(
            args.translation_num_experts
        ),
        depth_checkpoint_dir=depth_checkpoint_dir,
        depth_model_name=args.depth_model_name,
        depth_output_mode=args.depth_output_mode,
        freeze_depth=args.freeze_depth,
        use_semantic_cues=args.use_semantic_cues,
        use_depth_cues=args.use_depth_cues,
    )

    return model.to(device)

def configure_translation_head_only(
    model: nn.Module,
) -> None:
    """Freeze everything except the translation decoder."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    decoder_type = getattr(
        model.translation_head,
        "decoder_type",
        None,
    )

    if decoder_type == "pooled_linear":
        # ----------------------------------------------------------
        # Frozen Backbone + Linear Readout experiment.
        #
        # Everything is frozen except:
        #
        #   translation_head.dense.weight
        #   translation_head.dense.bias
        #
        # Critically, translation_head.conv remains frozen so the
        # audited compact 128-D representation is not allowed to
        # move during this experiment.
        # ----------------------------------------------------------
        translation_parameter_names = {
            name
            for name, parameter
            in model.named_parameters()
            if name.startswith(
                "translation_head.dense."
            )
        }

    elif decoder_type == "pooled_mlp":
        translation_parameter_names = {
            name
            for name, parameter
            in model.named_parameters()
            if (
                name.startswith(
                    "translation_head.conv."
                )
                or name.startswith(
                    "translation_head.dense."
                )
            )
        }

    elif decoder_type == "gated_expert":
        translation_parameter_names = {
            name
            for name, parameter
            in model.named_parameters()
            if (
                name.startswith(
                    "translation_head.conv."
                )
                or name.startswith(
                    "translation_head.gate."
                )
                or name.startswith(
                    "translation_head.experts."
                )
            )
        }

    else:
        # Preserve dense/MLP behavior.
        translation_parameter_names = {
            name
            for name, parameter
            in model.named_parameters()
            if name.startswith(
                "translation_head.dense."
            )
        }

    if not translation_parameter_names:
        raise RuntimeError(
            "No translation decoder parameters were found."
        )

    named_parameters = dict(
        model.named_parameters()
    )

    for name in translation_parameter_names:
        named_parameters[name].requires_grad = True

    actual_trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if decoder_type == "pooled_linear":
        expected_linear_parameters = {
            "translation_head.dense.weight",
            "translation_head.dense.bias",
        }

        if actual_trainable != expected_linear_parameters:
            raise RuntimeError(
                "Frozen-representation linear-readout audit failed.\n"
                f"Expected: {sorted(expected_linear_parameters)}\n"
                f"Actual:   {sorted(actual_trainable)}"
            )

    if actual_trainable != translation_parameter_names:
        raise RuntimeError(
            "Translation-head-only parameter audit failed.\n"
            f"Expected: {sorted(translation_parameter_names)}\n"
            f"Actual:   {sorted(actual_trainable)}"
        )

    print("=" * 88)
    print("Translation-head-only fine-tuning enabled")
    print("=" * 88)

    for name in sorted(actual_trainable):
        parameter = named_parameters[name]

        print(
            f"TRAINABLE: {name:<50} "
            f"shape={tuple(parameter.shape)}"
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("-" * 88)
    print(
        f"Trainable parameters: {trainable_count:,}"
    )
    print(
        f"Total parameters:     {total_count:,}"
    )
    print("=" * 88)

def configure_rotation_readout_only(
    model: nn.Module,
) -> Tuple[int, int]:
    """Freeze everything except the final rotation Linear(D, 3).

    The experiment preserves the complete learned rotation
    representation:

        fusion
            -> rotation_head.conv
            -> ReLU
            -> Dropout
            -> Flatten
            -> h

    and retrains only:

        h -> rotation_head.dense -> [rx, ry, rz]

    Returns
    -------
    trainable_parameters:
        Number of trainable scalar parameters.

    total_parameters:
        Total number of scalar model parameters.
    """

    # Freeze everything first.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if not hasattr(
        model,
        "rotation_head",
    ):
        raise AttributeError(
            "Model has no rotation_head."
        )

    rotation_head = model.rotation_head

    if not hasattr(
        rotation_head,
        "dense",
    ):
        raise AttributeError(
            "rotation_head has no dense readout."
        )

    dense = rotation_head.dense

    if not isinstance(
        dense,
        nn.Linear,
    ):
        raise TypeError(
            "Frozen-representation rotation experiment requires "
            "rotation_head.dense to be nn.Linear, but received "
            f"{type(dense).__name__}."
        )

    # Only fresh Linear(D, 3) is trainable.
    for parameter in dense.parameters():
        parameter.requires_grad_(True)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    expected_trainable = (
        dense.in_features
        * dense.out_features
        + dense.out_features
    )

    if trainable_parameters != expected_trainable:
        raise RuntimeError(
            "Unexpected number of trainable rotation-readout "
            "parameters. "
            f"Expected {expected_trainable}, "
            f"received {trainable_parameters}."
        )

    print()
    print("=" * 72)
    print("Frozen rotation representation experiment")
    print("=" * 72)

    print(
        "Rotation representation: frozen"
    )

    print(
        "Trainable readout:       "
        f"Linear({dense.in_features}, {dense.out_features})"
    )

    print(
        f"Trainable parameters:    {trainable_parameters:,}"
    )

    print(
        f"Total parameters:        {total_parameters:,}"
    )

    print(
        "Trainable tensors:"
    )

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(
                f"  {name:<45} "
                f"{tuple(parameter.shape)}"
            )

    print("=" * 72)

    return (
        trainable_parameters,
        total_parameters,
    )

def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> Adam:
    """Construct the Adam optimizer over trainable parameters."""

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    return Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> Dict[str, object]:
    """Restore model, optimizer, scheduler, and epoch state."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
    }

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing: "
            f"{sorted(missing_keys)}."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if (
        "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    training_metrics: EpochMetrics,
    validation_metrics: ValidationMetrics,
    best_validation_loss: float,
    args: argparse.Namespace,
    warm_start_metadata: Optional[
        Dict[str, Any]
    ] = None,
    translation_regime_configuration: Optional[
        TranslationRegimeConfiguration
    ] = None,
    rotation_geometry_loss_fn: Optional[
        RotationGeometryBank
    ] = None,
) -> None:
    """Save complete resumable training state."""

    warm_start_provenance: Dict[str, object] = {
        "enabled": warm_start_metadata is not None,
        "checkpoint_path": None,
        "source_epoch": None,
        "source_validation_loss": None,
        "missing_keys": [],
        "unexpected_keys": [],
        "reset_translation_head": False,
        "reset_rotation_readout": False,
        "excluded_keys": [],
    }

    if warm_start_metadata is not None:
        warm_start_provenance.update(
            {
                "checkpoint_path": warm_start_metadata.get(
                    "checkpoint_path"
                ),
                "source_epoch": warm_start_metadata.get(
                    "source_epoch"
                ),
                "source_validation_loss": (
                    warm_start_metadata.get(
                        "source_validation_loss"
                    )
                ),
                "missing_keys": list(
                    warm_start_metadata.get(
                        "missing_keys",
                        [],
                    )
                ),
                "unexpected_keys": list(
                    warm_start_metadata.get(
                        "unexpected_keys",
                        [],
                    )
                ),
                                "reset_translation_head": bool(
                    warm_start_metadata.get(
                        "reset_translation_head",
                        False,
                    )
                ),
                "reset_rotation_readout": bool(
                    warm_start_metadata.get(
                        "reset_rotation_readout",
                        False,
                    )
                ),
                "excluded_keys": list(
                    warm_start_metadata.get(
                        "excluded_keys",
                        [],
                    )
                ),
            }
        )

    checkpoint: Dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_metrics": training_metrics.as_dict(),
        "validation_metrics": validation_metrics.as_dict(),
        "best_validation_loss": best_validation_loss,
        "rotation_geometry_state": (
            rotation_geometry_loss_fn.state_dict()
            if rotation_geometry_loss_fn is not None
            else None
        ),

        # ------------------------------------------------------------
        # Experiment identity and provenance
        # ------------------------------------------------------------
        "experiment": {
            "name": args.experiment_name,
            "source_experiment": args.source_experiment,
            "experiment_type": (
                "frozen_rotation_representation_linear_readout"
                if args.rotation_readout_only
                else (
                    (
                        "frozen_representation_linear_readout"
                        if args.translation_decoder
                        == "pooled_linear"
                        else "translation_head_only_finetune"
                    )
                    if args.translation_head_only
                    else (
                        "continuous_so3_rotation_geometry"
                        if args.rotation_geometry_weight > 0.0
                        else (
                            "warm_start_cue_adaptation"
                            if warm_start_metadata is not None
                            else "standard_training"
                        )
                    )
                )
            ),
            "trainable_parameters": (
                sorted(
                    name
                    for name, parameter
                    in model.named_parameters()
                    if parameter.requires_grad
                )
                if (
                    args.translation_head_only
                    or args.rotation_readout_only
                )
                else None
            ),
            "translation_decoder": (
                args.translation_decoder
            ),
        },

        "warm_start": warm_start_provenance,

        "translation_loss_configuration": {
            "type": args.translation_loss,
            "regime_balancing": (
                translation_regime_configuration.as_dict()
                if translation_regime_configuration is not None
                else None
            ),
        },

        "cue_config": {
            "use_semantic_cues": (
                args.use_semantic_cues
            ),
            "use_depth_cues": (
                args.use_depth_cues
            ),
            "pretrained_semantic": (
                args.pretrained_semantic
            ),
            "freeze_semantic": (
                args.freeze_semantic
            ),
            "train_semantic_model": (
                args.train_semantic_model
            ),
            "freeze_depth": (
                args.freeze_depth
            ),
            "train_depth_model": (
                args.train_depth_model
            ),
            "use_internal_depth": (
                args.use_depth_cues
            ),
            "depth_checkpoint_dir": (
                str(
                    args.depth_checkpoint_dir.resolve()
                )
                if args.use_depth_cues
                else None
            ),
            "depth_model_name": (
                args.depth_model_name
                if args.use_depth_cues
                else None
            ),
            "depth_output_mode": (
                args.depth_output_mode
                if args.use_depth_cues
                else None
            ),
        },

        # ------------------------------------------------------------
        # Complete resolved training configuration
        # ------------------------------------------------------------
        "configuration": {
            "data_root": str(
                args.data_root.resolve()
            ),
            "train_sequences": list(
                args.train_sequences
            ),
            "validation_sequences": list(
                args.validation_sequences
            ),
            "sampling_strategy": (
                args.sampling_strategy
            ),
            "sampling_alpha": (
                args.sampling_alpha
            ),
            "camera": args.camera,
            "height": args.height,
            "width": args.width,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "rotation_loss_weight": (
                args.rotation_loss_weight
            ),
            # ------------------------------------------------------------
            # Track-A pose-loss / rotation-normalization configuration.
            #
            # A1:
            #     scale = 1.0
            #     pose_loss_type = "mse"
            #
            # A2:
            #     scale = 0.175
            #     pose_loss_type = "mae"
            # ------------------------------------------------------------
            "rotation_normalization_scale": (
                args.rotation_normalization_scale
            ),

            "pose_loss_type": (
                args.pose_loss_type
            ),
            "translation_loss_weight": (
                args.translation_loss_weight
            ),
            "translation_loss": (
                args.translation_loss
            ),
            "translation_regime_quantiles": list(
                args.translation_regime_quantiles
            ),
            "max_grad_norm": (
                args.max_grad_norm
            ),
            "scheduler_patience": (
                args.scheduler_patience
            ),
            "scheduler_factor": (
                args.scheduler_factor
            ),
            "use_ground_truth_rotation": (
                args.use_ground_truth_rotation
            ),
            "rotation_readout_only": (
                args.rotation_readout_only
            ),
            "reset_rotation_readout_on_init": (
                args.reset_rotation_readout_on_init
            ),
            "rotation_pool_size": list(
                args.rotation_pool_size
            ),

            "rotation_representation_dimension": int(
                model.rotation_head.representation_dim
            ),
            "rotation_geometry_weight": (
                args.rotation_geometry_weight
            ),

            "rotation_geometry_bank_size": (
                args.rotation_geometry_bank_size
            ),

            "rotation_geometry_temperature": (
                args.rotation_geometry_temperature
            ),
            "rotation_geometry_supervision": (
                "continuous_so3"
                if args.rotation_geometry_weight > 0.0
                else None
            ),
            "translation_head_only": (
                args.translation_head_only
            ),
            "translation_decoder": (
                args.translation_decoder
            ),
            "translation_mlp_hidden_dims": list(
                args.translation_mlp_hidden_dims
            ),
            "translation_projection_channels": (
                args.translation_projection_channels
            ),
            "translation_representation_dim": (
                args.translation_projection_channels
                * args.translation_pool_size[0]
                * args.translation_pool_size[1]
            ),
            "translation_pool_size": list(
                args.translation_pool_size
            ),
            "translation_aggregation_hidden_dim": (
                args.translation_aggregation_hidden_dim
            ),
            "translation_num_experts": (
                args.translation_num_experts
            ),
            "translation_pool_size": (
                args.translation_pool_size
            ),
            "share_aresunet_between_models": (
                args.share_aresunet_between_models
            ),

            # Semantic-cue configuration
            "use_semantic_cues": (
                args.use_semantic_cues
            ),
            "pretrained_semantic": (
                args.pretrained_semantic
            ),
            "freeze_semantic": (
                args.freeze_semantic
            ),
            "semantic_map_mode": args.semantic_map_mode,

            # Depth-cue configuration
            "use_depth_cues": (
                args.use_depth_cues
            ),
            "use_internal_depth": (
                args.use_depth_cues
            ),
            "depth_checkpoint_dir": (
                str(
                    args.depth_checkpoint_dir.resolve()
                )
                if args.use_depth_cues
                else None
            ),
            "depth_model_name": (
                args.depth_model_name
            ),
            "depth_output_mode": (
                args.depth_output_mode
            ),
            "freeze_depth": (
                args.freeze_depth
            ),

            # Reproducibility
            "seed": args.seed,
            "device": str(
                next(model.parameters()).device
            ),
            "torch_version": torch.__version__,
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(path)


def print_run_summary(
    args: argparse.Namespace,
    device: torch.device,
    training_dataset: DeepDCTTrainingDataset,
    validation_dataset: DeepDCTTrainingDataset,
    model: nn.Module,
    warm_start_metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> None:
    """Print the resolved training configuration."""

    if warm_start_metadata is not None:
        print(
            f"Baseline epoch:       "
            f"{warm_start_metadata.get('source_epoch')}"
        )
        print(
            f"Baseline best val:    "
            f"{warm_start_metadata.get('source_validation_loss')}"
        )
        print(
            f"Checkpoint missing:   "
            f"{len(warm_start_metadata.get('missing_keys', []))}"
        )
        print(
            f"Checkpoint unexpected:"
            f" {len(warm_start_metadata.get('unexpected_keys', []))}"
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 72)
    print("DeepDCT-VO training")
    print("=" * 72)
    print(f"Device:               {device}")
    print(f"Data root:            {args.data_root.resolve()}")
    print(f"Training sequences:   {args.train_sequences}")
    print(
        f"Validation sequences: "
        f"{args.validation_sequences}"
    )
    print(f"Training samples:     {len(training_dataset)}")
    print(
        f"Validation samples:   {len(validation_dataset)}"
    )
    print(
        f"Input size:           "
        f"{args.height} x {args.width}"
    )
    print(f"Batch size:           {args.batch_size}")
    print(
    f"Sampling strategy:    "
    f"{args.sampling_strategy}"
    )
    print(f"Epochs:               {args.epochs}")
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(f"Total parameters:     {total_parameters:,}")
    print(
        f"Semantic pretrained:  "
        f"{args.pretrained_semantic}"
    )
    print(
        f"Semantic frozen:      "
        f"{args.freeze_semantic}"
    )
    print("=" * 72)
    print(
        f"Experiment name:      "
        f"{args.experiment_name}"
    )

    print(
        f"Source experiment:    "
        f"{args.source_experiment}"
    )

    print(
        f"Warm-start enabled:   "
        f"{args.init_checkpoint is not None}"
    )

    print(
        f"Translation head only:"
        f" {args.translation_head_only}"
    )

    print(
        f"Rotation readout only: "
        f"{args.rotation_readout_only}"
    )

    print(
        f"Rotation geometry wt:  "
        f"{args.rotation_geometry_weight}"
    )

    print(
        f"Pose loss type:        "
        f"{args.pose_loss_type.upper()}"
    )

    print(
        f"Rotation norm scale:   "
        f"{args.rotation_normalization_scale:.6f}"
    )

    if args.rotation_geometry_weight > 0.0:
        print(
            f"Rotation geom bank:    "
            f"{args.rotation_geometry_bank_size}"
        )

        print(
            f"Rotation geom temp:    "
            f"{args.rotation_geometry_temperature}"
        )

        print(
            "Rotation supervision:  "
            "continuous SO(3)"
        )

        print(
            f"Translation decoder:  "
            f"{args.translation_decoder}"
        )

    print(
        f"Translation loss:     "
        f"{args.translation_loss}"
    )

    if args.translation_loss == "regime_balanced_mse":
        print(
            "Translation quantiles:"
            f" {tuple(args.translation_regime_quantiles)}"
        )

    if args.translation_decoder == "mlp":
        print(
            "Translation MLP dims:"
            f" {tuple(args.translation_mlp_hidden_dims)}"
        )

    if args.translation_decoder in {
        "pooled_mlp",
        "pooled_linear",
    }:
        compact_dimension = (
            args.translation_projection_channels
            * args.translation_pool_size[0]
            * args.translation_pool_size[1]
        )

        print(
            "Translation proj ch:  "
            f"{args.translation_projection_channels}"
        )

        print(
            "Translation pool:     "
            f"{tuple(args.translation_pool_size)}"
        )

        print(
            "Translation compact:  "
            f"{compact_dimension}"
        )

        if args.translation_decoder == "pooled_mlp":
            print(
                "Aggregation hidden:   "
                f"{args.translation_aggregation_hidden_dim}"
            )

        if args.translation_decoder == "pooled_linear":
            print(
                "Translation readout:  "
                f"Linear({compact_dimension}, 3)"
            )

    if args.translation_head_only:
        print(
            "Optimization target:  "
            "translation decoder only"
        )
        print(
            "Rotation loss weight: "
            f"{args.rotation_loss_weight}"
        )
        print(
            "Backbone mode:        "
            "eval (BN/dropout frozen)"
        )

    if args.rotation_readout_only:
        print(
            "Optimization target:  "
            "rotation readout only"
        )

        print(
            "Rotation representation:"
            " frozen"
        )

        print(
            "Rotation readout:     "
            f"Linear("
            f"{model.rotation_head.dense.in_features}, "
            f"{model.rotation_head.dense.out_features})"
        )

        print(
            "Rotation loss weight: "
            f"{args.rotation_loss_weight}"
        )

        print(
            "Translation loss wt:  "
            f"{args.translation_loss_weight}"
        )

        print(
            "Backbone mode:        "
            "eval (BN/dropout frozen)"
        )

    if args.init_checkpoint is not None:
        print(
            f"Warm-start checkpoint:"
            f" {args.init_checkpoint.expanduser().resolve()}"
        )

    print(
        f"Semantic cues:        "
        f"{args.use_semantic_cues}"
    )

    print(
        f"Depth cues:           "
        f"{args.use_depth_cues}"
    )

    print(
        f"Semantic frozen:      "
        f"{args.freeze_semantic}"
    )

    print(
        f"Depth frozen:         "
        f"{args.freeze_depth}"
    )

    if args.use_depth_cues:
        print(
            "Depth source:         "
            "internal Lite-Mono"
        )
        print(
            f"Depth checkpoint:     "
            f"{args.depth_checkpoint_dir.expanduser().resolve()}"
        )
        print(
            f"Depth output mode:    "
            f"{args.depth_output_mode}"
        )
    else:
        print(
            "Depth source:         "
            "dataset placeholder"
        )
    print("=" * 72)


def print_epoch_summary(
    epoch: int,
    training_metrics: EpochMetrics,
    validation_metrics: ValidationMetrics,
    learning_rate: float,
    is_best: bool,
) -> None:
    """Print end-of-epoch training and validation results."""

    best_marker = " [best]" if is_best else ""

    print()
    print("-" * 72)
    print(f"Epoch {epoch} complete{best_marker}")
    print("-" * 72)
    print(
        f"Train total loss:       "
        f"{training_metrics.total_loss:.6f}"
    )
    print(
        f"Train rotation loss:    "
        f"{training_metrics.rotation_loss:.6f}"
    )
    print(
        f"Train rotation geometry:"
        f" {training_metrics.rotation_geometry_loss:.6f}"
    )
    print(
        f"Train translation loss: "
        f"{training_metrics.translation_loss:.6f}"
    )
    print(
        f"Validation total loss:  "
        f"{validation_metrics.total_loss:.6f}"
    )
    print(
        f"Validation rotation:    "
        f"{validation_metrics.rotation_loss:.6f}"
    )
    print(
        f"Validation rot geometry:"
        f" {validation_metrics.rotation_geometry_loss:.6f}"
    )
    print(
        f"Validation translation: "
        f"{validation_metrics.translation_loss:.6f}"
    )
    print(
        f"Train time:              "
        f"{training_metrics.elapsed_seconds:.2f} s"
    )
    print(
        f"Validation time:         "
        f"{validation_metrics.elapsed_seconds:.2f} s"
    )
    print(f"Learning rate:           {learning_rate:.8f}")
    print("-" * 72)
    print()

def extract_model_state_dict(
    checkpoint: Any,
) -> Mapping[str, torch.Tensor]:
    """Extract a model state dictionary from common checkpoint formats."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a mapping, but received "
            f"{type(checkpoint).__name__}."
        )

    candidate_keys = (
        "model_state_dict",
        "state_dict",
        "model",
        "network",
    )

    for key in candidate_keys:
        candidate = checkpoint.get(key)

        if isinstance(candidate, Mapping):
            return candidate

    # A raw state_dict is itself a mapping from parameter names to tensors.
    if checkpoint and all(
        isinstance(key, str)
        and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        return checkpoint

    raise KeyError(
        "Could not locate model weights in the checkpoint. "
        f"Tried keys: {candidate_keys}."
    )


def remove_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove a DataParallel/DistributedDataParallel 'module.' prefix."""

    if not state_dict:
        return dict(state_dict)

    if all(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return dict(state_dict)


def load_warm_start_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict: bool = True,
    reset_translation_head: bool = False,
    reset_rotation_readout: bool = False
) -> Dict[str, Any]:
    """Load model weights without restoring optimizer or epoch state."""

    checkpoint_path = checkpoint_path.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Warm-start checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = remove_module_prefix(
        extract_model_state_dict(checkpoint)
    )

    excluded_keys = []

    if reset_translation_head:
        excluded_keys = sorted(
            key
            for key in state_dict
            if key.startswith(
                "translation_head."
            )
        )

        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(
                "translation_head."
            )
        }

        print(
            "Warm-start policy:     "
            "reset translation head"
        )

        print(
            "Excluded parameters:   "
            f"{len(excluded_keys)}"
        )

        for key in excluded_keys[:20]:
            print(
                f"  excluded:   {key}"
            )

    excluded_rotation_readout_keys = []

    if reset_rotation_readout:
        rotation_readout_prefixes = (
            "rotation_head.dense.weight",
            "rotation_head.dense.bias",
        )

        filtered_state_dict = {}

        for key, value in state_dict.items():
            if key in rotation_readout_prefixes:
                excluded_rotation_readout_keys.append(
                    key
                )
                continue

            filtered_state_dict[
                key
            ] = value

        state_dict = filtered_state_dict

        print(
            "Warm-start policy:     "
            "reset rotation readout"
        )

        print(
            "Excluded parameters:   "
            f"{len(excluded_rotation_readout_keys)}"
        )

        for key in sorted(
            excluded_rotation_readout_keys
        ):
            print(
                f"  excluded:   {key}"
            )

    # --------------------------------------------------------------
    # Load once using strict=False so controlled warm-start
    # exclusions can appear as missing keys.
    #
    # We then perform our own strict compatibility audit:
    #
    #   expected missing:
    #       parameters deliberately excluded by the reset policy
    #
    #   unexpected missing:
    #       anything else
    #
    # This avoids loading the checkpoint twice and allows controlled
    # rotation/translation reset experiments to remain auditable.
    # --------------------------------------------------------------
    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    expected_missing_keys = set(
        excluded_keys
        + excluded_rotation_readout_keys
    )

    actual_missing_keys = set(
        incompatible.missing_keys
    )

    unexpected_missing_keys = sorted(
        actual_missing_keys
        - expected_missing_keys
    )

    missing_expected_but_not_reported = sorted(
        expected_missing_keys
        - actual_missing_keys
    )

    unexpected_keys = sorted(
        incompatible.unexpected_keys
    )

    if missing_expected_but_not_reported:
        raise RuntimeError(
            "Warm-start reset audit failed. Parameters were "
            "explicitly excluded from the checkpoint but were not "
            "reported as missing by load_state_dict():\n"
            + "\n".join(
                f"  {key}"
                for key in missing_expected_but_not_reported
            )
        )

    if strict and (
        unexpected_missing_keys
        or unexpected_keys
    ):
        message = [
            "",
            "Warm-start checkpoint is incompatible with",
            "the current DeepDCTVO architecture.",
            "",
        ]

        if unexpected_missing_keys:
            message.append(
                "Unexpected missing parameters:"
            )
            message.extend(
                f"  {key}"
                for key in unexpected_missing_keys
            )
            message.append("")

        if unexpected_keys:
            message.append(
                "Unexpected checkpoint parameters:"
            )
            message.extend(
                f"  {key}"
                for key in unexpected_keys
            )

        raise RuntimeError(
            "\n".join(message)
        )

    metadata: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source_epoch": None,
        "source_validation_loss": None,

        "missing_keys": sorted(
            incompatible.missing_keys
        ),
        "expected_missing_keys": sorted(
            expected_missing_keys
        ),
        "unexpected_missing_keys": (
            unexpected_missing_keys
        ),
        "unexpected_keys": (
            unexpected_keys
        ),

        "reset_translation_head": (
            reset_translation_head
        ),
        "reset_rotation_readout": (
            reset_rotation_readout
        ),

        "excluded_keys": sorted(
            excluded_keys
            + excluded_rotation_readout_keys
        ),
    }

    if isinstance(checkpoint, Mapping):
        metadata["source_epoch"] = checkpoint.get(
            "epoch",
            checkpoint.get("checkpoint_epoch"),
        )

        metadata["source_validation_loss"] = checkpoint.get(
            "best_validation_loss",
            checkpoint.get(
                "validation_loss",
                checkpoint.get("val_loss"),
            ),
        )

    print("=" * 88)
    print("Warm-start checkpoint loaded")
    print("=" * 88)
    print(
        f"Checkpoint:              {metadata['checkpoint_path']}"
    )
    print(
        f"Source epoch:            {metadata['source_epoch']}"
    )
    print(
        "Source validation loss: "
        f"{metadata['source_validation_loss']}"
    )
    print(
        f"Strict loading:          {strict}"
    )
    print(
        f"Missing keys:            "
        f"{len(metadata['missing_keys'])}"
    )
    print(
        f"Unexpected keys:         "
        f"{len(metadata['unexpected_keys'])}"
    )

    if metadata["missing_keys"]:
        for key in metadata["missing_keys"][:20]:
            print(f"  missing:    {key}")

    if metadata["unexpected_keys"]:
        for key in metadata["unexpected_keys"][:20]:
            print(f"  unexpected: {key}")

    print("=" * 88)

    return metadata

def sanity_check_gated_expert(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Verify gated-expert shapes, mixture identity, and gradients."""

    if args.translation_decoder != "gated_expert":
        return

    print()
    print("=" * 72)
    print("Gated-expert translation sanity check")
    print("=" * 72)

    model.train()

    batch_size = 1

    image_prev = torch.rand(
        batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )

    image_curr = torch.rand(
        batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )

    # Use explicit zero depth to avoid invoking Lite-Mono during
    # this architecture/gradient sanity test.
    depth_curr = torch.zeros(
        batch_size,
        1,
        args.height,
        args.width,
        device=device,
    )

    outputs = model(
        image_prev=image_prev,
        image_curr=image_curr,
        depth_curr=depth_curr,
        use_ground_truth_rotation=False,
        return_intermediates=True,
    )

    translation = outputs[
        "directional_translation"
    ]

    representation = outputs[
        "translation_representation"
    ]

    gate_weights = outputs[
        "translation_gate_weights"
    ]

    expert_predictions = outputs[
        "translation_expert_predictions"
    ]

    expected_representation_dim = (
        args.translation_projection_channels
        * args.translation_pool_size[0]
        * args.translation_pool_size[1]
    )

    expected_translation_shape = (
        batch_size,
        3,
    )

    expected_gate_shape = (
        batch_size,
        args.translation_num_experts,
    )

    expected_expert_shape = (
        batch_size,
        args.translation_num_experts,
        3,
    )

    expected_representation_shape = (
        batch_size,
        expected_representation_dim,
    )

    if translation.shape != expected_translation_shape:
        raise RuntimeError(
            "Gated-expert translation shape mismatch: "
            f"expected {expected_translation_shape}, "
            f"received {tuple(translation.shape)}."
        )

    if representation.shape != expected_representation_shape:
        raise RuntimeError(
            "Gated-expert representation shape mismatch: "
            f"expected {expected_representation_shape}, "
            f"received {tuple(representation.shape)}."
        )

    if gate_weights.shape != expected_gate_shape:
        raise RuntimeError(
            "Gated-expert gate shape mismatch: "
            f"expected {expected_gate_shape}, "
            f"received {tuple(gate_weights.shape)}."
        )

    if expert_predictions.shape != expected_expert_shape:
        raise RuntimeError(
            "Gated-expert expert-output shape mismatch: "
            f"expected {expected_expert_shape}, "
            f"received {tuple(expert_predictions.shape)}."
        )

    gate_sums = gate_weights.sum(
        dim=1
    )

    if not torch.allclose(
        gate_sums,
        torch.ones_like(gate_sums),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError(
            "Gated-expert weights do not sum to 1."
        )

    # NOTE:
    # DeepDCTVO exposes detached diagnostics, so reconstruct the
    # mixture using the live tensors cached by the head only for
    # numerical verification.
    live_gate_weights = (
        model.translation_head
        .last_gate_weights
    )

    live_expert_predictions = (
        model.translation_head
        .last_expert_predictions
    )

    manual_translation = torch.sum(
        live_gate_weights.unsqueeze(-1)
        * live_expert_predictions,
        dim=1,
    )

    if not torch.allclose(
        translation.detach(),
        manual_translation,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError(
            "Gated-expert weighted-sum reconstruction failed."
        )

    # ----------------------------------------------------------
    # Gradient connectivity
    # ----------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    target = torch.zeros_like(
        translation
    )

    loss = torch.mean(
        (translation - target) ** 2
    )

    loss.backward()

    gate_gradient = (
        model.translation_head
        .gate.weight.grad
    )

    if gate_gradient is None:
        raise RuntimeError(
            "No gradient reached translation gate."
        )

    if not torch.isfinite(
        gate_gradient
    ).all():
        raise RuntimeError(
            "Translation gate gradient is non-finite."
        )

    for expert_index, expert in enumerate(
        model.translation_head.experts
    ):
        if expert.weight.grad is None:
            raise RuntimeError(
                "No gradient reached translation expert "
                f"{expert_index}."
            )

        if not torch.isfinite(
            expert.weight.grad
        ).all():
            raise RuntimeError(
                "Non-finite gradient in translation expert "
                f"{expert_index}."
            )

    print(
        "Representation shape:     "
        f"{tuple(representation.shape)}"
    )
    print(
        "Gate shape:               "
        f"{tuple(gate_weights.shape)}"
    )
    print(
        "Expert prediction shape:  "
        f"{tuple(expert_predictions.shape)}"
    )
    print(
        "Translation shape:        "
        f"{tuple(translation.shape)}"
    )
    print(
        "Gate row sums:            "
        f"{gate_sums.detach().cpu().numpy()}"
    )
    print(
        "Gate gradient norm:       "
        f"{gate_gradient.norm().item():.6e}"
    )

    for expert_index, expert in enumerate(
        model.translation_head.experts
    ):
        print(
            f"Expert {expert_index} gradient norm: "
            f"{expert.weight.grad.norm().item():.6e}"
        )

    print("Status:                    PASS")
    print("=" * 72)
    print()

    # Clear artificial sanity-check gradients before real training.
    model.zero_grad(
        set_to_none=True
    )


def main() -> None:
    """Run multi-epoch training and validation."""

    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    training_dataset = build_dataset(
        args=args,
        sequences=args.train_sequences,
    )

    validation_dataset = build_dataset(
        args=args,
        sequences=args.validation_sequences,
    )

    translation_regime_configuration = None

    if (
        args.translation_loss == "regime_balanced_mse"
    ):
        regime_quantiles = tuple(
            args.translation_regime_quantiles
        )

        translation_regime_configuration = (
            build_translation_regime_configuration(
                dataset=training_dataset,
                quantiles=regime_quantiles,
            )
        )

        print("=" * 72)
        print("Translation regime balancing")
        print("=" * 72)
        print(
            "Threshold source:      training sequences only"
        )
        print(
            "Quantiles:             "
            f"{translation_regime_configuration.low_quantile:.3f}, "
            f"{translation_regime_configuration.high_quantile:.3f}"
        )
        print(
            "Low threshold t_z:     "
            f"{translation_regime_configuration.low_threshold:.9f}"
        )
        print(
            "High threshold t_z:    "
            f"{translation_regime_configuration.high_threshold:.9f}"
        )
        print(
            "Regime counts:         "
            f"low={translation_regime_configuration.low_count}, "
            f"medium={translation_regime_configuration.medium_count}, "
            f"high={translation_regime_configuration.high_count}"
        )
        print(
            "Regime weights:        "
            f"low={translation_regime_configuration.low_weight:.6f}, "
            f"medium={translation_regime_configuration.medium_weight:.6f}, "
            f"high={translation_regime_configuration.high_weight:.6f}"
        )
        print("=" * 72)

    training_sampler: Optional[Sampler[int]] = None

    sequence_counts: Optional[Dict[str, int]] = None
    sequence_probabilities: Optional[Dict[str, float]] = None

    print(f"Batch size:           {args.batch_size}")
    print(
        f"Sampling strategy:    "
        f"{args.sampling_strategy}"
    )

    if args.sampling_strategy == "sequence_balanced":
        print(
            f"Sampling alpha:       "
            f"{args.sampling_alpha:.3f}"
        )
        (
            training_sampler,
            sequence_counts,
            sequence_probabilities,
        ) = build_sequence_balanced_sampler(
            training_dataset,
            alpha=args.sampling_alpha,
            seed=args.seed,
            num_samples=len(training_dataset),
            replacement=True,
        )

    print(f"Epochs:               {args.epochs}")

    training_loader = build_dataloader(
        dataset=training_dataset,
        args=args,
        device=device,
        shuffle=training_sampler is None,
        sampler=training_sampler,
    )

    validation_loader = build_dataloader(
        dataset=validation_dataset,
        args=args,
        device=device,
        shuffle=False,
        sampler=None,
    )

    model = build_model(
        args=args,
        device=device,
    )

    warm_start_metadata = None

    if args.init_checkpoint is not None:
        warm_start_metadata = load_warm_start_weights(
            model=model,
            checkpoint_path=args.init_checkpoint,
            device=device,
            strict=args.strict_init_checkpoint,
            reset_translation_head=(
                args.reset_translation_head_on_init
            ),
            reset_rotation_readout=(
                args.reset_rotation_readout_on_init
            ),
        )

    # --------------------------------------------------------------
    # Fresh rotation readout initialization.
    #
    # The source checkpoint intentionally does not restore:
    #
    #   rotation_head.dense.weight
    #   rotation_head.dense.bias
    #
    # Reset the final Linear(D, 3) explicitly while preserving the
    # complete upstream rotation representation extractor.
    # --------------------------------------------------------------
    if args.reset_rotation_readout_on_init:
        if args.init_checkpoint is None:
            raise ValueError(
                "--reset-rotation-readout-on-init requires "
                "--init-checkpoint."
            )

        if not isinstance(
            model.rotation_head.dense,
            nn.Linear,
        ):
            raise TypeError(
                "--reset-rotation-readout-on-init requires "
                "rotation_head.dense to be nn.Linear."
            )

        model.rotation_head.dense.reset_parameters()

        print(
            "Fresh rotation readout: "
            f"Linear("
            f"{model.rotation_head.dense.in_features}, "
            f"{model.rotation_head.dense.out_features})"
        )

    # --------------------------------------------------------------
    # Verify gated-expert architecture before optimizer creation.
    # --------------------------------------------------------------
    sanity_check_gated_expert(
        model=model,
        args=args,
        device=device,
    )

    # --------------------------------------------------------------
    # Configure the controlled trainable parameter set.
    # --------------------------------------------------------------
    if args.translation_head_only:
        configure_translation_head_only(
            model
        )

    if args.rotation_readout_only:
        configure_rotation_readout_only(
            model
        )

    if args.rotation_geometry_weight > 0.0:
        rotation_representation_trainable = any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith("rotation_head.conv.")
        )

        if not rotation_representation_trainable:
            raise RuntimeError(
                "Continuous rotation geometry is enabled, but "
                "rotation_head.conv has no trainable parameters. "
                "The geometry objective cannot shape the rotation "
                "representation."
            )
    # --------------------------------------------------------------
    # Translation-head-only fine-tuning.
    #
    # Rotation is frozen and must not contribute to optimization
    # or validation-based model selection.
    # --------------------------------------------------------------
    if args.translation_head_only:
        if args.rotation_loss_weight != 0.0:
            print(
                "Translation-head-only mode: overriding "
                f"rotation_loss_weight={args.rotation_loss_weight} "
                "with 0.0."
            )

        args.rotation_loss_weight = 0.0

    # --------------------------------------------------------------
    # Frozen rotation representation + fresh linear readout.
    #
    # Only rotation contributes to optimization/model selection.
    # Translation remains part of the forward pass but contributes
    # zero loss.
    # --------------------------------------------------------------
    if args.rotation_readout_only:
        if args.rotation_loss_weight != 1.0:
            print(
                "Rotation-readout-only mode: overriding "
                f"rotation_loss_weight={args.rotation_loss_weight} "
                "with 1.0."
            )

        if args.translation_loss_weight != 0.0:
            print(
                "Rotation-readout-only mode: overriding "
                f"translation_loss_weight={args.translation_loss_weight} "
                "with 0.0."
            )

        args.rotation_loss_weight = 1.0
        args.translation_loss_weight = 0.0

    # --------------------------------------------------------------
    # Create optimizer only after:
    #
    #   1. warm-start loading;
    #   2. fresh-readout initialization;
    #   3. trainable-parameter selection;
    #   4. loss-weight resolution.
    # --------------------------------------------------------------

    # Create the optimizer only after:
    #   1. warm-start weights have been loaded; and
    #   2. the trainable parameter set has been configured.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    # --------------------------------------------------------------
    # Pose regression objective
    #
    # A1:
    #     pose_loss_type = "mse"
    #
    # A2:
    #     pose_loss_type = "mae"
    #
    # A2 uses the same MAE/L1 objective for:
    #     - normalized rotation
    #     - directional translation
    # --------------------------------------------------------------
    if args.pose_loss_type == "mae":
        if args.translation_loss != "mse":
            raise RuntimeError(
                "Track-A MAE pose loss requires ordinary translation "
                "loss configuration; regime-balanced MSE cannot be "
                "combined with --pose-loss-type mae."
            )

        rotation_criterion = nn.L1Loss()
        translation_criterion = nn.L1Loss()

    elif args.pose_loss_type == "mse":
        rotation_criterion = nn.MSELoss()

        if args.translation_loss == "mse":
            translation_criterion = nn.MSELoss()

        elif args.translation_loss == "regime_balanced_mse":
            if translation_regime_configuration is None:
                raise RuntimeError(
                    "Regime-balanced translation loss was selected, "
                    "but no regime configuration was constructed."
                )

            translation_criterion = (
                RegimeBalancedTranslationMSE(
                    translation_regime_configuration
                )
            )

        else:
            raise RuntimeError(
                "Unsupported translation loss: "
                f"{args.translation_loss!r}."
            )

    else:
        raise RuntimeError(
            "Unsupported pose loss type: "
            f"{args.pose_loss_type!r}."
        )

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_epoch = 1
    best_validation_loss = float("inf")

    # --------------------------------------------------------------
    # Continuous SO(3)-supervised rotation-geometry loss.
    #
    # Construct before resume handling so the FIFO memory-bank state
    # can be restored from a resumable checkpoint.
    # --------------------------------------------------------------
    rotation_geometry_loss_fn = None

    if args.rotation_geometry_weight > 0.0:
        rotation_geometry_loss_fn = RotationGeometryBank(
            bank_size=args.rotation_geometry_bank_size,
            temperature=args.rotation_geometry_temperature,
            supervision="so3",
        )

    checkpoint = None

    if args.resume is not None:
        checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        resumed_epoch = int(
            checkpoint["epoch"]
        )

        start_epoch = resumed_epoch + 1

        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        print(
            f"Resumed from {args.resume} at epoch "
            f"{resumed_epoch}."
        )

        if (
            args.rotation_geometry_weight > 0.0
            and checkpoint.get(
                "rotation_geometry_state"
            ) is None
        ):
            raise RuntimeError(
                "This run enables continuous rotation geometry, "
                "but the resume checkpoint contains no "
                "'rotation_geometry_state'. Use --init-checkpoint "
                "for a fresh geometry-bank experiment instead of "
                "--resume."
            )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint resumes at epoch {start_epoch - 1}, "
            f"but --epochs is {args.epochs}. Increase --epochs."
        )

    print_run_summary(
        args=args,
        device=device,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        warm_start_metadata=warm_start_metadata,
    )

    if (
        sequence_counts is not None
        and sequence_probabilities is not None
    ):
        print(
            summarize_sequence_distribution(
                sequence_counts,
                sequence_probabilities,
            )
        )

    maximum_gradient_norm: Optional[float]

    if args.max_grad_norm > 0:
        maximum_gradient_norm = args.max_grad_norm
    else:
        maximum_gradient_norm = None

    epochs_without_improvement = 0

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print(f"\nStarting epoch {epoch}/{args.epochs}")

        training_metrics = train_one_epoch(
            model=model,
            dataloader=training_loader,
            optimizer=optimizer,
            device=device,
            rotation_criterion=rotation_criterion,
            translation_criterion=translation_criterion,
            rotation_loss_weight=(
                args.rotation_loss_weight
            ),
            translation_loss_weight=(
                args.translation_loss_weight
            ),
            rotation_geometry_weight=(
                args.rotation_geometry_weight
            ),
            rotation_geometry_loss_fn=(
                rotation_geometry_loss_fn
            ),
            use_ground_truth_rotation=(
                args.use_ground_truth_rotation
            ),
            max_grad_norm=maximum_gradient_norm,
            log_interval=args.log_interval,
            epoch_index=epoch,
            skip_nonfinite_batches=(
                args.skip_nonfinite_batches
            ),
            use_internal_depth=(
                args.use_depth_cues
            ),
            keep_model_in_eval_mode=(
                args.translation_head_only
                or args.rotation_readout_only
            ),

        )

        validation_metrics = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            rotation_criterion=rotation_criterion,
            translation_criterion=translation_criterion,
            rotation_loss_weight=(
                args.rotation_loss_weight
            ),
            translation_loss_weight=(
                args.translation_loss_weight
            ),
            use_ground_truth_rotation=(
                args.use_ground_truth_rotation
            ),
            log_interval=args.log_interval,
            epoch_index=epoch,
            skip_nonfinite_batches=(
                args.skip_nonfinite_batches
            ),
            use_internal_depth=(
                args.use_depth_cues
            ),
            rotation_geometry_weight=(
                args.rotation_geometry_weight
            ),
            rotation_geometry_loss_fn=(
                rotation_geometry_loss_fn
            ),
        )

        scheduler.step(
            validation_metrics.total_loss
        )

        validation_improvement = (
            best_validation_loss
            - validation_metrics.total_loss
        )

        is_best = (
            validation_improvement
            > args.early_stopping_min_delta
        )

        if is_best:
            best_validation_loss = (
                validation_metrics.total_loss
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if is_best:
            best_validation_loss = (
                validation_metrics.total_loss
            )

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        print_epoch_summary(
            epoch=epoch,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            learning_rate=current_learning_rate,
            is_best=is_best,
        )

        latest_path = (
            args.checkpoint_dir
            / "latest.pt"
        )

        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            best_validation_loss=best_validation_loss,
            args=args,
            warm_start_metadata=warm_start_metadata,
            translation_regime_configuration=(
                translation_regime_configuration
            ),
            rotation_geometry_loss_fn=(
                rotation_geometry_loss_fn
            ),
        )

        if args.save_every_epoch:
            epoch_path = (
                args.checkpoint_dir
                / f"deepdct_vo_epoch_{epoch:03d}.pt"
            )

            shutil.copy2(
                latest_path,
                epoch_path,
            )

        if is_best:
            best_path = (
                args.checkpoint_dir
                / "best_validation.pt"
            )

            shutil.copy2(
                latest_path,
                best_path,
            )

            print(
                "Saved new best-validation checkpoint: "
                f"{best_path}"
            )

        print(
            f"Saved latest checkpoint: {latest_path}"
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"no validation improvement greater than "
                f"{args.early_stopping_min_delta:.6g} for "
                f"{epochs_without_improvement} consecutive epochs."
            )

            print(
                f"Best validation loss: "
                f"{best_validation_loss:.6f}"
            )

            break


if __name__ == "__main__":
    main()