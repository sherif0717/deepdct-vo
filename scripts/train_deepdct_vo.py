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


CheckpointValue = Union[
    int,
    float,
    str,
    Dict[str, object],
]


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
        "--rotation-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to rotation loss.",
    )

    parser.add_argument(
        "--translation-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to directional-translation loss.",
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

    if args.translation_loss_weight < 0:
        raise ValueError(
            "--translation-loss-weight cannot be negative."
        )

    if (
        args.rotation_loss_weight == 0
        and args.translation_loss_weight == 0
    ):
        raise ValueError(
            "At least one loss weight must be positive."
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
        depth_checkpoint_dir=depth_checkpoint_dir,
        depth_model_name=args.depth_model_name,
        depth_output_mode=args.depth_output_mode,
        freeze_depth=args.freeze_depth,
        use_semantic_cues=args.use_semantic_cues,
        use_depth_cues=args.use_depth_cues,
    )

    return model.to(device)


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
) -> None:
    """Save complete resumable training state."""

    warm_start_provenance: Dict[str, object] = {
        "enabled": warm_start_metadata is not None,
        "checkpoint_path": None,
        "source_epoch": None,
        "source_validation_loss": None,
        "missing_keys": [],
        "unexpected_keys": [],
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

        # ------------------------------------------------------------
        # Experiment identity and provenance
        # ------------------------------------------------------------
        "experiment": {
            "name": args.experiment_name,
            "source_experiment": args.source_experiment,
            "experiment_type": (
                "warm_start_cue_adaptation"
                if warm_start_metadata is not None
                else "standard_training"
            ),
        },

        "warm_start": warm_start_provenance,

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
            "translation_loss_weight": (
                args.translation_loss_weight
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

    # Inspect compatibility before committing the load.
    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if strict:
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            message = [
                "",
                "Warm-start checkpoint is incompatible with",
                "the current DeepDCTVO architecture.",
                "",
            ]

            if incompatible.missing_keys:
                message.append("Missing parameters:")
                message.extend(
                    f"  {key}"
                    for key in incompatible.missing_keys
                )
                message.append("")

            if incompatible.unexpected_keys:
                message.append("Unexpected parameters:")
                message.extend(
                    f"  {key}"
                    for key in incompatible.unexpected_keys
                )

            raise RuntimeError(
                "\n".join(message)
            )
        
    # Perform the actual strict load once compatibility has been verified.
    model.load_state_dict(
        state_dict,
        strict=strict,
    )

    metadata: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source_epoch": None,
        "source_validation_loss": None,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
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
        )

    # Create the optimizer only after warm-start weights have been loaded.
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
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    rotation_criterion = nn.MSELoss()
    translation_criterion = nn.MSELoss()

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_epoch = 1
    best_validation_loss = float("inf")

    if args.resume is not None:
        checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        resumed_epoch = int(checkpoint["epoch"])
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
        )

        scheduler.step(
            validation_metrics.total_loss
        )

        is_best = (
            validation_metrics.total_loss
            < best_validation_loss
        )

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


if __name__ == "__main__":
    main()