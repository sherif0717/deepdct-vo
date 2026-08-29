#!/usr/bin/env python3
"""
DeepDCT-VO translation-representation decodability audit.

Purpose
-------

Previous A6 diagnostics established:

1. DCT geometry is correct.
2. GT-R -> Model-T conditioning is correct.
3. Forward translation is strongly range-compressed.
4. That compression already exists on the 00-08 training sequences.

The next discriminator is:

    Does the saved translation representation itself contain enough
    information to recover metric translation?

This script fits a fresh linear probe

    z_rep -> [t_x, t_y, t_z]

using ONLY saved representations from training sequences 00-08.

It then compares:

    actual Model-T output
vs
    fresh linear probe output

on:

    A) a held-out subset of 00-08 frames;
    B) optionally sequence 09;
    C) optionally sequence 10.

Interpretation
--------------

CASE A
    Linear probe recovers t_z well on held-out 00-08
    (slope near 1, high correlation, much lower RMSE)
    while Model-T head remains compressed.

    => Representation contains metric motion.
       Problem is downstream output-head optimization/mapping.

CASE B
    Linear probe is also compressed / weak on held-out 00-08.

    => Representation itself does not encode metric translation well.
       Problem is upstream fusion/representation learning.

CASE C
    Probe works on 00-08 holdout but collapses on 09/10.

    => Training representation is decodable but does not generalize
       sequence-independently.

Design notes
------------

The probe is intentionally simple:
    nn.Linear(representation_dim, 3)

No hidden layers are used.

Features and targets are standardized from the PROBE-TRAIN subset only.
The probe is optimized using AdamW with an L2 weight penalty.

A deterministic within-sequence holdout split is used:
    global/local sample index modulo holdout_modulus == holdout_remainder

By default this gives approximately 20% probe-holdout samples.

The original DeepDCT-VO checkpoint is NOT modified or retrained.

Expected per-sequence files
---------------------------

train_predictions/sequence_00/
    translation_representations.npz
    frame_predictions.csv

...
train_predictions/sequence_08/

Optional unseen sequence files:

evaluation_sequence_09/
    translation_representations.npz
    frame_predictions.csv

evaluation_sequence_10/
    translation_representations.npz
    frame_predictions.csv

Outputs
-------

<output-dir>/
    summary.json
    overall_metrics.csv
    per_sequence_metrics.csv
    per_axis_metrics.csv
    probe_training_history.csv

    probe_holdout_predictions.csv

    calibration_train_holdout.png
    calibration_seq09.png
    calibration_seq10.png

    residual_train_holdout.png
    residual_seq09.png
    residual_seq10.png

Example
-------

python scripts/audit_translation_representation_decodability.py \
    --train-root \
      experiments/track_a/paper_reproduction/\
a6_translation_train_test_calibration/train_predictions \
    --predictions-09 \
      experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_09/frame_predictions.csv \
    --representations-09 \
      experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_09/translation_representations.npz \
    --predictions-10 \
      experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_10/frame_predictions.csv \
    --representations-10 \
      experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_10/translation_representations.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


DEFAULT_TRAIN_SEQUENCES = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
]

AXES = ("x", "y", "z")


# ============================================================================
# CLI
# ============================================================================


def normalize_sequence(
    value: str,
) -> str:
    value = str(value).strip()

    if value.isdigit():
        return f"{int(value):02d}"

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether saved DeepDCT-VO Model-T representations "
            "linearly encode metric translation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help=(
            "Directory containing sequence_00 ... sequence_08, "
            "each with translation_representations.npz and "
            "frame_predictions.csv."
        ),
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=DEFAULT_TRAIN_SEQUENCES,
        help="Sequences used for probe training/holdout.",
    )

    parser.add_argument(
        "--predictions-09",
        type=Path,
        default=None,
        help="Optional seq09 frame_predictions.csv.",
    )

    parser.add_argument(
        "--representations-09",
        type=Path,
        default=None,
        help="Optional seq09 translation_representations.npz.",
    )

    parser.add_argument(
        "--predictions-10",
        type=Path,
        default=None,
        help="Optional seq10 frame_predictions.csv.",
    )

    parser.add_argument(
        "--representations-10",
        type=Path,
        default=None,
        help="Optional seq10 translation_representations.npz.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/track_a/paper_reproduction/"
            "a6_translation_representation_decodability"
        ),
        help="Audit output directory.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Linear-probe training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Probe optimization batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="Probe learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-4,
        help="Probe L2/AdamW weight decay.",
    )

    parser.add_argument(
        "--holdout-modulus",
        type=int,
        default=5,
        help="Modulo used for deterministic probe holdout.",
    )

    parser.add_argument(
        "--holdout-remainder",
        type=int,
        default=0,
        help="Modulo remainder assigned to probe holdout.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Probe-training device.",
    )

    parser.add_argument(
        "--feature-epsilon",
        type=float,
        default=1.0e-6,
        help="Minimum feature std during standardization.",
    )

    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=1.0e-8,
        help="Minimum target std during standardization.",
    )

    args = parser.parse_args()

    args.train_sequences = [
        normalize_sequence(sequence)
        for sequence in args.train_sequences
    ]

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.learning_rate <= 0.0:
        raise ValueError(
            "--learning-rate must be positive."
        )

    if args.weight_decay < 0.0:
        raise ValueError(
            "--weight-decay cannot be negative."
        )

    if args.holdout_modulus < 2:
        raise ValueError(
            "--holdout-modulus must be >= 2."
        )

    if not (
        0
        <= args.holdout_remainder
        < args.holdout_modulus
    ):
        raise ValueError(
            "--holdout-remainder must satisfy "
            "0 <= remainder < modulus."
        )

    return args


# ============================================================================
# General helpers
# ============================================================================


def choose_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but CUDA is unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Representation loading
# ============================================================================


def find_representation_array(
    path: Path,
) -> np.ndarray:
    """
    Robustly locate the representation matrix inside an NPZ.

    Expected shape:
        [N, D]

    If several arrays exist, choose the largest valid 2-D numeric array.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Representation file not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        candidates = []

        for key in data.files:
            value = np.asarray(
                data[key]
            )

            if (
                value.ndim == 2
                and value.shape[0] > 0
                and value.shape[1] > 0
                and np.issubdtype(
                    value.dtype,
                    np.number,
                )
            ):
                candidates.append(
                    (
                        value.size,
                        key,
                        value.astype(
                            np.float32,
                            copy=True,
                        ),
                    )
                )

        if not candidates:
            raise ValueError(
                "Could not find a numeric [N,D] representation "
                f"array in {path}. Available keys: {data.files}"
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _, key, array = candidates[0]

    if not np.isfinite(array).all():
        raise ValueError(
            f"NaN/Inf detected in representation array: {path}"
        )

    print(
        f"Loaded representation: {path}"
    )
    print(
        f"  selected NPZ key: {key}"
    )
    print(
        f"  shape:            {array.shape}"
    )

    return array


# ============================================================================
# frame_predictions.csv loading
# ============================================================================


def load_predictions(
    path: Path,
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found: {path}"
        )

    required = [
        "translation_gt_x",
        "translation_gt_y",
        "translation_gt_z",
        "translation_pred_x",
        "translation_pred_y",
        "translation_pred_z",
    ]

    values = {
        key: []
        for key in required
    }

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        missing = [
            key
            for key in required
            if key not in reader.fieldnames
        ]

        if missing:
            raise KeyError(
                f"{path} missing columns: {missing}"
            )

        for row in reader:
            for key in required:
                values[key].append(
                    float(row[key])
                )

    def stack(
        prefix: str,
    ) -> np.ndarray:
        return np.column_stack(
            [
                values[f"{prefix}_x"],
                values[f"{prefix}_y"],
                values[f"{prefix}_z"],
            ]
        ).astype(
            np.float64,
            copy=False,
        )

    gt = stack(
        "translation_gt"
    )

    pred = stack(
        "translation_pred"
    )

    if (
        not np.isfinite(gt).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError(
            f"NaN/Inf in {path}"
        )

    return {
        "gt": gt,
        "head_pred": pred,
    }


# ============================================================================
# Sequence loading
# ============================================================================


def training_sequence_paths(
    root: Path,
    sequence: str,
) -> Tuple[Path, Path]:
    directory = (
        root
        / f"sequence_{sequence}"
    )

    return (
        directory
        / "translation_representations.npz",
        directory
        / "frame_predictions.csv",
    )


def load_sequence(
    representation_path: Path,
    prediction_path: Path,
) -> Dict[str, np.ndarray]:
    representation = (
        find_representation_array(
            representation_path
        )
    )

    prediction = load_predictions(
        prediction_path
    )

    if representation.shape[0] != prediction[
        "gt"
    ].shape[0]:
        raise ValueError(
            "Representation/prediction row mismatch:\n"
            f"  {representation_path}: "
            f"{representation.shape[0]}\n"
            f"  {prediction_path}: "
            f"{prediction['gt'].shape[0]}"
        )

    return {
        "representation": representation,
        "gt": prediction["gt"],
        "head_pred": prediction[
            "head_pred"
        ],
    }


# ============================================================================
# Deterministic probe split
# ============================================================================


def holdout_mask(
    count: int,
    modulus: int,
    remainder: int,
    sequence_offset: int,
) -> np.ndarray:
    """
    Sequence offset prevents exactly the same local positions from always
    belonging to the same partition across every sequence.
    """

    indices = (
        np.arange(
            count,
            dtype=np.int64,
        )
        + sequence_offset
    )

    return (
        indices % modulus
        == remainder
    )


# ============================================================================
# Feature statistics
# ============================================================================


def compute_training_statistics(
    train_sequences: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    masks: Mapping[
        str,
        np.ndarray,
    ],
    feature_epsilon: float,
    target_epsilon: float,
) -> Dict[str, np.ndarray]:
    feature_sum = None
    feature_sq_sum = None

    target_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    target_sq_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    total_count = 0

    for sequence, data in (
        train_sequences.items()
    ):
        train_mask = ~masks[
            sequence
        ]

        x = data[
            "representation"
        ][
            train_mask
        ].astype(
            np.float64,
            copy=False,
        )

        y = data[
            "gt"
        ][
            train_mask
        ].astype(
            np.float64,
            copy=False,
        )

        if feature_sum is None:
            feature_sum = np.zeros(
                x.shape[1],
                dtype=np.float64,
            )

            feature_sq_sum = np.zeros(
                x.shape[1],
                dtype=np.float64,
            )

        feature_sum += np.sum(
            x,
            axis=0,
        )

        feature_sq_sum += np.sum(
            x * x,
            axis=0,
        )

        target_sum += np.sum(
            y,
            axis=0,
        )

        target_sq_sum += np.sum(
            y * y,
            axis=0,
        )

        total_count += x.shape[0]

    if (
        feature_sum is None
        or feature_sq_sum is None
        or total_count == 0
    ):
        raise ValueError(
            "No probe-training samples."
        )

    feature_mean = (
        feature_sum
        / total_count
    )

    feature_variance = (
        feature_sq_sum
        / total_count
        - feature_mean ** 2
    )

    feature_variance = np.maximum(
        feature_variance,
        0.0,
    )

    feature_std = np.sqrt(
        feature_variance
    )

    feature_std = np.maximum(
        feature_std,
        feature_epsilon,
    )

    target_mean = (
        target_sum
        / total_count
    )

    target_variance = (
        target_sq_sum
        / total_count
        - target_mean ** 2
    )

    target_variance = np.maximum(
        target_variance,
        0.0,
    )

    target_std = np.sqrt(
        target_variance
    )

    target_std = np.maximum(
        target_std,
        target_epsilon,
    )

    return {
        "feature_mean": (
            feature_mean.astype(
                np.float32
            )
        ),
        "feature_std": (
            feature_std.astype(
                np.float32
            )
        ),
        "target_mean": (
            target_mean.astype(
                np.float32
            )
        ),
        "target_std": (
            target_std.astype(
                np.float32
            )
        ),
        "count": np.asarray(
            total_count,
            dtype=np.int64,
        ),
    }


# ============================================================================
# Probe
# ============================================================================


class LinearTranslationProbe(
    nn.Module,
):
    def __init__(
        self,
        input_dim: int,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            3,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(x)


def iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.RandomState,
) -> Iterable[
    Tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    count = x.shape[0]

    indices = np.arange(
        count,
        dtype=np.int64,
    )

    if shuffle:
        rng.shuffle(indices)

    for start in range(
        0,
        count,
        batch_size,
    ):
        batch_indices = indices[
            start:
            start + batch_size
        ]

        yield (
            x[batch_indices],
            y[batch_indices],
        )


def train_probe(
    *,
    model: LinearTranslationProbe,
    train_sequences: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    holdout_masks: Mapping[
        str,
        np.ndarray,
    ],
    statistics: Mapping[
        str,
        np.ndarray,
    ],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    model.to(device)

    feature_mean = statistics[
        "feature_mean"
    ]

    feature_std = statistics[
        "feature_std"
    ]

    target_mean = statistics[
        "target_mean"
    ]

    target_std = statistics[
        "target_std"
    ]

    history = []

    rng = np.random.RandomState(
        seed
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        total_samples = 0

        sequence_order = list(
            train_sequences.keys()
        )

        rng.shuffle(
            sequence_order
        )

        for sequence in (
            sequence_order
        ):
            data = train_sequences[
                sequence
            ]

            mask = ~holdout_masks[
                sequence
            ]

            x = data[
                "representation"
            ][mask]

            y = data[
                "gt"
            ][mask]

            for (
                batch_x,
                batch_y,
            ) in iter_batches(
                x,
                y,
                batch_size,
                shuffle=True,
                rng=rng,
            ):
                batch_x = (
                    batch_x
                    - feature_mean
                ) / feature_std

                batch_y = (
                    batch_y
                    - target_mean
                ) / target_std

                x_tensor = torch.from_numpy(
                    batch_x.astype(
                        np.float32,
                        copy=False,
                    )
                ).to(
                    device,
                    non_blocking=True,
                )

                y_tensor = torch.from_numpy(
                    batch_y.astype(
                        np.float32,
                        copy=False,
                    )
                ).to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = model(
                    x_tensor
                )

                loss = criterion(
                    prediction,
                    y_tensor,
                )

                loss.backward()

                optimizer.step()

                samples = (
                    batch_x.shape[0]
                )

                total_loss += (
                    float(
                        loss.detach().cpu()
                    )
                    * samples
                )

                total_samples += (
                    samples
                )

        mean_loss = (
            total_loss
            / max(
                total_samples,
                1,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "standardized_mse": (
                    mean_loss
                ),
            }
        )

        print(
            f"probe epoch={epoch:02d}/{epochs:02d} "
            f"standardized_mse={mean_loss:.8f}"
        )

    return history


# ============================================================================
# Probe inference
# ============================================================================


def predict_probe(
    model: nn.Module,
    representation: np.ndarray,
    statistics: Mapping[
        str,
        np.ndarray,
    ],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    feature_mean = statistics[
        "feature_mean"
    ]

    feature_std = statistics[
        "feature_std"
    ]

    target_mean = statistics[
        "target_mean"
    ]

    target_std = statistics[
        "target_std"
    ]

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            representation.shape[0],
            batch_size,
        ):
            x = representation[
                start:
                start + batch_size
            ]

            x = (
                x
                - feature_mean
            ) / feature_std

            tensor = torch.from_numpy(
                x.astype(
                    np.float32,
                    copy=False,
                )
            ).to(
                device,
                non_blocking=True,
            )

            pred_standard = (
                model(tensor)
                .detach()
                .cpu()
                .numpy()
            )

            pred = (
                pred_standard
                * target_std
                + target_mean
            )

            outputs.append(
                pred
            )

    return np.concatenate(
        outputs,
        axis=0,
    ).astype(
        np.float64,
        copy=False,
    )


# ============================================================================
# Metrics
# ============================================================================


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if (
        a.size < 2
        or np.std(a) < 1.0e-12
        or np.std(b) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def calibration_fit(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    design = np.column_stack(
        [
            gt,
            np.ones_like(gt),
        ]
    )

    parameters, _, _, _ = (
        np.linalg.lstsq(
            design,
            pred,
            rcond=None,
        )
    )

    slope = float(
        parameters[0]
    )

    intercept = float(
        parameters[1]
    )

    fitted = (
        slope * gt
        + intercept
    )

    ss_res = float(
        np.sum(
            (pred - fitted) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (pred - np.mean(pred)) ** 2
        )
    )

    if ss_tot <= 1.0e-15:
        r_squared = float("nan")
    else:
        r_squared = float(
            1.0
            - ss_res / ss_tot
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def one_axis_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    error = (
        pred - gt
    )

    fit = calibration_fit(
        gt,
        pred,
    )

    gt_std = float(
        np.std(gt)
    )

    pred_std = float(
        np.std(pred)
    )

    return {
        "count": int(
            gt.shape[0]
        ),
        "gt_mean": float(
            np.mean(gt)
        ),
        "pred_mean": float(
            np.mean(pred)
        ),
        "gt_std": gt_std,
        "pred_std": pred_std,
        "std_ratio": (
            pred_std / gt_std
            if gt_std > 1.0e-12
            else float("nan")
        ),
        "bias": float(
            np.mean(error)
        ),
        "mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "correlation": (
            safe_correlation(
                gt,
                pred,
            )
        ),
        "slope": fit[
            "slope"
        ],
        "intercept": fit[
            "intercept"
        ],
        "r_squared": fit[
            "r_squared"
        ],
    }


def all_axis_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[
    str,
    Dict[str, float],
]:
    return {
        axis: one_axis_metrics(
            gt[:, axis_index],
            pred[:, axis_index],
        )
        for (
            axis_index,
            axis,
        ) in enumerate(AXES)
    }


# ============================================================================
# CSV output
# ============================================================================


def write_rows(
    path: Path,
    rows: Sequence[
        Mapping[str, object]
    ],
) -> None:
    rows = list(rows)

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================================
# Plotting
# ============================================================================


def plot_forward_calibration(
    path: Path,
    gt: np.ndarray,
    head_pred: np.ndarray,
    probe_pred: np.ndarray,
    title: str,
) -> None:
    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        gt[:, 2],
        head_pred[:, 2],
        s=7,
        alpha=0.25,
        label="Model-T head",
    )

    plt.scatter(
        gt[:, 2],
        probe_pred[:, 2],
        s=7,
        alpha=0.25,
        label="Linear probe",
    )

    minimum = float(
        min(
            np.min(gt[:, 2]),
            np.min(head_pred[:, 2]),
            np.min(probe_pred[:, 2]),
        )
    )

    maximum = float(
        max(
            np.max(gt[:, 2]),
            np.max(head_pred[:, 2]),
            np.max(probe_pred[:, 2]),
        )
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="ideal y=x",
    )

    plt.xlabel(
        "GT directional t_z [m]"
    )

    plt.ylabel(
        "Predicted directional t_z [m]"
    )

    plt.title(
        title
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_forward_residual(
    path: Path,
    gt: np.ndarray,
    head_pred: np.ndarray,
    probe_pred: np.ndarray,
    title: str,
) -> None:
    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        gt[:, 2],
        head_pred[:, 2] - gt[:, 2],
        s=7,
        alpha=0.25,
        label="Model-T head",
    )

    plt.scatter(
        gt[:, 2],
        probe_pred[:, 2] - gt[:, 2],
        s=7,
        alpha=0.25,
        label="Linear probe",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "GT directional t_z [m]"
    )

    plt.ylabel(
        "Residual pred_z - gt_z [m]"
    )

    plt.title(
        title
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


# ============================================================================
# Reporting
# ============================================================================


def print_forward_comparison(
    dataset_name: str,
    gt: np.ndarray,
    head_pred: np.ndarray,
    probe_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    head = one_axis_metrics(
        gt[:, 2],
        head_pred[:, 2],
    )

    probe = one_axis_metrics(
        gt[:, 2],
        probe_pred[:, 2],
    )

    print()
    print(
        "=" * 132
    )
    print(
        f"{dataset_name} — FORWARD t_z DECODABILITY"
    )
    print(
        "=" * 132
    )

    print(
        f"{'Predictor':<18}"
        f"{'N':>8}"
        f"{'Bias':>12}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'Corr':>10}"
        f"{'StdRat':>10}"
        f"{'Slope':>10}"
        f"{'Offset':>12}"
        f"{'R2':>10}"
    )

    print(
        "-" * 112
    )

    for name, values in (
        (
            "Model-T head",
            head,
        ),
        (
            "Linear probe",
            probe,
        ),
    ):
        print(
            f"{name:<18}"
            f"{values['count']:>8d}"
            f"{values['bias']:>12.6f}"
            f"{values['mae']:>12.6f}"
            f"{values['rmse']:>12.6f}"
            f"{values['correlation']:>10.4f}"
            f"{values['std_ratio']:>10.4f}"
            f"{values['slope']:>10.4f}"
            f"{values['intercept']:>12.6f}"
            f"{values['r_squared']:>10.4f}"
        )

    return {
        "head": head,
        "probe": probe,
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = choose_device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 112
    )
    print(
        "DeepDCT-VO translation representation decodability audit"
    )
    print(
        "=" * 112
    )
    print(
        f"Training representation root: {args.train_root}"
    )
    print(
        "Training sequences:           "
        f"{','.join(args.train_sequences)}"
    )
    print(
        f"Probe epochs:                 {args.epochs}"
    )
    print(
        f"Probe batch size:             {args.batch_size}"
    )
    print(
        f"Learning rate:                {args.learning_rate}"
    )
    print(
        f"Weight decay:                 {args.weight_decay}"
    )
    print(
        f"Device:                       {device}"
    )
    print(
        f"Output directory:             {args.output_dir}"
    )
    print(
        "=" * 112
    )

    # ------------------------------------------------------------------------
    # Load all 00-08 representations.
    # ------------------------------------------------------------------------

    train_sequences = {}

    representation_dimension = None

    holdout_masks = {}

    for sequence_index, sequence in enumerate(
        args.train_sequences
    ):
        representation_path, prediction_path = (
            training_sequence_paths(
                args.train_root,
                sequence,
            )
        )

        data = load_sequence(
            representation_path,
            prediction_path,
        )

        current_dimension = (
            data[
                "representation"
            ].shape[1]
        )

        if representation_dimension is None:
            representation_dimension = (
                current_dimension
            )

        elif current_dimension != representation_dimension:
            raise ValueError(
                "Representation dimension mismatch: "
                f"sequence {sequence} has "
                f"{current_dimension}, expected "
                f"{representation_dimension}."
            )

        train_sequences[
            sequence
        ] = data

        holdout_masks[
            sequence
        ] = holdout_mask(
            data[
                "representation"
            ].shape[0],
            modulus=(
                args.holdout_modulus
            ),
            remainder=(
                args.holdout_remainder
            ),
            sequence_offset=(
                sequence_index
            ),
        )

    assert (
        representation_dimension
        is not None
    )

    train_count = sum(
        int(
            np.count_nonzero(
                ~holdout_masks[
                    sequence
                ]
            )
        )
        for sequence in (
            args.train_sequences
        )
    )

    holdout_count = sum(
        int(
            np.count_nonzero(
                holdout_masks[
                    sequence
                ]
            )
        )
        for sequence in (
            args.train_sequences
        )
    )

    print()
    print(
        f"Representation dimension:     {representation_dimension}"
    )
    print(
        f"Probe-train samples:          {train_count}"
    )
    print(
        f"Probe-holdout samples:        {holdout_count}"
    )

    # ------------------------------------------------------------------------
    # Compute normalization from probe-training samples only.
    # ------------------------------------------------------------------------

    print()
    print(
        "Computing probe-training feature/target statistics..."
    )

    statistics = compute_training_statistics(
        train_sequences,
        holdout_masks,
        feature_epsilon=(
            args.feature_epsilon
        ),
        target_epsilon=(
            args.target_epsilon
        ),
    )

    print(
        "Target mean [x,y,z]: "
        f"{statistics['target_mean'].tolist()}"
    )

    print(
        "Target std  [x,y,z]: "
        f"{statistics['target_std'].tolist()}"
    )

    # ------------------------------------------------------------------------
    # Fit fresh linear probe.
    # ------------------------------------------------------------------------

    model = LinearTranslationProbe(
        representation_dimension
    )

    print()
    print(
        "=" * 112
    )
    print(
        "LINEAR PROBE TRAINING"
    )
    print(
        "=" * 112
    )

    history = train_probe(
        model=model,
        train_sequences=(
            train_sequences
        ),
        holdout_masks=(
            holdout_masks
        ),
        statistics=statistics,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=(
            args.learning_rate
        ),
        weight_decay=(
            args.weight_decay
        ),
        seed=args.seed,
    )

    # ------------------------------------------------------------------------
    # Evaluate pooled holdout.
    # ------------------------------------------------------------------------

    pooled_gt = []
    pooled_head = []
    pooled_probe = []
    pooled_sequence = []
    pooled_local_index = []

    per_sequence_rows = []
    per_axis_rows = []

    for sequence in (
        args.train_sequences
    ):
        data = train_sequences[
            sequence
        ]

        mask = holdout_masks[
            sequence
        ]

        representation = data[
            "representation"
        ][mask]

        gt = data[
            "gt"
        ][mask]

        head_pred = data[
            "head_pred"
        ][mask]

        probe_pred = predict_probe(
            model,
            representation,
            statistics,
            device,
            args.batch_size,
        )

        pooled_gt.append(
            gt
        )

        pooled_head.append(
            head_pred
        )

        pooled_probe.append(
            probe_pred
        )

        indices = np.flatnonzero(
            mask
        )

        pooled_sequence.extend(
            [sequence]
            * len(indices)
        )

        pooled_local_index.extend(
            indices.tolist()
        )

        for predictor_name, prediction in (
            (
                "head",
                head_pred,
            ),
            (
                "probe",
                probe_pred,
            ),
        ):
            metrics = all_axis_metrics(
                gt,
                prediction,
            )

            for axis in AXES:
                row = {
                    "dataset": (
                        f"train-{sequence}-holdout"
                    ),
                    "predictor": (
                        predictor_name
                    ),
                    "axis": axis,
                    **metrics[axis],
                }

                per_axis_rows.append(
                    row
                )

            z = metrics["z"]

            per_sequence_rows.append(
                {
                    "sequence": (
                        sequence
                    ),
                    "predictor": (
                        predictor_name
                    ),
                    **z,
                }
            )

    pooled_gt_array = np.concatenate(
        pooled_gt,
        axis=0,
    )

    pooled_head_array = np.concatenate(
        pooled_head,
        axis=0,
    )

    pooled_probe_array = np.concatenate(
        pooled_probe,
        axis=0,
    )

    result_summary = {}

    result_summary[
        "train_holdout"
    ] = print_forward_comparison(
        "POOLED 00-08 PROBE HOLDOUT",
        pooled_gt_array,
        pooled_head_array,
        pooled_probe_array,
    )

    # ------------------------------------------------------------------------
    # Save pooled holdout prediction CSV.
    # ------------------------------------------------------------------------

    holdout_rows = []

    for index in range(
        pooled_gt_array.shape[0]
    ):
        holdout_rows.append(
            {
                "sequence": (
                    pooled_sequence[
                        index
                    ]
                ),
                "local_index": (
                    pooled_local_index[
                        index
                    ]
                ),
                "gt_tx": float(
                    pooled_gt_array[
                        index,
                        0,
                    ]
                ),
                "gt_ty": float(
                    pooled_gt_array[
                        index,
                        1,
                    ]
                ),
                "gt_tz": float(
                    pooled_gt_array[
                        index,
                        2,
                    ]
                ),
                "head_tx": float(
                    pooled_head_array[
                        index,
                        0,
                    ]
                ),
                "head_ty": float(
                    pooled_head_array[
                        index,
                        1,
                    ]
                ),
                "head_tz": float(
                    pooled_head_array[
                        index,
                        2,
                    ]
                ),
                "probe_tx": float(
                    pooled_probe_array[
                        index,
                        0,
                    ]
                ),
                "probe_ty": float(
                    pooled_probe_array[
                        index,
                        1,
                    ]
                ),
                "probe_tz": float(
                    pooled_probe_array[
                        index,
                        2,
                    ]
                ),
            }
        )

    write_rows(
        args.output_dir
        / "probe_holdout_predictions.csv",
        holdout_rows,
    )

    plot_forward_calibration(
        args.output_dir
        / "calibration_train_holdout.png",
        pooled_gt_array,
        pooled_head_array,
        pooled_probe_array,
        "00-08 held-out representation decodability",
    )

    plot_forward_residual(
        args.output_dir
        / "residual_train_holdout.png",
        pooled_gt_array,
        pooled_head_array,
        pooled_probe_array,
        "00-08 held-out forward residual",
    )

    # ------------------------------------------------------------------------
    # Optional seq09 / seq10 external evaluation.
    # ------------------------------------------------------------------------

    external = {
        "09": (
            args.representations_09,
            args.predictions_09,
        ),
        "10": (
            args.representations_10,
            args.predictions_10,
        ),
    }

    for sequence, (
        representation_path,
        prediction_path,
    ) in external.items():
        if (
            representation_path is None
            and prediction_path is None
        ):
            continue

        if (
            representation_path is None
            or prediction_path is None
        ):
            raise ValueError(
                f"Sequence {sequence}: provide both "
                f"--representations-{sequence} and "
                f"--predictions-{sequence}."
            )

        data = load_sequence(
            representation_path,
            prediction_path,
        )

        if (
            data[
                "representation"
            ].shape[1]
            != representation_dimension
        ):
            raise ValueError(
                f"Sequence {sequence} representation dimension "
                "does not match training representation."
            )

        probe_pred = predict_probe(
            model,
            data[
                "representation"
            ],
            statistics,
            device,
            args.batch_size,
        )

        result_summary[
            f"seq{sequence}"
        ] = print_forward_comparison(
            f"SEQUENCE {sequence}",
            data["gt"],
            data["head_pred"],
            probe_pred,
        )

        for predictor_name, prediction in (
            (
                "head",
                data["head_pred"],
            ),
            (
                "probe",
                probe_pred,
            ),
        ):
            metrics = all_axis_metrics(
                data["gt"],
                prediction,
            )

            for axis in AXES:
                per_axis_rows.append(
                    {
                        "dataset": (
                            f"seq{sequence}"
                        ),
                        "predictor": (
                            predictor_name
                        ),
                        "axis": axis,
                        **metrics[axis],
                    }
                )

        plot_forward_calibration(
            args.output_dir
            / f"calibration_seq{sequence}.png",
            data["gt"],
            data["head_pred"],
            probe_pred,
            (
                f"Sequence {sequence}: "
                "head vs linear representation probe"
            ),
        )

        plot_forward_residual(
            args.output_dir
            / f"residual_seq{sequence}.png",
            data["gt"],
            data["head_pred"],
            probe_pred,
            (
                f"Sequence {sequence}: "
                "forward residual"
            ),
        )

    # ------------------------------------------------------------------------
    # Primary interpretation.
    # ------------------------------------------------------------------------

    head_holdout = result_summary[
        "train_holdout"
    ][
        "head"
    ]

    probe_holdout = result_summary[
        "train_holdout"
    ][
        "probe"
    ]

    print()
    print(
        "=" * 112
    )
    print(
        "PRIMARY DIAGNOSTIC"
    )
    print(
        "=" * 112
    )

    print(
        "00-08 holdout Model-T head:"
    )
    print(
        f"  slope:       {head_holdout['slope']:.6f}"
    )
    print(
        f"  correlation: {head_holdout['correlation']:.6f}"
    )
    print(
        f"  RMSE:        {head_holdout['rmse']:.6f}"
    )
    print(
        f"  std ratio:   {head_holdout['std_ratio']:.6f}"
    )

    print()
    print(
        "00-08 holdout linear probe:"
    )
    print(
        f"  slope:       {probe_holdout['slope']:.6f}"
    )
    print(
        f"  correlation: {probe_holdout['correlation']:.6f}"
    )
    print(
        f"  RMSE:        {probe_holdout['rmse']:.6f}"
    )
    print(
        f"  std ratio:   {probe_holdout['std_ratio']:.6f}"
    )

    # A deliberately conservative interpretation rule.
    if (
        probe_holdout["slope"] >= 0.80
        and probe_holdout[
            "correlation"
        ] >= 0.80
        and probe_holdout[
            "rmse"
        ] <= 0.75
        * head_holdout[
            "rmse"
        ]
    ):
        interpretation = (
            "REPRESENTATION_DECODABLE_HEAD_MAPPING_LIMITED"
        )

        print()
        print(
            "[RESULT] The held-out training representation contains "
            "substantially more recoverable metric t_z information "
            "than the current Model-T output mapping uses."
        )

        print(
            "Primary suspect: downstream translation-head/output "
            "optimization rather than loss of metric information "
            "in the representation."
        )

    elif (
        probe_holdout["slope"] < 0.60
        or probe_holdout[
            "correlation"
        ] < 0.60
    ):
        interpretation = (
            "REPRESENTATION_ITSELF_WEAKLY_DECODABLE"
        )

        print()
        print(
            "[RESULT] A fresh linear probe cannot recover metric "
            "forward translation reliably from held-out 00-08 "
            "representations."
        )

        print(
            "Primary suspect: the fused translation representation "
            "itself does not preserve sufficient metric-motion "
            "information."
        )

    else:
        interpretation = (
            "REPRESENTATION_PARTIALLY_DECODABLE"
        )

        print()
        print(
            "[RESULT] The representation contains useful metric "
            "translation information, but a simple linear probe "
            "does not fully recover the target range."
        )

        print(
            "Both representation quality and downstream mapping "
            "may contribute."
        )

    # ------------------------------------------------------------------------
    # Save metrics.
    # ------------------------------------------------------------------------

    overall_rows = []

    for dataset_name, values in (
        result_summary.items()
    ):
        for predictor, metrics in (
            values.items()
        ):
            overall_rows.append(
                {
                    "dataset": (
                        dataset_name
                    ),
                    "predictor": (
                        predictor
                    ),
                    **metrics,
                }
            )

    write_rows(
        args.output_dir
        / "overall_metrics.csv",
        overall_rows,
    )

    write_rows(
        args.output_dir
        / "per_sequence_metrics.csv",
        per_sequence_rows,
    )

    write_rows(
        args.output_dir
        / "per_axis_metrics.csv",
        per_axis_rows,
    )

    write_rows(
        args.output_dir
        / "probe_training_history.csv",
        history,
    )

    summary = {
        "train_root": str(
            args.train_root
        ),
        "train_sequences": (
            args.train_sequences
        ),
        "representation_dimension": int(
            representation_dimension
        ),
        "probe_train_samples": int(
            train_count
        ),
        "probe_holdout_samples": int(
            holdout_count
        ),
        "probe_configuration": {
            "epochs": (
                args.epochs
            ),
            "batch_size": (
                args.batch_size
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "holdout_modulus": (
                args.holdout_modulus
            ),
            "holdout_remainder": (
                args.holdout_remainder
            ),
            "seed": (
                args.seed
            ),
            "device": str(
                device
            ),
        },
        "normalization": {
            "target_mean": (
                statistics[
                    "target_mean"
                ].tolist()
            ),
            "target_std": (
                statistics[
                    "target_std"
                ].tolist()
            ),
        },
        "results": (
            result_summary
        ),
        "primary_interpretation": (
            interpretation
        ),
    }

    with (
        args.output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "=" * 112
    )
    print(
        "AUDIT COMPLETE"
    )
    print(
        "=" * 112
    )
    print(
        f"Interpretation: {interpretation}"
    )
    print(
        f"Outputs:        {args.output_dir}"
    )
    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()