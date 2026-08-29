#!/usr/bin/env python3
"""
DeepDCT-VO A6 train-vs-test translation calibration audit.

Primary question
----------------

The preceding A6 audits established that:

1. DCT forward/inverse geometry is correct.
2. Existing *_dct.txt labels are geometrically compatible with KITTI poses.
3. Model-T is conditioned on ground-truth rotation in the paper-style path.
4. Unseen sequences 09/10 exhibit strong forward-motion range compression.
5. Almost all 09/10 GT t_z values lie inside the 00-08 training support.

This script tests the remaining discriminator:

    Did Model-T learn metric forward translation on its TRAINING data?

Two outcomes are especially informative.

A) Good training calibration, bad unseen calibration:

       train:  pred_z ~= 1.0 * gt_z + small offset
       test:   pred_z ~= slope << 1 * gt_z + large offset

   Interpretation:
       representation/generalization failure.

B) Training calibration is already compressed:

       train: slope << 1

   Interpretation:
       Model-T did not learn forward-motion magnitude properly even on
       its optimization distribution. Investigate architecture/loss/
       optimization before interpreting A7.

Design
------

The script deliberately reuses scripts/evaluate_deepdct_vo.py for inference
instead of reimplementing:

    - checkpoint loading,
    - model reconstruction,
    - LR-ASPP configuration,
    - Lite-Mono configuration,
    - normalization,
    - Model-T GT-rotation conditioning,
    - dataset construction.

When --generate-train-predictions is supplied, it evaluates the A6 checkpoint
on each training sequence 00-08 with:

    --use-ground-truth-rotation
    --skip-trajectory

and saves:

    <output-dir>/train_predictions/sequence_00/frame_predictions.csv
    ...
    <output-dir>/train_predictions/sequence_08/frame_predictions.csv

Afterward, the script analyzes:

    pooled train 00-08
    each individual training sequence
    unseen sequence 09
    unseen sequence 10

Metrics
-------

For forward directional translation t_z:

    GT mean/std
    predicted mean/std
    bias
    MAE
    RMSE
    correlation
    prediction std / GT std
    calibration slope
    calibration intercept
    calibration R^2

The script also uses TRAIN-derived bins and TRAIN-derived low/medium/high
regimes for all datasets so that comparisons use identical thresholds.

Important
---------

Use the same A6 final checkpoint that generated the 09/10 frame predictions.

Do not apply A5 post-processing scaling for this audit. We want the native
Model-T output.

Example
-------

First run: generate 00-08 predictions and audit everything:

python scripts/audit_translation_train_test_calibration.py \
    --checkpoint \
      experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/checkpoints/latest.pt \
    --data-root data \
    --predictions-09 \
      experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/evaluation_sequence_09/frame_predictions.csv \
    --predictions-10 \
      experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/evaluation_sequence_10/frame_predictions.csv \
    --generate-train-predictions

Subsequent analysis-only rerun:

python scripts/audit_translation_train_test_calibration.py \
    --checkpoint \
      experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/checkpoints/latest.pt \
    --data-root data \
    --predictions-09 \
      experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/evaluation_sequence_09/frame_predictions.csv \
    --predictions-10 \
      experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/evaluation_sequence_10/frame_predictions.csv

Outputs
-------

<output-dir>/
    summary.json
    calibration_summary.csv
    per_sequence_training_calibration.csv
    frozen_regime_metrics.csv
    calibration_bins.csv

    train_predictions/
        sequence_00/frame_predictions.csv
        ...
        sequence_08/frame_predictions.csv

    train_vs_test_scatter.png
    train_vs_test_calibration_curve.png
    train_vs_test_residual_curve.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


# ============================================================================
# CLI
# ============================================================================


def normalize_sequence(value: str) -> str:
    value = str(value).strip()

    if value.isdigit():
        return f"{int(value):02d}"

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DeepDCT-VO Model-T translation calibration "
            "on A6 training sequences 00-08 versus unseen 09/10."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "A6 final checkpoint. Use the same checkpoint used to "
            "generate the sequence-09/10 predictions."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root.",
    )

    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("scripts/evaluate_deepdct_vo.py"),
        help="Existing DeepDCT-VO evaluation script.",
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=DEFAULT_TRAIN_SEQUENCES,
        help="Training sequences to pool.",
    )

    parser.add_argument(
        "--predictions-09",
        type=Path,
        required=True,
        help="Existing A6 frame_predictions.csv for sequence 09.",
    )

    parser.add_argument(
        "--predictions-10",
        type=Path,
        required=True,
        help="Existing A6 frame_predictions.csv for sequence 10.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/track_a/paper_reproduction/"
            "a6_translation_train_test_calibration"
        ),
        help="Audit output directory.",
    )

    parser.add_argument(
        "--generate-train-predictions",
        action="store_true",
        help=(
            "Run evaluate_deepdct_vo.py over 00-08 before analysis. "
            "Without this flag, existing generated CSVs are reused."
        ),
    )

    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help=(
            "Regenerate training prediction CSVs even when they already exist."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size passed to evaluate_deepdct_vo.py.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Evaluation worker count.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Evaluator progress interval.",
    )

    parser.add_argument(
        "--low-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Pooled-training low-motion quantile.",
    )

    parser.add_argument(
        "--high-quantile",
        type=float,
        default=2.0 / 3.0,
        help="Pooled-training high-motion quantile.",
    )

    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=12,
        help="Number of pooled-training quantile calibration bins.",
    )

    parser.add_argument(
        "--minimum-bin-count",
        type=int,
        default=20,
        help="Minimum count for reporting a calibration bin as reliable.",
    )

    args = parser.parse_args()

    args.train_sequences = [
        normalize_sequence(sequence)
        for sequence in args.train_sequences
    ]

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if args.log_interval <= 0:
        raise ValueError(
            "--log-interval must be positive."
        )

    if not (
        0.0
        < args.low_quantile
        < args.high_quantile
        < 1.0
    ):
        raise ValueError(
            "Require 0 < low-quantile < high-quantile < 1."
        )

    if args.calibration_bins < 2:
        raise ValueError(
            "--calibration-bins must be >= 2."
        )

    if args.minimum_bin_count < 1:
        raise ValueError(
            "--minimum-bin-count must be >= 1."
        )

    return args


# ============================================================================
# Prediction generation using the existing evaluator
# ============================================================================


def training_prediction_path(
    output_dir: Path,
    sequence: str,
) -> Path:
    return (
        output_dir
        / "train_predictions"
        / f"sequence_{sequence}"
        / "frame_predictions.csv"
    )


def run_training_evaluation(
    *,
    evaluator: Path,
    checkpoint: Path,
    data_root: Path,
    sequence: str,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    log_interval: int,
) -> None:
    """
    Reuse the project's evaluator.

    Explicit GT rotation is supplied even though an A6 checkpoint should
    already restore that conditioning configuration. This makes the intended
    Model-T diagnostic unambiguous.
    """

    sequence_output = (
        output_dir
        / "train_predictions"
        / f"sequence_{sequence}"
    )

    sequence_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--sequence",
        sequence,
        "--output-dir",
        str(sequence_output),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
        "--log-interval",
        str(log_interval),
        "--use-ground-truth-rotation",
        "--skip-trajectory",
    ]

    print()
    print("=" * 104)
    print(
        f"Generating training predictions: sequence {sequence}"
    )
    print("=" * 104)
    print(
        " ".join(command)
    )
    print("=" * 104)

    subprocess.run(
        command,
        check=True,
    )

    expected = (
        sequence_output
        / "frame_predictions.csv"
    )

    if not expected.is_file():
        raise RuntimeError(
            "Evaluator completed but did not create "
            f"{expected}"
        )


def prepare_training_predictions(
    args: argparse.Namespace,
) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.evaluator.is_file():
        raise FileNotFoundError(
            f"Evaluator not found: {args.evaluator}"
        )

    for sequence in args.train_sequences:
        target = training_prediction_path(
            args.output_dir,
            sequence,
        )

        if (
            target.is_file()
            and not args.force_regenerate
        ):
            print(
                f"[reuse] sequence {sequence}: {target}"
            )
            continue

        run_training_evaluation(
            evaluator=args.evaluator,
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            sequence=sequence,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            log_interval=args.log_interval,
        )


# ============================================================================
# CSV loading
# ============================================================================


def load_frame_predictions(
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

    columns: Dict[str, List[float]] = {
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
                columns[key].append(
                    float(row[key])
                )

    if not columns["translation_gt_z"]:
        raise ValueError(
            f"No prediction rows in {path}"
        )

    def stack(prefix: str) -> np.ndarray:
        return np.column_stack(
            [
                columns[f"{prefix}_x"],
                columns[f"{prefix}_y"],
                columns[f"{prefix}_z"],
            ]
        ).astype(
            np.float64,
            copy=False,
        )

    gt = stack("translation_gt")
    pred = stack("translation_pred")

    if gt.shape != pred.shape:
        raise ValueError(
            f"GT/prediction shape mismatch in {path}: "
            f"{gt.shape} vs {pred.shape}"
        )

    if (
        not np.isfinite(gt).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError(
            f"NaN/Inf detected in {path}"
        )

    return {
        "gt": gt,
        "pred": pred,
    }


# ============================================================================
# Calibration metrics
# ============================================================================


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        a.size < 2
        or np.std(a) < 1.0e-12
        or np.std(b) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def linear_fit(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """
    Fit:

        pred = slope * gt + intercept
    """

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
        r_squared = (
            1.0
            - ss_res / ss_tot
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": float(
            r_squared
        ),
    }


def calibration_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    error = (
        pred - gt
    )

    fit = linear_fit(
        gt,
        pred,
    )

    gt_std = float(
        np.std(gt)
    )

    pred_std = float(
        np.std(pred)
    )

    if gt_std < 1.0e-12:
        std_ratio = float("nan")
    else:
        std_ratio = (
            pred_std / gt_std
        )

    return {
        "count": int(
            gt.shape[0]
        ),
        "gt_mean": float(
            np.mean(gt)
        ),
        "gt_std": gt_std,
        "gt_min": float(
            np.min(gt)
        ),
        "gt_q01": float(
            np.quantile(gt, 0.01)
        ),
        "gt_q25": float(
            np.quantile(gt, 0.25)
        ),
        "gt_median": float(
            np.median(gt)
        ),
        "gt_q75": float(
            np.quantile(gt, 0.75)
        ),
        "gt_q99": float(
            np.quantile(gt, 0.99)
        ),
        "gt_max": float(
            np.max(gt)
        ),
        "pred_mean": float(
            np.mean(pred)
        ),
        "pred_std": pred_std,
        "prediction_std_ratio": float(
            std_ratio
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
        "calibration_slope": (
            fit["slope"]
        ),
        "calibration_intercept": (
            fit["intercept"]
        ),
        "calibration_r_squared": (
            fit["r_squared"]
        ),
    }


# ============================================================================
# Frozen regimes
# ============================================================================


def regime_masks(
    gt_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> Dict[str, np.ndarray]:
    return {
        "low": (
            gt_z
            <= low_threshold
        ),
        "medium": (
            (gt_z > low_threshold)
            & (gt_z <= high_threshold)
        ),
        "high": (
            gt_z
            > high_threshold
        ),
    }


def compute_regime_rows(
    dataset_name: str,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[
        Dict[str, object]
    ] = []

    masks = regime_masks(
        gt_z,
        low_threshold,
        high_threshold,
    )

    for regime, mask in masks.items():
        count = int(
            np.count_nonzero(mask)
        )

        if count == 0:
            rows.append(
                {
                    "dataset": dataset_name,
                    "regime": regime,
                    "count": 0,
                    "gt_mean": float("nan"),
                    "pred_mean": float("nan"),
                    "bias": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "correlation": float("nan"),
                    "slope": float("nan"),
                    "intercept": float("nan"),
                }
            )
            continue

        gt_subset = gt_z[mask]
        pred_subset = pred_z[mask]

        metrics = calibration_metrics(
            gt_subset,
            pred_subset,
        )

        rows.append(
            {
                "dataset": dataset_name,
                "regime": regime,
                "count": count,
                "gt_mean": metrics[
                    "gt_mean"
                ],
                "pred_mean": metrics[
                    "pred_mean"
                ],
                "bias": metrics["bias"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "correlation": metrics[
                    "correlation"
                ],
                "slope": metrics[
                    "calibration_slope"
                ],
                "intercept": metrics[
                    "calibration_intercept"
                ],
            }
        )

    return rows


# ============================================================================
# Training-derived calibration bins
# ============================================================================


def build_training_bin_edges(
    train_gt_z: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    quantiles = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    edges = np.quantile(
        train_gt_z,
        quantiles,
    )

    edges = np.unique(
        edges
    )

    if edges.shape[0] < 3:
        raise ValueError(
            "Insufficient unique GT translation values "
            "for calibration bins."
        )

    epsilon = 1.0e-12

    edges[0] -= epsilon
    edges[-1] += epsilon

    return edges


def compute_bin_rows(
    dataset_name: str,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    edges: np.ndarray,
    minimum_count: int,
) -> List[Dict[str, object]]:
    rows: List[
        Dict[str, object]
    ] = []

    for bin_index in range(
        edges.shape[0] - 1
    ):
        low = float(
            edges[bin_index]
        )

        high = float(
            edges[bin_index + 1]
        )

        if bin_index == (
            edges.shape[0] - 2
        ):
            mask = (
                (gt_z >= low)
                & (gt_z <= high)
            )
        else:
            mask = (
                (gt_z >= low)
                & (gt_z < high)
            )

        count = int(
            np.count_nonzero(mask)
        )

        if count == 0:
            continue

        gt_subset = gt_z[mask]
        pred_subset = pred_z[mask]

        error = (
            pred_subset
            - gt_subset
        )

        rows.append(
            {
                "dataset": dataset_name,
                "bin_index": bin_index,
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "sufficient_count": (
                    count >= minimum_count
                ),
                "gt_mean": float(
                    np.mean(gt_subset)
                ),
                "pred_mean": float(
                    np.mean(pred_subset)
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
                        gt_subset,
                        pred_subset,
                    )
                ),
            }
        )

    return rows


# ============================================================================
# CSV helpers
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

        for row in rows:
            writer.writerow(row)


# ============================================================================
# Plot helpers
# ============================================================================


def subsample_for_plot(
    gt: np.ndarray,
    pred: np.ndarray,
    maximum: int = 5000,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    if gt.shape[0] <= maximum:
        return gt, pred

    indices = np.linspace(
        0,
        gt.shape[0] - 1,
        maximum,
    ).astype(
        np.int64
    )

    return (
        gt[indices],
        pred[indices],
    )


def plot_train_test_scatter(
    path: Path,
    datasets: Mapping[
        str,
        Tuple[
            np.ndarray,
            np.ndarray,
        ],
    ],
) -> None:
    plt.figure(
        figsize=(9, 8)
    )

    minimum = float("inf")
    maximum = float("-inf")

    for dataset_name, (
        gt_z,
        pred_z,
    ) in datasets.items():
        plot_gt, plot_pred = (
            subsample_for_plot(
                gt_z,
                pred_z,
            )
        )

        plt.scatter(
            plot_gt,
            plot_pred,
            s=8,
            alpha=0.25,
            label=dataset_name,
        )

        minimum = min(
            minimum,
            float(np.min(gt_z)),
            float(np.min(pred_z)),
        )

        maximum = max(
            maximum,
            float(np.max(gt_z)),
            float(np.max(pred_z)),
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
        "Model-T forward translation: train vs unseen"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_calibration_curves(
    path: Path,
    rows: Sequence[
        Mapping[str, object]
    ],
) -> None:
    plt.figure(
        figsize=(9, 8)
    )

    datasets = sorted(
        {
            str(row["dataset"])
            for row in rows
        }
    )

    all_values = []

    for dataset_name in datasets:
        subset = [
            row
            for row in rows
            if (
                str(row["dataset"])
                == dataset_name
                and bool(
                    row[
                        "sufficient_count"
                    ]
                )
            )
        ]

        if not subset:
            continue

        x = np.asarray(
            [
                float(row["gt_mean"])
                for row in subset
            ]
        )

        y = np.asarray(
            [
                float(row["pred_mean"])
                for row in subset
            ]
        )

        all_values.extend(
            x.tolist()
        )

        all_values.extend(
            y.tolist()
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=dataset_name,
        )

    if all_values:
        minimum = min(
            all_values
        )

        maximum = max(
            all_values
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            label="ideal y=x",
        )

    plt.xlabel(
        "Mean GT t_z in training-derived bin [m]"
    )

    plt.ylabel(
        "Mean predicted t_z [m]"
    )

    plt.title(
        "Train-derived forward-motion calibration"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_residual_curves(
    path: Path,
    rows: Sequence[
        Mapping[str, object]
    ],
) -> None:
    plt.figure(
        figsize=(9, 7)
    )

    datasets = sorted(
        {
            str(row["dataset"])
            for row in rows
        }
    )

    for dataset_name in datasets:
        subset = [
            row
            for row in rows
            if (
                str(row["dataset"])
                == dataset_name
                and bool(
                    row[
                        "sufficient_count"
                    ]
                )
            )
        ]

        if not subset:
            continue

        x = np.asarray(
            [
                float(row["gt_mean"])
                for row in subset
            ]
        )

        y = np.asarray(
            [
                float(row["bias"])
                for row in subset
            ]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=dataset_name,
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Mean GT t_z in training-derived bin [m]"
    )

    plt.ylabel(
        "Mean residual pred_z - gt_z [m]"
    )

    plt.title(
        "Forward translation residual: train vs unseen"
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
# Console reporting
# ============================================================================


def print_calibration_header() -> None:
    print(
        f"{'Dataset':<14}"
        f"{'N':>8}"
        f"{'GTmean':>11}"
        f"{'Pmean':>11}"
        f"{'GTstd':>11}"
        f"{'Pstd':>11}"
        f"{'StdRat':>10}"
        f"{'Bias':>11}"
        f"{'RMSE':>11}"
        f"{'Corr':>9}"
        f"{'Slope':>10}"
        f"{'Offset':>11}"
        f"{'R2':>9}"
    )

    print(
        "-" * 136
    )


def print_calibration_row(
    name: str,
    values: Mapping[
        str,
        float,
    ],
) -> None:
    print(
        f"{name:<14}"
        f"{int(values['count']):>8d}"
        f"{values['gt_mean']:>11.5f}"
        f"{values['pred_mean']:>11.5f}"
        f"{values['gt_std']:>11.5f}"
        f"{values['pred_std']:>11.5f}"
        f"{values['prediction_std_ratio']:>10.4f}"
        f"{values['bias']:>11.5f}"
        f"{values['rmse']:>11.5f}"
        f"{values['correlation']:>9.4f}"
        f"{values['calibration_slope']:>10.4f}"
        f"{values['calibration_intercept']:>11.5f}"
        f"{values['calibration_r_squared']:>9.4f}"
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 112
    )
    print(
        "DeepDCT-VO A6 train-vs-test translation calibration audit"
    )
    print(
        "=" * 112
    )
    print(
        f"Checkpoint:                   {args.checkpoint}"
    )
    print(
        "Training sequences:           "
        f"{','.join(args.train_sequences)}"
    )
    print(
        f"Sequence-09 predictions:      {args.predictions_09}"
    )
    print(
        f"Sequence-10 predictions:      {args.predictions_10}"
    )
    print(
        f"Generate train predictions:   {args.generate_train_predictions}"
    )
    print(
        f"Output directory:             {args.output_dir}"
    )
    print(
        "=" * 112
    )

    # ------------------------------------------------------------------------
    # Optional evaluation of training sequences.
    # ------------------------------------------------------------------------

    if args.generate_train_predictions:
        prepare_training_predictions(
            args
        )

    # ------------------------------------------------------------------------
    # Load per-training-sequence predictions.
    # ------------------------------------------------------------------------

    train_by_sequence: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    pooled_gt = []
    pooled_pred = []

    for sequence in args.train_sequences:
        prediction_path = (
            training_prediction_path(
                args.output_dir,
                sequence,
            )
        )

        if not prediction_path.is_file():
            raise FileNotFoundError(
                "\nTraining prediction CSV is missing:\n"
                f"  {prediction_path}\n\n"
                "Run this script once with:\n"
                "  --generate-train-predictions\n"
            )

        prediction = (
            load_frame_predictions(
                prediction_path
            )
        )

        train_by_sequence[
            sequence
        ] = prediction

        pooled_gt.append(
            prediction["gt"]
        )

        pooled_pred.append(
            prediction["pred"]
        )

    pooled_train_gt = np.concatenate(
        pooled_gt,
        axis=0,
    )

    pooled_train_pred = np.concatenate(
        pooled_pred,
        axis=0,
    )

    # ------------------------------------------------------------------------
    # Load unseen predictions.
    # ------------------------------------------------------------------------

    sequence_09 = (
        load_frame_predictions(
            args.predictions_09
        )
    )

    sequence_10 = (
        load_frame_predictions(
            args.predictions_10
        )
    )

    datasets = {
        "train00-08": (
            pooled_train_gt[:, 2],
            pooled_train_pred[:, 2],
        ),
        "seq09": (
            sequence_09["gt"][:, 2],
            sequence_09["pred"][:, 2],
        ),
        "seq10": (
            sequence_10["gt"][:, 2],
            sequence_10["pred"][:, 2],
        ),
    }

    # ------------------------------------------------------------------------
    # Overall calibration.
    # ------------------------------------------------------------------------

    summary_metrics: Dict[
        str,
        Dict[str, float],
    ] = {}

    summary_rows: List[
        Dict[str, object]
    ] = []

    print()
    print(
        "=" * 136
    )
    print(
        "POOLED TRAIN VS UNSEEN FORWARD CALIBRATION"
    )
    print(
        "=" * 136
    )

    print_calibration_header()

    for dataset_name, (
        gt_z,
        pred_z,
    ) in datasets.items():
        metrics = calibration_metrics(
            gt_z,
            pred_z,
        )

        summary_metrics[
            dataset_name
        ] = metrics

        print_calibration_row(
            dataset_name,
            metrics,
        )

        summary_rows.append(
            {
                "dataset": dataset_name,
                **metrics,
            }
        )

    # ------------------------------------------------------------------------
    # Individual training sequences.
    # ------------------------------------------------------------------------

    per_sequence_rows = []

    print()
    print(
        "=" * 136
    )
    print(
        "PER-TRAINING-SEQUENCE CALIBRATION"
    )
    print(
        "=" * 136
    )

    print_calibration_header()

    for sequence in (
        args.train_sequences
    ):
        prediction = (
            train_by_sequence[
                sequence
            ]
        )

        metrics = (
            calibration_metrics(
                prediction["gt"][:, 2],
                prediction["pred"][:, 2],
            )
        )

        print_calibration_row(
            f"train-{sequence}",
            metrics,
        )

        per_sequence_rows.append(
            {
                "sequence": sequence,
                **metrics,
            }
        )

    # ------------------------------------------------------------------------
    # Frozen pooled-training thresholds.
    # ------------------------------------------------------------------------

    train_gt_z = (
        pooled_train_gt[:, 2]
    )

    low_threshold = float(
        np.quantile(
            train_gt_z,
            args.low_quantile,
        )
    )

    high_threshold = float(
        np.quantile(
            train_gt_z,
            args.high_quantile,
        )
    )

    print()
    print(
        "=" * 112
    )
    print(
        "FROZEN TRAIN-DERIVED MOTION THRESHOLDS"
    )
    print(
        "=" * 112
    )
    print(
        f"Low:     t_z <= {low_threshold:.6f}"
    )
    print(
        f"Medium:  {low_threshold:.6f} < "
        f"t_z <= {high_threshold:.6f}"
    )
    print(
        f"High:    t_z > {high_threshold:.6f}"
    )

    # ------------------------------------------------------------------------
    # Frozen regimes.
    # ------------------------------------------------------------------------

    regime_rows = []

    print()
    print(
        "=" * 112
    )
    print(
        "FROZEN TRAIN-DERIVED MOTION-REGIME CALIBRATION"
    )
    print(
        "=" * 112
    )

    for dataset_name, (
        gt_z,
        pred_z,
    ) in datasets.items():
        rows = compute_regime_rows(
            dataset_name,
            gt_z,
            pred_z,
            low_threshold,
            high_threshold,
        )

        regime_rows.extend(
            rows
        )

        print(
            f"\n{dataset_name}"
        )

        for row in rows:
            print(
                f"  {str(row['regime']):<7s}"
                f" n={int(row['count']):5d}"
                f" gt={float(row['gt_mean']):.6f}"
                f" pred={float(row['pred_mean']):.6f}"
                f" bias={float(row['bias']):+.6f}"
                f" rmse={float(row['rmse']):.6f}"
                f" corr={float(row['correlation']):.4f}"
                f" slope={float(row['slope']):.4f}"
            )

    # ------------------------------------------------------------------------
    # Frozen training-derived bins.
    # ------------------------------------------------------------------------

    edges = build_training_bin_edges(
        train_gt_z,
        args.calibration_bins,
    )

    bin_rows = []

    print()
    print(
        "=" * 112
    )
    print(
        "TRAIN-DERIVED CALIBRATION BINS"
    )
    print(
        "=" * 112
    )

    for dataset_name, (
        gt_z,
        pred_z,
    ) in datasets.items():
        rows = compute_bin_rows(
            dataset_name,
            gt_z,
            pred_z,
            edges,
            args.minimum_bin_count,
        )

        bin_rows.extend(
            rows
        )

        print(
            f"\n{dataset_name}"
        )

        for row in rows:
            marker = (
                " "
                if bool(
                    row[
                        "sufficient_count"
                    ]
                )
                else "*"
            )

            print(
                f"{marker} bin {int(row['bin_index']):02d} "
                f"[{float(row['bin_low']):.3f}, "
                f"{float(row['bin_high']):.3f}) "
                f"n={int(row['count']):5d} "
                f"gt={float(row['gt_mean']):.4f} "
                f"pred={float(row['pred_mean']):.4f} "
                f"bias={float(row['bias']):+.4f}"
            )

    # ------------------------------------------------------------------------
    # Main discriminator.
    # ------------------------------------------------------------------------

    train_slope = (
        summary_metrics[
            "train00-08"
        ][
            "calibration_slope"
        ]
    )

    test09_slope = (
        summary_metrics[
            "seq09"
        ][
            "calibration_slope"
        ]
    )

    test10_slope = (
        summary_metrics[
            "seq10"
        ][
            "calibration_slope"
        ]
    )

    train_std_ratio = (
        summary_metrics[
            "train00-08"
        ][
            "prediction_std_ratio"
        ]
    )

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
        f"Pooled training calibration slope: {train_slope:.6f}"
    )
    print(
        f"Sequence 09 calibration slope:     {test09_slope:.6f}"
    )
    print(
        f"Sequence 10 calibration slope:     {test10_slope:.6f}"
    )
    print(
        "Pooled training prediction/GT "
        f"std ratio: {train_std_ratio:.6f}"
    )

    print()

    if (
        train_slope >= 0.80
        and train_std_ratio >= 0.75
    ):
        interpretation = (
            "TRAIN_CALIBRATION_GOOD_TEST_COLLAPSE"
        )

        print(
            "[RESULT] Training calibration preserves much more "
            "of the GT translation range."
        )
        print(
            "The severe seq09/10 compression is therefore most "
            "consistent with representation/generalization failure."
        )

    elif (
        train_slope < 0.60
        or train_std_ratio < 0.60
    ):
        interpretation = (
            "TRAIN_CALIBRATION_ALREADY_COMPRESSED"
        )

        print(
            "[RESULT] Forward-motion compression is already "
            "substantial on training data."
        )
        print(
            "The root issue is therefore not purely unseen-sequence "
            "generalization; Model-T did not learn the training "
            "translation range adequately."
        )

    else:
        interpretation = (
            "INTERMEDIATE_TRAIN_TEST_DEGRADATION"
        )

        print(
            "[RESULT] Training calibration is imperfect and "
            "degrades further on unseen sequences."
        )
        print(
            "Both in-distribution fitting and sequence "
            "generalization likely contribute."
        )

    # ------------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------------

    write_rows(
        args.output_dir
        / "calibration_summary.csv",
        summary_rows,
    )

    write_rows(
        args.output_dir
        / "per_sequence_training_calibration.csv",
        per_sequence_rows,
    )

    write_rows(
        args.output_dir
        / "frozen_regime_metrics.csv",
        regime_rows,
    )

    write_rows(
        args.output_dir
        / "calibration_bins.csv",
        bin_rows,
    )

    plot_train_test_scatter(
        args.output_dir
        / "train_vs_test_scatter.png",
        datasets,
    )

    plot_calibration_curves(
        args.output_dir
        / "train_vs_test_calibration_curve.png",
        bin_rows,
    )

    plot_residual_curves(
        args.output_dir
        / "train_vs_test_residual_curve.png",
        bin_rows,
    )

    summary = {
        "checkpoint": str(
            args.checkpoint
        ),
        "train_sequences": (
            args.train_sequences
        ),
        "training_samples": int(
            pooled_train_gt.shape[0]
        ),
        "sequence_09_samples": int(
            sequence_09[
                "gt"
            ].shape[0]
        ),
        "sequence_10_samples": int(
            sequence_10[
                "gt"
            ].shape[0]
        ),
        "frozen_training_thresholds": {
            "low": (
                low_threshold
            ),
            "high": (
                high_threshold
            ),
        },
        "overall_calibration": (
            summary_metrics
        ),
        "per_training_sequence": {
            row["sequence"]: row
            for row in per_sequence_rows
        },
        "regime_metrics": (
            regime_rows
        ),
        "calibration_bins": (
            bin_rows
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