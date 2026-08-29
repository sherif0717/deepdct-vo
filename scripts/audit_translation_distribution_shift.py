#!/usr/bin/env python3
"""
Audit DeepDCT-VO translation distribution shift and forward-motion calibration.

Purpose
-------

This script analyzes whether the A6 unseen-sequence translation error can be
explained by a mismatch between the training motion distribution (KITTI 00-08)
and the unseen test distributions (09 and 10).

It focuses primarily on directional forward translation t_z because the prior
A6 translation-error audit showed that t_z dominates accumulated trajectory
error.

Questions answered
------------------

1. What is the training distribution of directional t_z over sequences 00-08?
2. How do sequences 09 and 10 differ from that distribution?
3. What frozen low / medium / high motion thresholds are obtained from 00-08?
4. How much of each test sequence lies outside the central training range?
5. How does Model-T calibration behave as a function of true t_z?
6. Is predicted t_z approximately:

       pred_z = slope * gt_z + intercept

   or is there clear nonlinear range compression?
7. Does Model-T fail even in regions well supported by training data?
8. Are seq09 and seq10 errors mainly caused by distribution shift or by
   conditional prediction bias?

Inputs
------

Training labels:

    data/out_csv/00_dct.txt
    ...
    data/out_csv/08_dct.txt

Optional test prediction CSVs:

    evaluation_sequence_09/frame_predictions.csv
    evaluation_sequence_10/frame_predictions.csv

Expected DCT convention:

    tx ty tz rx ry rz

Outputs
-------

<output-dir>/
    summary.json
    sequence_distribution_metrics.csv
    frozen_regime_metrics.csv
    calibration_bins.csv
    training_tz_quantiles.csv

    training_test_tz_histogram.png
    calibration_curve_seq09.png
    calibration_curve_seq10.png
    prediction_scatter_seq09.png
    prediction_scatter_seq10.png
    residual_vs_gt_seq09.png
    residual_vs_gt_seq10.png

Examples
--------

Primary A6 audit:

    python scripts/audit_translation_distribution_shift.py \
        --data-root data \
        --predictions-09 \
          experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_09/frame_predictions.csv \
        --predictions-10 \
          experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_10/frame_predictions.csv

Training-only distribution audit:

    python scripts/audit_translation_distribution_shift.py \
        --data-root data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


AXIS_NAMES = ("x", "y", "z")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit DeepDCT-VO translation distribution shift "
            "between training sequences 00-08 and unseen sequences 09/10."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root containing out_csv/<sequence>_dct.txt.",
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
        help="Sequences defining the frozen training motion distribution.",
    )

    parser.add_argument(
        "--test-sequences",
        nargs="+",
        default=[
            "09",
            "10",
        ],
        help="Test label sequences to compare against training.",
    )

    parser.add_argument(
        "--predictions-09",
        type=Path,
        default=None,
        help="Optional A6 frame_predictions.csv for sequence 09.",
    )

    parser.add_argument(
        "--predictions-10",
        type=Path,
        default=None,
        help="Optional A6 frame_predictions.csv for sequence 10.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/track_a/paper_reproduction/"
            "a6_translation_distribution_shift"
        ),
        help="Audit output directory.",
    )

    parser.add_argument(
        "--low-quantile",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Training-derived lower threshold for low/medium/high "
            "motion regimes."
        ),
    )

    parser.add_argument(
        "--high-quantile",
        type=float,
        default=2.0 / 3.0,
        help=(
            "Training-derived upper threshold for low/medium/high "
            "motion regimes."
        ),
    )

    parser.add_argument(
        "--central-range-low-quantile",
        type=float,
        default=0.01,
        help="Lower training quantile defining central support.",
    )

    parser.add_argument(
        "--central-range-high-quantile",
        type=float,
        default=0.99,
        help="Upper training quantile defining central support.",
    )

    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=12,
        help="Number of fixed training-range bins for calibration analysis.",
    )

    parser.add_argument(
        "--minimum-bin-count",
        type=int,
        default=10,
        help="Minimum test samples required to report one calibration bin.",
    )

    parser.add_argument(
        "--translation-scale-factor-09",
        type=float,
        default=1.0,
        help="Optional post-processing scale applied to seq09 predictions.",
    )

    parser.add_argument(
        "--translation-scale-factor-10",
        type=float,
        default=1.0,
        help="Optional post-processing scale applied to seq10 predictions.",
    )

    args = parser.parse_args()

    args.train_sequences = [
        normalize_sequence(value)
        for value in args.train_sequences
    ]

    args.test_sequences = [
        normalize_sequence(value)
        for value in args.test_sequences
    ]

    if not (
        0.0
        < args.low_quantile
        < args.high_quantile
        < 1.0
    ):
        raise ValueError(
            "Regime quantiles must satisfy 0 < low < high < 1."
        )

    if not (
        0.0
        <= args.central_range_low_quantile
        < args.central_range_high_quantile
        <= 1.0
    ):
        raise ValueError(
            "Central support quantiles must satisfy "
            "0 <= low < high <= 1."
        )

    if args.calibration_bins < 2:
        raise ValueError(
            "--calibration-bins must be at least 2."
        )

    if args.minimum_bin_count < 1:
        raise ValueError(
            "--minimum-bin-count must be positive."
        )

    for value, name in (
        (
            args.translation_scale_factor_09,
            "--translation-scale-factor-09",
        ),
        (
            args.translation_scale_factor_10,
            "--translation-scale-factor-10",
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and positive."
            )

    return args


def normalize_sequence(
    sequence: str,
) -> str:
    value = str(sequence).strip()

    if value.isdigit():
        return f"{int(value):02d}"

    return value


# ============================================================================
# Loading
# ============================================================================


def load_dct_labels(
    path: Path,
) -> np.ndarray:
    """
    Load a DeepDCT label file as [N,6].

    Convention:
        tx ty tz rx ry rz
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"DCT label file not found: {path}"
        )

    rows: List[List[float]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            tokens = (
                line
                .replace(",", " ")
                .replace(";", " ")
                .split()
            )

            try:
                values = [
                    float(token)
                    for token in tokens
                ]
            except ValueError as error:
                if not rows:
                    # Permit one textual header.
                    continue

                raise ValueError(
                    f"Non-numeric DCT row in {path} "
                    f"at line {line_number}."
                ) from error

            if len(values) < 6:
                raise ValueError(
                    f"{path}:{line_number} contains "
                    f"{len(values)} values; expected at least six."
                )

            rows.append(
                values[-6:]
            )

    if not rows:
        raise ValueError(
            f"No numeric DCT labels found in {path}."
        )

    labels = np.asarray(
        rows,
        dtype=np.float64,
    )

    if not np.isfinite(labels).all():
        raise ValueError(
            f"DCT labels contain NaN/Inf: {path}"
        )

    return labels


def load_prediction_csv(
    path: Path,
) -> Dict[str, np.ndarray]:
    """
    Load GT and predicted translations from frame_predictions.csv.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    required = (
        "translation_gt_x",
        "translation_gt_y",
        "translation_gt_z",
        "translation_pred_x",
        "translation_pred_y",
        "translation_pred_z",
    )

    values: Dict[str, List[float]] = {
        name: []
        for name in required
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
            name
            for name in required
            if name not in reader.fieldnames
        ]

        if missing:
            raise KeyError(
                f"{path} is missing columns: {missing}"
            )

        for row in reader:
            for name in required:
                values[name].append(
                    float(row[name])
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

    return {
        "translation_gt": stack(
            "translation_gt"
        ),
        "translation_pred": stack(
            "translation_pred"
        ),
    }


# ============================================================================
# Statistical helpers
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


def linear_calibration_fit(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """
    Fit:

        pred = slope * gt + intercept
    """

    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    design = np.column_stack(
        [
            gt,
            np.ones_like(gt),
        ]
    )

    solution, _, _, _ = (
        np.linalg.lstsq(
            design,
            pred,
            rcond=None,
        )
    )

    slope = float(
        solution[0]
    )

    intercept = float(
        solution[1]
    )

    fitted = (
        slope * gt
        + intercept
    )

    residual = (
        pred - fitted
    )

    total = float(
        np.sum(
            (pred - np.mean(pred)) ** 2
        )
    )

    unexplained = float(
        np.sum(
            residual ** 2
        )
    )

    if total <= 1.0e-15:
        r_squared = float("nan")
    else:
        r_squared = float(
            1.0
            - unexplained / total
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def distribution_metrics(
    values: np.ndarray,
) -> Dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    quantiles = np.quantile(
        values,
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    return {
        "count": int(
            values.shape[0]
        ),
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(values)
        ),
        "min": float(
            np.min(values)
        ),
        "q01": float(
            quantiles[0]
        ),
        "q05": float(
            quantiles[1]
        ),
        "q10": float(
            quantiles[2]
        ),
        "q25": float(
            quantiles[3]
        ),
        "median": float(
            quantiles[4]
        ),
        "q75": float(
            quantiles[5]
        ),
        "q90": float(
            quantiles[6]
        ),
        "q95": float(
            quantiles[7]
        ),
        "q99": float(
            quantiles[8]
        ),
        "max": float(
            np.max(values)
        ),
    }


def prediction_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    error = (
        pred - gt
    )

    calibration = (
        linear_calibration_fit(
            gt,
            pred,
        )
    )

    return {
        "count": int(
            gt.shape[0]
        ),
        "gt_mean": float(
            np.mean(gt)
        ),
        "gt_std": float(
            np.std(gt)
        ),
        "pred_mean": float(
            np.mean(pred)
        ),
        "pred_std": float(
            np.std(pred)
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
            calibration["slope"]
        ),
        "calibration_intercept": (
            calibration["intercept"]
        ),
        "calibration_r_squared": (
            calibration["r_squared"]
        ),
    }


# ============================================================================
# Training distribution
# ============================================================================


def load_training_distributions(
    data_root: Path,
    sequences: Sequence[str],
) -> Tuple[
    np.ndarray,
    Dict[str, np.ndarray],
]:
    by_sequence: Dict[
        str,
        np.ndarray,
    ] = {}

    pooled = []

    for sequence in sequences:
        path = (
            data_root
            / "out_csv"
            / f"{sequence}_dct.txt"
        )

        labels = load_dct_labels(
            path
        )

        translation = (
            labels[:, :3]
        )

        by_sequence[
            sequence
        ] = translation

        pooled.append(
            translation
        )

    pooled_array = (
        np.concatenate(
            pooled,
            axis=0,
        )
    )

    return (
        pooled_array,
        by_sequence,
    )


# ============================================================================
# Frozen training regimes
# ============================================================================


def frozen_regime_masks(
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


def compute_frozen_regime_metrics(
    sequence: str,
    gt: np.ndarray,
    pred: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> List[Dict[str, float]]:
    gt_z = (
        gt[:, 2]
    )

    pred_z = (
        pred[:, 2]
    )

    masks = frozen_regime_masks(
        gt_z,
        low_threshold,
        high_threshold,
    )

    rows = []

    for regime, mask in (
        masks.items()
    ):
        count = int(
            np.count_nonzero(
                mask
            )
        )

        if count == 0:
            rows.append(
                {
                    "sequence": sequence,
                    "regime": regime,
                    "count": 0,
                    "gt_z_mean": float("nan"),
                    "pred_z_mean": float("nan"),
                    "bias_z": float("nan"),
                    "mae_z": float("nan"),
                    "rmse_z": float("nan"),
                    "corr_z": float("nan"),
                }
            )
            continue

        subset_gt = (
            gt_z[mask]
        )

        subset_pred = (
            pred_z[mask]
        )

        error = (
            subset_pred
            - subset_gt
        )

        rows.append(
            {
                "sequence": sequence,
                "regime": regime,
                "count": count,
                "gt_z_mean": float(
                    np.mean(
                        subset_gt
                    )
                ),
                "pred_z_mean": float(
                    np.mean(
                        subset_pred
                    )
                ),
                "bias_z": float(
                    np.mean(
                        error
                    )
                ),
                "mae_z": float(
                    np.mean(
                        np.abs(
                            error
                        )
                    )
                ),
                "rmse_z": float(
                    np.sqrt(
                        np.mean(
                            error ** 2
                        )
                    )
                ),
                "corr_z": (
                    safe_correlation(
                        subset_gt,
                        subset_pred,
                    )
                ),
            }
        )

    return rows


# ============================================================================
# Calibration bins
# ============================================================================


def build_training_bin_edges(
    train_z: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """
    Use quantile-spaced training bins.

    This ensures the training distribution contributes roughly equal
    support to each calibration interval.
    """

    quantiles = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    edges = np.quantile(
        train_z,
        quantiles,
    )

    edges = np.unique(
        edges
    )

    if edges.size < 3:
        raise ValueError(
            "Training forward-motion distribution "
            "does not contain enough unique values."
        )

    # Expand outer boundaries slightly so exact endpoints are included.
    epsilon = 1.0e-12

    edges[0] -= epsilon
    edges[-1] += epsilon

    return edges


def calibration_rows(
    sequence: str,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    edges: np.ndarray,
    minimum_count: int,
) -> List[Dict[str, float]]:
    rows = []

    for index in range(
        edges.shape[0] - 1
    ):
        low = float(
            edges[index]
        )

        high = float(
            edges[index + 1]
        )

        if index == (
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
            np.count_nonzero(
                mask
            )
        )

        if count == 0:
            continue

        subset_gt = (
            gt_z[mask]
        )

        subset_pred = (
            pred_z[mask]
        )

        error = (
            subset_pred
            - subset_gt
        )

        rows.append(
            {
                "sequence": sequence,
                "bin_index": index,
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "sufficient_count": (
                    count >= minimum_count
                ),
                "gt_mean": float(
                    np.mean(
                        subset_gt
                    )
                ),
                "gt_std": float(
                    np.std(
                        subset_gt
                    )
                ),
                "pred_mean": float(
                    np.mean(
                        subset_pred
                    )
                ),
                "pred_std": float(
                    np.std(
                        subset_pred
                    )
                ),
                "bias": float(
                    np.mean(
                        error
                    )
                ),
                "mae": float(
                    np.mean(
                        np.abs(
                            error
                        )
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
                        subset_gt,
                        subset_pred,
                    )
                ),
            }
        )

    return rows


# ============================================================================
# Support / distribution-shift metrics
# ============================================================================


def support_metrics(
    gt_z: np.ndarray,
    training_z: np.ndarray,
    central_low: float,
    central_high: float,
) -> Dict[str, float]:
    lower = float(
        np.quantile(
            training_z,
            central_low,
        )
    )

    upper = float(
        np.quantile(
            training_z,
            central_high,
        )
    )

    below = (
        gt_z < lower
    )

    above = (
        gt_z > upper
    )

    outside = (
        below | above
    )

    return {
        "central_training_low": lower,
        "central_training_high": upper,
        "below_training_central_range_count": int(
            np.count_nonzero(
                below
            )
        ),
        "above_training_central_range_count": int(
            np.count_nonzero(
                above
            )
        ),
        "outside_training_central_range_count": int(
            np.count_nonzero(
                outside
            )
        ),
        "outside_training_central_range_fraction": float(
            np.mean(
                outside
            )
        ),
        "below_fraction": float(
            np.mean(
                below
            )
        ),
        "above_fraction": float(
            np.mean(
                above
            )
        ),
    }


# ============================================================================
# Output helpers
# ============================================================================


def write_rows(
    path: Path,
    rows: Sequence[Mapping],
) -> None:
    rows = list(rows)

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


# ============================================================================
# Plots
# ============================================================================


def plot_training_test_histogram(
    path: Path,
    train_z: np.ndarray,
    test_gt: Mapping[
        str,
        np.ndarray,
    ],
) -> None:
    plt.figure(
        figsize=(11, 7)
    )

    plt.hist(
        train_z,
        bins=60,
        density=True,
        alpha=0.45,
        label="train 00-08",
    )

    for sequence, gt in (
        test_gt.items()
    ):
        plt.hist(
            gt[:, 2],
            bins=60,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=f"seq {sequence}",
        )

    plt.xlabel(
        "Directional forward translation t_z [m]"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Training vs unseen forward-motion distributions"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_prediction_scatter(
    path: Path,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    sequence: str,
) -> None:
    fit = linear_calibration_fit(
        gt_z,
        pred_z,
    )

    minimum = float(
        min(
            np.min(gt_z),
            np.min(pred_z),
        )
    )

    maximum = float(
        max(
            np.max(gt_z),
            np.max(pred_z),
        )
    )

    line = np.linspace(
        minimum,
        maximum,
        200,
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        gt_z,
        pred_z,
        s=8,
        alpha=0.35,
        label="frames",
    )

    plt.plot(
        line,
        line,
        linestyle="--",
        label="ideal y=x",
    )

    plt.plot(
        line,
        fit["slope"] * line
        + fit["intercept"],
        label=(
            f"fit: y={fit['slope']:.3f}x"
            f"{fit['intercept']:+.3f}"
        ),
    )

    plt.xlabel(
        "GT directional t_z [m]"
    )

    plt.ylabel(
        "Predicted directional t_z [m]"
    )

    plt.title(
        f"Sequence {sequence}: forward-motion calibration"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_residual_vs_gt(
    path: Path,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    sequence: str,
) -> None:
    residual = (
        pred_z - gt_z
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        gt_z,
        residual,
        s=8,
        alpha=0.35,
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "GT directional t_z [m]"
    )

    plt.ylabel(
        "Prediction residual pred_z - gt_z [m]"
    )

    plt.title(
        f"Sequence {sequence}: residual vs forward motion"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_calibration_curve(
    path: Path,
    rows: Sequence[Mapping],
    sequence: str,
) -> None:
    usable = [
        row
        for row in rows
        if bool(
            row["sufficient_count"]
        )
    ]

    if not usable:
        return

    gt_mean = np.asarray(
        [
            float(
                row["gt_mean"]
            )
            for row in usable
        ],
        dtype=np.float64,
    )

    pred_mean = np.asarray(
        [
            float(
                row["pred_mean"]
            )
            for row in usable
        ],
        dtype=np.float64,
    )

    minimum = float(
        min(
            np.min(gt_mean),
            np.min(pred_mean),
        )
    )

    maximum = float(
        max(
            np.max(gt_mean),
            np.max(pred_mean),
        )
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        gt_mean,
        pred_mean,
        marker="o",
        label="binned calibration",
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="ideal y=x",
    )

    plt.xlabel(
        "Mean GT t_z per training-derived bin [m]"
    )

    plt.ylabel(
        "Mean predicted t_z [m]"
    )

    plt.title(
        f"Sequence {sequence}: binned forward calibration"
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


def print_distribution_row(
    name: str,
    metrics: Mapping[
        str,
        float,
    ],
) -> None:
    print(
        f"{name:<12s}"
        f"{metrics['count']:>8d}"
        f"{metrics['mean']:>12.6f}"
        f"{metrics['std']:>12.6f}"
        f"{metrics['q01']:>12.6f}"
        f"{metrics['q25']:>12.6f}"
        f"{metrics['median']:>12.6f}"
        f"{metrics['q75']:>12.6f}"
        f"{metrics['q99']:>12.6f}"
    )


def print_prediction_metrics(
    sequence: str,
    metrics: Mapping[
        str,
        float,
    ],
) -> None:
    print(
        f"Sequence {sequence}"
    )

    print(
        f"  GT mean/std:        "
        f"{metrics['gt_mean']:.6f} / "
        f"{metrics['gt_std']:.6f}"
    )

    print(
        f"  Pred mean/std:      "
        f"{metrics['pred_mean']:.6f} / "
        f"{metrics['pred_std']:.6f}"
    )

    print(
        f"  Bias:               "
        f"{metrics['bias']:+.6f}"
    )

    print(
        f"  MAE / RMSE:         "
        f"{metrics['mae']:.6f} / "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"  Correlation:        "
        f"{metrics['correlation']:.4f}"
    )

    print(
        f"  Calibration slope:  "
        f"{metrics['calibration_slope']:.6f}"
    )

    print(
        f"  Calibration offset: "
        f"{metrics['calibration_intercept']:+.6f}"
    )

    print(
        f"  Calibration R^2:    "
        f"{metrics['calibration_r_squared']:.4f}"
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
        "=" * 104
    )
    print(
        "DeepDCT-VO A6 translation distribution-shift audit"
    )
    print(
        "=" * 104
    )
    print(
        "Training sequences:          "
        f"{','.join(args.train_sequences)}"
    )
    print(
        "Test sequences:              "
        f"{','.join(args.test_sequences)}"
    )
    print(
        f"Data root:                   {args.data_root}"
    )
    print(
        f"Calibration bins:            {args.calibration_bins}"
    )
    print(
        f"Output directory:            {args.output_dir}"
    )
    print(
        "=" * 104
    )

    # ========================================================================
    # Load training distribution.
    # ========================================================================

    (
        train_translation,
        train_by_sequence,
    ) = load_training_distributions(
        args.data_root,
        args.train_sequences,
    )

    train_z = (
        train_translation[:, 2]
    )

    train_distribution = (
        distribution_metrics(
            train_z
        )
    )

    low_threshold = float(
        np.quantile(
            train_z,
            args.low_quantile,
        )
    )

    high_threshold = float(
        np.quantile(
            train_z,
            args.high_quantile,
        )
    )

    central_low = float(
        np.quantile(
            train_z,
            args.central_range_low_quantile,
        )
    )

    central_high = float(
        np.quantile(
            train_z,
            args.central_range_high_quantile,
        )
    )

    bin_edges = (
        build_training_bin_edges(
            train_z,
            args.calibration_bins,
        )
    )

    # ========================================================================
    # Load GT distributions for 09 / 10.
    # ========================================================================

    test_gt_labels: Dict[
        str,
        np.ndarray,
    ] = {}

    for sequence in (
        args.test_sequences
    ):
        path = (
            args.data_root
            / "out_csv"
            / f"{sequence}_dct.txt"
        )

        labels = load_dct_labels(
            path
        )

        test_gt_labels[
            sequence
        ] = labels[:, :3]

    # ========================================================================
    # Distribution summary.
    # ========================================================================

    sequence_distribution_rows = []

    train_row = {
        "sequence": "train_00_08",
        **train_distribution,
    }

    sequence_distribution_rows.append(
        train_row
    )

    print()
    print(
        "=" * 104
    )
    print(
        "FORWARD-MOTION DISTRIBUTIONS"
    )
    print(
        "=" * 104
    )

    print(
        f"{'Dataset':<12}"
        f"{'N':>8}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Q01':>12}"
        f"{'Q25':>12}"
        f"{'Median':>12}"
        f"{'Q75':>12}"
        f"{'Q99':>12}"
    )

    print(
        "-" * 104
    )

    print_distribution_row(
        "train00-08",
        train_distribution,
    )

    test_distribution_metrics = {}

    for sequence, translation in (
        test_gt_labels.items()
    ):
        metrics = (
            distribution_metrics(
                translation[:, 2]
            )
        )

        test_distribution_metrics[
            sequence
        ] = metrics

        sequence_distribution_rows.append(
            {
                "sequence": sequence,
                **metrics,
            }
        )

        print_distribution_row(
            f"seq{sequence}",
            metrics,
        )

    print()
    print(
        "Frozen training motion thresholds:"
    )
    print(
        f"  low:    t_z <= {low_threshold:.6f}"
    )
    print(
        f"  medium: {low_threshold:.6f} < "
        f"t_z <= {high_threshold:.6f}"
    )
    print(
        f"  high:   t_z > {high_threshold:.6f}"
    )

    print()
    print(
        "Central training support:"
    )
    print(
        f"  q={args.central_range_low_quantile:.3f}: "
        f"{central_low:.6f}"
    )
    print(
        f"  q={args.central_range_high_quantile:.3f}: "
        f"{central_high:.6f}"
    )

    # ========================================================================
    # Per-training-sequence distributions.
    # ========================================================================

    print()
    print(
        "=" * 104
    )
    print(
        "TRAINING SEQUENCE MOTION COVERAGE"
    )
    print(
        "=" * 104
    )

    training_sequence_rows = []

    for sequence in (
        args.train_sequences
    ):
        metrics = distribution_metrics(
            train_by_sequence[
                sequence
            ][:, 2]
        )

        training_sequence_rows.append(
            {
                "sequence": sequence,
                **metrics,
            }
        )

        print(
            f"seq {sequence}: "
            f"n={metrics['count']:5d} "
            f"mean={metrics['mean']:.6f} "
            f"std={metrics['std']:.6f} "
            f"q01={metrics['q01']:.6f} "
            f"median={metrics['median']:.6f} "
            f"q99={metrics['q99']:.6f}"
        )

    # ========================================================================
    # Test support metrics.
    # ========================================================================

    support_summary = {}

    print()
    print(
        "=" * 104
    )
    print(
        "TEST COVERAGE RELATIVE TO TRAINING SUPPORT"
    )
    print(
        "=" * 104
    )

    for sequence, translation in (
        test_gt_labels.items()
    ):
        metrics = support_metrics(
            translation[:, 2],
            train_z,
            args.central_range_low_quantile,
            args.central_range_high_quantile,
        )

        support_summary[
            sequence
        ] = metrics

        print(
            f"Sequence {sequence}"
        )

        print(
            "  outside central training range: "
            f"{metrics['outside_training_central_range_count']} / "
            f"{translation.shape[0]} "
            f"({100.0 * metrics['outside_training_central_range_fraction']:.2f}%)"
        )

        print(
            "  below range: "
            f"{100.0 * metrics['below_fraction']:.2f}%"
        )

        print(
            "  above range: "
            f"{100.0 * metrics['above_fraction']:.2f}%"
        )

    # ========================================================================
    # Optional Model-T prediction analysis.
    # ========================================================================

    prediction_paths = {
        "09": args.predictions_09,
        "10": args.predictions_10,
    }

    scale_factors = {
        "09": (
            args.translation_scale_factor_09
        ),
        "10": (
            args.translation_scale_factor_10
        ),
    }

    prediction_summaries = {}
    frozen_regime_rows = []
    calibration_bin_rows = []

    for sequence in (
        "09",
        "10",
    ):
        prediction_path = (
            prediction_paths[
                sequence
            ]
        )

        if prediction_path is None:
            continue

        predictions = (
            load_prediction_csv(
                prediction_path
            )
        )

        gt = predictions[
            "translation_gt"
        ]

        pred = (
            predictions[
                "translation_pred"
            ]
            * scale_factors[
                sequence
            ]
        )

        if sequence in (
            test_gt_labels
        ):
            expected = (
                test_gt_labels[
                    sequence
                ]
            )

            if (
                expected.shape
                != gt.shape
            ):
                raise ValueError(
                    f"Prediction/label count mismatch "
                    f"for sequence {sequence}: "
                    f"{gt.shape} vs {expected.shape}."
                )

            consistency_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            gt
                            - expected
                        )
                        ** 2
                    )
                )
            )

            if consistency_rmse > 1.0e-5:
                raise ValueError(
                    f"Prediction CSV GT does not match "
                    f"{sequence}_dct.txt; RMSE="
                    f"{consistency_rmse:.9g}"
                )

        metrics = prediction_metrics(
            gt[:, 2],
            pred[:, 2],
        )

        prediction_summaries[
            sequence
        ] = {
            **metrics,
            "scale_factor": float(
                scale_factors[
                    sequence
                ]
            ),
        }

        print()
        print(
            "=" * 104
        )
        print(
            f"MODEL-T FORWARD CALIBRATION — SEQUENCE {sequence}"
        )
        print(
            "=" * 104
        )

        print_prediction_metrics(
            sequence,
            metrics,
        )

        # ------------------------------------------------------------
        # Frozen train-derived regimes.
        # ------------------------------------------------------------

        regime_rows = (
            compute_frozen_regime_metrics(
                sequence,
                gt,
                pred,
                low_threshold,
                high_threshold,
            )
        )

        frozen_regime_rows.extend(
            regime_rows
        )

        print()
        print(
            "Frozen train-derived motion regimes:"
        )

        for row in regime_rows:
            print(
                f"  {row['regime']:<6s} "
                f"n={row['count']:4d} "
                f"gt_mean={row['gt_z_mean']:.6f} "
                f"pred_mean={row['pred_z_mean']:.6f} "
                f"bias={row['bias_z']:+.6f} "
                f"rmse={row['rmse_z']:.6f} "
                f"corr={row['corr_z']:.4f}"
            )

        # ------------------------------------------------------------
        # Training-derived calibration bins.
        # ------------------------------------------------------------

        bin_rows = calibration_rows(
            sequence,
            gt[:, 2],
            pred[:, 2],
            bin_edges,
            args.minimum_bin_count,
        )

        calibration_bin_rows.extend(
            bin_rows
        )

        print()
        print(
            "Training-derived calibration bins:"
        )

        for row in bin_rows:
            marker = (
                " "
                if row[
                    "sufficient_count"
                ]
                else "*"
            )

            print(
                f"{marker} bin {row['bin_index']:02d} "
                f"[{row['bin_low']:.3f}, "
                f"{row['bin_high']:.3f}) "
                f"n={row['count']:4d} "
                f"gt={row['gt_mean']:.4f} "
                f"pred={row['pred_mean']:.4f} "
                f"bias={row['bias']:+.4f}"
            )

        # ------------------------------------------------------------
        # Plots.
        # ------------------------------------------------------------

        plot_prediction_scatter(
            args.output_dir
            / f"prediction_scatter_seq{sequence}.png",
            gt[:, 2],
            pred[:, 2],
            sequence,
        )

        plot_residual_vs_gt(
            args.output_dir
            / f"residual_vs_gt_seq{sequence}.png",
            gt[:, 2],
            pred[:, 2],
            sequence,
        )

        plot_calibration_curve(
            args.output_dir
            / f"calibration_curve_seq{sequence}.png",
            bin_rows,
            sequence,
        )

    # ========================================================================
    # Save training/test distribution plot.
    # ========================================================================

    plot_training_test_histogram(
        args.output_dir
        / "training_test_tz_histogram.png",
        train_z,
        test_gt_labels,
    )

    # ========================================================================
    # Quantile output.
    # ========================================================================

    quantile_rows = []

    quantile_grid = (
        0.00,
        0.01,
        0.05,
        0.10,
        0.20,
        0.25,
        1.0 / 3.0,
        0.50,
        2.0 / 3.0,
        0.75,
        0.80,
        0.90,
        0.95,
        0.99,
        1.00,
    )

    for quantile in (
        quantile_grid
    ):
        quantile_rows.append(
            {
                "quantile": (
                    quantile
                ),
                "train_tz": float(
                    np.quantile(
                        train_z,
                        quantile,
                    )
                ),
            }
        )

    # ========================================================================
    # Save CSV outputs.
    # ========================================================================

    write_rows(
        args.output_dir
        / "sequence_distribution_metrics.csv",
        sequence_distribution_rows,
    )

    write_rows(
        args.output_dir
        / "training_sequence_distribution_metrics.csv",
        training_sequence_rows,
    )

    if frozen_regime_rows:
        write_rows(
            args.output_dir
            / "frozen_regime_metrics.csv",
            frozen_regime_rows,
        )

    if calibration_bin_rows:
        write_rows(
            args.output_dir
            / "calibration_bins.csv",
            calibration_bin_rows,
        )

    write_rows(
        args.output_dir
        / "training_tz_quantiles.csv",
        quantile_rows,
    )

    # ========================================================================
    # Interpretation helper.
    # ========================================================================

    print()
    print(
        "=" * 104
    )
    print(
        "INTERPRETATION GUIDE"
    )
    print(
        "=" * 104
    )

    print(
        "Distribution-shift explanation is supported if:"
    )
    print(
        "  - a large fraction of seq09/10 lies outside "
        "the central training t_z range, and"
    )
    print(
        "  - the largest errors occur primarily in those "
        "poorly represented regions."
    )

    print()
    print(
        "Dynamic-range compression is supported if:"
    )
    print(
        "  - calibration slope is substantially below 1, and/or"
    )
    print(
        "  - low t_z is overpredicted while high t_z is "
        "underpredicted, and"
    )
    print(
        "  - this occurs inside well-supported training bins."
    )

    print()
    print(
        "Sequence-specific generalization is supported if:"
    )
    print(
        "  - seq09 and seq10 occupy similar training-supported "
        "regions but show strongly different conditional biases."
    )

    # ========================================================================
    # JSON summary.
    # ========================================================================

    summary = {
        "train_sequences": (
            args.train_sequences
        ),
        "test_sequences": (
            args.test_sequences
        ),
        "training_distribution": (
            train_distribution
        ),
        "training_motion_thresholds": {
            "low_quantile": (
                args.low_quantile
            ),
            "high_quantile": (
                args.high_quantile
            ),
            "low_threshold": (
                low_threshold
            ),
            "high_threshold": (
                high_threshold
            ),
        },
        "central_training_support": {
            "low_quantile": (
                args.central_range_low_quantile
            ),
            "high_quantile": (
                args.central_range_high_quantile
            ),
            "low_value": (
                central_low
            ),
            "high_value": (
                central_high
            ),
        },
        "test_distributions": (
            test_distribution_metrics
        ),
        "support_metrics": (
            support_summary
        ),
        "prediction_calibration": (
            prediction_summaries
        ),
        "training_sequence_distributions": {
            row["sequence"]: row
            for row in training_sequence_rows
        },
        "frozen_regime_metrics": (
            frozen_regime_rows
        ),
        "calibration_bins": (
            calibration_bin_rows
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
        "=" * 104
    )
    print(
        "AUDIT COMPLETE"
    )
    print(
        "=" * 104
    )
    print(
        f"Outputs: {args.output_dir}"
    )
    print(
        "=" * 104
    )


if __name__ == "__main__":
    main()