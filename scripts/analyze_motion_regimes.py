#!/usr/bin/env python3
"""
Analyze DeepDCT-VO translation prediction quality conditioned on motion regime.

The script compares one or more frame_predictions.csv files while assigning
motion regimes exclusively from ground-truth motion. This ensures that every
experiment is evaluated on exactly the same frames in every regime.

Primary use case
----------------

python scripts/analyze_motion_regimes.py \
    --input baseline=experiments/baseline_identity_output/evaluation_seq10/frame_predictions.csv \
    --input head_ft=experiments/baseline_identity_output_translation_head_ft/evaluation_seq10/frame_predictions.csv \
    --output-dir experiments/motion_regime_analysis/sequence_10

Outputs
-------
<output-dir>/
    summary.json
    regime_definitions.json
    frame_regime_assignments.csv
    regime_metrics.csv
    regime_comparison.csv
    plots/
        vector_rmse_by_translation_magnitude.png
        z_rmse_by_forward_motion.png
        z_bias_by_forward_motion.png
        z_correlation_by_forward_motion.png
        vector_rmse_by_turn_intensity.png
        vector_rmse_by_motion_composition.png
        improvement_by_forward_motion.png
        improvement_by_translation_magnitude.png

Regime families
---------------
1. translation_magnitude
       low / medium / high
       based on quantiles of ||t_gt||

2. forward_motion
       low / medium / high
       based on quantiles of |z_gt|

3. turn_intensity
       straight / moderate_turn / strong_turn
       based on quantiles of ||rotation_gt||

4. motion_composition
       forward_dominant / lateral_dominant / mixed
       based on relative forward and lateral translation magnitudes

Notes
-----
* Regimes are defined using ground truth only.
* Quantile thresholds are computed from the first/reference experiment.
* All experiment CSVs must contain matching ground-truth motion.
* Correlation is reported as NaN when a regime does not contain enough
  variation for a meaningful Pearson correlation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AXES = ("x", "y", "z")

REGIME_FAMILIES = (
    "translation_magnitude",
    "forward_motion",
    "turn_intensity",
    "motion_composition",
)

QUANTILE_REGIME_ORDER = (
    "low",
    "medium",
    "high",
)

TURN_REGIME_ORDER = (
    "straight",
    "moderate_turn",
    "strong_turn",
)

COMPOSITION_REGIME_ORDER = (
    "forward_dominant",
    "mixed",
    "lateral_dominant",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_named_input(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must use NAME=PATH syntax. "
            "Example: baseline=experiments/baseline/frame_predictions.csv"
        )

    name, path_text = value.split("=", 1)

    name = name.strip()
    path_text = path_text.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            "Experiment name before '=' cannot be empty."
        )

    if not path_text:
        raise argparse.ArgumentTypeError(
            "CSV path after '=' cannot be empty."
        )

    return name, Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DeepDCT-VO translation errors conditioned on "
            "ground-truth motion regimes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        action="append",
        type=parse_named_input,
        required=True,
        metavar="NAME=CSV",
        help=(
            "Named frame_predictions.csv input. May be specified multiple "
            "times. The first input defines the ground-truth regime bins."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for tables, JSON summaries, and plots.",
    )

    parser.add_argument(
        "--low-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Boundary between low and medium motion regimes.",
    )

    parser.add_argument(
        "--high-quantile",
        type=float,
        default=2.0 / 3.0,
        help="Boundary between medium and high motion regimes.",
    )

    parser.add_argument(
        "--composition-ratio",
        type=float,
        default=1.5,
        help=(
            "Dominance ratio for motion composition. Forward motion is "
            "forward-dominant when |z| >= ratio*lateral magnitude; lateral "
            "motion is lateral-dominant when lateral >= ratio*|z|."
        ),
    )

    parser.add_argument(
        "--minimum-correlation-samples",
        type=int,
        default=3,
        help="Minimum regime size required for Pearson correlation.",
    )

    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Experiment used as comparison reference. Defaults to the first "
            "--input experiment."
        ),
    )

    parser.add_argument(
        "--comparison",
        type=str,
        default=None,
        help=(
            "Experiment compared against --reference in improvement tables. "
            "Defaults to the second input when exactly two or more are given."
        ),
    )

    parser.add_argument(
        "--gt-tolerance",
        type=float,
        default=1.0e-8,
        help="Maximum allowed GT difference between experiment CSVs.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.input) < 1:
        raise ValueError("At least one --input is required.")

    names = [name for name, _ in args.input]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Experiment names must be unique. Received: {names}"
        )

    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError(
            "Require 0 < --low-quantile < --high-quantile < 1."
        )

    if args.composition_ratio <= 1.0:
        raise ValueError(
            "--composition-ratio must be greater than 1.0."
        )

    if args.minimum_correlation_samples < 2:
        raise ValueError(
            "--minimum-correlation-samples must be at least 2."
        )

    if args.gt_tolerance < 0:
        raise ValueError("--gt-tolerance cannot be negative.")

    if args.reference is not None and args.reference not in names:
        raise ValueError(
            f"--reference {args.reference!r} is not one of {names}."
        )

    if args.comparison is not None and args.comparison not in names:
        raise ValueError(
            f"--comparison {args.comparison!r} is not one of {names}."
        )


# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------


def find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    *,
    description: str,
) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        f"Could not find {description}. Tried columns:\n  "
        + "\n  ".join(candidates)
        + "\nAvailable columns:\n  "
        + "\n  ".join(dataframe.columns)
    )


def resolve_columns(
    dataframe: pd.DataFrame,
) -> Dict[str, str]:
    columns: Dict[str, str] = {}

    for axis in AXES:
        columns[f"translation_gt_{axis}"] = find_column(
            dataframe,
            (
                f"translation_gt_{axis}",
                f"gt_translation_{axis}",
                f"translation_{axis}_gt",
            ),
            description=f"ground-truth translation {axis}",
        )

        columns[f"translation_pred_{axis}"] = find_column(
            dataframe,
            (
                f"translation_pred_{axis}",
                f"pred_translation_{axis}",
                f"translation_{axis}_pred",
            ),
            description=f"predicted translation {axis}",
        )

        columns[f"rotation_gt_{axis}"] = find_column(
            dataframe,
            (
                f"rotation_gt_{axis}",
                f"gt_rotation_{axis}",
                f"rotation_{axis}_gt",
            ),
            description=f"ground-truth rotation {axis}",
        )

    return columns


# ---------------------------------------------------------------------------
# Data loading / ground-truth validation
# ---------------------------------------------------------------------------


def load_experiments(
    inputs: Sequence[Tuple[str, Path]],
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[str, str]],
]:
    experiments: Dict[str, pd.DataFrame] = {}
    column_maps: Dict[str, Dict[str, str]] = {}

    for name, path in inputs:
        path = path.expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Input CSV does not exist: {path}"
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(
                f"Input CSV is empty: {path}"
            )

        experiments[name] = dataframe
        column_maps[name] = resolve_columns(
            dataframe
        )

        print(
            f"Loaded {name:<20} "
            f"samples={len(dataframe):5d} "
            f"path={path}"
        )

    return experiments, column_maps


def extract_ground_truth(
    dataframe: pd.DataFrame,
    columns: Mapping[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    translation = np.column_stack(
        [
            dataframe[
                columns[f"translation_gt_{axis}"]
            ].to_numpy(dtype=np.float64)
            for axis in AXES
        ]
    )

    rotation = np.column_stack(
        [
            dataframe[
                columns[f"rotation_gt_{axis}"]
            ].to_numpy(dtype=np.float64)
            for axis in AXES
        ]
    )

    return translation, rotation


def validate_matching_ground_truth(
    experiments: Mapping[str, pd.DataFrame],
    column_maps: Mapping[str, Mapping[str, str]],
    *,
    reference_name: str,
    tolerance: float,
) -> None:
    reference_df = experiments[reference_name]
    reference_columns = column_maps[reference_name]

    reference_t, reference_r = extract_ground_truth(
        reference_df,
        reference_columns,
    )

    for name, dataframe in experiments.items():
        if name == reference_name:
            continue

        if len(dataframe) != len(reference_df):
            raise ValueError(
                f"Sample-count mismatch: reference {reference_name} has "
                f"{len(reference_df)} rows, but {name} has "
                f"{len(dataframe)}."
            )

        translation, rotation = extract_ground_truth(
            dataframe,
            column_maps[name],
        )

        t_diff = float(
            np.max(
                np.abs(
                    translation - reference_t
                )
            )
        )

        r_diff = float(
            np.max(
                np.abs(
                    rotation - reference_r
                )
            )
        )

        if t_diff > tolerance or r_diff > tolerance:
            raise ValueError(
                "Ground-truth mismatch between experiments.\n"
                f"Reference: {reference_name}\n"
                f"Compared:  {name}\n"
                f"Max translation GT difference: {t_diff:.12e}\n"
                f"Max rotation GT difference:    {r_diff:.12e}\n"
                f"Tolerance:                     {tolerance:.12e}"
            )

        print(
            f"Ground-truth audit {name:<20} PASS "
            f"translation_diff={t_diff:.3e} "
            f"rotation_diff={r_diff:.3e}"
        )


# ---------------------------------------------------------------------------
# Regime construction
# ---------------------------------------------------------------------------


def quantile_labels(
    values: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    labels = np.full(
        values.shape,
        "medium",
        dtype=object,
    )

    labels[values <= low_threshold] = "low"
    labels[values > high_threshold] = "high"

    return labels


def build_regime_assignments(
    translation_gt: np.ndarray,
    rotation_gt: np.ndarray,
    *,
    low_quantile: float,
    high_quantile: float,
    composition_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tx = translation_gt[:, 0]
    ty = translation_gt[:, 1]
    tz = translation_gt[:, 2]

    translation_magnitude = np.linalg.norm(
        translation_gt,
        axis=1,
    )

    forward_magnitude = np.abs(tz)

    lateral_magnitude = np.sqrt(
        tx ** 2 + ty ** 2
    )

    rotation_magnitude = np.linalg.norm(
        rotation_gt,
        axis=1,
    )

    translation_low = float(
        np.quantile(
            translation_magnitude,
            low_quantile,
        )
    )

    translation_high = float(
        np.quantile(
            translation_magnitude,
            high_quantile,
        )
    )

    forward_low = float(
        np.quantile(
            forward_magnitude,
            low_quantile,
        )
    )

    forward_high = float(
        np.quantile(
            forward_magnitude,
            high_quantile,
        )
    )

    rotation_low = float(
        np.quantile(
            rotation_magnitude,
            low_quantile,
        )
    )

    rotation_high = float(
        np.quantile(
            rotation_magnitude,
            high_quantile,
        )
    )

    magnitude_labels = quantile_labels(
        translation_magnitude,
        translation_low,
        translation_high,
    )

    forward_labels = quantile_labels(
        forward_magnitude,
        forward_low,
        forward_high,
    )

    raw_turn_labels = quantile_labels(
        rotation_magnitude,
        rotation_low,
        rotation_high,
    )

    turn_labels = np.empty(
        len(raw_turn_labels),
        dtype=object,
    )

    turn_labels[
        raw_turn_labels == "low"
    ] = "straight"

    turn_labels[
        raw_turn_labels == "medium"
    ] = "moderate_turn"

    turn_labels[
        raw_turn_labels == "high"
    ] = "strong_turn"

    composition_labels = np.full(
        len(translation_gt),
        "mixed",
        dtype=object,
    )

    epsilon = 1.0e-12

    forward_dominant = (
        forward_magnitude
        >= composition_ratio
        * np.maximum(
            lateral_magnitude,
            epsilon,
        )
    )

    lateral_dominant = (
        lateral_magnitude
        >= composition_ratio
        * np.maximum(
            forward_magnitude,
            epsilon,
        )
    )

    composition_labels[
        forward_dominant
    ] = "forward_dominant"

    composition_labels[
        lateral_dominant
    ] = "lateral_dominant"

    assignments = pd.DataFrame(
        {
            "frame_index": np.arange(
                len(translation_gt),
                dtype=np.int64,
            ),
            "translation_gt_x": tx,
            "translation_gt_y": ty,
            "translation_gt_z": tz,
            "translation_magnitude": (
                translation_magnitude
            ),
            "forward_magnitude": (
                forward_magnitude
            ),
            "lateral_magnitude": (
                lateral_magnitude
            ),
            "rotation_magnitude": (
                rotation_magnitude
            ),
            "translation_magnitude_regime": (
                magnitude_labels
            ),
            "forward_motion_regime": (
                forward_labels
            ),
            "turn_intensity_regime": (
                turn_labels
            ),
            "motion_composition_regime": (
                composition_labels
            ),
        }
    )

    definitions: Dict[str, Any] = {
        "quantiles": {
            "low_quantile": low_quantile,
            "high_quantile": high_quantile,
        },
        "translation_magnitude": {
            "quantity": "||translation_gt||_2",
            "low_upper": translation_low,
            "medium_upper": translation_high,
            "labels": list(
                QUANTILE_REGIME_ORDER
            ),
        },
        "forward_motion": {
            "quantity": "|translation_gt_z|",
            "low_upper": forward_low,
            "medium_upper": forward_high,
            "labels": list(
                QUANTILE_REGIME_ORDER
            ),
        },
        "turn_intensity": {
            "quantity": "||rotation_gt||_2",
            "straight_upper": rotation_low,
            "moderate_turn_upper": rotation_high,
            "labels": list(
                TURN_REGIME_ORDER
            ),
        },
        "motion_composition": {
            "forward_quantity": "|translation_gt_z|",
            "lateral_quantity": (
                "sqrt(translation_gt_x^2 + "
                "translation_gt_y^2)"
            ),
            "dominance_ratio": composition_ratio,
            "forward_dominant": (
                "|z| >= ratio * lateral"
            ),
            "lateral_dominant": (
                "lateral >= ratio * |z|"
            ),
            "mixed": "neither dominance condition",
            "labels": list(
                COMPOSITION_REGIME_ORDER
            ),
        },
    }

    return assignments, definitions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def safe_correlation(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    minimum_samples: int,
) -> float:
    valid = (
        np.isfinite(target)
        & np.isfinite(prediction)
    )

    target = target[valid]
    prediction = prediction[valid]

    if len(target) < minimum_samples:
        return float("nan")

    if np.std(target) <= 1.0e-12:
        return float("nan")

    if np.std(prediction) <= 1.0e-12:
        return float("nan")

    return float(
        np.corrcoef(
            target,
            prediction,
        )[0, 1]
    )


def safe_linear_fit(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Tuple[float, float]:
    valid = (
        np.isfinite(target)
        & np.isfinite(prediction)
    )

    x = target[valid]
    y = prediction[valid]

    if len(x) < 2:
        return float("nan"), float("nan")

    if np.std(x) <= 1.0e-12:
        return float("nan"), float("nan")

    slope, intercept = np.polyfit(
        x,
        y,
        deg=1,
    )

    return float(slope), float(intercept)


def compute_metrics(
    translation_gt: np.ndarray,
    translation_pred: np.ndarray,
    *,
    minimum_correlation_samples: int,
) -> Dict[str, float]:
    error = (
        translation_pred
        - translation_gt
    )

    squared_error = error ** 2

    vector_error = np.linalg.norm(
        error,
        axis=1,
    )

    # sqrt(mean(||error||^2))
    vector_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    squared_error,
                    axis=1,
                )
            )
        )
    )

    metrics: Dict[str, float] = {
        "samples": int(
            len(translation_gt)
        ),
        "vector_rmse": vector_rmse,
        "mean_translation_l2": float(
            np.mean(
                vector_error
            )
        ),
        "median_translation_l2": float(
            np.median(
                vector_error
            )
        ),
        "max_translation_l2": float(
            np.max(
                vector_error
            )
        ),
    }

    for axis_index, axis in enumerate(
        AXES
    ):
        gt = translation_gt[
            :,
            axis_index,
        ]

        pred = translation_pred[
            :,
            axis_index,
        ]

        axis_error = (
            pred - gt
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    axis_error ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    axis_error
                )
            )
        )

        bias = float(
            np.mean(
                axis_error
            )
        )

        correlation = safe_correlation(
            gt,
            pred,
            minimum_samples=(
                minimum_correlation_samples
            ),
        )

        slope, intercept = safe_linear_fit(
            gt,
            pred,
        )

        metrics[
            f"{axis}_rmse"
        ] = rmse

        metrics[
            f"{axis}_mae"
        ] = mae

        metrics[
            f"{axis}_bias"
        ] = bias

        metrics[
            f"{axis}_correlation"
        ] = correlation

        metrics[
            f"{axis}_slope"
        ] = slope

        metrics[
            f"{axis}_intercept"
        ] = intercept

    return metrics


# ---------------------------------------------------------------------------
# Per-regime analysis
# ---------------------------------------------------------------------------


def prediction_array(
    dataframe: pd.DataFrame,
    columns: Mapping[str, str],
) -> np.ndarray:
    return np.column_stack(
        [
            dataframe[
                columns[
                    f"translation_pred_{axis}"
                ]
            ].to_numpy(
                dtype=np.float64
            )
            for axis in AXES
        ]
    )


def regime_column_name(
    family: str,
) -> str:
    mapping = {
        "translation_magnitude": (
            "translation_magnitude_regime"
        ),
        "forward_motion": (
            "forward_motion_regime"
        ),
        "turn_intensity": (
            "turn_intensity_regime"
        ),
        "motion_composition": (
            "motion_composition_regime"
        ),
    }

    return mapping[family]


def ordered_regimes(
    family: str,
) -> Tuple[str, ...]:
    if family in (
        "translation_magnitude",
        "forward_motion",
    ):
        return QUANTILE_REGIME_ORDER

    if family == "turn_intensity":
        return TURN_REGIME_ORDER

    if family == "motion_composition":
        return COMPOSITION_REGIME_ORDER

    raise KeyError(
        f"Unknown regime family: {family}"
    )


def compute_regime_metrics(
    experiments: Mapping[str, pd.DataFrame],
    column_maps: Mapping[str, Mapping[str, str]],
    assignments: pd.DataFrame,
    *,
    minimum_correlation_samples: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    reference_name = next(
        iter(experiments)
    )

    translation_gt, _ = extract_ground_truth(
        experiments[reference_name],
        column_maps[reference_name],
    )

    for experiment_name, dataframe in (
        experiments.items()
    ):
        translation_pred = prediction_array(
            dataframe,
            column_maps[experiment_name],
        )

        # Overall row.
        overall_metrics = compute_metrics(
            translation_gt,
            translation_pred,
            minimum_correlation_samples=(
                minimum_correlation_samples
            ),
        )

        rows.append(
            {
                "experiment": experiment_name,
                "regime_family": "overall",
                "regime": "all",
                **overall_metrics,
            }
        )

        for family in REGIME_FAMILIES:
            column = regime_column_name(
                family
            )

            for regime in ordered_regimes(
                family
            ):
                mask = (
                    assignments[
                        column
                    ].to_numpy()
                    == regime
                )

                sample_count = int(
                    np.sum(mask)
                )

                if sample_count == 0:
                    continue

                metrics = compute_metrics(
                    translation_gt[mask],
                    translation_pred[mask],
                    minimum_correlation_samples=(
                        minimum_correlation_samples
                    ),
                )

                rows.append(
                    {
                        "experiment": (
                            experiment_name
                        ),
                        "regime_family": family,
                        "regime": regime,
                        **metrics,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def percent_improvement(
    reference_value: float,
    comparison_value: float,
) -> float:
    if not np.isfinite(
        reference_value
    ):
        return float("nan")

    if abs(reference_value) <= 1.0e-15:
        return float("nan")

    return float(
        100.0
        * (
            reference_value
            - comparison_value
        )
        / reference_value
    )


def build_comparison_table(
    metrics: pd.DataFrame,
    *,
    reference_name: str,
    comparison_name: str,
) -> pd.DataFrame:
    reference = metrics[
        metrics["experiment"]
        == reference_name
    ].copy()

    comparison = metrics[
        metrics["experiment"]
        == comparison_name
    ].copy()

    keys = [
        "regime_family",
        "regime",
    ]

    merged = reference.merge(
        comparison,
        on=keys,
        how="inner",
        suffixes=(
            "_reference",
            "_comparison",
        ),
    )

    rows: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        result: Dict[str, Any] = {
            "reference_experiment": (
                reference_name
            ),
            "comparison_experiment": (
                comparison_name
            ),
            "regime_family": row[
                "regime_family"
            ],
            "regime": row["regime"],
            "samples": int(
                row["samples_reference"]
            ),
        }

        error_metrics = (
            "vector_rmse",
            "mean_translation_l2",
            "x_rmse",
            "y_rmse",
            "z_rmse",
            "x_mae",
            "y_mae",
            "z_mae",
        )

        for metric in error_metrics:
            reference_value = float(
                row[
                    f"{metric}_reference"
                ]
            )

            comparison_value = float(
                row[
                    f"{metric}_comparison"
                ]
            )

            result[
                f"{metric}_reference"
            ] = reference_value

            result[
                f"{metric}_comparison"
            ] = comparison_value

            result[
                f"{metric}_difference"
            ] = (
                comparison_value
                - reference_value
            )

            result[
                f"{metric}_improvement_percent"
            ] = percent_improvement(
                reference_value,
                comparison_value,
            )

        for axis in AXES:
            for metric in (
                "bias",
                "correlation",
                "slope",
            ):
                key = (
                    f"{axis}_{metric}"
                )

                result[
                    f"{key}_reference"
                ] = row[
                    f"{key}_reference"
                ]

                result[
                    f"{key}_comparison"
                ] = row[
                    f"{key}_comparison"
                ]

                result[
                    f"{key}_difference"
                ] = (
                    row[
                        f"{key}_comparison"
                    ]
                    - row[
                        f"{key}_reference"
                    ]
                )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_metric_by_regime(
    metrics: pd.DataFrame,
    *,
    family: str,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    subset = metrics[
        metrics["regime_family"]
        == family
    ]

    regimes = list(
        ordered_regimes(
            family
        )
    )

    experiments = list(
        subset["experiment"].unique()
    )

    if subset.empty:
        return

    x = np.arange(
        len(regimes),
        dtype=np.float64,
    )

    width = (
        0.8
        / max(
            len(experiments),
            1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for experiment_index, experiment in (
        enumerate(experiments)
    ):
        experiment_subset = subset[
            subset["experiment"]
            == experiment
        ]

        values: List[float] = []

        for regime in regimes:
            row = experiment_subset[
                experiment_subset[
                    "regime"
                ]
                == regime
            ]

            if row.empty:
                values.append(
                    float("nan")
                )
            else:
                values.append(
                    float(
                        row.iloc[0][
                            metric
                        ]
                    )
                )

        offset = (
            experiment_index
            - (
                len(experiments)
                - 1
            )
            / 2.0
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=experiment,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        regimes
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_improvement(
    comparison: pd.DataFrame,
    *,
    family: str,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    subset = comparison[
        comparison[
            "regime_family"
        ]
        == family
    ]

    if subset.empty:
        return

    regimes = list(
        ordered_regimes(
            family
        )
    )

    values: List[float] = []

    improvement_column = (
        f"{metric}_improvement_percent"
    )

    for regime in regimes:
        row = subset[
            subset["regime"]
            == regime
        ]

        if row.empty:
            values.append(
                float("nan")
            )
        else:
            values.append(
                float(
                    row.iloc[0][
                        improvement_column
                    ]
                )
            )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = np.arange(
        len(regimes)
    )

    ax.bar(
        x,
        values,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        regimes
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def create_plots(
    metrics: pd.DataFrame,
    comparison: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    plot_dir = (
        output_dir / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_metric_by_regime(
        metrics,
        family="translation_magnitude",
        metric="vector_rmse",
        ylabel="Translation vector RMSE",
        title=(
            "Translation Error by "
            "Ground-Truth Motion Magnitude"
        ),
        output_path=(
            plot_dir
            / "vector_rmse_by_translation_magnitude.png"
        ),
    )

    plot_metric_by_regime(
        metrics,
        family="forward_motion",
        metric="z_rmse",
        ylabel="z RMSE",
        title=(
            "Forward Translation Error "
            "by Ground-Truth Forward Motion"
        ),
        output_path=(
            plot_dir
            / "z_rmse_by_forward_motion.png"
        ),
    )

    plot_metric_by_regime(
        metrics,
        family="forward_motion",
        metric="z_bias",
        ylabel="z prediction bias",
        title=(
            "Forward Translation Bias "
            "by Ground-Truth Forward Motion"
        ),
        output_path=(
            plot_dir
            / "z_bias_by_forward_motion.png"
        ),
    )

    plot_metric_by_regime(
        metrics,
        family="forward_motion",
        metric="z_correlation",
        ylabel="Pearson correlation",
        title=(
            "Forward Translation Correlation "
            "by Ground-Truth Forward Motion"
        ),
        output_path=(
            plot_dir
            / "z_correlation_by_forward_motion.png"
        ),
    )

    plot_metric_by_regime(
        metrics,
        family="turn_intensity",
        metric="vector_rmse",
        ylabel="Translation vector RMSE",
        title=(
            "Translation Error "
            "by Turn Intensity"
        ),
        output_path=(
            plot_dir
            / "vector_rmse_by_turn_intensity.png"
        ),
    )

    plot_metric_by_regime(
        metrics,
        family="motion_composition",
        metric="vector_rmse",
        ylabel="Translation vector RMSE",
        title=(
            "Translation Error "
            "by Motion Composition"
        ),
        output_path=(
            plot_dir
            / "vector_rmse_by_motion_composition.png"
        ),
    )

    if comparison is not None:
        plot_improvement(
            comparison,
            family="forward_motion",
            metric="z_rmse",
            ylabel="z RMSE improvement (%)",
            title=(
                "Head Fine-Tuning Improvement "
                "by Forward-Motion Regime"
            ),
            output_path=(
                plot_dir
                / "improvement_by_forward_motion.png"
            ),
        )

        plot_improvement(
            comparison,
            family="translation_magnitude",
            metric="vector_rmse",
            ylabel=(
                "Vector RMSE improvement (%)"
            ),
            title=(
                "Head Fine-Tuning Improvement "
                "by Translation-Magnitude Regime"
            ),
            output_path=(
                plot_dir
                / "improvement_by_translation_magnitude.png"
            ),
        )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def json_safe_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        value = float(value)

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            json_safe_value(
                item
            )
            for item in value.tolist()
        ]

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe_value(
                item
            )
            for item in value
        ]

    return value


def save_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe_value(
                data
            ),
            handle,
            indent=2,
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def print_regime_counts(
    assignments: pd.DataFrame,
) -> None:
    print()
    print("=" * 88)
    print("Motion-regime distribution")
    print("=" * 88)

    for family in REGIME_FAMILIES:
        column = regime_column_name(
            family
        )

        print(
            f"\n{family}"
        )

        counts = assignments[
            column
        ].value_counts()

        for regime in ordered_regimes(
            family
        ):
            print(
                f"  {regime:<20} "
                f"{int(counts.get(regime, 0)):5d}"
            )

    print("=" * 88)


def print_overall_metrics(
    metrics: pd.DataFrame,
) -> None:
    print()
    print("=" * 88)
    print("Overall translation metrics")
    print("=" * 88)

    subset = metrics[
        metrics["regime_family"]
        == "overall"
    ]

    for _, row in subset.iterrows():
        print(
            f"{row['experiment']:<20} "
            f"vector_RMSE={row['vector_rmse']:.6f} "
            f"x={row['x_rmse']:.6f} "
            f"y={row['y_rmse']:.6f} "
            f"z={row['z_rmse']:.6f} "
            f"z_bias={row['z_bias']:+.6f} "
            f"z_corr={row['z_correlation']:.4f}"
        )

    print("=" * 88)


def print_forward_comparison(
    comparison: pd.DataFrame,
) -> None:
    subset = comparison[
        comparison["regime_family"]
        == "forward_motion"
    ]

    if subset.empty:
        return

    print()
    print("=" * 88)
    print(
        "Forward-motion conditioned comparison"
    )
    print("=" * 88)

    print(
        f"{'Regime':<16}"
        f"{'Samples':>10}"
        f"{'Ref z RMSE':>16}"
        f"{'Cmp z RMSE':>16}"
        f"{'Improvement':>16}"
    )

    print("-" * 88)

    for regime in QUANTILE_REGIME_ORDER:
        row = subset[
            subset["regime"]
            == regime
        ]

        if row.empty:
            continue

        item = row.iloc[0]

        print(
            f"{regime:<16}"
            f"{int(item['samples']):>10d}"
            f"{item['z_rmse_reference']:>16.6f}"
            f"{item['z_rmse_comparison']:>16.6f}"
            f"{item['z_rmse_improvement_percent']:>15.2f}%"
        )

    print("=" * 88)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments, column_maps = (
        load_experiments(
            args.input
        )
    )

    experiment_names = list(
        experiments.keys()
    )

    reference_name = (
        args.reference
        if args.reference is not None
        else experiment_names[0]
    )

    comparison_name: Optional[str]

    if args.comparison is not None:
        comparison_name = (
            args.comparison
        )
    elif len(experiment_names) >= 2:
        comparison_name = (
            experiment_names[1]
        )
    else:
        comparison_name = None

    print()
    print("=" * 88)
    print("Motion-Regime Conditioning Analysis")
    print("=" * 88)
    print(
        f"Reference experiment:  {reference_name}"
    )
    print(
        f"Comparison experiment: {comparison_name}"
    )
    print(
        f"Output directory:      {output_dir}"
    )
    print("=" * 88)

    validate_matching_ground_truth(
        experiments,
        column_maps,
        reference_name=reference_name,
        tolerance=args.gt_tolerance,
    )

    reference_df = experiments[
        reference_name
    ]

    reference_columns = column_maps[
        reference_name
    ]

    translation_gt, rotation_gt = (
        extract_ground_truth(
            reference_df,
            reference_columns,
        )
    )

    assignments, definitions = (
        build_regime_assignments(
            translation_gt,
            rotation_gt,
            low_quantile=(
                args.low_quantile
            ),
            high_quantile=(
                args.high_quantile
            ),
            composition_ratio=(
                args.composition_ratio
            ),
        )
    )

    assignments.to_csv(
        output_dir
        / "frame_regime_assignments.csv",
        index=False,
    )

    save_json(
        output_dir
        / "regime_definitions.json",
        definitions,
    )

    print_regime_counts(
        assignments
    )

    metrics = compute_regime_metrics(
        experiments,
        column_maps,
        assignments,
        minimum_correlation_samples=(
            args.minimum_correlation_samples
        ),
    )

    metrics.to_csv(
        output_dir
        / "regime_metrics.csv",
        index=False,
    )

    comparison: Optional[pd.DataFrame] = None

    if (
        comparison_name is not None
        and comparison_name != reference_name
    ):
        comparison = (
            build_comparison_table(
                metrics,
                reference_name=reference_name,
                comparison_name=(
                    comparison_name
                ),
            )
        )

        comparison.to_csv(
            output_dir
            / "regime_comparison.csv",
            index=False,
        )

    create_plots(
        metrics,
        comparison,
        output_dir,
    )

    print_overall_metrics(
        metrics
    )

    if comparison is not None:
        print_forward_comparison(
            comparison
        )

    overall_records = (
        metrics[
            metrics["regime_family"]
            == "overall"
        ]
        .to_dict(
            orient="records"
        )
    )

    summary: Dict[str, Any] = {
        "reference_experiment": (
            reference_name
        ),
        "comparison_experiment": (
            comparison_name
        ),
        "experiments": {
            name: {
                "csv": str(
                    path.expanduser().resolve()
                ),
                "samples": len(
                    experiments[name]
                ),
            }
            for name, path in args.input
        },
        "sample_count": len(
            reference_df
        ),
        "ground_truth_audit": "PASS",
        "regime_definitions": definitions,
        "overall_metrics": (
            overall_records
        ),
    }

    if comparison is not None:
        overall_comparison = comparison[
            comparison["regime_family"]
            == "overall"
        ]

        if not overall_comparison.empty:
            summary[
                "overall_comparison"
            ] = (
                overall_comparison.iloc[0]
                .to_dict()
            )

    save_json(
        output_dir
        / "summary.json",
        summary,
    )

    print()
    print("=" * 88)
    print("Analysis complete")
    print("=" * 88)
    print(
        "Regime metrics:      "
        f"{output_dir / 'regime_metrics.csv'}"
    )

    if comparison is not None:
        print(
            "Comparison:          "
            f"{output_dir / 'regime_comparison.csv'}"
        )

    print(
        "Regime assignments:  "
        f"{output_dir / 'frame_regime_assignments.csv'}"
    )
    print(
        "Definitions:         "
        f"{output_dir / 'regime_definitions.json'}"
    )
    print(
        "Summary:             "
        f"{output_dir / 'summary.json'}"
    )
    print(
        "Plots:               "
        f"{output_dir / 'plots'}"
    )
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )