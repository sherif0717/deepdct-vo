"""Analyze DeepDCT-VO translation errors by motion regime.

The script divides frame-level translation predictions into low-, medium-,
and high-motion regimes and measures how model behavior changes across those
regimes.

Supported regime variables
--------------------------
z
    Bin by ground-truth translation z.

magnitude
    Bin by the Euclidean norm of the ground-truth translation vector.

Supported binning methods
-------------------------
quantile
    Divide samples into approximately equal-sized bins. This is the default
    and is useful for comparing models without sparse high-motion bins.

fixed
    Use explicit user-provided bin edges in the original target units.

Generated outputs
-----------------
motion_regime_analysis/
├── summary.json
├── regime_statistics.csv
├── regime_axis_statistics.csv
├── frame_regime_assignments.csv
└── plots/
    ├── regime_sample_counts.png
    ├── regime_vector_rmse.png
    ├── regime_axis_bias.png
    ├── regime_axis_rmse.png
    ├── regime_axis_correlation.png
    ├── regime_axis_regression_slope.png
    ├── regime_magnitude_ratio.png
    ├── regime_direction_error.png
    ├── translation_z_timeseries_by_regime.png
    └── translation_z_scatter_by_regime.png

Examples
--------
Three equal-population regimes based on ground-truth z:

    python3 scripts/analyze_translation_motion_regimes.py \
        --input-csv \
        evaluation/sequence_10_semantic_depth_identity_output/frame_predictions.csv \
        --output-dir \
        motion_regime_analysis/semantic_depth_z_quantiles \
        --regime-variable z \
        --binning quantile \
        --num-regimes 3

Fixed z regimes:

    python3 scripts/analyze_translation_motion_regimes.py \
        --input-csv \
        evaluation/sequence_10_semantic_depth_identity_output/frame_predictions.csv \
        --output-dir \
        motion_regime_analysis/semantic_depth_z_fixed \
        --regime-variable z \
        --binning fixed \
        --bin-edges 0.0 0.5 1.0 1.6

Important
---------
The analysis uses the translation representation stored in the evaluator CSV.
If the CSV stores directional-coordinate translation, all component and
magnitude results are in directional-coordinate space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AXES: Tuple[str, ...] = ("x", "y", "z")

GT_COLUMNS: Mapping[str, str] = {
    "x": "translation_gt_x",
    "y": "translation_gt_y",
    "z": "translation_gt_z",
}

PRED_COLUMNS: Mapping[str, str] = {
    "x": "translation_pred_x",
    "y": "translation_pred_y",
    "z": "translation_pred_z",
}

EPSILON = 1.0e-12


@dataclass(frozen=True)
class RegimeStatistics:
    """Vector-level statistics for one motion regime."""

    regime_index: int
    regime_name: str
    lower_bound: float
    upper_bound: float
    include_upper_bound: bool
    count: int
    fraction: float

    regime_variable_gt_mean: float
    regime_variable_gt_median: float
    regime_variable_gt_minimum: float
    regime_variable_gt_maximum: float

    gt_magnitude_mean: float
    pred_magnitude_mean: float
    magnitude_bias: float
    magnitude_mae: float
    magnitude_rmse: float
    magnitude_ratio_mean: float
    magnitude_ratio_median: float

    direction_error_mean_degrees: float
    direction_error_median_degrees: float
    direction_error_percentile_90_degrees: float

    vector_error_norm_mean: float
    vector_error_norm_median: float
    vector_error_norm_rmse: float
    vector_error_norm_maximum: float


@dataclass(frozen=True)
class RegimeAxisStatistics:
    """Component-level statistics for one motion regime."""

    regime_index: int
    regime_name: str
    axis: str
    count: int

    gt_mean: float
    gt_standard_deviation: float
    pred_mean: float
    pred_standard_deviation: float

    bias: float
    residual_standard_deviation: float
    mae: float
    mse: float
    rmse: float

    pearson_correlation: float
    regression_slope: float
    regression_intercept: float
    coefficient_of_determination: float

    cumulative_final_error: float
    cumulative_maximum_absolute_error: float


@dataclass(frozen=True)
class RegimeDefinition:
    """One numerical motion-regime interval."""

    index: int
    name: str
    lower_bound: float
    upper_bound: float
    include_upper_bound: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze translation prediction errors by motion regime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Evaluator frame_predictions.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("motion_regime_analysis"),
        help="Analysis output directory.",
    )

    parser.add_argument(
        "--regime-variable",
        choices=("z", "magnitude"),
        default="z",
        help="Ground-truth quantity used to define motion regimes.",
    )

    parser.add_argument(
        "--binning",
        choices=("quantile", "fixed"),
        default="quantile",
        help="Method used to define regime boundaries.",
    )

    parser.add_argument(
        "--num-regimes",
        type=int,
        default=3,
        help="Number of quantile regimes.",
    )

    parser.add_argument(
        "--bin-edges",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Strictly increasing boundaries for fixed binning. For three "
            "regimes, provide four edges."
        ),
    )

    parser.add_argument(
        "--regime-names",
        nargs="+",
        default=None,
        help=(
            "Optional names for the regimes. The number of names must equal "
            "the number of intervals."
        ),
    )

    parser.add_argument(
        "--direction-min-norm",
        type=float,
        default=1.0e-6,
        help="Minimum GT and prediction norms for direction-error analysis.",
    )

    parser.add_argument(
        "--magnitude-ratio-min-gt-norm",
        type=float,
        default=1.0e-6,
        help="Minimum GT norm for magnitude-ratio analysis.",
    )

    parser.add_argument(
        "--scatter-alpha",
        type=float,
        default=0.35,
        help="Scatter-plot point opacity.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved plot resolution.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if not args.input_csv.is_file():
        raise FileNotFoundError(
            f"Input prediction CSV does not exist: {args.input_csv}"
        )

    if args.num_regimes < 2:
        raise ValueError("--num-regimes must be at least 2.")

    if args.binning == "fixed":
        if args.bin_edges is None:
            raise ValueError(
                "--bin-edges is required when --binning fixed is selected."
            )

        if len(args.bin_edges) < 3:
            raise ValueError(
                "--bin-edges must contain at least three boundaries."
            )

        edges = np.asarray(args.bin_edges, dtype=np.float64)

        if not np.isfinite(edges).all():
            raise ValueError("--bin-edges must be finite.")

        if not np.all(np.diff(edges) > 0.0):
            raise ValueError(
                "--bin-edges must be strictly increasing."
            )

    if args.regime_names is not None:
        expected_count = (
            args.num_regimes
            if args.binning == "quantile"
            else len(args.bin_edges) - 1
        )

        if len(args.regime_names) != expected_count:
            raise ValueError(
                f"Expected {expected_count} regime names, received "
                f"{len(args.regime_names)}."
            )

    if args.direction_min_norm < 0.0:
        raise ValueError("--direction-min-norm cannot be negative.")

    if args.magnitude_ratio_min_gt_norm < 0.0:
        raise ValueError(
            "--magnitude-ratio-min-gt-norm cannot be negative."
        )

    if not 0.0 < args.scatter_alpha <= 1.0:
        raise ValueError("--scatter-alpha must lie in (0, 1].")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate frame-level translation predictions."""

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Input CSV contains no rows: {path}")

    required_columns = [
        *GT_COLUMNS.values(),
        *PRED_COLUMNS.values(),
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"Input CSV is missing required columns: {missing}. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    for column in required_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_values = dataframe[
        required_columns
    ].to_numpy(dtype=np.float64)

    invalid_mask = ~np.isfinite(required_values).all(axis=1)

    if invalid_mask.any():
        invalid_rows = np.flatnonzero(invalid_mask)

        raise ValueError(
            "Non-finite translation values found at row indices "
            f"{invalid_rows[:20].tolist()}."
        )

    dataframe = dataframe.reset_index(drop=True)

    if "frame_curr" in dataframe.columns:
        dataframe["analysis_frame"] = pd.to_numeric(
            dataframe["frame_curr"],
            errors="coerce",
        )
    elif "frame_prev" in dataframe.columns:
        dataframe["analysis_frame"] = (
            pd.to_numeric(
                dataframe["frame_prev"],
                errors="coerce",
            )
            + 1
        )
    else:
        dataframe["analysis_frame"] = np.arange(
            len(dataframe),
            dtype=np.int64,
        )

    if not np.isfinite(
        dataframe["analysis_frame"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Frame indices contain non-finite values.")

    return dataframe


def extract_translation_arrays(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract arrays shaped [N, 3]."""

    ground_truth = dataframe[
        [GT_COLUMNS[axis] for axis in AXES]
    ].to_numpy(dtype=np.float64)

    prediction = dataframe[
        [PRED_COLUMNS[axis] for axis in AXES]
    ].to_numpy(dtype=np.float64)

    return ground_truth, prediction


def sample_standard_deviation(values: np.ndarray) -> float:
    """Return sample standard deviation."""

    if values.size <= 1:
        return 0.0

    return float(np.std(values, ddof=1))


def safe_pearson(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Compute Pearson correlation for nonconstant arrays."""

    if ground_truth.size < 2:
        return float("nan")

    if (
        np.std(ground_truth) <= EPSILON
        or np.std(prediction) <= EPSILON
    ):
        return float("nan")

    return float(
        np.corrcoef(ground_truth, prediction)[0, 1]
    )


def linear_regression(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit prediction = slope * ground_truth + intercept."""

    if ground_truth.size < 2:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    if np.std(ground_truth) <= EPSILON:
        return (
            float("nan"),
            float(np.mean(prediction)),
            float("nan"),
        )

    slope, intercept = np.polyfit(
        ground_truth,
        prediction,
        deg=1,
    )

    fitted = slope * ground_truth + intercept

    residual_sum_squares = float(
        np.sum((prediction - fitted) ** 2)
    )

    total_sum_squares = float(
        np.sum(
            (
                prediction
                - np.mean(prediction)
            )
            ** 2
        )
    )

    r_squared = (
        float("nan")
        if total_sum_squares <= EPSILON
        else 1.0
        - residual_sum_squares
        / total_sum_squares
    )

    return (
        float(slope),
        float(intercept),
        float(r_squared),
    )


def default_regime_names(count: int) -> List[str]:
    """Return readable names for a regime count."""

    if count == 3:
        return [
            "low_motion",
            "medium_motion",
            "high_motion",
        ]

    return [
        f"regime_{index + 1:02d}"
        for index in range(count)
    ]


def make_strictly_increasing_edges(
    edges: np.ndarray,
) -> np.ndarray:
    """Repair duplicate quantile edges by a minimal numerical increment."""

    repaired = edges.astype(np.float64, copy=True)

    for index in range(1, repaired.size):
        if repaired[index] <= repaired[index - 1]:
            repaired[index] = np.nextafter(
                repaired[index - 1],
                np.inf,
            )

    return repaired


def build_regimes(
    regime_values: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[RegimeDefinition], np.ndarray]:
    """Build regime boundaries and assign each sample."""

    if args.binning == "quantile":
        quantiles = np.linspace(
            0.0,
            1.0,
            args.num_regimes + 1,
        )

        edges = np.quantile(
            regime_values,
            quantiles,
        )

        edges = make_strictly_increasing_edges(edges)
    else:
        edges = np.asarray(
            args.bin_edges,
            dtype=np.float64,
        )

    regime_count = edges.size - 1

    names = (
        list(args.regime_names)
        if args.regime_names is not None
        else default_regime_names(regime_count)
    )

    definitions: List[RegimeDefinition] = []

    assignments = np.full(
        regime_values.shape,
        -1,
        dtype=np.int64,
    )

    for index in range(regime_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        include_upper = index == regime_count - 1

        if include_upper:
            mask = (
                (regime_values >= lower)
                & (regime_values <= upper)
            )
        else:
            mask = (
                (regime_values >= lower)
                & (regime_values < upper)
            )

        assignments[mask] = index

        definitions.append(
            RegimeDefinition(
                index=index,
                name=names[index],
                lower_bound=lower,
                upper_bound=upper,
                include_upper_bound=include_upper,
            )
        )

    unassigned = np.flatnonzero(assignments < 0)

    if unassigned.size:
        raise ValueError(
            f"{unassigned.size} samples were outside the configured regime "
            "edges. For fixed binning, ensure the first and last edges cover "
            "the full data range."
        )

    return assignments, definitions, edges


def compute_direction_errors(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    minimum_norm: float,
) -> np.ndarray:
    """Compute translation direction error in degrees."""

    gt_norm = np.linalg.norm(
        ground_truth,
        axis=1,
    )

    pred_norm = np.linalg.norm(
        prediction,
        axis=1,
    )

    valid = (
        (gt_norm > minimum_norm)
        & (pred_norm > minimum_norm)
    )

    errors = np.full(
        gt_norm.shape,
        np.nan,
        dtype=np.float64,
    )

    if valid.any():
        dot_products = np.sum(
            ground_truth[valid]
            * prediction[valid],
            axis=1,
        )

        cosine = dot_products / (
            gt_norm[valid]
            * pred_norm[valid]
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        errors[valid] = np.degrees(
            np.arccos(cosine)
        )

    return errors


def finite_mean(values: np.ndarray) -> float:
    """Mean of finite values, or NaN if none exist."""

    finite = values[np.isfinite(values)]

    return (
        float(np.mean(finite))
        if finite.size
        else float("nan")
    )


def finite_median(values: np.ndarray) -> float:
    """Median of finite values, or NaN if none exist."""

    finite = values[np.isfinite(values)]

    return (
        float(np.median(finite))
        if finite.size
        else float("nan")
    )


def finite_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    """Percentile of finite values."""

    finite = values[np.isfinite(values)]

    return (
        float(np.percentile(finite, percentile))
        if finite.size
        else float("nan")
    )


def compute_regime_statistics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    regime_values: np.ndarray,
    assignments: np.ndarray,
    definitions: Sequence[RegimeDefinition],
    direction_errors: np.ndarray,
    ratio_minimum_gt_norm: float,
) -> Tuple[
    List[RegimeStatistics],
    List[RegimeAxisStatistics],
]:
    """Compute vector- and axis-level statistics for each regime."""

    total_count = ground_truth.shape[0]

    gt_norm_all = np.linalg.norm(
        ground_truth,
        axis=1,
    )

    pred_norm_all = np.linalg.norm(
        prediction,
        axis=1,
    )

    vector_residual_all = prediction - ground_truth

    vector_error_norm_all = np.linalg.norm(
        vector_residual_all,
        axis=1,
    )

    vector_rows: List[RegimeStatistics] = []
    axis_rows: List[RegimeAxisStatistics] = []

    for definition in definitions:
        mask = assignments == definition.index
        indices = np.flatnonzero(mask)

        if indices.size == 0:
            raise ValueError(
                f"Regime {definition.name} contains no samples."
            )

        gt = ground_truth[mask]
        pred = prediction[mask]
        regime_subset = regime_values[mask]

        gt_norm = gt_norm_all[mask]
        pred_norm = pred_norm_all[mask]
        magnitude_error = pred_norm - gt_norm
        vector_error_norm = vector_error_norm_all[mask]
        direction_subset = direction_errors[mask]

        magnitude_ratio = np.full(
            gt_norm.shape,
            np.nan,
            dtype=np.float64,
        )

        ratio_mask = gt_norm > ratio_minimum_gt_norm

        magnitude_ratio[ratio_mask] = (
            pred_norm[ratio_mask]
            / gt_norm[ratio_mask]
        )

        vector_rows.append(
            RegimeStatistics(
                regime_index=definition.index,
                regime_name=definition.name,
                lower_bound=definition.lower_bound,
                upper_bound=definition.upper_bound,
                include_upper_bound=(
                    definition.include_upper_bound
                ),
                count=int(indices.size),
                fraction=float(
                    indices.size / total_count
                ),
                regime_variable_gt_mean=float(
                    np.mean(regime_subset)
                ),
                regime_variable_gt_median=float(
                    np.median(regime_subset)
                ),
                regime_variable_gt_minimum=float(
                    np.min(regime_subset)
                ),
                regime_variable_gt_maximum=float(
                    np.max(regime_subset)
                ),
                gt_magnitude_mean=float(
                    np.mean(gt_norm)
                ),
                pred_magnitude_mean=float(
                    np.mean(pred_norm)
                ),
                magnitude_bias=float(
                    np.mean(magnitude_error)
                ),
                magnitude_mae=float(
                    np.mean(
                        np.abs(magnitude_error)
                    )
                ),
                magnitude_rmse=float(
                    math.sqrt(
                        float(
                            np.mean(
                                magnitude_error ** 2
                            )
                        )
                    )
                ),
                magnitude_ratio_mean=finite_mean(
                    magnitude_ratio
                ),
                magnitude_ratio_median=finite_median(
                    magnitude_ratio
                ),
                direction_error_mean_degrees=finite_mean(
                    direction_subset
                ),
                direction_error_median_degrees=finite_median(
                    direction_subset
                ),
                direction_error_percentile_90_degrees=(
                    finite_percentile(
                        direction_subset,
                        90.0,
                    )
                ),
                vector_error_norm_mean=float(
                    np.mean(vector_error_norm)
                ),
                vector_error_norm_median=float(
                    np.median(vector_error_norm)
                ),
                vector_error_norm_rmse=float(
                    math.sqrt(
                        float(
                            np.mean(
                                vector_error_norm ** 2
                            )
                        )
                    )
                ),
                vector_error_norm_maximum=float(
                    np.max(vector_error_norm)
                ),
            )
        )

        for axis_index, axis in enumerate(AXES):
            gt_axis = gt[:, axis_index]
            pred_axis = pred[:, axis_index]
            residual = pred_axis - gt_axis

            slope, intercept, r_squared = linear_regression(
                gt_axis,
                pred_axis,
            )

            mse = float(
                np.mean(residual ** 2)
            )

            cumulative = np.cumsum(residual)

            axis_rows.append(
                RegimeAxisStatistics(
                    regime_index=definition.index,
                    regime_name=definition.name,
                    axis=axis,
                    count=int(gt_axis.size),
                    gt_mean=float(
                        np.mean(gt_axis)
                    ),
                    gt_standard_deviation=(
                        sample_standard_deviation(
                            gt_axis
                        )
                    ),
                    pred_mean=float(
                        np.mean(pred_axis)
                    ),
                    pred_standard_deviation=(
                        sample_standard_deviation(
                            pred_axis
                        )
                    ),
                    bias=float(
                        np.mean(residual)
                    ),
                    residual_standard_deviation=(
                        sample_standard_deviation(
                            residual
                        )
                    ),
                    mae=float(
                        np.mean(
                            np.abs(residual)
                        )
                    ),
                    mse=mse,
                    rmse=float(
                        math.sqrt(mse)
                    ),
                    pearson_correlation=safe_pearson(
                        gt_axis,
                        pred_axis,
                    ),
                    regression_slope=slope,
                    regression_intercept=intercept,
                    coefficient_of_determination=(
                        r_squared
                    ),
                    cumulative_final_error=float(
                        cumulative[-1]
                    ),
                    cumulative_maximum_absolute_error=float(
                        np.max(
                            np.abs(cumulative)
                        )
                    ),
                )
            )

    return vector_rows, axis_rows


def write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
) -> None:
    """Write dataclass instances as CSV rows."""

    if not rows:
        raise ValueError(
            f"Cannot write an empty CSV: {path}"
        )

    dictionaries = [
        asdict(row)
        for row in rows
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                dictionaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(dictionaries)


def write_assignments_csv(
    path: Path,
    dataframe: pd.DataFrame,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    regime_values: np.ndarray,
    assignments: np.ndarray,
    definitions: Sequence[RegimeDefinition],
) -> None:
    """Save frame-level motion-regime assignments and residuals."""

    output = pd.DataFrame(
        {
            "analysis_frame": dataframe[
                "analysis_frame"
            ].to_numpy(),
            "regime_value": regime_values,
            "regime_index": assignments,
            "regime_name": [
                definitions[index].name
                for index in assignments
            ],
        }
    )

    for axis_index, axis in enumerate(AXES):
        output[f"translation_gt_{axis}"] = (
            ground_truth[:, axis_index]
        )

        output[f"translation_pred_{axis}"] = (
            prediction[:, axis_index]
        )

        output[f"translation_residual_{axis}"] = (
            prediction[:, axis_index]
            - ground_truth[:, axis_index]
        )

    output["translation_gt_magnitude"] = np.linalg.norm(
        ground_truth,
        axis=1,
    )

    output["translation_pred_magnitude"] = np.linalg.norm(
        prediction,
        axis=1,
    )

    output["translation_vector_error_norm"] = np.linalg.norm(
        prediction - ground_truth,
        axis=1,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        path,
        index=False,
    )


def save_figure(
    figure: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    """Save and close a Matplotlib figure."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_regime_metric(
    regime_rows: Sequence[RegimeStatistics],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot one vector-level metric by regime."""

    labels = [
        row.regime_name
        for row in regime_rows
    ]

    values = [
        float(
            getattr(row, metric_name)
        )
        for row in regime_rows
    ]

    figure, axes = plt.subplots(
        figsize=(9, 6)
    )

    positions = np.arange(
        len(labels)
    )

    axes.bar(
        positions,
        values,
    )

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylabel(ylabel)
    axes.set_xlabel("Motion regime")
    axes.set_title(title)
    axes.grid(
        axis="y",
        alpha=0.3,
    )

    for position, value in zip(
        positions,
        values,
    ):
        axes.text(
            position,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    save_figure(
        figure,
        output_path,
        dpi,
    )


def plot_sample_counts(
    regime_rows: Sequence[RegimeStatistics],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot sample count by regime."""

    labels = [
        row.regime_name
        for row in regime_rows
    ]

    counts = [
        row.count
        for row in regime_rows
    ]

    figure, axes = plt.subplots(
        figsize=(9, 6)
    )

    positions = np.arange(
        len(labels)
    )

    axes.bar(
        positions,
        counts,
    )

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlabel("Motion regime")
    axes.set_ylabel("Samples")
    axes.set_title(
        "Sample contribution by motion regime"
    )
    axes.grid(
        axis="y",
        alpha=0.3,
    )

    for position, count in zip(
        positions,
        counts,
    ):
        axes.text(
            position,
            count,
            str(count),
            ha="center",
            va="bottom",
        )

    save_figure(
        figure,
        output_path,
        dpi,
    )


def plot_axis_metric(
    axis_rows: Sequence[RegimeAxisStatistics],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot x/y/z component metrics grouped by regime."""

    regime_names = []

    for row in axis_rows:
        if row.regime_name not in regime_names:
            regime_names.append(
                row.regime_name
            )

    positions = np.arange(
        len(regime_names)
    )

    width = 0.24

    figure, axes = plt.subplots(
        figsize=(11, 7)
    )

    for axis_index, axis in enumerate(AXES):
        axis_lookup = {
            row.regime_name: row
            for row in axis_rows
            if row.axis == axis
        }

        values = [
            float(
                getattr(
                    axis_lookup[name],
                    metric_name,
                )
            )
            for name in regime_names
        ]

        axes.bar(
            positions
            + (axis_index - 1) * width,
            values,
            width=width,
            label=f"Axis {axis}",
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(regime_names)
    axes.set_xlabel("Motion regime")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(
        axis="y",
        alpha=0.3,
    )
    axes.legend()

    save_figure(
        figure,
        output_path,
        dpi,
    )


def plot_z_timeseries_by_regime(
    frames: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    definitions: Sequence[RegimeDefinition],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot z targets and shade samples by regime."""

    figure, axes = plt.subplots(
        figsize=(15, 7)
    )

    axes.plot(
        frames,
        ground_truth[:, 2],
        label="Ground truth z",
        linewidth=1.3,
    )

    axes.plot(
        frames,
        prediction[:, 2],
        label="Predicted z",
        linewidth=1.0,
    )

    for definition in definitions:
        mask = assignments == definition.index

        axes.scatter(
            frames[mask],
            ground_truth[mask, 2],
            s=12,
            alpha=0.35,
            label=f"{definition.name} samples",
        )

    axes.set_xlabel("Frame")
    axes.set_ylabel("Translation z")
    axes.set_title(
        "Translation z over time by ground-truth motion regime"
    )
    axes.grid(alpha=0.3)
    axes.legend(
        ncol=2,
    )

    save_figure(
        figure,
        output_path,
        dpi,
    )


def plot_z_scatter_by_regime(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    definitions: Sequence[RegimeDefinition],
    output_path: Path,
    dpi: int,
    scatter_alpha: float,
) -> None:
    """Plot z prediction calibration separately by regime."""

    gt_z = ground_truth[:, 2]
    pred_z = prediction[:, 2]

    lower = float(
        min(
            np.min(gt_z),
            np.min(pred_z),
        )
    )

    upper = float(
        max(
            np.max(gt_z),
            np.max(pred_z),
        )
    )

    if math.isclose(
        lower,
        upper,
    ):
        padding = max(
            abs(lower) * 0.05,
            1.0e-3,
        )

        lower -= padding
        upper += padding

    figure, axes = plt.subplots(
        figsize=(8, 8)
    )

    for definition in definitions:
        mask = assignments == definition.index

        axes.scatter(
            gt_z[mask],
            pred_z[mask],
            s=14,
            alpha=scatter_alpha,
            label=definition.name,
        )

        if np.count_nonzero(mask) >= 2:
            slope, intercept, _ = linear_regression(
                gt_z[mask],
                pred_z[mask],
            )

            if math.isfinite(slope):
                regime_x = np.asarray(
                    [
                        np.min(gt_z[mask]),
                        np.max(gt_z[mask]),
                    ]
                )

                axes.plot(
                    regime_x,
                    slope * regime_x + intercept,
                    linewidth=1.3,
                    label=(
                        f"{definition.name} fit: "
                        f"{slope:.3f}x{intercept:+.3f}"
                    ),
                )

    axes.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.3,
        label="Ideal prediction",
    )

    axes.set_xlim(
        lower,
        upper,
    )

    axes.set_ylim(
        lower,
        upper,
    )

    axes.set_aspect(
        "equal",
        adjustable="box",
    )

    axes.set_xlabel(
        "Ground-truth translation z"
    )

    axes.set_ylabel(
        "Predicted translation z"
    )

    axes.set_title(
        "Translation z calibration by motion regime"
    )

    axes.grid(alpha=0.3)

    axes.legend(
        fontsize=8,
    )

    save_figure(
        figure,
        output_path,
        dpi,
    )


def json_safe(value: object) -> object:
    """Convert NumPy values and nonfinite floats for strict JSON."""

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        numeric = float(value)

        return (
            numeric
            if math.isfinite(numeric)
            else None
        )

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):
        return int(value)

    return value


def print_summary(
    regime_rows: Sequence[RegimeStatistics],
    axis_rows: Sequence[RegimeAxisStatistics],
    output_dir: Path,
) -> None:
    """Print the most relevant regime results."""

    z_lookup = {
        row.regime_name: row
        for row in axis_rows
        if row.axis == "z"
    }

    print()
    print("=" * 124)
    print("Translation motion-regime analysis")
    print("=" * 124)

    print(
        f"{'Regime':<18}"
        f"{'Count':>10}"
        f"{'GT z mean':>14}"
        f"{'z bias':>14}"
        f"{'z RMSE':>14}"
        f"{'z corr':>14}"
        f"{'z slope':>14}"
        f"{'Vector RMSE':>16}"
    )

    print("-" * 124)

    for regime in regime_rows:
        z_row = z_lookup[
            regime.regime_name
        ]

        print(
            f"{regime.regime_name:<18}"
            f"{regime.count:>10d}"
            f"{z_row.gt_mean:>14.6f}"
            f"{z_row.bias:>14.6f}"
            f"{z_row.rmse:>14.6f}"
            f"{z_row.pearson_correlation:>14.6f}"
            f"{z_row.regression_slope:>14.6f}"
            f"{regime.vector_error_norm_rmse:>16.6f}"
        )

    print("-" * 124)
    print(
        f"Output directory: {output_dir.resolve()}"
    )
    print("=" * 124)


def main() -> None:
    """Run the motion-regime analysis."""

    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    plot_dir = output_dir / "plots"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_predictions(
        args.input_csv
    )

    ground_truth, prediction = extract_translation_arrays(
        dataframe
    )

    frames = dataframe[
        "analysis_frame"
    ].to_numpy(dtype=np.float64)

    gt_magnitude = np.linalg.norm(
        ground_truth,
        axis=1,
    )

    if args.regime_variable == "z":
        regime_values = ground_truth[:, 2]
    else:
        regime_values = gt_magnitude

    (
        assignments,
        definitions,
        edges,
    ) = build_regimes(
        regime_values,
        args,
    )

    direction_errors = compute_direction_errors(
        ground_truth,
        prediction,
        minimum_norm=args.direction_min_norm,
    )

    (
        regime_statistics,
        regime_axis_statistics,
    ) = compute_regime_statistics(
        ground_truth=ground_truth,
        prediction=prediction,
        regime_values=regime_values,
        assignments=assignments,
        definitions=definitions,
        direction_errors=direction_errors,
        ratio_minimum_gt_norm=(
            args.magnitude_ratio_min_gt_norm
        ),
    )

    write_dataclass_csv(
        output_dir / "regime_statistics.csv",
        regime_statistics,
    )

    write_dataclass_csv(
        output_dir / "regime_axis_statistics.csv",
        regime_axis_statistics,
    )

    write_assignments_csv(
        path=output_dir / "frame_regime_assignments.csv",
        dataframe=dataframe,
        ground_truth=ground_truth,
        prediction=prediction,
        regime_values=regime_values,
        assignments=assignments,
        definitions=definitions,
    )

    plot_sample_counts(
        regime_rows=regime_statistics,
        output_path=(
            plot_dir
            / "regime_sample_counts.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_metric(
        regime_rows=regime_statistics,
        metric_name="vector_error_norm_rmse",
        ylabel="Translation vector-error norm RMSE",
        title="Translation vector error by motion regime",
        output_path=(
            plot_dir
            / "regime_vector_rmse.png"
        ),
        dpi=args.dpi,
    )

    plot_axis_metric(
        axis_rows=regime_axis_statistics,
        metric_name="bias",
        ylabel="Mean residual",
        title="Translation bias by motion regime",
        output_path=(
            plot_dir
            / "regime_axis_bias.png"
        ),
        dpi=args.dpi,
    )

    plot_axis_metric(
        axis_rows=regime_axis_statistics,
        metric_name="rmse",
        ylabel="RMSE",
        title="Translation component RMSE by motion regime",
        output_path=(
            plot_dir
            / "regime_axis_rmse.png"
        ),
        dpi=args.dpi,
    )

    plot_axis_metric(
        axis_rows=regime_axis_statistics,
        metric_name="pearson_correlation",
        ylabel="Pearson correlation",
        title="Prediction correlation by motion regime",
        output_path=(
            plot_dir
            / "regime_axis_correlation.png"
        ),
        dpi=args.dpi,
    )

    plot_axis_metric(
        axis_rows=regime_axis_statistics,
        metric_name="regression_slope",
        ylabel="Prediction-versus-GT slope",
        title="Regression slope by motion regime",
        output_path=(
            plot_dir
            / "regime_axis_regression_slope.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_metric(
        regime_rows=regime_statistics,
        metric_name="magnitude_ratio_median",
        ylabel="Median predicted/GT magnitude",
        title="Translation magnitude ratio by motion regime",
        output_path=(
            plot_dir
            / "regime_magnitude_ratio.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_metric(
        regime_rows=regime_statistics,
        metric_name="direction_error_median_degrees",
        ylabel="Median direction error (degrees)",
        title="Translation direction error by motion regime",
        output_path=(
            plot_dir
            / "regime_direction_error.png"
        ),
        dpi=args.dpi,
    )

    plot_z_timeseries_by_regime(
        frames=frames,
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        definitions=definitions,
        output_path=(
            plot_dir
            / "translation_z_timeseries_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_z_scatter_by_regime(
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        definitions=definitions,
        output_path=(
            plot_dir
            / "translation_z_scatter_by_regime.png"
        ),
        dpi=args.dpi,
        scatter_alpha=args.scatter_alpha,
    )

    summary = {
        "input_csv": str(
            args.input_csv.resolve()
        ),
        "sample_count": int(
            ground_truth.shape[0]
        ),
        "regime_variable": args.regime_variable,
        "binning": args.binning,
        "bin_edges": edges.tolist(),
        "residual_definition": (
            "prediction_minus_ground_truth"
        ),
        "translation_representation_note": (
            "Statistics describe the translation representation stored in "
            "the input CSV."
        ),
        "regime_definitions": [
            asdict(definition)
            for definition in definitions
        ],
        "regime_statistics": [
            asdict(row)
            for row in regime_statistics
        ],
        "regime_axis_statistics": [
            asdict(row)
            for row in regime_axis_statistics
        ],
    }

    with (
        output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(summary),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    print_summary(
        regime_rows=regime_statistics,
        axis_rows=regime_axis_statistics,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()