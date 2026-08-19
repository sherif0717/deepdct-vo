"""Evaluate DeepDCT translation prediction errors in seven fixed tc_z regimes.

The script reads evaluator-produced frame_predictions.csv files with:

    translation_gt_x
    translation_gt_y
    translation_gt_z
    translation_pred_x
    translation_pred_y
    translation_pred_z

Regimes are defined from ground-truth translation_gt_z:

    very_low:    z < 0.25
    low:         0.25 <= z < 0.50
    medium_low:  0.50 <= z < 0.75
    medium:      0.75 <= z < 1.00
    high:        1.00 <= z < 1.25
    very_high:   1.25 <= z < 1.50
    extreme:     z >= 1.50

Outputs
-------
refined_prediction_regime_analysis/
├── summary.json
├── regime_statistics.csv
├── regime_axis_statistics.csv
├── frame_regime_assignments.csv
└── plots/
    ├── z_bias_by_regime.png
    ├── z_rmse_by_regime.png
    ├── z_correlation_by_regime.png
    ├── z_regression_slope_by_regime.png
    ├── vector_rmse_by_regime.png
    ├── magnitude_ratio_by_regime.png
    ├── direction_error_by_regime.png
    ├── z_gt_pred_mean_by_regime.png
    ├── z_scatter_by_regime.png
    └── z_timeseries_by_regime.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AXES = ("x", "y", "z")

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

REGIME_NAMES = (
    "very_low",
    "low",
    "medium_low",
    "medium",
    "high",
    "very_high",
    "extreme",
)

REGIME_BOUNDARIES = np.asarray(
    [0.25, 0.50, 0.75, 1.00, 1.25, 1.50],
    dtype=np.float64,
)

EPSILON = 1.0e-12


@dataclass(frozen=True)
class RegimeStatistics:
    regime_index: int
    regime_name: str
    count: int
    fraction: float

    gt_z_mean: float
    pred_z_mean: float
    z_bias: float
    z_mae: float
    z_rmse: float

    gt_magnitude_mean: float
    pred_magnitude_mean: float
    magnitude_bias: float
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DeepDCT translation prediction errors using "
            "seven fixed ground-truth tc_z motion regimes."
        ),
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
        default=Path("refined_prediction_regime_analysis"),
        help="Analysis output directory.",
    )

    parser.add_argument(
        "--direction-min-norm",
        type=float,
        default=1.0e-6,
        help="Minimum GT/pred norm for direction-error calculation.",
    )

    parser.add_argument(
        "--magnitude-ratio-min-gt-norm",
        type=float,
        default=1.0e-6,
        help="Minimum GT norm for magnitude-ratio calculation.",
    )

    parser.add_argument(
        "--scatter-alpha",
        type=float,
        default=0.30,
        help="Scatter point opacity.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved plot resolution.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_csv.is_file():
        raise FileNotFoundError(
            f"Input CSV does not exist: {args.input_csv}"
        )

    if args.direction_min_norm < 0:
        raise ValueError(
            "--direction-min-norm cannot be negative."
        )

    if args.magnitude_ratio_min_gt_norm < 0:
        raise ValueError(
            "--magnitude-ratio-min-gt-norm cannot be negative."
        )

    if not 0.0 < args.scatter_alpha <= 1.0:
        raise ValueError(
            "--scatter-alpha must lie in (0, 1]."
        )

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def load_predictions(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"CSV is empty: {path}")

    required = [
        *GT_COLUMNS.values(),
        *PRED_COLUMNS.values(),
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )

    for column in required:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    values = dataframe[
        required
    ].to_numpy(dtype=np.float64)

    invalid = ~np.isfinite(values).all(axis=1)

    if invalid.any():
        indices = np.flatnonzero(invalid)

        raise ValueError(
            "Non-finite prediction values at rows "
            f"{indices[:20].tolist()}."
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

    return dataframe


def extract_translation_arrays(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    ground_truth = dataframe[
        [GT_COLUMNS[axis] for axis in AXES]
    ].to_numpy(dtype=np.float64)

    prediction = dataframe[
        [PRED_COLUMNS[axis] for axis in AXES]
    ].to_numpy(dtype=np.float64)

    return ground_truth, prediction


def classify_regime(
    gt_z: np.ndarray,
) -> np.ndarray:
    return np.digitize(
        gt_z,
        REGIME_BOUNDARIES,
        right=False,
    ).astype(np.int64)


def sample_standard_deviation(
    values: np.ndarray,
) -> float:
    if values.size <= 1:
        return 0.0

    return float(np.std(values, ddof=1))


def safe_pearson(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if ground_truth.size < 2:
        return float("nan")

    if (
        np.std(ground_truth) <= EPSILON
        or np.std(prediction) <= EPSILON
    ):
        return float("nan")

    return float(
        np.corrcoef(
            ground_truth,
            prediction,
        )[0, 1]
    )


def linear_regression(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> Tuple[float, float, float]:
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
        else (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )
    )

    return (
        float(slope),
        float(intercept),
        float(r_squared),
    )


def compute_direction_errors(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    minimum_norm: float,
) -> np.ndarray:
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

    result = np.full(
        gt_norm.shape,
        np.nan,
        dtype=np.float64,
    )

    if valid.any():
        dot = np.sum(
            ground_truth[valid]
            * prediction[valid],
            axis=1,
        )

        cosine = dot / (
            gt_norm[valid]
            * pred_norm[valid]
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        result[valid] = np.degrees(
            np.arccos(cosine)
        )

    return result


def finite_mean(
    values: np.ndarray,
) -> float:
    finite = values[
        np.isfinite(values)
    ]

    return (
        float(np.mean(finite))
        if finite.size
        else float("nan")
    )


def finite_median(
    values: np.ndarray,
) -> float:
    finite = values[
        np.isfinite(values)
    ]

    return (
        float(np.median(finite))
        if finite.size
        else float("nan")
    )


def finite_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    finite = values[
        np.isfinite(values)
    ]

    return (
        float(
            np.percentile(
                finite,
                percentile,
            )
        )
        if finite.size
        else float("nan")
    )


def compute_statistics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    direction_errors: np.ndarray,
    ratio_minimum_gt_norm: float,
) -> Tuple[
    List[RegimeStatistics],
    List[RegimeAxisStatistics],
]:
    regime_rows: List[RegimeStatistics] = []
    axis_rows: List[RegimeAxisStatistics] = []

    total_count = ground_truth.shape[0]

    gt_norm_all = np.linalg.norm(
        ground_truth,
        axis=1,
    )

    pred_norm_all = np.linalg.norm(
        prediction,
        axis=1,
    )

    vector_error_all = np.linalg.norm(
        prediction - ground_truth,
        axis=1,
    )

    for regime_index, regime_name in enumerate(
        REGIME_NAMES
    ):
        mask = assignments == regime_index
        count = int(np.count_nonzero(mask))

        if count == 0:
            continue

        gt = ground_truth[mask]
        pred = prediction[mask]

        gt_norm = gt_norm_all[mask]
        pred_norm = pred_norm_all[mask]
        vector_error = vector_error_all[mask]
        direction_subset = direction_errors[mask]

        magnitude_error = (
            pred_norm - gt_norm
        )

        ratio = np.full(
            gt_norm.shape,
            np.nan,
            dtype=np.float64,
        )

        valid_ratio = (
            gt_norm
            > ratio_minimum_gt_norm
        )

        ratio[valid_ratio] = (
            pred_norm[valid_ratio]
            / gt_norm[valid_ratio]
        )

        z_residual = (
            pred[:, 2]
            - gt[:, 2]
        )

        z_mse = float(
            np.mean(
                z_residual ** 2
            )
        )

        regime_rows.append(
            RegimeStatistics(
                regime_index=regime_index,
                regime_name=regime_name,
                count=count,
                fraction=float(
                    count
                    / total_count
                ),
                gt_z_mean=float(
                    np.mean(gt[:, 2])
                ),
                pred_z_mean=float(
                    np.mean(pred[:, 2])
                ),
                z_bias=float(
                    np.mean(z_residual)
                ),
                z_mae=float(
                    np.mean(
                        np.abs(
                            z_residual
                        )
                    )
                ),
                z_rmse=float(
                    math.sqrt(z_mse)
                ),
                gt_magnitude_mean=float(
                    np.mean(gt_norm)
                ),
                pred_magnitude_mean=float(
                    np.mean(pred_norm)
                ),
                magnitude_bias=float(
                    np.mean(
                        magnitude_error
                    )
                ),
                magnitude_ratio_mean=finite_mean(
                    ratio
                ),
                magnitude_ratio_median=finite_median(
                    ratio
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
                    np.mean(vector_error)
                ),
                vector_error_norm_median=float(
                    np.median(vector_error)
                ),
                vector_error_norm_rmse=float(
                    math.sqrt(
                        float(
                            np.mean(
                                vector_error ** 2
                            )
                        )
                    )
                ),
                vector_error_norm_maximum=float(
                    np.max(vector_error)
                ),
            )
        )

        for axis_index, axis in enumerate(
            AXES
        ):
            gt_axis = gt[:, axis_index]
            pred_axis = pred[:, axis_index]

            residual = (
                pred_axis
                - gt_axis
            )

            mse = float(
                np.mean(
                    residual ** 2
                )
            )

            slope, intercept, r_squared = (
                linear_regression(
                    gt_axis,
                    pred_axis,
                )
            )

            cumulative = np.cumsum(
                residual
            )

            axis_rows.append(
                RegimeAxisStatistics(
                    regime_index=regime_index,
                    regime_name=regime_name,
                    axis=axis,
                    count=count,
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
                            np.abs(
                                residual
                            )
                        )
                    ),
                    mse=mse,
                    rmse=float(
                        math.sqrt(mse)
                    ),
                    pearson_correlation=(
                        safe_pearson(
                            gt_axis,
                            pred_axis,
                        )
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
                            np.abs(
                                cumulative
                            )
                        )
                    ),
                )
            )

    return (
        regime_rows,
        axis_rows,
    )


def write_assignments(
    dataframe: pd.DataFrame,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    path: Path,
) -> None:
    output = pd.DataFrame(
        {
            "analysis_frame": dataframe[
                "analysis_frame"
            ].to_numpy(),
            "regime_index": assignments,
            "regime_name": [
                REGIME_NAMES[index]
                for index in assignments
            ],
        }
    )

    for axis_index, axis in enumerate(
        AXES
    ):
        output[
            f"translation_gt_{axis}"
        ] = ground_truth[
            :,
            axis_index,
        ]

        output[
            f"translation_pred_{axis}"
        ] = prediction[
            :,
            axis_index,
        ]

        output[
            f"translation_residual_{axis}"
        ] = (
            prediction[:, axis_index]
            - ground_truth[:, axis_index]
        )

    output[
        "translation_gt_magnitude"
    ] = np.linalg.norm(
        ground_truth,
        axis=1,
    )

    output[
        "translation_pred_magnitude"
    ] = np.linalg.norm(
        prediction,
        axis=1,
    )

    output[
        "translation_vector_error_norm"
    ] = np.linalg.norm(
        prediction - ground_truth,
        axis=1,
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


def plot_regime_bar(
    regime_rows: Sequence[RegimeStatistics],
    metric_name: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    labels = [
        row.regime_name
        for row in regime_rows
    ]

    values = [
        float(
            getattr(
                row,
                metric_name,
            )
        )
        for row in regime_rows
    ]

    positions = np.arange(
        len(labels)
    )

    figure, axes = plt.subplots(
        figsize=(11, 6)
    )

    bars = axes.bar(
        positions,
        values,
    )

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    axes.set_ylabel(
        ylabel
    )

    axes.set_xlabel(
        "Ground-truth tc_z regime"
    )

    axes.set_title(
        title
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        if math.isfinite(value):
            axes.text(
                bar.get_x()
                + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_figure(
        figure,
        path,
        dpi,
    )


def get_z_axis_rows(
    axis_rows: Sequence[
        RegimeAxisStatistics
    ],
) -> List[RegimeAxisStatistics]:
    return [
        row
        for row in axis_rows
        if row.axis == "z"
    ]


def plot_z_axis_metric(
    axis_rows: Sequence[
        RegimeAxisStatistics
    ],
    metric_name: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    rows = get_z_axis_rows(
        axis_rows
    )

    labels = [
        row.regime_name
        for row in rows
    ]

    values = [
        float(
            getattr(
                row,
                metric_name,
            )
        )
        for row in rows
    ]

    positions = np.arange(
        len(labels)
    )

    figure, axes = plt.subplots(
        figsize=(11, 6)
    )

    bars = axes.bar(
        positions,
        values,
    )

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    axes.set_ylabel(ylabel)
    axes.set_xlabel(
        "Ground-truth tc_z regime"
    )
    axes.set_title(title)
    axes.grid(
        axis="y",
        alpha=0.3,
    )

    if metric_name in (
        "bias",
        "pearson_correlation",
        "regression_slope",
    ):
        axes.axhline(
            0.0,
            linewidth=1.0,
        )

    if metric_name == "regression_slope":
        axes.axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            label="Ideal slope",
        )
        axes.legend()

    for bar, value in zip(
        bars,
        values,
    ):
        if math.isfinite(value):
            axes.text(
                bar.get_x()
                + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_z_gt_pred_mean(
    regime_rows: Sequence[
        RegimeStatistics
    ],
    path: Path,
    dpi: int,
) -> None:
    labels = [
        row.regime_name
        for row in regime_rows
    ]

    gt_values = [
        row.gt_z_mean
        for row in regime_rows
    ]

    pred_values = [
        row.pred_z_mean
        for row in regime_rows
    ]

    positions = np.arange(
        len(labels)
    )

    width = 0.38

    figure, axes = plt.subplots(
        figsize=(12, 6)
    )

    axes.bar(
        positions - width / 2.0,
        gt_values,
        width=width,
        label="Ground truth",
    )

    axes.bar(
        positions + width / 2.0,
        pred_values,
        width=width,
        label="Prediction",
    )

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    axes.set_xlabel(
        "Ground-truth tc_z regime"
    )

    axes.set_ylabel(
        "Mean translation z"
    )

    axes.set_title(
        "Mean ground-truth versus predicted tc_z by regime"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.legend()

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_z_scatter_by_regime(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    path: Path,
    dpi: int,
    scatter_alpha: float,
) -> None:
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

    figure, axes = plt.subplots(
        figsize=(9, 9)
    )

    for regime_index, regime_name in enumerate(
        REGIME_NAMES
    ):
        mask = assignments == regime_index

        if not mask.any():
            continue

        axes.scatter(
            gt_z[mask],
            pred_z[mask],
            s=14,
            alpha=scatter_alpha,
            label=regime_name,
        )

    axes.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.2,
        label="Ideal prediction",
    )

    axes.set_xlabel(
        "Ground-truth translation z"
    )

    axes.set_ylabel(
        "Predicted translation z"
    )

    axes.set_title(
        "Semantic-depth tc_z prediction by refined motion regime"
    )

    axes.grid(alpha=0.3)
    axes.legend(
        fontsize=8,
        ncol=2,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_z_timeseries_by_regime(
    dataframe: pd.DataFrame,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    path: Path,
    dpi: int,
) -> None:
    frames = dataframe[
        "analysis_frame"
    ].to_numpy(
        dtype=np.float64
    )

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

    for regime_index, regime_name in enumerate(
        REGIME_NAMES
    ):
        mask = assignments == regime_index

        if not mask.any():
            continue

        axes.scatter(
            frames[mask],
            ground_truth[mask, 2],
            s=10,
            alpha=0.25,
            label=regime_name,
        )

    axes.set_xlabel("Frame")
    axes.set_ylabel(
        "Translation z"
    )

    axes.set_title(
        "Semantic-depth tc_z prediction over time by refined regime"
    )

    axes.grid(alpha=0.3)
    axes.legend(
        fontsize=8,
        ncol=3,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def json_safe(
    value: object,
) -> object:
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
            np.integer,
            int,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        numeric = float(value)

        if not math.isfinite(
            numeric
        ):
            return None

        return numeric

    return value


def print_summary(
    regime_rows: Sequence[
        RegimeStatistics
    ],
    axis_rows: Sequence[
        RegimeAxisStatistics
    ],
    output_dir: Path,
) -> None:
    z_lookup = {
        row.regime_name: row
        for row in axis_rows
        if row.axis == "z"
    }

    print()
    print("=" * 134)
    print(
        "Semantic-depth refined tc_z prediction-regime analysis"
    )
    print("=" * 134)

    print(
        f"{'Regime':<15}"
        f"{'N':>7}"
        f"{'Frac %':>10}"
        f"{'GT mean':>12}"
        f"{'Pred mean':>12}"
        f"{'Bias':>12}"
        f"{'RMSE':>12}"
        f"{'Corr':>12}"
        f"{'Slope':>12}"
        f"{'Vec RMSE':>14}"
    )

    print("-" * 134)

    for regime in regime_rows:
        z_row = z_lookup[
            regime.regime_name
        ]

        print(
            f"{regime.regime_name:<15}"
            f"{regime.count:>7d}"
            f"{100.0 * regime.fraction:>10.2f}"
            f"{regime.gt_z_mean:>12.6f}"
            f"{regime.pred_z_mean:>12.6f}"
            f"{z_row.bias:>12.6f}"
            f"{z_row.rmse:>12.6f}"
            f"{z_row.pearson_correlation:>12.6f}"
            f"{z_row.regression_slope:>12.6f}"
            f"{regime.vector_error_norm_rmse:>14.6f}"
        )

    print("-" * 134)
    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )
    print("=" * 134)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    plots_dir = (
        output_dir
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_predictions(
        args.input_csv
    )

    ground_truth, prediction = (
        extract_translation_arrays(
            dataframe
        )
    )

    assignments = classify_regime(
        ground_truth[:, 2]
    )

    direction_errors = compute_direction_errors(
        ground_truth,
        prediction,
        minimum_norm=(
            args.direction_min_norm
        ),
    )

    (
        regime_statistics,
        regime_axis_statistics,
    ) = compute_statistics(
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        direction_errors=direction_errors,
        ratio_minimum_gt_norm=(
            args.magnitude_ratio_min_gt_norm
        ),
    )

    pd.DataFrame(
        [
            asdict(row)
            for row in regime_statistics
        ]
    ).to_csv(
        output_dir
        / "regime_statistics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            asdict(row)
            for row in regime_axis_statistics
        ]
    ).to_csv(
        output_dir
        / "regime_axis_statistics.csv",
        index=False,
    )

    write_assignments(
        dataframe=dataframe,
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        path=(
            output_dir
            / "frame_regime_assignments.csv"
        ),
    )

    plot_z_axis_metric(
        regime_axis_statistics,
        "bias",
        "Mean residual",
        "Translation z bias by refined motion regime",
        plots_dir
        / "z_bias_by_regime.png",
        args.dpi,
    )

    plot_z_axis_metric(
        regime_axis_statistics,
        "rmse",
        "RMSE",
        "Translation z RMSE by refined motion regime",
        plots_dir
        / "z_rmse_by_regime.png",
        args.dpi,
    )

    plot_z_axis_metric(
        regime_axis_statistics,
        "pearson_correlation",
        "Pearson correlation",
        "Translation z correlation by refined motion regime",
        plots_dir
        / "z_correlation_by_regime.png",
        args.dpi,
    )

    plot_z_axis_metric(
        regime_axis_statistics,
        "regression_slope",
        "Prediction versus GT slope",
        "Translation z regression slope by refined motion regime",
        plots_dir
        / "z_regression_slope_by_regime.png",
        args.dpi,
    )

    plot_regime_bar(
        regime_statistics,
        "vector_error_norm_rmse",
        "Vector-error norm RMSE",
        "Translation vector error by refined motion regime",
        plots_dir
        / "vector_rmse_by_regime.png",
        args.dpi,
    )

    plot_regime_bar(
        regime_statistics,
        "magnitude_ratio_median",
        "Median predicted / GT magnitude",
        "Translation magnitude ratio by refined motion regime",
        plots_dir
        / "magnitude_ratio_by_regime.png",
        args.dpi,
    )

    plot_regime_bar(
        regime_statistics,
        "direction_error_median_degrees",
        "Median direction error (deg)",
        "Translation direction error by refined motion regime",
        plots_dir
        / "direction_error_by_regime.png",
        args.dpi,
    )

    plot_z_gt_pred_mean(
        regime_statistics,
        plots_dir
        / "z_gt_pred_mean_by_regime.png",
        args.dpi,
    )

    plot_z_scatter_by_regime(
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        path=(
            plots_dir
            / "z_scatter_by_regime.png"
        ),
        dpi=args.dpi,
        scatter_alpha=args.scatter_alpha,
    )

    plot_z_timeseries_by_regime(
        dataframe=dataframe,
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        path=(
            plots_dir
            / "z_timeseries_by_regime.png"
        ),
        dpi=args.dpi,
    )

    summary = {
        "input_csv": str(
            args.input_csv.resolve()
        ),
        "sample_count": int(
            len(dataframe)
        ),
        "residual_definition": (
            "prediction_minus_ground_truth"
        ),
        "regime_definitions": [
            {
                "name": "very_low",
                "condition": "z < 0.25",
            },
            {
                "name": "low",
                "condition": (
                    "0.25 <= z < 0.50"
                ),
            },
            {
                "name": "medium_low",
                "condition": (
                    "0.50 <= z < 0.75"
                ),
            },
            {
                "name": "medium",
                "condition": (
                    "0.75 <= z < 1.00"
                ),
            },
            {
                "name": "high",
                "condition": (
                    "1.00 <= z < 1.25"
                ),
            },
            {
                "name": "very_high",
                "condition": (
                    "1.25 <= z < 1.50"
                ),
            },
            {
                "name": "extreme",
                "condition": (
                    "z >= 1.50"
                ),
            },
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
        output_dir
        / "summary.json"
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
        regime_statistics,
        regime_axis_statistics,
        output_dir,
    )


if __name__ == "__main__":
    main()