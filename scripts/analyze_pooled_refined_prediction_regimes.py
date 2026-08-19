"""Pool DeepDCT prediction CSVs and analyze seven tc_z motion regimes.

This script pools evaluator-produced frame_predictions.csv files for
multiple KITTI sequences and evaluates prediction behavior across the same
seven fixed ground-truth directional-translation regimes used in the
refined sequence-10 analysis.

Default sequences
-----------------
00 01 02 03 04 05 06 07 08

Default input layout
--------------------
evaluation/training_semantic_depth_identity_output/
├── sequence_00/frame_predictions.csv
├── sequence_01/frame_predictions.csv
...
└── sequence_08/frame_predictions.csv

Refined tc_z regimes
--------------------
very_low:    z < 0.25
low:         0.25 <= z < 0.50
medium_low:  0.50 <= z < 0.75
medium:      0.75 <= z < 1.00
high:        1.00 <= z < 1.25
very_high:   1.25 <= z < 1.50
extreme:     z >= 1.50

Expected outputs
----------------
pooled_refined_prediction_regime_analysis/
├── summary.json
├── pooled_frame_predictions.csv
├── regime_statistics.csv
├── regime_axis_statistics.csv
├── per_sequence_regime_statistics.csv
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
    ├── regime_fraction_by_sequence.png
    └── z_residual_by_sequence_and_regime.png
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

DEFAULT_SEQUENCES = (
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
)

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
    [
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ],
    dtype=np.float64,
)

EPSILON = 1.0e-12


@dataclass(frozen=True)
class RegimeStatistics:
    """Vector-level statistics for one pooled motion regime."""

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
    """Component-level statistics for one pooled motion regime."""

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Pool DeepDCT frame predictions across KITTI sequences and "
            "analyze seven fixed tc_z motion regimes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "evaluation/"
            "training_semantic_depth_identity_output"
        ),
        help=(
            "Root containing sequence_<NN>/frame_predictions.csv."
        ),
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help="KITTI sequences to pool.",
    )

    parser.add_argument(
        "--prediction-filename",
        type=str,
        default="frame_predictions.csv",
        help="Prediction CSV name within each sequence directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "pooled_refined_prediction_regime_analysis/"
            "semantic_depth_training_00_08"
        ),
        help="Output analysis directory.",
    )

    parser.add_argument(
        "--expected-total-samples",
        type=int,
        default=20400,
        help=(
            "Expected pooled transition count. Set to 0 to disable "
            "the check."
        ),
    )

    parser.add_argument(
        "--direction-min-norm",
        type=float,
        default=1.0e-6,
        help=(
            "Minimum GT and predicted vector norms required for "
            "direction-error calculation."
        ),
    )

    parser.add_argument(
        "--magnitude-ratio-min-gt-norm",
        type=float,
        default=1.0e-6,
        help=(
            "Minimum GT norm required for magnitude-ratio calculation."
        ),
    )

    parser.add_argument(
        "--scatter-alpha",
        type=float,
        default=0.20,
        help="Scatter plot opacity.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved plot resolution.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line configuration."""

    if not args.input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: {args.input_root}"
        )

    if not args.sequences:
        raise ValueError(
            "At least one sequence is required."
        )

    if args.expected_total_samples < 0:
        raise ValueError(
            "--expected-total-samples cannot be negative."
        )

    if args.direction_min_norm < 0.0:
        raise ValueError(
            "--direction-min-norm cannot be negative."
        )

    if args.magnitude_ratio_min_gt_norm < 0.0:
        raise ValueError(
            "--magnitude-ratio-min-gt-norm cannot be negative."
        )

    if not 0.0 < args.scatter_alpha <= 1.0:
        raise ValueError(
            "--scatter-alpha must lie in (0, 1]."
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )


def required_prediction_columns() -> List[str]:
    """Return columns required from each evaluator output."""

    return [
        *GT_COLUMNS.values(),
        *PRED_COLUMNS.values(),
    ]


def load_sequence_predictions(
    *,
    input_root: Path,
    sequence: str,
    prediction_filename: str,
) -> pd.DataFrame:
    """Load and validate one sequence-level prediction CSV."""

    sequence = str(sequence).zfill(2)

    path = (
        input_root
        / f"sequence_{sequence}"
        / prediction_filename
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction CSV for sequence {sequence}: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"Prediction CSV contains no rows: {path}"
        )

    required = required_prediction_columns()

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    for column in required:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    values = dataframe[
        required
    ].to_numpy(
        dtype=np.float64
    )

    invalid = (
        ~np.isfinite(values).all(axis=1)
    )

    if invalid.any():
        indices = np.flatnonzero(
            invalid
        )

        raise ValueError(
            f"Non-finite translation data in sequence {sequence}, "
            f"rows {indices[:20].tolist()}."
        )

    dataframe = dataframe.reset_index(
        drop=True
    )

    # Preserve any existing metadata but explicitly attach the
    # sequence identifier used for pooling.
    dataframe.insert(
        0,
        "pooled_sequence",
        sequence,
    )

    dataframe.insert(
        1,
        "sequence_row_index",
        np.arange(
            len(dataframe),
            dtype=np.int64,
        ),
    )

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


def load_pooled_predictions(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and concatenate every requested sequence."""

    frames: List[pd.DataFrame] = []
    sequence_counts: Dict[str, int] = {}

    for sequence_value in args.sequences:
        sequence = str(
            sequence_value
        ).zfill(2)

        dataframe = load_sequence_predictions(
            input_root=args.input_root,
            sequence=sequence,
            prediction_filename=(
                args.prediction_filename
            ),
        )

        sequence_counts[sequence] = int(
            len(dataframe)
        )

        frames.append(
            dataframe
        )

    pooled = pd.concat(
        frames,
        ignore_index=True,
    )

    pooled.insert(
        2,
        "pooled_global_index",
        np.arange(
            len(pooled),
            dtype=np.int64,
        ),
    )

    if (
        args.expected_total_samples > 0
        and len(pooled)
        != args.expected_total_samples
    ):
        raise ValueError(
            "Pooled sample-count mismatch: "
            f"expected {args.expected_total_samples}, "
            f"received {len(pooled)}. "
            f"Per-sequence counts: {sequence_counts}"
        )

    return (
        pooled,
        sequence_counts,
    )


def extract_translation_arrays(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract GT and prediction arrays shaped [N, 3]."""

    ground_truth = dataframe[
        [GT_COLUMNS[axis] for axis in AXES]
    ].to_numpy(
        dtype=np.float64
    )

    prediction = dataframe[
        [PRED_COLUMNS[axis] for axis in AXES]
    ].to_numpy(
        dtype=np.float64
    )

    return (
        ground_truth,
        prediction,
    )


def classify_regime(
    gt_z: np.ndarray,
) -> np.ndarray:
    """Assign one of the seven fixed motion regimes."""

    assignments = np.digitize(
        gt_z,
        REGIME_BOUNDARIES,
        right=False,
    ).astype(
        np.int64,
        copy=False,
    )

    if (
        np.any(assignments < 0)
        or np.any(
            assignments
            >= len(REGIME_NAMES)
        )
    ):
        raise RuntimeError(
            "Invalid motion-regime assignment."
        )

    return assignments


def sample_standard_deviation(
    values: np.ndarray,
) -> float:
    """Return sample standard deviation."""

    if values.size <= 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def safe_pearson(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Return Pearson correlation where defined."""

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
) -> Tuple[
    float,
    float,
    float,
]:
    """Fit prediction = slope * ground_truth + intercept."""

    if ground_truth.size < 2:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    if np.std(
        ground_truth
    ) <= EPSILON:
        return (
            float("nan"),
            float(
                np.mean(
                    prediction
                )
            ),
            float("nan"),
        )

    slope, intercept = np.polyfit(
        ground_truth,
        prediction,
        deg=1,
    )

    fitted = (
        slope * ground_truth
        + intercept
    )

    residual_sum_squares = float(
        np.sum(
            (
                prediction
                - fitted
            )
            ** 2
        )
    )

    total_sum_squares = float(
        np.sum(
            (
                prediction
                - np.mean(
                    prediction
                )
            )
            ** 2
        )
    )

    if (
        total_sum_squares
        <= EPSILON
    ):
        r_squared = float(
            "nan"
        )
    else:
        r_squared = (
            1.0
            - residual_sum_squares
            / total_sum_squares
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
    """Compute angular translation-direction errors."""

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
        dot_product = np.sum(
            ground_truth[valid]
            * prediction[valid],
            axis=1,
        )

        cosine = (
            dot_product
            / (
                gt_norm[valid]
                * pred_norm[valid]
            )
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        result[valid] = np.degrees(
            np.arccos(
                cosine
            )
        )

    return result


def finite_mean(
    values: np.ndarray,
) -> float:
    finite = values[
        np.isfinite(values)
    ]

    return (
        float(
            np.mean(
                finite
            )
        )
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
        float(
            np.median(
                finite
            )
        )
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


def compute_pooled_statistics(
    *,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    direction_errors: np.ndarray,
    ratio_minimum_gt_norm: float,
) -> Tuple[
    List[RegimeStatistics],
    List[RegimeAxisStatistics],
]:
    """Compute pooled vector- and axis-level regime statistics."""

    regime_rows: List[
        RegimeStatistics
    ] = []

    axis_rows: List[
        RegimeAxisStatistics
    ] = []

    total_count = (
        ground_truth.shape[0]
    )

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

    for (
        regime_index,
        regime_name,
    ) in enumerate(
        REGIME_NAMES
    ):
        mask = (
            assignments
            == regime_index
        )

        count = int(
            np.count_nonzero(
                mask
            )
        )

        if count == 0:
            continue

        gt = ground_truth[
            mask
        ]

        pred = prediction[
            mask
        ]

        gt_norm = gt_norm_all[
            mask
        ]

        pred_norm = pred_norm_all[
            mask
        ]

        vector_error = (
            vector_error_all[
                mask
            ]
        )

        direction_subset = (
            direction_errors[
                mask
            ]
        )

        magnitude_error = (
            pred_norm
            - gt_norm
        )

        magnitude_ratio = np.full(
            gt_norm.shape,
            np.nan,
            dtype=np.float64,
        )

        valid_ratio = (
            gt_norm
            > ratio_minimum_gt_norm
        )

        magnitude_ratio[
            valid_ratio
        ] = (
            pred_norm[
                valid_ratio
            ]
            / gt_norm[
                valid_ratio
            ]
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
                regime_index=(
                    regime_index
                ),
                regime_name=(
                    regime_name
                ),
                count=count,
                fraction=float(
                    count
                    / total_count
                ),
                gt_z_mean=float(
                    np.mean(
                        gt[:, 2]
                    )
                ),
                pred_z_mean=float(
                    np.mean(
                        pred[:, 2]
                    )
                ),
                z_bias=float(
                    np.mean(
                        z_residual
                    )
                ),
                z_mae=float(
                    np.mean(
                        np.abs(
                            z_residual
                        )
                    )
                ),
                z_rmse=float(
                    math.sqrt(
                        z_mse
                    )
                ),
                gt_magnitude_mean=float(
                    np.mean(
                        gt_norm
                    )
                ),
                pred_magnitude_mean=float(
                    np.mean(
                        pred_norm
                    )
                ),
                magnitude_bias=float(
                    np.mean(
                        magnitude_error
                    )
                ),
                magnitude_ratio_mean=(
                    finite_mean(
                        magnitude_ratio
                    )
                ),
                magnitude_ratio_median=(
                    finite_median(
                        magnitude_ratio
                    )
                ),
                direction_error_mean_degrees=(
                    finite_mean(
                        direction_subset
                    )
                ),
                direction_error_median_degrees=(
                    finite_median(
                        direction_subset
                    )
                ),
                direction_error_percentile_90_degrees=(
                    finite_percentile(
                        direction_subset,
                        90.0,
                    )
                ),
                vector_error_norm_mean=float(
                    np.mean(
                        vector_error
                    )
                ),
                vector_error_norm_median=float(
                    np.median(
                        vector_error
                    )
                ),
                vector_error_norm_rmse=float(
                    math.sqrt(
                        float(
                            np.mean(
                                vector_error
                                ** 2
                            )
                        )
                    )
                ),
                vector_error_norm_maximum=float(
                    np.max(
                        vector_error
                    )
                ),
            )
        )

        for (
            axis_index,
            axis,
        ) in enumerate(
            AXES
        ):
            gt_axis = (
                gt[:, axis_index]
            )

            pred_axis = (
                pred[:, axis_index]
            )

            residual = (
                pred_axis
                - gt_axis
            )

            mse = float(
                np.mean(
                    residual ** 2
                )
            )

            (
                slope,
                intercept,
                r_squared,
            ) = linear_regression(
                gt_axis,
                pred_axis,
            )

            axis_rows.append(
                RegimeAxisStatistics(
                    regime_index=(
                        regime_index
                    ),
                    regime_name=(
                        regime_name
                    ),
                    axis=axis,
                    count=count,
                    gt_mean=float(
                        np.mean(
                            gt_axis
                        )
                    ),
                    gt_standard_deviation=(
                        sample_standard_deviation(
                            gt_axis
                        )
                    ),
                    pred_mean=float(
                        np.mean(
                            pred_axis
                        )
                    ),
                    pred_standard_deviation=(
                        sample_standard_deviation(
                            pred_axis
                        )
                    ),
                    bias=float(
                        np.mean(
                            residual
                        )
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
                        math.sqrt(
                            mse
                        )
                    ),
                    pearson_correlation=(
                        safe_pearson(
                            gt_axis,
                            pred_axis,
                        )
                    ),
                    regression_slope=(
                        slope
                    ),
                    regression_intercept=(
                        intercept
                    ),
                    coefficient_of_determination=(
                        r_squared
                    ),
                )
            )

    return (
        regime_rows,
        axis_rows,
    )


def compute_per_sequence_regime_statistics(
    dataframe: pd.DataFrame,
    assignments: np.ndarray,
) -> pd.DataFrame:
    """Compute tc_z prediction statistics per sequence and regime."""

    working = dataframe.copy()

    working[
        "regime_index"
    ] = assignments

    working[
        "regime_name"
    ] = [
        REGIME_NAMES[index]
        for index in assignments
    ]

    working[
        "z_residual"
    ] = (
        working[
            "translation_pred_z"
        ]
        - working[
            "translation_gt_z"
        ]
    )

    rows: List[
        Dict[str, object]
    ] = []

    for sequence in sorted(
        working[
            "pooled_sequence"
        ].unique()
    ):
        sequence_frame = working[
            working[
                "pooled_sequence"
            ]
            == sequence
        ]

        sequence_count = int(
            len(
                sequence_frame
            )
        )

        for (
            regime_index,
            regime_name,
        ) in enumerate(
            REGIME_NAMES
        ):
            subset = sequence_frame[
                sequence_frame[
                    "regime_name"
                ]
                == regime_name
            ]

            count = int(
                len(
                    subset
                )
            )

            row: Dict[
                str,
                object,
            ] = {
                "sequence": sequence,
                "sequence_count": (
                    sequence_count
                ),
                "regime_index": (
                    regime_index
                ),
                "regime_name": (
                    regime_name
                ),
                "count": count,
                "fraction_of_sequence": (
                    count
                    / sequence_count
                    if sequence_count
                    else float("nan")
                ),
            }

            if count:
                gt_z = subset[
                    "translation_gt_z"
                ].to_numpy(
                    dtype=np.float64
                )

                pred_z = subset[
                    "translation_pred_z"
                ].to_numpy(
                    dtype=np.float64
                )

                residual = (
                    pred_z
                    - gt_z
                )

                row.update(
                    {
                        "gt_z_mean": float(
                            np.mean(
                                gt_z
                            )
                        ),
                        "pred_z_mean": float(
                            np.mean(
                                pred_z
                            )
                        ),
                        "z_bias": float(
                            np.mean(
                                residual
                            )
                        ),
                        "z_mae": float(
                            np.mean(
                                np.abs(
                                    residual
                                )
                            )
                        ),
                        "z_rmse": float(
                            math.sqrt(
                                float(
                                    np.mean(
                                        residual
                                        ** 2
                                    )
                                )
                            )
                        ),
                        "z_correlation": (
                            safe_pearson(
                                gt_z,
                                pred_z,
                            )
                        ),
                    }
                )

            else:
                row.update(
                    {
                        "gt_z_mean": float(
                            "nan"
                        ),
                        "pred_z_mean": float(
                            "nan"
                        ),
                        "z_bias": float(
                            "nan"
                        ),
                        "z_mae": float(
                            "nan"
                        ),
                        "z_rmse": float(
                            "nan"
                        ),
                        "z_correlation": float(
                            "nan"
                        ),
                    }
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def write_frame_assignments(
    *,
    dataframe: pd.DataFrame,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    path: Path,
) -> None:
    """Save per-frame regime assignment and residual information."""

    output = pd.DataFrame(
        {
            "sequence": dataframe[
                "pooled_sequence"
            ].to_numpy(),
            "sequence_row_index": dataframe[
                "sequence_row_index"
            ].to_numpy(),
            "pooled_global_index": dataframe[
                "pooled_global_index"
            ].to_numpy(),
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

    for (
        axis_index,
        axis,
    ) in enumerate(
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
            prediction[
                :,
                axis_index,
            ]
            - ground_truth[
                :,
                axis_index,
            ]
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
        prediction
        - ground_truth,
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
    """Save and close one Matplotlib figure."""

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

    plt.close(
        figure
    )


def z_axis_rows(
    axis_rows: Sequence[
        RegimeAxisStatistics
    ],
) -> List[
    RegimeAxisStatistics
]:
    """Return only z-axis statistics."""

    return [
        row
        for row in axis_rows
        if row.axis == "z"
    ]


def plot_z_axis_metric(
    *,
    axis_rows: Sequence[
        RegimeAxisStatistics
    ],
    metric_name: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot one z-axis metric by pooled regime."""

    rows = z_axis_rows(
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
        len(
            labels
        )
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

    axes.set_xlabel(
        "Ground-truth tc_z regime"
    )

    axes.set_ylabel(
        ylabel
    )

    axes.set_title(
        title
    )

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

    if (
        metric_name
        == "regression_slope"
    ):
        axes.axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            label="Ideal slope",
        )

        axes.legend()

    for (
        bar,
        value,
    ) in zip(
        bars,
        values,
    ):
        if math.isfinite(
            value
        ):
            axes.text(
                (
                    bar.get_x()
                    + bar.get_width()
                    / 2.0
                ),
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


def plot_regime_metric(
    *,
    regime_rows: Sequence[
        RegimeStatistics
    ],
    metric_name: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot one vector metric by regime."""

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
        len(
            labels
        )
    )

    figure, axes = plt.subplots(
        figsize=(11, 6)
    )

    axes.bar(
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

    axes.set_xlabel(
        "Ground-truth tc_z regime"
    )

    axes.set_ylabel(
        ylabel
    )

    axes.set_title(
        title
    )

    axes.grid(
        axis="y",
        alpha=0.3,
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
    """Plot mean pooled GT and predicted z."""

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
        len(
            labels
        )
    )

    width = 0.38

    figure, axes = plt.subplots(
        figsize=(12, 6)
    )

    axes.bar(
        positions
        - width / 2.0,
        gt_values,
        width=width,
        label="Ground truth",
    )

    axes.bar(
        positions
        + width / 2.0,
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
        "Pooled training GT versus predicted tc_z"
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


def plot_z_scatter(
    *,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    path: Path,
    dpi: int,
    scatter_alpha: float,
) -> None:
    """Plot pooled tc_z prediction scatter by regime."""

    gt_z = ground_truth[
        :,
        2,
    ]

    pred_z = prediction[
        :,
        2,
    ]

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

    for (
        regime_index,
        regime_name,
    ) in enumerate(
        REGIME_NAMES
    ):
        mask = (
            assignments
            == regime_index
        )

        if not mask.any():
            continue

        axes.scatter(
            gt_z[
                mask
            ],
            pred_z[
                mask
            ],
            s=9,
            alpha=scatter_alpha,
            label=regime_name,
        )

    axes.plot(
        [
            lower,
            upper,
        ],
        [
            lower,
            upper,
        ],
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
        "Pooled semantic-depth training tc_z predictions"
    )

    axes.grid(
        alpha=0.3
    )

    axes.legend(
        ncol=2,
        fontsize=8,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_regime_fraction_by_sequence(
    per_sequence: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot regime fraction for every pooled sequence."""

    sequences = sorted(
        per_sequence[
            "sequence"
        ].unique()
    )

    positions = np.arange(
        len(
            sequences
        )
    )

    bottom = np.zeros(
        len(
            sequences
        ),
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        figsize=(14, 7)
    )

    for regime_name in REGIME_NAMES:
        values = []

        for sequence in sequences:
            row = per_sequence[
                (
                    per_sequence[
                        "sequence"
                    ]
                    == sequence
                )
                & (
                    per_sequence[
                        "regime_name"
                    ]
                    == regime_name
                )
            ]

            if row.empty:
                value = 0.0
            else:
                value = (
                    100.0
                    * float(
                        row.iloc[
                            0
                        ][
                            "fraction_of_sequence"
                        ]
                    )
                )

            values.append(
                value
            )

        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        axes.bar(
            positions,
            values_array,
            bottom=bottom,
            label=regime_name,
        )

        bottom += (
            values_array
        )

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        sequences
    )

    axes.set_ylim(
        0.0,
        100.0,
    )

    axes.set_xlabel(
        "KITTI training sequence"
    )

    axes.set_ylabel(
        "Transitions (%)"
    )

    axes.set_title(
        "Refined motion-regime composition of pooled training predictions"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.legend(
        ncol=2,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_z_residual_heatmap(
    per_sequence: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot mean z residual by sequence and regime."""

    pivot = per_sequence.pivot(
        index="sequence",
        columns="regime_name",
        values="z_bias",
    )

    pivot = pivot.reindex(
        columns=list(
            REGIME_NAMES
        )
    )

    values = pivot.to_numpy(
        dtype=np.float64
    )

    masked = np.ma.masked_invalid(
        values
    )

    figure, axes = plt.subplots(
        figsize=(12, 7)
    )

    image = axes.imshow(
        masked,
        aspect="auto",
    )

    axes.set_xticks(
        np.arange(
            len(
                REGIME_NAMES
            )
        )
    )

    axes.set_xticklabels(
        REGIME_NAMES,
        rotation=30,
        ha="right",
    )

    axes.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    axes.set_yticklabels(
        pivot.index.tolist()
    )

    axes.set_xlabel(
        "Ground-truth tc_z regime"
    )

    axes.set_ylabel(
        "KITTI sequence"
    )

    axes.set_title(
        "Mean tc_z residual by training sequence and motion regime"
    )

    figure.colorbar(
        image,
        ax=axes,
        label=(
            "Mean residual "
            "(prediction - ground truth)"
        ),
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            value = values[
                row_index,
                column_index,
            ]

            if np.isfinite(
                value
            ):
                axes.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    save_figure(
        figure,
        path,
        dpi,
    )


def json_safe(
    value: object,
) -> object:
    """Convert output to strict JSON-compatible values."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for (
                key,
                item,
            ) in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        number = float(
            value
        )

        if not math.isfinite(
            number
        ):
            return None

        return number

    return value


def print_summary(
    *,
    regime_rows: Sequence[
        RegimeStatistics
    ],
    axis_rows: Sequence[
        RegimeAxisStatistics
    ],
    sequence_counts: Mapping[
        str,
        int,
    ],
    output_dir: Path,
) -> None:
    """Print headline pooled results."""

    z_lookup = {
        row.regime_name: row
        for row in axis_rows
        if row.axis == "z"
    }

    print()
    print(
        "=" * 140
    )

    print(
        "Pooled semantic-depth training tc_z prediction-regime analysis"
    )

    print(
        "=" * 140
    )

    print(
        "Sequence counts:"
    )

    for (
        sequence,
        count,
    ) in sorted(
        sequence_counts.items()
    ):
        print(
            f"  {sequence}: {count}"
        )

    print(
        f"  total: "
        f"{sum(sequence_counts.values())}"
    )

    print(
        "-" * 140
    )

    print(
        f"{'Regime':<15}"
        f"{'N':>8}"
        f"{'Frac %':>10}"
        f"{'GT mean':>12}"
        f"{'Pred mean':>12}"
        f"{'Bias':>12}"
        f"{'RMSE':>12}"
        f"{'Corr':>12}"
        f"{'Slope':>12}"
        f"{'Vec RMSE':>14}"
    )

    print(
        "-" * 140
    )

    for regime in regime_rows:
        z_row = z_lookup[
            regime.regime_name
        ]

        print(
            f"{regime.regime_name:<15}"
            f"{regime.count:>8d}"
            f"{100.0 * regime.fraction:>10.2f}"
            f"{regime.gt_z_mean:>12.6f}"
            f"{regime.pred_z_mean:>12.6f}"
            f"{z_row.bias:>12.6f}"
            f"{z_row.rmse:>12.6f}"
            f"{z_row.pearson_correlation:>12.6f}"
            f"{z_row.regression_slope:>12.6f}"
            f"{regime.vector_error_norm_rmse:>14.6f}"
        )

    print(
        "-" * 140
    )

    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )

    print(
        "=" * 140
    )


def main() -> None:
    """Run pooled prediction analysis."""

    args = parse_args()
    validate_args(args)

    output_dir = (
        args.output_dir
    )

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

    (
        dataframe,
        sequence_counts,
    ) = load_pooled_predictions(
        args
    )

    # Preserve the exact pooled input table used by the analysis.
    dataframe.to_csv(
        output_dir
        / "pooled_frame_predictions.csv",
        index=False,
    )

    (
        ground_truth,
        prediction,
    ) = extract_translation_arrays(
        dataframe
    )

    assignments = classify_regime(
        ground_truth[
            :,
            2,
        ]
    )

    direction_errors = compute_direction_errors(
        ground_truth=ground_truth,
        prediction=prediction,
        minimum_norm=(
            args.direction_min_norm
        ),
    )

    (
        regime_statistics,
        axis_statistics,
    ) = compute_pooled_statistics(
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        direction_errors=(
            direction_errors
        ),
        ratio_minimum_gt_norm=(
            args.magnitude_ratio_min_gt_norm
        ),
    )

    per_sequence_statistics = (
        compute_per_sequence_regime_statistics(
            dataframe,
            assignments,
        )
    )

    pd.DataFrame(
        [
            asdict(
                row
            )
            for row in regime_statistics
        ]
    ).to_csv(
        output_dir
        / "regime_statistics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            asdict(
                row
            )
            for row in axis_statistics
        ]
    ).to_csv(
        output_dir
        / "regime_axis_statistics.csv",
        index=False,
    )

    per_sequence_statistics.to_csv(
        output_dir
        / "per_sequence_regime_statistics.csv",
        index=False,
    )

    write_frame_assignments(
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
        axis_rows=axis_statistics,
        metric_name="bias",
        ylabel="Mean residual",
        title=(
            "Pooled training tc_z bias by motion regime"
        ),
        path=(
            plots_dir
            / "z_bias_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_z_axis_metric(
        axis_rows=axis_statistics,
        metric_name="rmse",
        ylabel="RMSE",
        title=(
            "Pooled training tc_z RMSE by motion regime"
        ),
        path=(
            plots_dir
            / "z_rmse_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_z_axis_metric(
        axis_rows=axis_statistics,
        metric_name="pearson_correlation",
        ylabel="Pearson correlation",
        title=(
            "Pooled training tc_z correlation by motion regime"
        ),
        path=(
            plots_dir
            / "z_correlation_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_z_axis_metric(
        axis_rows=axis_statistics,
        metric_name="regression_slope",
        ylabel="Prediction versus GT slope",
        title=(
            "Pooled training tc_z regression slope by motion regime"
        ),
        path=(
            plots_dir
            / "z_regression_slope_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_metric(
        regime_rows=(
            regime_statistics
        ),
        metric_name=(
            "vector_error_norm_rmse"
        ),
        ylabel=(
            "Vector-error norm RMSE"
        ),
        title=(
            "Pooled translation vector error by motion regime"
        ),
        path=(
            plots_dir
            / "vector_rmse_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_metric(
        regime_rows=(
            regime_statistics
        ),
        metric_name=(
            "magnitude_ratio_median"
        ),
        ylabel=(
            "Median predicted / GT magnitude"
        ),
        title=(
            "Pooled translation magnitude ratio by motion regime"
        ),
        path=(
            plots_dir
            / "magnitude_ratio_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_metric(
        regime_rows=(
            regime_statistics
        ),
        metric_name=(
            "direction_error_median_degrees"
        ),
        ylabel=(
            "Median direction error (degrees)"
        ),
        title=(
            "Pooled translation direction error by motion regime"
        ),
        path=(
            plots_dir
            / "direction_error_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_z_gt_pred_mean(
        regime_rows=(
            regime_statistics
        ),
        path=(
            plots_dir
            / "z_gt_pred_mean_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_z_scatter(
        ground_truth=ground_truth,
        prediction=prediction,
        assignments=assignments,
        path=(
            plots_dir
            / "z_scatter_by_regime.png"
        ),
        dpi=args.dpi,
        scatter_alpha=(
            args.scatter_alpha
        ),
    )

    plot_regime_fraction_by_sequence(
        per_sequence=(
            per_sequence_statistics
        ),
        path=(
            plots_dir
            / "regime_fraction_by_sequence.png"
        ),
        dpi=args.dpi,
    )

    plot_z_residual_heatmap(
        per_sequence=(
            per_sequence_statistics
        ),
        path=(
            plots_dir
            / "z_residual_by_sequence_and_regime.png"
        ),
        dpi=args.dpi,
    )

    summary = {
        "input_root": str(
            args.input_root.resolve()
        ),
        "sequences": [
            str(
                sequence
            ).zfill(2)
            for sequence
            in args.sequences
        ],
        "sequence_counts": (
            sequence_counts
        ),
        "total_samples": int(
            len(
                dataframe
            )
        ),
        "residual_definition": (
            "prediction_minus_ground_truth"
        ),
        "regime_definitions": [
            {
                "name": "very_low",
                "condition": (
                    "translation_gt_z < 0.25"
                ),
            },
            {
                "name": "low",
                "condition": (
                    "0.25 <= translation_gt_z < 0.50"
                ),
            },
            {
                "name": "medium_low",
                "condition": (
                    "0.50 <= translation_gt_z < 0.75"
                ),
            },
            {
                "name": "medium",
                "condition": (
                    "0.75 <= translation_gt_z < 1.00"
                ),
            },
            {
                "name": "high",
                "condition": (
                    "1.00 <= translation_gt_z < 1.25"
                ),
            },
            {
                "name": "very_high",
                "condition": (
                    "1.25 <= translation_gt_z < 1.50"
                ),
            },
            {
                "name": "extreme",
                "condition": (
                    "translation_gt_z >= 1.50"
                ),
            },
        ],
        "regime_statistics": [
            asdict(
                row
            )
            for row
            in regime_statistics
        ],
        "regime_axis_statistics": [
            asdict(
                row
            )
            for row
            in axis_statistics
        ],
        "per_sequence_regime_statistics": (
            per_sequence_statistics.to_dict(
                orient="records"
            )
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                summary
            ),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    print_summary(
        regime_rows=(
            regime_statistics
        ),
        axis_rows=(
            axis_statistics
        ),
        sequence_counts=(
            sequence_counts
        ),
        output_dir=(
            output_dir
        ),
    )


if __name__ == "__main__":
    main()