#!/usr/bin/env python3
"""
Analyze structured DeepDCT-VO translation representations.

Purpose
-------
This script audits the translation representation produced by the structured
aggregation experiment:

    translation fusion
        -> Conv C -> K
        -> ReLU
        -> Dropout
        -> AdaptiveAvgPool
        -> compact representation
        -> small MLP
        -> translation prediction

The script is intentionally architecture-agnostic. The representation stored
in the NPZ may be:

    [N, D]
    [N, C, H, W]
    [N, ...]

All dimensions after N are flattened before linear-decoding analysis.

Primary questions
-----------------
1. Does the compact representation retain translation information?
2. Can a single global linear decoder recover x/y/z?
3. Does decoding quality still depend strongly on forward-motion regime?
4. Do regime-specific decoders substantially outperform the global decoder?
5. Are low/medium/high-motion samples geometrically separated in the
   representation?

Expected evaluator outputs
--------------------------
representation directory:
    translation_representations.npz
    frame_predictions.csv

Preferred NPZ key:
    translation_rep

Required CSV columns:
    translation_gt_x
    translation_gt_y
    translation_gt_z

Optional prediction columns:
    translation_pred_x
    translation_pred_y
    translation_pred_z

Outputs
-------
<output-dir>/
├── summary.json
├── representation_statistics.csv
├── decoder_statistics.csv
├── frame_assignments.csv
├── centroid_distances.csv
└── plots/
    ├── representation_pca_by_regime.png
    ├── global_decoder_z_scatter.png
    ├── decoder_z_rmse_by_regime.png
    ├── decoder_z_bias_by_regime.png
    └── decoder_z_correlation_by_regime.png

Example
-------
python scripts/analyze_translation_aggregation_representation.py \\
    --representation-dir \\
    evaluation/sequence_10_baseline_pooled_translation_head \\
    --output-dir \\
    experiments/translation_aggregation_representation/baseline_pooled \\
    --regime-variable z \\
    --num-regimes 3 \\
    --ridge 1e-4

Python compatibility
--------------------
Python 3.8+
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AXES: Tuple[str, ...] = (
    "x",
    "y",
    "z",
)

GT_COLUMNS: Tuple[str, ...] = (
    "translation_gt_x",
    "translation_gt_y",
    "translation_gt_z",
)

PRED_COLUMNS: Tuple[str, ...] = (
    "translation_pred_x",
    "translation_pred_y",
    "translation_pred_z",
)

REPRESENTATION_KEYS: Tuple[str, ...] = (
    "translation_rep",
    "translation_compact_rep",
    "compact_translation_rep",
    "translation_representation",
    "representation",
    "features",
)

EPSILON = 1.0e-12


@dataclass(frozen=True)
class RepresentationStatistics:
    """Geometry statistics for one representation subset."""

    subset: str
    count: int
    dimension: int

    mean_norm: float
    norm_standard_deviation: float

    centroid_norm: float
    within_spread: float

    mean_feature_standard_deviation: float
    maximum_feature_standard_deviation: float


@dataclass(frozen=True)
class DecoderStatistics:
    """Prediction statistics for one decoder/regime/axis."""

    decoder: str
    evaluation_subset: str
    axis: str
    train_count: int
    test_count: int

    gt_mean: float
    gt_standard_deviation: float

    pred_mean: float
    pred_standard_deviation: float

    bias: float
    mae: float
    rmse: float

    pearson_correlation: float
    regression_slope: float
    regression_intercept: float


@dataclass(frozen=True)
class RidgeModel:
    """Stored parameters for one multivariate ridge decoder."""

    weight: np.ndarray
    intercept: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze structured DeepDCT-VO translation representations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--representation-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing translation_representations.npz "
            "and frame_predictions.csv."
        ),
    )

    parser.add_argument(
        "--representation-file",
        type=str,
        default="translation_representations.npz",
        help="Representation NPZ filename.",
    )

    parser.add_argument(
        "--csv-file",
        type=str,
        default="frame_predictions.csv",
        help="Matching evaluator prediction CSV filename.",
    )

    parser.add_argument(
        "--rep-key",
        type=str,
        default=None,
        help=(
            "Explicit NPZ representation key. When omitted, the script "
            "searches known translation-representation keys."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Analysis output directory.",
    )

    parser.add_argument(
        "--regime-variable",
        choices=(
            "z",
            "magnitude",
        ),
        default="z",
        help=(
            "Ground-truth translation quantity used to define "
            "motion regimes."
        ),
    )

    parser.add_argument(
        "--num-regimes",
        type=int,
        default=3,
        help="Number of equal-population motion regimes.",
    )

    parser.add_argument(
        "--regime-names",
        nargs="+",
        default=None,
        help=(
            "Optional regime names. Must match --num-regimes."
        ),
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.30,
        help=(
            "Held-out fraction used for ridge-decoder evaluation. "
            "Splitting is stratified by motion regime."
        ),
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0e-4,
        help="Ridge regularization coefficient.",
    )

    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help=(
            "Disable feature standardization before ridge fitting. "
            "Target centering and an intercept are still used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test partitioning.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved plot resolution.",
    )

    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip the representation PCA visualization.",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate command-line configuration."""

    if not args.representation_dir.is_dir():
        raise NotADirectoryError(
            "Representation directory does not exist: "
            f"{args.representation_dir}"
        )

    if args.num_regimes < 2:
        raise ValueError(
            "--num-regimes must be at least 2."
        )

    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError(
            "--test-fraction must lie strictly between 0 and 1."
        )

    if args.ridge < 0.0:
        raise ValueError(
            "--ridge cannot be negative."
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )

    if args.regime_names is not None:
        if len(args.regime_names) != args.num_regimes:
            raise ValueError(
                "The number of --regime-names must match "
                "--num-regimes."
            )


def resolve_input_paths(
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    """Resolve representation and CSV inputs."""

    representation_path = (
        args.representation_dir
        / args.representation_file
    )

    csv_path = (
        args.representation_dir
        / args.csv_file
    )

    if not representation_path.is_file():
        raise FileNotFoundError(
            "Representation file not found: "
            f"{representation_path}"
        )

    if not csv_path.is_file():
        raise FileNotFoundError(
            "Prediction CSV not found: "
            f"{csv_path}"
        )

    return (
        representation_path,
        csv_path,
    )


def load_representation(
    path: Path,
    requested_key: Optional[str],
) -> Tuple[np.ndarray, str, Tuple[int, ...]]:
    """Load and flatten one translation representation."""

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        available_keys = list(
            data.keys()
        )

        if requested_key is not None:
            if requested_key not in data:
                raise KeyError(
                    f"Representation key {requested_key!r} "
                    f"not found in {path}.\n"
                    f"Available keys: {available_keys}"
                )

            selected_key = requested_key

        else:
            selected_key = next(
                (
                    key
                    for key in REPRESENTATION_KEYS
                    if key in data
                ),
                None,
            )

            if selected_key is None:
                raise KeyError(
                    "Could not identify a translation representation "
                    f"in {path}.\n"
                    f"Expected one of: {REPRESENTATION_KEYS}\n"
                    f"Available keys: {available_keys}"
                )

        representation = np.asarray(
            data[selected_key],
            dtype=np.float64,
        )

    raw_shape = tuple(
        representation.shape
    )

    if representation.ndim < 2:
        raise ValueError(
            "Representation must have shape [N, ...], "
            f"but received {raw_shape}."
        )

    if representation.shape[0] == 0:
        raise ValueError(
            "Representation contains zero samples."
        )

    representation = representation.reshape(
        representation.shape[0],
        -1,
    )

    if not np.isfinite(
        representation
    ).all():
        raise ValueError(
            "Representation contains NaN or infinite values."
        )

    return (
        representation,
        selected_key,
        raw_shape,
    )


def load_prediction_csv(
    path: Path,
) -> pd.DataFrame:
    """Load and validate matching translation targets."""

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"Prediction CSV contains no rows: {path}"
        )

    missing = [
        column
        for column in GT_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            "Prediction CSV is missing required ground-truth "
            f"columns: {missing}"
        )

    numeric_columns = list(
        GT_COLUMNS
    )

    for column in PRED_COLUMNS:
        if column in dataframe.columns:
            numeric_columns.append(
                column
            )

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid = ~np.isfinite(
        dataframe[
            list(GT_COLUMNS)
        ].to_numpy(
            dtype=np.float64
        )
    ).all(
        axis=1
    )

    if invalid.any():
        indices = np.flatnonzero(
            invalid
        )

        raise ValueError(
            "Ground-truth translation contains non-finite "
            f"values at CSV rows {indices[:20].tolist()}."
        )

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe


def extract_ground_truth(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Return translation GT shaped [N, 3]."""

    return dataframe[
        list(GT_COLUMNS)
    ].to_numpy(
        dtype=np.float64
    )


def extract_model_prediction(
    dataframe: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Return evaluator translation prediction when available."""

    if not all(
        column in dataframe.columns
        for column in PRED_COLUMNS
    ):
        return None

    prediction = dataframe[
        list(PRED_COLUMNS)
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        prediction
    ).all():
        return None

    return prediction


def sample_standard_deviation(
    values: np.ndarray,
) -> float:
    """Return population standard deviation."""

    if values.size == 0:
        return float("nan")

    return float(
        np.std(
            values,
            ddof=0,
        )
    )


def safe_pearson(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Return Pearson correlation or NaN for constant arrays."""

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


def linear_fit(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> Tuple[
    float,
    float,
]:
    """Fit prediction = slope * ground_truth + intercept."""

    if ground_truth.size < 2:
        return (
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
        )

    slope, intercept = np.polyfit(
        ground_truth,
        prediction,
        deg=1,
    )

    return (
        float(slope),
        float(intercept),
    )


def default_regime_names(
    count: int,
) -> List[str]:
    """Generate readable regime names."""

    if count == 3:
        return [
            "low",
            "medium",
            "high",
        ]

    return [
        f"regime_{index + 1:02d}"
        for index in range(count)
    ]


def make_strict_edges(
    edges: np.ndarray,
) -> np.ndarray:
    """Repair duplicate quantile boundaries."""

    result = np.asarray(
        edges,
        dtype=np.float64,
    ).copy()

    for index in range(
        1,
        result.size,
    ):
        if (
            result[index]
            <= result[index - 1]
        ):
            result[index] = np.nextafter(
                result[index - 1],
                np.inf,
            )

    return result


def build_motion_regimes(
    ground_truth: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
]:
    """Assign each sample to a quantile motion regime."""

    if args.regime_variable == "z":
        regime_values = ground_truth[
            :,
            2,
        ]

    else:
        regime_values = np.linalg.norm(
            ground_truth,
            axis=1,
        )

    quantiles = np.linspace(
        0.0,
        1.0,
        args.num_regimes + 1,
    )

    edges = np.quantile(
        regime_values,
        quantiles,
    )

    edges = make_strict_edges(
        edges
    )

    names = (
        list(
            args.regime_names
        )
        if args.regime_names is not None
        else default_regime_names(
            args.num_regimes
        )
    )

    assignments = np.full(
        regime_values.shape,
        -1,
        dtype=np.int64,
    )

    for index in range(
        args.num_regimes
    ):
        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if (
            index
            == args.num_regimes - 1
        ):
            mask = (
                (regime_values >= lower)
                & (regime_values <= upper)
            )

        else:
            mask = (
                (regime_values >= lower)
                & (regime_values < upper)
            )

        assignments[
            mask
        ] = index

    if np.any(
        assignments < 0
    ):
        raise RuntimeError(
            "Some samples could not be assigned to a motion regime."
        )

    return (
        assignments,
        regime_values,
        names,
        edges,
    )


def stratified_train_test_split(
    assignments: np.ndarray,
    regime_count: int,
    test_fraction: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create a deterministic regime-stratified holdout split."""

    random = np.random.RandomState(
        seed
    )

    train_indices: List[int] = []
    test_indices: List[int] = []

    for regime_index in range(
        regime_count
    ):
        indices = np.flatnonzero(
            assignments
            == regime_index
        )

        if indices.size < 3:
            raise ValueError(
                f"Regime {regime_index} contains only "
                f"{indices.size} samples; at least 3 are required."
            )

        indices = indices.copy()

        random.shuffle(
            indices
        )

        test_count = int(
            round(
                indices.size
                * test_fraction
            )
        )

        test_count = max(
            1,
            test_count,
        )

        test_count = min(
            test_count,
            indices.size - 2,
        )

        test_indices.extend(
            indices[
                :test_count
            ].tolist()
        )

        train_indices.extend(
            indices[
                test_count:
            ].tolist()
        )

    train = np.asarray(
        sorted(
            train_indices
        ),
        dtype=np.int64,
    )

    test = np.asarray(
        sorted(
            test_indices
        ),
        dtype=np.int64,
    )

    if np.intersect1d(
        train,
        test,
    ).size:
        raise RuntimeError(
            "Train/test split contains overlapping samples."
        )

    return (
        train,
        test,
    )


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    standardize: bool,
) -> RidgeModel:
    """Fit a three-output ridge regression model.

    Uses either the primal or dual form depending on feature dimensionality,
    avoiding a D x D matrix when D is much larger than the number of samples.
    """

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [N, D]."
        )

    if (
        targets.ndim != 2
        or targets.shape[1] != 3
    ):
        raise ValueError(
            "targets must have shape [N, 3]."
        )

    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "Feature and target sample counts do not match."
        )

    feature_mean = np.mean(
        features,
        axis=0,
    )

    centered_features = (
        features
        - feature_mean
    )

    if standardize:
        feature_scale = np.std(
            centered_features,
            axis=0,
            ddof=0,
        )

        feature_scale[
            feature_scale
            <= EPSILON
        ] = 1.0

    else:
        feature_scale = np.ones(
            features.shape[1],
            dtype=np.float64,
        )

    normalized_features = (
        centered_features
        / feature_scale
    )

    target_mean = np.mean(
        targets,
        axis=0,
    )

    centered_targets = (
        targets
        - target_mean
    )

    sample_count = (
        normalized_features.shape[0]
    )

    feature_count = (
        normalized_features.shape[1]
    )

    if feature_count <= sample_count:
        # Primal form:
        #
        #   W = (X^T X + λI)^-1 X^T Y
        gram = (
            normalized_features.T
            @ normalized_features
        )

        regularized = (
            gram
            + ridge
            * np.eye(
                feature_count,
                dtype=np.float64,
            )
        )

        right_hand_side = (
            normalized_features.T
            @ centered_targets
        )

        try:
            normalized_weight = np.linalg.solve(
                regularized,
                right_hand_side,
            )

        except np.linalg.LinAlgError:
            normalized_weight = (
                np.linalg.pinv(
                    regularized
                )
                @ right_hand_side
            )

    else:
        # Dual form:
        #
        #   W = X^T (X X^T + λI)^-1 Y
        #
        # This is important for the original 14,400-D representation.
        sample_gram = (
            normalized_features
            @ normalized_features.T
        )

        regularized = (
            sample_gram
            + ridge
            * np.eye(
                sample_count,
                dtype=np.float64,
            )
        )

        try:
            dual_solution = np.linalg.solve(
                regularized,
                centered_targets,
            )

        except np.linalg.LinAlgError:
            dual_solution = (
                np.linalg.pinv(
                    regularized
                )
                @ centered_targets
            )

        normalized_weight = (
            normalized_features.T
            @ dual_solution
        )

    # Convert the standardized-feature model back into the original
    # feature coordinate system.
    weight = (
        normalized_weight
        / feature_scale[:, None]
    )

    intercept = (
        target_mean
        - feature_mean
        @ weight
    )

    return RidgeModel(
        weight=weight,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def ridge_predict(
    model: RidgeModel,
    features: np.ndarray,
) -> np.ndarray:
    """Apply a fitted ridge decoder."""

    return (
        features
        @ model.weight
        + model.intercept
    )


def build_decoder_rows(
    decoder_name: str,
    evaluation_subset: str,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    train_count: int,
) -> List[
    DecoderStatistics
]:
    """Build x/y/z decoder statistics."""

    rows: List[
        DecoderStatistics
    ] = []

    for axis_index, axis in enumerate(
        AXES
    ):
        gt_axis = ground_truth[
            :,
            axis_index,
        ]

        pred_axis = prediction[
            :,
            axis_index,
        ]

        residual = (
            pred_axis
            - gt_axis
        )

        slope, intercept = linear_fit(
            gt_axis,
            pred_axis,
        )

        rows.append(
            DecoderStatistics(
                decoder=decoder_name,
                evaluation_subset=(
                    evaluation_subset
                ),
                axis=axis,
                train_count=int(
                    train_count
                ),
                test_count=int(
                    gt_axis.size
                ),
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
                mae=float(
                    np.mean(
                        np.abs(
                            residual
                        )
                    )
                ),
                rmse=float(
                    math.sqrt(
                        float(
                            np.mean(
                                residual ** 2
                            )
                        )
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
            )
        )

    return rows


def compute_representation_statistics(
    representation: np.ndarray,
    assignments: np.ndarray,
    regime_names: Sequence[str],
) -> List[
    RepresentationStatistics
]:
    """Compute overall and within-regime representation geometry."""

    rows: List[
        RepresentationStatistics
    ] = []

    subsets: List[
        Tuple[str, np.ndarray]
    ] = [
        (
            "all",
            np.arange(
                representation.shape[0]
            ),
        )
    ]

    for regime_index, regime_name in enumerate(
        regime_names
    ):
        subsets.append(
            (
                regime_name,
                np.flatnonzero(
                    assignments
                    == regime_index
                ),
            )
        )

    for subset_name, indices in subsets:
        features = representation[
            indices
        ]

        centroid = np.mean(
            features,
            axis=0,
        )

        sample_norms = np.linalg.norm(
            features,
            axis=1,
        )

        centered = (
            features
            - centroid
        )

        within_spread = float(
            math.sqrt(
                float(
                    np.mean(
                        np.sum(
                            centered ** 2,
                            axis=1,
                        )
                    )
                )
            )
        )

        feature_standard_deviation = np.std(
            features,
            axis=0,
            ddof=0,
        )

        rows.append(
            RepresentationStatistics(
                subset=subset_name,
                count=int(
                    features.shape[0]
                ),
                dimension=int(
                    features.shape[1]
                ),
                mean_norm=float(
                    np.mean(
                        sample_norms
                    )
                ),
                norm_standard_deviation=float(
                    np.std(
                        sample_norms,
                        ddof=0,
                    )
                ),
                centroid_norm=float(
                    np.linalg.norm(
                        centroid
                    )
                ),
                within_spread=(
                    within_spread
                ),
                mean_feature_standard_deviation=float(
                    np.mean(
                        feature_standard_deviation
                    )
                ),
                maximum_feature_standard_deviation=float(
                    np.max(
                        feature_standard_deviation
                    )
                ),
            )
        )

    return rows


def compute_centroid_distances(
    representation: np.ndarray,
    assignments: np.ndarray,
    regime_names: Sequence[str],
) -> pd.DataFrame:
    """Compute pairwise Euclidean distances between regime centroids."""

    centroids = {}

    for regime_index, name in enumerate(
        regime_names
    ):
        features = representation[
            assignments
            == regime_index
        ]

        centroids[
            name
        ] = np.mean(
            features,
            axis=0,
        )

    rows: List[
        Dict[str, object]
    ] = []

    for source_name in regime_names:
        for target_name in regime_names:
            distance = np.linalg.norm(
                centroids[source_name]
                - centroids[target_name]
            )

            rows.append(
                {
                    "source_regime": (
                        source_name
                    ),
                    "target_regime": (
                        target_name
                    ),
                    "centroid_distance": float(
                        distance
                    ),
                }
            )

    return pd.DataFrame.from_records(
        rows
    )


def run_decoder_analysis(
    representation: np.ndarray,
    ground_truth: np.ndarray,
    assignments: np.ndarray,
    regime_names: Sequence[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    ridge: float,
    standardize: bool,
) -> Tuple[
    List[DecoderStatistics],
    np.ndarray,
]:
    """Fit global and regime-specific ridge decoders."""

    rows: List[
        DecoderStatistics
    ] = []

    global_model = fit_ridge(
        features=representation[
            train_indices
        ],
        targets=ground_truth[
            train_indices
        ],
        ridge=ridge,
        standardize=standardize,
    )

    global_test_prediction = ridge_predict(
        global_model,
        representation[
            test_indices
        ],
    )

    rows.extend(
        build_decoder_rows(
            decoder_name="global_ridge",
            evaluation_subset="all",
            ground_truth=ground_truth[
                test_indices
            ],
            prediction=global_test_prediction,
            train_count=int(
                train_indices.size
            ),
        )
    )

    # Evaluate the same global mapping within each held-out regime.
    for regime_index, regime_name in enumerate(
        regime_names
    ):
        local_test_mask = (
            assignments[
                test_indices
            ]
            == regime_index
        )

        regime_test_indices = (
            test_indices[
                local_test_mask
            ]
        )

        regime_prediction = ridge_predict(
            global_model,
            representation[
                regime_test_indices
            ],
        )

        rows.extend(
            build_decoder_rows(
                decoder_name="global_ridge",
                evaluation_subset=regime_name,
                ground_truth=ground_truth[
                    regime_test_indices
                ],
                prediction=regime_prediction,
                train_count=int(
                    train_indices.size
                ),
            )
        )

    # Train a separate ridge decoder within each regime.
    for regime_index, regime_name in enumerate(
        regime_names
    ):
        regime_train_indices = train_indices[
            assignments[
                train_indices
            ]
            == regime_index
        ]

        regime_test_indices = test_indices[
            assignments[
                test_indices
            ]
            == regime_index
        ]

        if (
            regime_train_indices.size
            < 2
        ):
            raise ValueError(
                f"Regime {regime_name!r} has insufficient "
                "training samples."
            )

        regime_model = fit_ridge(
            features=representation[
                regime_train_indices
            ],
            targets=ground_truth[
                regime_train_indices
            ],
            ridge=ridge,
            standardize=standardize,
        )

        regime_prediction = ridge_predict(
            regime_model,
            representation[
                regime_test_indices
            ],
        )

        rows.extend(
            build_decoder_rows(
                decoder_name=(
                    "within_regime_ridge"
                ),
                evaluation_subset=(
                    regime_name
                ),
                ground_truth=ground_truth[
                    regime_test_indices
                ],
                prediction=regime_prediction,
                train_count=int(
                    regime_train_indices.size
                ),
            )
        )

    return (
        rows,
        global_test_prediction,
    )


def write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
) -> None:
    """Write dataclass rows to CSV."""

    if not rows:
        raise ValueError(
            f"Cannot write empty table: {path}"
        )

    dictionaries = [
        asdict(
            row
        )
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
                dictionaries[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            dictionaries
        )


def save_figure(
    figure: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    """Save one Matplotlib figure."""

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


def compute_pca_2d(
    representation: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """Compute a two-dimensional PCA projection.

    Uses an eigen-decomposition of the sample Gram matrix when the
    representation has more dimensions than samples. This avoids forming
    a potentially very large D x D covariance matrix.
    """

    centered = (
        representation
        - np.mean(
            representation,
            axis=0,
        )
    )

    sample_count = centered.shape[
        0
    ]

    feature_count = centered.shape[
        1
    ]

    if feature_count <= sample_count:
        _, singular_values, vt = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        components = vt[
            :2
        ].T

        projection = (
            centered
            @ components
        )

        denominator = max(
            sample_count - 1,
            1,
        )

        explained_variance = (
            singular_values[
                :2
            ]
            ** 2
            / denominator
        )

    else:
        gram = (
            centered
            @ centered.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            gram
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = np.maximum(
            eigenvalues[
                order
            ],
            0.0,
        )

        eigenvectors = eigenvectors[
            :,
            order,
        ]

        top_values = eigenvalues[
            :2
        ]

        top_vectors = eigenvectors[
            :,
            :2
        ]

        projection = (
            top_vectors
            * np.sqrt(
                top_values
            )[None, :]
        )

        denominator = max(
            sample_count - 1,
            1,
        )

        explained_variance = (
            top_values
            / denominator
        )

    total_variance = float(
        np.sum(
            np.var(
                centered,
                axis=0,
                ddof=1,
            )
        )
    )

    if total_variance > EPSILON:
        explained_ratio = (
            explained_variance
            / total_variance
        )

    else:
        explained_ratio = np.zeros(
            2,
            dtype=np.float64,
        )

    return (
        projection,
        explained_ratio,
    )


def plot_pca(
    representation: np.ndarray,
    assignments: np.ndarray,
    regime_names: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot the first two PCA dimensions by motion regime."""

    projection, explained_ratio = (
        compute_pca_2d(
            representation
        )
    )

    figure, axes = plt.subplots(
        figsize=(9, 7)
    )

    for regime_index, name in enumerate(
        regime_names
    ):
        mask = (
            assignments
            == regime_index
        )

        axes.scatter(
            projection[
                mask,
                0,
            ],
            projection[
                mask,
                1,
            ],
            s=14,
            alpha=0.45,
            label=name,
        )

    axes.set_xlabel(
        "PC1 "
        f"({100.0 * explained_ratio[0]:.2f}% variance)"
    )

    axes.set_ylabel(
        "PC2 "
        f"({100.0 * explained_ratio[1]:.2f}% variance)"
    )

    axes.set_title(
        "Translation representation by motion regime"
    )

    axes.grid(
        alpha=0.25
    )

    axes.legend()

    save_figure(
        figure,
        output_path,
        dpi,
    )


def plot_global_decoder_z_scatter(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    assignments: np.ndarray,
    regime_names: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot held-out z prediction from the global representation probe."""

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
            np.min(
                gt_z
            ),
            np.min(
                pred_z
            ),
        )
    )

    upper = float(
        max(
            np.max(
                gt_z
            ),
            np.max(
                pred_z
            ),
        )
    )

    figure, axes = plt.subplots(
        figsize=(8, 8)
    )

    for regime_index, name in enumerate(
        regime_names
    ):
        mask = (
            assignments
            == regime_index
        )

        axes.scatter(
            gt_z[
                mask
            ],
            pred_z[
                mask
            ],
            s=18,
            alpha=0.50,
            label=name,
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
        label="Ideal",
    )

    axes.set_xlabel(
        "Ground-truth translation z"
    )

    axes.set_ylabel(
        "Global ridge decoded z"
    )

    axes.set_title(
        "Held-out linear decoding of forward translation"
    )

    axes.grid(
        alpha=0.25
    )

    axes.legend()

    save_figure(
        figure,
        output_path,
        dpi,
    )


def plot_decoder_metric(
    decoder_rows: Sequence[DecoderStatistics],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Compare z-axis global and regime-specific decoder statistics."""

    z_rows = [
        row
        for row in decoder_rows
        if (
            row.axis == "z"
            and row.evaluation_subset
            != "all"
        )
    ]

    regime_names: List[str] = []

    for row in z_rows:
        if (
            row.evaluation_subset
            not in regime_names
        ):
            regime_names.append(
                row.evaluation_subset
            )

    decoder_names = [
        "global_ridge",
        "within_regime_ridge",
    ]

    lookup = {
        (
            row.decoder,
            row.evaluation_subset,
        ): row
        for row in z_rows
    }

    positions = np.arange(
        len(
            regime_names
        )
    )

    width = 0.36

    figure, axes = plt.subplots(
        figsize=(10, 6)
    )

    for decoder_index, decoder_name in enumerate(
        decoder_names
    ):
        values = []

        for regime_name in regime_names:
            row = lookup.get(
                (
                    decoder_name,
                    regime_name,
                )
            )

            value = (
                float(
                    getattr(
                        row,
                        metric,
                    )
                )
                if row is not None
                else float("nan")
            )

            values.append(
                value
            )

        offset = (
            decoder_index
            - 0.5
        ) * width

        axes.bar(
            positions
            + offset,
            values,
            width=width,
            label=decoder_name,
        )

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        regime_names
    )

    axes.set_xlabel(
        "Motion regime"
    )

    axes.set_ylabel(
        ylabel
    )

    axes.set_title(
        title
    )

    axes.grid(
        axis="y",
        alpha=0.25,
    )

    axes.legend()

    save_figure(
        figure,
        output_path,
        dpi,
    )


def dataframe_json_safe(
    dataframe: pd.DataFrame,
) -> List[
    Dict[str, object]
]:
    """Convert a dataframe to strict JSON-compatible records."""

    output: List[
        Dict[str, object]
    ] = []

    for raw_record in dataframe.to_dict(
        orient="records"
    ):
        record: Dict[
            str,
            object,
        ] = {}

        for key, value in raw_record.items():
            if pd.isna(
                value
            ):
                record[
                    str(key)
                ] = None

            elif isinstance(
                value,
                (
                    np.integer,
                    int,
                ),
            ):
                record[
                    str(key)
                ] = int(
                    value
                )

            elif isinstance(
                value,
                (
                    np.floating,
                    float,
                ),
            ):
                numeric = float(
                    value
                )

                record[
                    str(key)
                ] = (
                    numeric
                    if math.isfinite(
                        numeric
                    )
                    else None
                )

            else:
                record[
                    str(key)
                ] = value

        output.append(
            record
        )

    return output


def json_safe(
    value: object,
) -> object:
    """Recursively convert NumPy and non-finite values."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): json_safe(
                item
            )
            for key, item in value.items()
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
        numeric = float(
            value
        )

        return (
            numeric
            if math.isfinite(
                numeric
            )
            else None
        )

    return value


def build_frame_assignments(
    dataframe: pd.DataFrame,
    assignments: np.ndarray,
    regime_values: np.ndarray,
    regime_names: Sequence[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> pd.DataFrame:
    """Build frame-level regime and train/test assignments."""

    output = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(
                    dataframe
                ),
                dtype=np.int64,
            ),
            "regime_value": (
                regime_values
            ),
            "regime_index": (
                assignments
            ),
            "regime_name": [
                regime_names[
                    index
                ]
                for index in assignments
            ],
        }
    )

    split = np.full(
        len(
            dataframe
        ),
        "",
        dtype=object,
    )

    split[
        train_indices
    ] = "train"

    split[
        test_indices
    ] = "test"

    output[
        "probe_split"
    ] = split

    for column in (
        "sequence",
        "frame_prev",
        "frame_curr",
    ):
        if column in dataframe.columns:
            output[
                column
            ] = dataframe[
                column
            ].to_numpy()

    for column in GT_COLUMNS:
        output[
            column
        ] = dataframe[
            column
        ].to_numpy(
            dtype=np.float64
        )

    for column in PRED_COLUMNS:
        if column in dataframe.columns:
            output[
                column
            ] = dataframe[
                column
            ].to_numpy(
                dtype=np.float64
            )

    return output


def print_summary(
    representation_key: str,
    raw_shape: Tuple[int, ...],
    representation: np.ndarray,
    representation_rows: Sequence[
        RepresentationStatistics
    ],
    decoder_rows: Sequence[
        DecoderStatistics
    ],
    output_dir: Path,
) -> None:
    """Print the principal audit findings."""

    print()
    print("=" * 88)
    print(
        "Translation Aggregation Representation Audit"
    )
    print("=" * 88)

    print(
        f"Representation key:     "
        f"{representation_key}"
    )

    print(
        f"Raw representation:     "
        f"{raw_shape}"
    )

    print(
        f"Flattened shape:        "
        f"{representation.shape}"
    )

    print("-" * 88)
    print(
        "Representation statistics"
    )
    print("-" * 88)

    for row in representation_rows:
        print(
            f"{row.subset:<10} "
            f"n={row.count:4d} "
            f"norm={row.mean_norm:.6f} "
            f"centroid_norm={row.centroid_norm:.6f} "
            f"within_spread={row.within_spread:.6f}"
        )

    print()
    print("=" * 88)
    print(
        "Linear z decoding by forward-motion regime"
    )
    print("=" * 88)

    z_rows = [
        row
        for row in decoder_rows
        if (
            row.axis == "z"
            and row.evaluation_subset
            != "all"
        )
    ]

    for decoder_name in (
        "global_ridge",
        "within_regime_ridge",
    ):
        for row in z_rows:
            if (
                row.decoder
                != decoder_name
            ):
                continue

            print(
                f"{decoder_name:<24} "
                f"{row.evaluation_subset:<8} "
                f"n={row.test_count:4d} "
                f"z_rmse={row.rmse:.6f} "
                f"z_bias={row.bias:+.6f} "
                f"z_corr={row.pearson_correlation:.4f} "
                f"z_slope={row.regression_slope:.4f}"
            )

    print("-" * 88)

    print(
        "Output directory:       "
        f"{output_dir.resolve()}"
    )

    print("=" * 88)


def main() -> None:
    """Run the representation audit."""

    args = parse_args()

    validate_args(
        args
    )

    representation_path, csv_path = (
        resolve_input_paths(
            args
        )
    )

    output_dir = args.output_dir

    plot_dir = (
        output_dir
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        representation,
        representation_key,
        raw_shape,
    ) = load_representation(
        representation_path,
        args.rep_key,
    )

    dataframe = load_prediction_csv(
        csv_path
    )

    ground_truth = extract_ground_truth(
        dataframe
    )

    model_prediction = extract_model_prediction(
        dataframe
    )

    if (
        representation.shape[0]
        != ground_truth.shape[0]
    ):
        raise ValueError(
            "Representation and prediction CSV sample counts "
            "do not match: "
            f"{representation.shape[0]} vs "
            f"{ground_truth.shape[0]}."
        )

    (
        assignments,
        regime_values,
        regime_names,
        regime_edges,
    ) = build_motion_regimes(
        ground_truth,
        args,
    )

    (
        train_indices,
        test_indices,
    ) = stratified_train_test_split(
        assignments=assignments,
        regime_count=args.num_regimes,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    representation_rows = (
        compute_representation_statistics(
            representation=representation,
            assignments=assignments,
            regime_names=regime_names,
        )
    )

    centroid_distances = (
        compute_centroid_distances(
            representation=representation,
            assignments=assignments,
            regime_names=regime_names,
        )
    )

    (
        decoder_rows,
        global_test_prediction,
    ) = run_decoder_analysis(
        representation=representation,
        ground_truth=ground_truth,
        assignments=assignments,
        regime_names=regime_names,
        train_indices=train_indices,
        test_indices=test_indices,
        ridge=args.ridge,
        standardize=(
            not args.no_standardize
        ),
    )

    # Include the actual neural-network output as an optional reference.
    if model_prediction is not None:
        decoder_rows.extend(
            build_decoder_rows(
                decoder_name="network_output",
                evaluation_subset="all",
                ground_truth=ground_truth,
                prediction=model_prediction,
                train_count=0,
            )
        )

        for regime_index, regime_name in enumerate(
            regime_names
        ):
            indices = np.flatnonzero(
                assignments
                == regime_index
            )

            decoder_rows.extend(
                build_decoder_rows(
                    decoder_name="network_output",
                    evaluation_subset=regime_name,
                    ground_truth=ground_truth[
                        indices
                    ],
                    prediction=model_prediction[
                        indices
                    ],
                    train_count=0,
                )
            )

    write_dataclass_csv(
        output_dir
        / "representation_statistics.csv",
        representation_rows,
    )

    write_dataclass_csv(
        output_dir
        / "decoder_statistics.csv",
        decoder_rows,
    )

    centroid_distances.to_csv(
        output_dir
        / "centroid_distances.csv",
        index=False,
    )

    frame_assignments = build_frame_assignments(
        dataframe=dataframe,
        assignments=assignments,
        regime_values=regime_values,
        regime_names=regime_names,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    frame_assignments.to_csv(
        output_dir
        / "frame_assignments.csv",
        index=False,
    )

    if not args.skip_pca:
        plot_pca(
            representation=representation,
            assignments=assignments,
            regime_names=regime_names,
            output_path=(
                plot_dir
                / "representation_pca_by_regime.png"
            ),
            dpi=args.dpi,
        )

    test_regime_assignments = assignments[
        test_indices
    ]

    plot_global_decoder_z_scatter(
        ground_truth=ground_truth[
            test_indices
        ],
        prediction=global_test_prediction,
        assignments=(
            test_regime_assignments
        ),
        regime_names=regime_names,
        output_path=(
            plot_dir
            / "global_decoder_z_scatter.png"
        ),
        dpi=args.dpi,
    )

    plot_decoder_metric(
        decoder_rows=decoder_rows,
        metric="rmse",
        ylabel="Held-out z RMSE",
        title=(
            "Global versus regime-specific z decoding"
        ),
        output_path=(
            plot_dir
            / "decoder_z_rmse_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_decoder_metric(
        decoder_rows=decoder_rows,
        metric="bias",
        ylabel="Held-out z bias",
        title=(
            "Global versus regime-specific z bias"
        ),
        output_path=(
            plot_dir
            / "decoder_z_bias_by_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_decoder_metric(
        decoder_rows=decoder_rows,
        metric="pearson_correlation",
        ylabel="Held-out z correlation",
        title=(
            "Global versus regime-specific z correlation"
        ),
        output_path=(
            plot_dir
            / "decoder_z_correlation_by_regime.png"
        ),
        dpi=args.dpi,
    )

    decoder_dataframe = pd.DataFrame(
        [
            asdict(
                row
            )
            for row in decoder_rows
        ]
    )

    representation_dataframe = pd.DataFrame(
        [
            asdict(
                row
            )
            for row in representation_rows
        ]
    )

    summary: Dict[
        str,
        object,
    ] = {
        "representation": {
            "file": str(
                representation_path.resolve()
            ),
            "key": (
                representation_key
            ),
            "raw_shape": list(
                raw_shape
            ),
            "flattened_shape": list(
                representation.shape
            ),
            "dimension": int(
                representation.shape[1]
            ),
        },
        "prediction_csv": str(
            csv_path.resolve()
        ),
        "sample_count": int(
            representation.shape[0]
        ),
        "regime_definition": {
            "variable": (
                args.regime_variable
            ),
            "count": int(
                args.num_regimes
            ),
            "names": list(
                regime_names
            ),
            "edges": [
                float(
                    value
                )
                for value in regime_edges
            ],
        },
        "probe_configuration": {
            "ridge": float(
                args.ridge
            ),
            "standardize": bool(
                not args.no_standardize
            ),
            "test_fraction": float(
                args.test_fraction
            ),
            "train_count": int(
                train_indices.size
            ),
            "test_count": int(
                test_indices.size
            ),
            "seed": int(
                args.seed
            ),
            "evaluation_note": (
                "Ridge decoders are fitted only on the probe training "
                "partition and evaluated only on held-out samples. "
                "The split is stratified by motion regime."
            ),
        },
        "representation_statistics": (
            dataframe_json_safe(
                representation_dataframe
            )
        ),
        "decoder_statistics": (
            dataframe_json_safe(
                decoder_dataframe
            )
        ),
        "centroid_distances": (
            dataframe_json_safe(
                centroid_distances
            )
        ),
        "network_prediction_available": bool(
            model_prediction
            is not None
        ),
        "interpretation": {
            "global_vs_within_regime": (
                "If within-regime ridge decoding substantially "
                "outperforms global ridge decoding, translation remains "
                "organized in a motion-regime-dependent geometry."
            ),
            "desired_result": (
                "The structured aggregation hypothesis is supported when "
                "global held-out decoding achieves strong z correlation, "
                "low bias/RMSE across all motion regimes, and the gap "
                "between global and within-regime decoders becomes small."
            ),
        },
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
            sort_keys=False,
            allow_nan=False,
        )

        file.write(
            "\n"
        )

    print_summary(
        representation_key=representation_key,
        raw_shape=raw_shape,
        representation=representation,
        representation_rows=representation_rows,
        decoder_rows=decoder_rows,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()