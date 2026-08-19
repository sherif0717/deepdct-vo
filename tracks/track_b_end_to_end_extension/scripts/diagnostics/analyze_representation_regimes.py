#!/usr/bin/env python3
"""
Audit DeepDCT-VO translation representations conditioned on motion regime.

This script operates entirely on exported artifacts:

    translation_representations.npz
    frame_regime_assignments.csv

No model inference or retraining is performed.

Primary questions
-----------------
1. Does representation magnitude/distribution change with motion regime?
2. Are low / medium / high forward-motion regimes separable in the
   frozen translation representation?
3. How linearly decodable is GT translation from the representation?
4. Does decodability change across motion regimes?

Typical usage
-------------

python scripts/analyze_representation_regimes.py \
    --representation \
        experiments/baseline_identity_output/evaluation_seq10/translation_representations.npz \
    --regimes \
        experiments/motion_regime_analysis/sequence_10/frame_regime_assignments.csv \
    --output-dir \
        experiments/representation_regime_audit/sequence_10

Outputs
-------
<output-dir>/
    summary.json
    regime_representation_statistics.csv
    regime_linear_decode.csv
    regime_centroid_distances.csv
    pca_regime_summary.csv
    plots/
        representation_norm_by_forward_regime.png
        pca_forward_regimes.png
        centroid_distance_matrix.png
        linear_decode_rmse_by_regime.png

Notes
-----
The NPZ is expected to contain translation representations with shape:

    (N, D)

For the current experiment:

    N = 1200
    D = 14400

The script does not use predicted translation. Regimes and targets are
ground-truth derived.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORWARD_REGIME_ORDER = (
    "low",
    "medium",
    "high",
)

GT_COLUMNS = (
    "translation_gt_x",
    "translation_gt_y",
    "translation_gt_z",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DeepDCT-VO translation representations "
            "conditioned on ground-truth motion regime."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--representation",
        type=Path,
        required=True,
        help="translation_representations.npz",
    )

    parser.add_argument(
        "--regimes",
        type=Path,
        required=True,
        help="frame_regime_assignments.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for audit outputs.",
    )

    parser.add_argument(
        "--representation-key",
        type=str,
        default="translation_rep",
        help="NPZ array key.",
    )

    parser.add_argument(
        "--regime-column",
        type=str,
        default="forward_motion_regime",
        help="Regime column to analyze.",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help=(
            "Number of principal components retained for "
            "representation-space summaries."
        ),
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0e-3,
        help="Ridge regularization used for linear decoding.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help=(
            "Fraction of each regime used to fit its diagnostic "
            "linear decoder."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=14400,
        help=(
            "Maximum representation dimensions used directly "
            "for the ridge decoder."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pca_components <= 0:
        raise ValueError(
            "--pca-components must be positive."
        )

    if args.ridge_alpha < 0:
        raise ValueError(
            "--ridge-alpha cannot be negative."
        )

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError(
            "--train-fraction must lie strictly between 0 and 1."
        )

    if args.max_features <= 0:
        raise ValueError(
            "--max-features must be positive."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_representation(
    path: Path,
    key: str,
) -> np.ndarray:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Representation file not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    if key not in data:
        raise KeyError(
            f"Representation key {key!r} not found.\n"
            f"Available keys: {list(data.keys())}"
        )

    representation = np.asarray(
        data[key],
        dtype=np.float64,
    )

    if representation.ndim < 2:
        raise ValueError(
            "Representation must have at least two dimensions."
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

    return representation


def load_regimes(
    path: Path,
) -> pd.DataFrame:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Regime assignment file not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            "Regime assignment CSV is empty."
        )

    return dataframe


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_alignment(
    representation: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
) -> None:
    if len(representation) != len(regimes):
        raise ValueError(
            "Representation/regime row-count mismatch:\n"
            f"representation rows = {len(representation)}\n"
            f"regime rows         = {len(regimes)}"
        )

    required = set(
        GT_COLUMNS
    ) | {regime_column}

    missing = (
        required
        - set(regimes.columns)
    )

    if missing:
        raise KeyError(
            "Regime CSV is missing required columns:\n  "
            + "\n  ".join(
                sorted(missing)
            )
        )

    expected_regimes = set(
        FORWARD_REGIME_ORDER
    )

    actual_regimes = set(
        regimes[
            regime_column
        ].dropna().astype(str)
    )

    missing_regimes = (
        expected_regimes
        - actual_regimes
    )

    if missing_regimes:
        raise ValueError(
            "Expected motion regimes are missing:\n  "
            + "\n  ".join(
                sorted(missing_regimes)
            )
        )


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------


def compute_regime_statistics(
    representation: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
) -> pd.DataFrame:
    norms = np.linalg.norm(
        representation,
        axis=1,
    )

    rows: List[Dict[str, Any]] = []

    for regime in FORWARD_REGIME_ORDER:
        mask = (
            regimes[
                regime_column
            ].to_numpy()
            == regime
        )

        subset = representation[
            mask
        ]

        subset_norms = norms[
            mask
        ]

        centroid = np.mean(
            subset,
            axis=0,
        )

        centered = (
            subset
            - centroid
        )

        mean_feature_variance = float(
            np.mean(
                np.var(
                    subset,
                    axis=0,
                )
            )
        )

        mean_distance_to_centroid = float(
            np.mean(
                np.linalg.norm(
                    centered,
                    axis=1,
                )
            )
        )

        rows.append(
            {
                "regime": regime,
                "samples": int(
                    np.sum(mask)
                ),
                "representation_norm_mean": float(
                    np.mean(
                        subset_norms
                    )
                ),
                "representation_norm_std": float(
                    np.std(
                        subset_norms
                    )
                ),
                "representation_norm_median": float(
                    np.median(
                        subset_norms
                    )
                ),
                "centroid_norm": float(
                    np.linalg.norm(
                        centroid
                    )
                ),
                "mean_feature_variance": (
                    mean_feature_variance
                ),
                "mean_distance_to_centroid": (
                    mean_distance_to_centroid
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def fit_pca(
    representation: np.ndarray,
    components: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    PCA using economical SVD.

    Returns
    -------
    projected:
        N x K PCA coordinates.

    explained_variance_ratio:
        K-element variance ratios.

    components_matrix:
        K x D principal directions.
    """

    mean = np.mean(
        representation,
        axis=0,
        keepdims=True,
    )

    centered = (
        representation
        - mean
    )

    max_components = min(
        components,
        centered.shape[0] - 1,
        centered.shape[1],
    )

    if max_components <= 0:
        raise ValueError(
            "Not enough samples for PCA."
        )

    u, singular_values, vt = (
        np.linalg.svd(
            centered,
            full_matrices=False,
        )
    )

    vt = vt[
        :max_components
    ]

    singular_values = singular_values[
        :max_components
    ]

    projected = (
        centered
        @ vt.T
    )

    all_variance = float(
        np.sum(
            np.var(
                centered,
                axis=0,
                ddof=1,
            )
        )
    )

    explained_variance = (
        singular_values ** 2
        / (
            centered.shape[0]
            - 1
        )
    )

    explained_variance_ratio = (
        explained_variance
        / all_variance
        if all_variance > 0
        else np.zeros_like(
            explained_variance
        )
    )

    return (
        projected,
        explained_variance_ratio,
        vt,
    )


def summarize_pca_by_regime(
    projected: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for regime in FORWARD_REGIME_ORDER:
        mask = (
            regimes[
                regime_column
            ].to_numpy()
            == regime
        )

        subset = projected[
            mask
        ]

        row: Dict[str, Any] = {
            "regime": regime,
            "samples": int(
                np.sum(mask)
            ),
        }

        for index in range(
            subset.shape[1]
        ):
            row[
                f"pc{index + 1}_mean"
            ] = float(
                np.mean(
                    subset[:, index]
                )
            )

            row[
                f"pc{index + 1}_std"
            ] = float(
                np.std(
                    subset[:, index]
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Centroid distances
# ---------------------------------------------------------------------------


def compute_centroid_distances(
    representation: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
) -> pd.DataFrame:
    centroids: Dict[
        str,
        np.ndarray,
    ] = {}

    for regime in FORWARD_REGIME_ORDER:
        mask = (
            regimes[
                regime_column
            ].to_numpy()
            == regime
        )

        centroids[
            regime
        ] = np.mean(
            representation[
                mask
            ],
            axis=0,
        )

    rows: List[Dict[str, Any]] = []

    for regime_a in FORWARD_REGIME_ORDER:
        for regime_b in FORWARD_REGIME_ORDER:
            distance = float(
                np.linalg.norm(
                    centroids[
                        regime_a
                    ]
                    - centroids[
                        regime_b
                    ]
                )
            )

            cosine_denominator = (
                np.linalg.norm(
                    centroids[
                        regime_a
                    ]
                )
                * np.linalg.norm(
                    centroids[
                        regime_b
                    ]
                )
            )

            if cosine_denominator > 0:
                cosine_similarity = float(
                    np.dot(
                        centroids[
                            regime_a
                        ],
                        centroids[
                            regime_b
                        ],
                    )
                    / cosine_denominator
                )
            else:
                cosine_similarity = float(
                    "nan"
                )

            rows.append(
                {
                    "regime_a": regime_a,
                    "regime_b": regime_b,
                    "euclidean_distance": (
                        distance
                    ),
                    "cosine_similarity": (
                        cosine_similarity
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Ridge decoding
# ---------------------------------------------------------------------------


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Fit multi-output ridge regression with an explicit intercept.

    Uses the dual form when feature dimension exceeds sample count,
    which is appropriate for D=14400 and only a few hundred samples.
    """

    x_mean = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    y_mean = np.mean(
        y,
        axis=0,
        keepdims=True,
    )

    xc = x - x_mean
    yc = y - y_mean

    n_samples = xc.shape[0]

    gram = (
        xc @ xc.T
    )

    regularized = (
        gram
        + alpha
        * np.eye(
            n_samples,
            dtype=np.float64,
        )
    )

    dual = np.linalg.solve(
        regularized,
        yc,
    )

    weights = (
        xc.T
        @ dual
    )

    bias = (
        y_mean.reshape(-1)
        - x_mean.reshape(-1)
        @ weights
    )

    return weights, bias


def predict_ridge(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return (
        x @ weights
        + bias
    )


def translation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    error = (
        y_pred - y_true
    )

    result: Dict[
        str,
        float,
    ] = {
        "vector_rmse": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        error ** 2,
                        axis=1,
                    )
                )
            )
        )
    }

    for index, axis in enumerate(
        ("x", "y", "z")
    ):
        axis_error = (
            error[:, index]
        )

        result[
            f"{axis}_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    axis_error ** 2
                )
            )
        )

        result[
            f"{axis}_bias"
        ] = float(
            np.mean(
                axis_error
            )
        )

        target = (
            y_true[:, index]
        )

        prediction = (
            y_pred[:, index]
        )

        if (
            len(target) >= 3
            and np.std(
                target
            ) > 1.0e-12
            and np.std(
                prediction
            ) > 1.0e-12
        ):
            correlation = float(
                np.corrcoef(
                    target,
                    prediction,
                )[0, 1]
            )
        else:
            correlation = float(
                "nan"
            )

        result[
            f"{axis}_correlation"
        ] = correlation

    return result


def stratified_regime_split(
    regimes: pd.DataFrame,
    regime_column: str,
    train_fraction: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(
        seed
    )

    train_indices: List[int] = []
    test_indices: List[int] = []

    labels = regimes[
        regime_column
    ].to_numpy()

    for regime in FORWARD_REGIME_ORDER:
        indices = np.where(
            labels == regime
        )[0]

        shuffled = (
            rng.permutation(
                indices
            )
        )

        count_train = int(
            math.floor(
                train_fraction
                * len(
                    shuffled
                )
            )
        )

        count_train = max(
            1,
            min(
                count_train,
                len(shuffled) - 1,
            ),
        )

        train_indices.extend(
            shuffled[
                :count_train
            ].tolist()
        )

        test_indices.extend(
            shuffled[
                count_train:
            ].tolist()
        )

    return (
        np.asarray(
            sorted(
                train_indices
            ),
            dtype=np.int64,
        ),
        np.asarray(
            sorted(
                test_indices
            ),
            dtype=np.int64,
        ),
    )


def run_global_linear_decode(
    representation: np.ndarray,
    targets: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
    *,
    alpha: float,
    train_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """
    Fit one global linear decoder, then evaluate it separately
    within each motion regime.

    This is the most important decoder diagnostic because it asks
    whether a single linear mapping from the frozen representation
    generalizes across regimes.
    """

    train_indices, test_indices = (
        stratified_regime_split(
            regimes,
            regime_column,
            train_fraction,
            seed,
        )
    )

    weights, bias = fit_ridge(
        representation[
            train_indices
        ],
        targets[
            train_indices
        ],
        alpha,
    )

    prediction = predict_ridge(
        representation[
            test_indices
        ],
        weights,
        bias,
    )

    test_labels = (
        regimes.iloc[
            test_indices
        ][
            regime_column
        ].to_numpy()
    )

    rows: List[
        Dict[str, Any]
    ] = []

    overall_metrics = (
        translation_metrics(
            targets[
                test_indices
            ],
            prediction,
        )
    )

    rows.append(
        {
            "decoder": (
                "global_ridge"
            ),
            "regime": "all",
            "samples": len(
                test_indices
            ),
            **overall_metrics,
        }
    )

    for regime in FORWARD_REGIME_ORDER:
        mask = (
            test_labels
            == regime
        )

        metrics = (
            translation_metrics(
                targets[
                    test_indices[
                        mask
                    ]
                ],
                prediction[
                    mask
                ],
            )
        )

        rows.append(
            {
                "decoder": (
                    "global_ridge"
                ),
                "regime": regime,
                "samples": int(
                    np.sum(
                        mask
                    )
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def run_within_regime_linear_decode(
    representation: np.ndarray,
    targets: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
    *,
    alpha: float,
    train_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """
    Fit an independent ridge decoder inside each regime.

    Comparison against the global ridge decoder tells us whether a
    single mapping is inadequate even when each regime individually
    remains linearly decodable.
    """

    rng = np.random.default_rng(
        seed
    )

    labels = regimes[
        regime_column
    ].to_numpy()

    rows: List[
        Dict[str, Any]
    ] = []

    for regime in FORWARD_REGIME_ORDER:
        indices = np.where(
            labels == regime
        )[0]

        indices = (
            rng.permutation(
                indices
            )
        )

        train_count = int(
            math.floor(
                train_fraction
                * len(
                    indices
                )
            )
        )

        train_count = max(
            1,
            min(
                train_count,
                len(indices) - 1,
            ),
        )

        train_indices = (
            indices[
                :train_count
            ]
        )

        test_indices = (
            indices[
                train_count:
            ]
        )

        weights, bias = (
            fit_ridge(
                representation[
                    train_indices
                ],
                targets[
                    train_indices
                ],
                alpha,
            )
        )

        prediction = (
            predict_ridge(
                representation[
                    test_indices
                ],
                weights,
                bias,
            )
        )

        metrics = (
            translation_metrics(
                targets[
                    test_indices
                ],
                prediction,
            )
        )

        rows.append(
            {
                "decoder": (
                    "within_regime_ridge"
                ),
                "regime": regime,
                "samples": len(
                    test_indices
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_representation_norm(
    representation: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
    output_path: Path,
) -> None:
    norms = np.linalg.norm(
        representation,
        axis=1,
    )

    values = [
        norms[
            regimes[
                regime_column
            ].to_numpy()
            == regime
        ]
        for regime in (
            FORWARD_REGIME_ORDER
        )
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        values,
        labels=(
            FORWARD_REGIME_ORDER
        ),
        showfliers=False,
    )

    ax.set_xlabel(
        "Forward-motion regime"
    )

    ax.set_ylabel(
        "Representation L2 norm"
    )

    ax.set_title(
        "Translation Representation Norm "
        "by Forward-Motion Regime"
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


def plot_pca(
    projected: np.ndarray,
    regimes: pd.DataFrame,
    regime_column: str,
    output_path: Path,
) -> None:
    if projected.shape[1] < 2:
        return

    labels = regimes[
        regime_column
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for regime in FORWARD_REGIME_ORDER:
        mask = (
            labels
            == regime
        )

        ax.scatter(
            projected[
                mask,
                0,
            ],
            projected[
                mask,
                1,
            ],
            s=12,
            alpha=0.55,
            label=regime,
        )

    ax.set_xlabel(
        "PC1"
    )

    ax.set_ylabel(
        "PC2"
    )

    ax.set_title(
        "PCA of Translation Representation "
        "by Forward-Motion Regime"
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
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


def plot_centroid_matrix(
    distances: pd.DataFrame,
    output_path: Path,
) -> None:
    matrix = np.zeros(
        (
            len(
                FORWARD_REGIME_ORDER
            ),
            len(
                FORWARD_REGIME_ORDER
            ),
        ),
        dtype=np.float64,
    )

    for i, regime_a in enumerate(
        FORWARD_REGIME_ORDER
    ):
        for j, regime_b in enumerate(
            FORWARD_REGIME_ORDER
        ):
            row = distances[
                (
                    distances[
                        "regime_a"
                    ]
                    == regime_a
                )
                & (
                    distances[
                        "regime_b"
                    ]
                    == regime_b
                )
            ]

            matrix[
                i,
                j,
            ] = float(
                row.iloc[0][
                    "euclidean_distance"
                ]
            )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        range(
            len(
                FORWARD_REGIME_ORDER
            )
        )
    )

    ax.set_xticklabels(
        FORWARD_REGIME_ORDER
    )

    ax.set_yticks(
        range(
            len(
                FORWARD_REGIME_ORDER
            )
        )
    )

    ax.set_yticklabels(
        FORWARD_REGIME_ORDER
    )

    ax.set_xlabel(
        "Regime"
    )

    ax.set_ylabel(
        "Regime"
    )

    ax.set_title(
        "Representation Centroid Distances"
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Euclidean distance",
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


def plot_decode_rmse(
    decode_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    subset = decode_metrics[
        decode_metrics[
            "regime"
        ].isin(
            FORWARD_REGIME_ORDER
        )
    ]

    decoders = list(
        subset[
            "decoder"
        ].unique()
    )

    x = np.arange(
        len(
            FORWARD_REGIME_ORDER
        )
    )

    width = (
        0.8
        / max(
            1,
            len(
                decoders
            ),
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for index, decoder in enumerate(
        decoders
    ):
        values: List[
            float
        ] = []

        for regime in (
            FORWARD_REGIME_ORDER
        ):
            row = subset[
                (
                    subset[
                        "decoder"
                    ]
                    == decoder
                )
                & (
                    subset[
                        "regime"
                    ]
                    == regime
                )
            ]

            values.append(
                float(
                    row.iloc[0][
                        "z_rmse"
                    ]
                )
            )

        offset = (
            index
            - (
                len(
                    decoders
                )
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=decoder,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        FORWARD_REGIME_ORDER
    )

    ax.set_ylabel(
        "Decoded z RMSE"
    )

    ax.set_xlabel(
        "Forward-motion regime"
    )

    ax.set_title(
        "Linear Decodability of Translation "
        "Representation"
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


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        value = float(
            value
        )

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
        Mapping,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(
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
            json_safe(
                data
            ),
            handle,
            indent=2,
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    validate_args(
        args
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

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

    representation = (
        load_representation(
            args.representation,
            args.representation_key,
        )
    )

    regimes = load_regimes(
        args.regimes
    )

    audit_alignment(
        representation,
        regimes,
        args.regime_column,
    )

    print("=" * 88)
    print(
        "Translation Representation-Regime Audit"
    )
    print("=" * 88)
    print(
        f"Representation:    {args.representation.resolve()}"
    )
    print(
        f"Representation key:{args.representation_key}"
    )
    print(
        f"Shape:             {representation.shape}"
    )
    print(
        f"Regime assignments:{args.regimes.resolve()}"
    )
    print(
        f"Regime column:     {args.regime_column}"
    )
    print(
        f"Samples:           {len(regimes)}"
    )
    print("=" * 88)

    # --------------------------------------------------------------
    # Ground-truth targets
    # --------------------------------------------------------------

    targets = regimes[
        list(
            GT_COLUMNS
        )
    ].to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------------
    # Optional feature truncation
    # --------------------------------------------------------------

    if (
        representation.shape[1]
        > args.max_features
    ):
        representation_decode = (
            representation[
                :,
                :args.max_features,
            ]
        )
    else:
        representation_decode = (
            representation
        )

    # --------------------------------------------------------------
    # Representation statistics
    # --------------------------------------------------------------

    statistics = (
        compute_regime_statistics(
            representation,
            regimes,
            args.regime_column,
        )
    )

    statistics.to_csv(
        output_dir
        / "regime_representation_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # PCA
    # --------------------------------------------------------------

    (
        projected,
        explained_variance_ratio,
        _,
    ) = fit_pca(
        representation,
        args.pca_components,
    )

    pca_summary = (
        summarize_pca_by_regime(
            projected,
            regimes,
            args.regime_column,
        )
    )

    pca_summary.to_csv(
        output_dir
        / "pca_regime_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Centroid distances
    # --------------------------------------------------------------

    centroid_distances = (
        compute_centroid_distances(
            representation,
            regimes,
            args.regime_column,
        )
    )

    centroid_distances.to_csv(
        output_dir
        / "regime_centroid_distances.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Linear decode experiments
    # --------------------------------------------------------------

    global_decode = (
        run_global_linear_decode(
            representation_decode,
            targets,
            regimes,
            args.regime_column,
            alpha=args.ridge_alpha,
            train_fraction=(
                args.train_fraction
            ),
            seed=args.seed,
        )
    )

    within_decode = (
        run_within_regime_linear_decode(
            representation_decode,
            targets,
            regimes,
            args.regime_column,
            alpha=args.ridge_alpha,
            train_fraction=(
                args.train_fraction
            ),
            seed=args.seed,
        )
    )

    decode_metrics = pd.concat(
        [
            global_decode,
            within_decode,
        ],
        ignore_index=True,
    )

    decode_metrics.to_csv(
        output_dir
        / "regime_linear_decode.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------

    plot_representation_norm(
        representation,
        regimes,
        args.regime_column,
        plot_dir
        / "representation_norm_by_forward_regime.png",
    )

    plot_pca(
        projected,
        regimes,
        args.regime_column,
        plot_dir
        / "pca_forward_regimes.png",
    )

    plot_centroid_matrix(
        centroid_distances,
        plot_dir
        / "centroid_distance_matrix.png",
    )

    plot_decode_rmse(
        decode_metrics,
        plot_dir
        / "linear_decode_rmse_by_regime.png",
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary: Dict[
        str,
        Any,
    ] = {
        "representation_path": str(
            args.representation
            .expanduser()
            .resolve()
        ),
        "representation_key": (
            args.representation_key
        ),
        "representation_shape": list(
            representation.shape
        ),
        "regime_path": str(
            args.regimes
            .expanduser()
            .resolve()
        ),
        "regime_column": (
            args.regime_column
        ),
        "sample_count": int(
            len(
                representation
            )
        ),
        "feature_count": int(
            representation.shape[1]
        ),
        "decode_feature_count": int(
            representation_decode.shape[1]
        ),
        "pca_components": int(
            projected.shape[1]
        ),
        "pca_explained_variance_ratio": (
            explained_variance_ratio.tolist()
        ),
        "pca_explained_variance_ratio_sum": float(
            np.sum(
                explained_variance_ratio
            )
        ),
        "ridge_alpha": (
            args.ridge_alpha
        ),
        "train_fraction": (
            args.train_fraction
        ),
        "representation_statistics": (
            statistics.to_dict(
                orient="records"
            )
        ),
        "linear_decode": (
            decode_metrics.to_dict(
                orient="records"
            )
        ),
    }

    save_json(
        output_dir
        / "summary.json",
        summary,
    )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print()
    print("=" * 88)
    print(
        "Representation statistics"
    )
    print("=" * 88)

    for _, row in (
        statistics.iterrows()
    ):
        print(
            f"{row['regime']:<10} "
            f"n={int(row['samples']):4d} "
            f"norm={row['representation_norm_mean']:.6f} "
            f"centroid_norm={row['centroid_norm']:.6f} "
            f"within_spread={row['mean_distance_to_centroid']:.6f}"
        )

    print()
    print("=" * 88)
    print(
        "Linear z decoding by forward-motion regime"
    )
    print("=" * 88)

    for _, row in (
        decode_metrics.iterrows()
    ):
        if row[
            "regime"
        ] == "all":
            continue

        print(
            f"{row['decoder']:<24} "
            f"{row['regime']:<8} "
            f"n={int(row['samples']):4d} "
            f"z_rmse={row['z_rmse']:.6f} "
            f"z_bias={row['z_bias']:+.6f} "
            f"z_corr={row['z_correlation']:.4f}"
        )

    print()
    print("=" * 88)
    print(
        "Audit complete"
    )
    print("=" * 88)
    print(
        f"Statistics:       "
        f"{output_dir / 'regime_representation_statistics.csv'}"
    )
    print(
        f"Linear decoding:  "
        f"{output_dir / 'regime_linear_decode.csv'}"
    )
    print(
        f"Centroids:        "
        f"{output_dir / 'regime_centroid_distances.csv'}"
    )
    print(
        f"PCA summary:      "
        f"{output_dir / 'pca_regime_summary.csv'}"
    )
    print(
        f"Summary:          "
        f"{output_dir / 'summary.json'}"
    )
    print(
        f"Plots:            "
        f"{plot_dir}"
    )
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )