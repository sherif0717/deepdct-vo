#!/usr/bin/env python3
"""
Continuous SO(3) Rotation-Geometry Alignment Audit for DeepDCT-VO.

Purpose
-------
Audit whether the learned compact rotation representation preserves the
continuous geometry of ground-truth relative rotations on SO(3).

This script is intended for the continuous SO(3)-supervised
rotation-geometry experiment implemented by:

    deepdct/training/rotation_geometry.py

It answers the central post-training question:

    Do pairs of samples that are close on SO(3) also lie close in the
    learned compact rotation representation?

The audit deliberately does NOT use:
    - straight / moderate / strong rotation classes;
    - Euler-z "heading" bins;
    - discrete motion-regime prototypes.

Instead, it compares:

    ground-truth SO(3) geodesic distance
            versus
    learned representation distance.

Primary diagnostics
-------------------
1. Pearson correlation:
       SO(3) distance vs representation distance

2. Spearman rank correlation:
       tests monotonic geometric ordering

3. Linear fit / R^2:
       representation distance ~= a * SO(3) distance + b

4. k-nearest-neighbor overlap:
       whether SO(3)-nearest samples are also latent-nearest samples

5. Distance-bin statistics:
       whether latent distance increases with SO(3) separation

6. Local-vs-global geometry:
       separate correlations for near, medium, and far SO(3) pairs

7. Nearest-neighbor audit:
       for every frame, report the nearest neighbor in:
           a) SO(3)
           b) representation space

Inputs
------
Evaluation directory containing:

    frame_predictions.csv
    rotation_representations.npz

The current DeepDCT-VO evaluator exports these artifacts.

Supported representation keys
-----------------------------
The script automatically accepts:

    rotation_representation
    rotation_rep
    rotation_representations

Ground-truth rotation source
----------------------------
Preferred:
    rotation_gt_array inside rotation_representations.npz

Fallback:
    rotation_gt_x
    rotation_gt_y
    rotation_gt_z
from frame_predictions.csv

Euler convention
----------------
Default:

    extrinsic xyz, radians

which corresponds to:

    R = Rz(z) @ Ry(y) @ Rx(x)

Examples
--------
Typical sequence-10 audit:

    python scripts/analyze_rotation_geometry_alignment.py \
        --evaluation-dir \
        experiments/continuous_so3_rotation_geometry/evaluation_sequence_10 \
        --output-dir \
        experiments/continuous_so3_rotation_geometry/rotation_geometry_alignment

Explicit files:

    python scripts/analyze_rotation_geometry_alignment.py \
        --representations \
        experiments/continuous_so3_rotation_geometry/evaluation_sequence_10/rotation_representations.npz \
        --predictions \
        experiments/continuous_so3_rotation_geometry/evaluation_sequence_10/frame_predictions.csv \
        --output-dir \
        experiments/continuous_so3_rotation_geometry/rotation_geometry_alignment

For reproducible pair subsampling:

    python scripts/analyze_rotation_geometry_alignment.py \
        --evaluation-dir \
        experiments/continuous_so3_rotation_geometry/evaluation_sequence_10 \
        --output-dir \
        experiments/continuous_so3_rotation_geometry/rotation_geometry_alignment \
        --max-pairs 200000 \
        --seed 42

Outputs
-------
<output-dir>/
    summary.json
    pairwise_alignment_metrics.csv
    distance_bins.csv
    neighborhood_metrics.csv
    nearest_neighbor_audit.csv
    sampled_pair_distances.csv
    so3_vs_representation_distance.png
    so3_distance_bins.png
    local_geometry_correlations.png
    nearest_neighbor_so3_distance.png

Python:
    3.8+

Dependencies:
    numpy
    pandas
    matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Constants
# ============================================================================


REPRESENTATION_KEYS = (
    "rotation_representation",
    "rotation_rep",
    "rotation_representations",
)

GT_ARRAY_KEYS = (
    "rotation_gt_array",
    "rotation_gt",
)

GT_CSV_COLUMNS = (
    "rotation_gt_x",
    "rotation_gt_y",
    "rotation_gt_z",
)

FRAME_ID_COLUMNS = (
    "sequence",
    "frame_prev",
    "frame_curr",
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit continuous SO(3) alignment of the learned "
            "DeepDCT-VO rotation representation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation directory containing frame_predictions.csv "
            "and rotation_representations.npz."
        ),
    )

    parser.add_argument(
        "--representations",
        type=Path,
        default=None,
        help="Explicit rotation_representations.npz path.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Explicit frame_predictions.csv path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which audit outputs are written.",
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help=(
            "Interpret ground-truth Euler angles as degrees. "
            "Default is radians."
        ),
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200000,
        help=(
            "Maximum number of unique sample pairs used for the "
            "global pairwise correlation audit. All pairs are used "
            "when their count is below this value."
        ),
    )

    parser.add_argument(
        "--scatter-pairs",
        type=int,
        default=50000,
        help=(
            "Maximum number of pairs drawn in the distance scatter plot."
        ),
    )

    parser.add_argument(
        "--distance-bins",
        type=int,
        default=10,
        help="Number of SO(3)-distance quantile bins.",
    )

    parser.add_argument(
        "--neighbor-k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Values of k used for neighborhood-overlap analysis.",
    )

    parser.add_argument(
        "--local-quantiles",
        type=float,
        nargs=2,
        default=[0.33, 0.67],
        metavar=("Q_NEAR", "Q_FAR"),
        help=(
            "SO(3) pair-distance quantiles separating near, medium, "
            "and far rotation geometry."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for pair subsampling.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Plot resolution.",
    )

    args = parser.parse_args()
    validate_args(args)
    resolve_input_paths(args)

    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.evaluation_dir is None:
        if args.representations is None:
            raise ValueError(
                "Provide either --evaluation-dir or --representations."
            )

        if args.predictions is None:
            raise ValueError(
                "Provide either --evaluation-dir or --predictions."
            )

    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive.")

    if args.scatter_pairs <= 0:
        raise ValueError("--scatter-pairs must be positive.")

    if args.distance_bins < 2:
        raise ValueError("--distance-bins must be at least 2.")

    if not args.neighbor_k:
        raise ValueError("--neighbor-k must contain at least one value.")

    if any(k <= 0 for k in args.neighbor_k):
        raise ValueError("--neighbor-k values must be positive.")

    q_near, q_far = args.local_quantiles

    if not (
        0.0 < q_near < q_far < 1.0
    ):
        raise ValueError(
            "--local-quantiles must satisfy "
            "0 < Q_NEAR < Q_FAR < 1."
        )

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def resolve_input_paths(
    args: argparse.Namespace,
) -> None:
    if args.evaluation_dir is not None:
        evaluation_dir = (
            args.evaluation_dir.expanduser()
        )

        if args.representations is None:
            args.representations = (
                evaluation_dir
                / "rotation_representations.npz"
            )

        if args.predictions is None:
            args.predictions = (
                evaluation_dir
                / "frame_predictions.csv"
            )

    args.representations = (
        args.representations.expanduser()
    )

    args.predictions = (
        args.predictions.expanduser()
    )

    args.output_dir = (
        args.output_dir.expanduser()
    )


# ============================================================================
# General utilities
# ============================================================================


def safe_float(
    value: object,
) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def write_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


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


def standardize_representation(
    representation: np.ndarray,
) -> np.ndarray:
    representation = np.asarray(
        representation,
        dtype=np.float64,
    )

    if representation.ndim < 2:
        raise ValueError(
            "Rotation representation must have at least two "
            f"dimensions, received {representation.shape}."
        )

    representation = representation.reshape(
        representation.shape[0],
        -1,
    )

    if representation.shape[1] == 0:
        raise ValueError(
            "Rotation representation has zero feature dimension."
        )

    if not np.all(
        np.isfinite(representation)
    ):
        raise ValueError(
            "Rotation representation contains NaN or infinity."
        )

    return representation


# ============================================================================
# Input loading
# ============================================================================


def load_npz(
    path: Path,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    str,
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Representation file does not exist: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        representation_key = None

        for candidate in REPRESENTATION_KEYS:
            if candidate in archive:
                representation_key = candidate
                break

        if representation_key is None:
            raise KeyError(
                "Could not find the rotation representation in "
                f"{path}. Available keys: {list(archive.keys())}. "
                f"Supported keys: {REPRESENTATION_KEYS}."
            )

        representation = standardize_representation(
            archive[representation_key]
        )

        rotation_gt = None

        for candidate in GT_ARRAY_KEYS:
            if candidate in archive:
                rotation_gt = np.asarray(
                    archive[candidate],
                    dtype=np.float64,
                )
                break

    if rotation_gt is not None:
        if (
            rotation_gt.ndim != 2
            or rotation_gt.shape[1] != 3
        ):
            raise ValueError(
                "Ground-truth rotation array in NPZ must have "
                f"shape [N, 3], received {rotation_gt.shape}."
            )

    return (
        representation,
        rotation_gt,
        representation_key,
    )


def load_predictions(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {path}"
        )

    table = pd.read_csv(path)

    if len(table) == 0:
        raise ValueError(
            f"Prediction CSV is empty: {path}"
        )

    return table


def rotation_gt_from_csv(
    table: pd.DataFrame,
) -> np.ndarray:
    missing = [
        column
        for column in GT_CSV_COLUMNS
        if column not in table.columns
    ]

    if missing:
        raise KeyError(
            "frame_predictions.csv does not contain the required "
            f"ground-truth rotation columns: {missing}."
        )

    result = table[
        list(GT_CSV_COLUMNS)
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Ground-truth rotation columns contain NaN or infinity."
        )

    return result


# ============================================================================
# Euler xyz -> SO(3)
# ============================================================================


def euler_xyz_to_rotation_matrix(
    angles: np.ndarray,
    degrees: bool,
) -> np.ndarray:
    """
    Extrinsic xyz Euler conversion.

        R = Rz(z) @ Ry(y) @ Rx(x)
    """

    angles = np.asarray(
        angles,
        dtype=np.float64,
    )

    if (
        angles.ndim != 2
        or angles.shape[1] != 3
    ):
        raise ValueError(
            "Euler angles must have shape [N, 3]."
        )

    if degrees:
        angles = np.radians(angles)

    x = angles[:, 0]
    y = angles[:, 1]
    z = angles[:, 2]

    cx = np.cos(x)
    sx = np.sin(x)

    cy = np.cos(y)
    sy = np.sin(y)

    cz = np.cos(z)
    sz = np.sin(z)

    matrices = np.empty(
        (angles.shape[0], 3, 3),
        dtype=np.float64,
    )

    matrices[:, 0, 0] = cz * cy
    matrices[:, 0, 1] = (
        cz * sy * sx
        - sz * cx
    )
    matrices[:, 0, 2] = (
        cz * sy * cx
        + sz * sx
    )

    matrices[:, 1, 0] = sz * cy
    matrices[:, 1, 1] = (
        sz * sy * sx
        + cz * cx
    )
    matrices[:, 1, 2] = (
        sz * sy * cx
        - cz * sx
    )

    matrices[:, 2, 0] = -sy
    matrices[:, 2, 1] = cy * sx
    matrices[:, 2, 2] = cy * cx

    return matrices


# ============================================================================
# Pair-distance computations
# ============================================================================


def so3_pair_distance(
    rotation_matrices: np.ndarray,
    index_a: np.ndarray,
    index_b: np.ndarray,
) -> np.ndarray:
    a = rotation_matrices[index_a]
    b = rotation_matrices[index_b]

    relative = (
        np.transpose(
            a,
            axes=(0, 2, 1),
        )
        @ b
    )

    trace = np.trace(
        relative,
        axis1=1,
        axis2=2,
    )

    cosine = (
        trace - 1.0
    ) * 0.5

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return np.arccos(cosine)


def normalized_representation(
    representation: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        representation,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1.0e-12,
    )

    return representation / norms


def representation_pair_distance(
    representation_normalized: np.ndarray,
    index_a: np.ndarray,
    index_b: np.ndarray,
) -> np.ndarray:
    cosine = np.sum(
        representation_normalized[index_a]
        * representation_normalized[index_b],
        axis=1,
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    # Cosine distance is the quantity most directly related to the
    # training objective:
    #
    #     similarity = z_i^T z_j
    #
    return 1.0 - cosine


def all_unique_pair_indices(
    sample_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(
        sample_count,
        k=1,
    )


def sampled_pair_indices(
    sample_count: int,
    maximum_pairs: int,
    rng: np.random.Generator,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    total_pairs = (
        sample_count
        * (sample_count - 1)
        // 2
    )

    if total_pairs <= maximum_pairs:
        i, j = all_unique_pair_indices(
            sample_count
        )

        return (
            i.astype(np.int64),
            j.astype(np.int64),
            total_pairs,
        )

    # Draw random unique flattened pair IDs. We intentionally
    # oversample candidates and deduplicate until max_pairs is reached.
    pair_set = set()

    while len(pair_set) < maximum_pairs:
        needed = (
            maximum_pairs
            - len(pair_set)
        )

        draw_count = max(
            needed * 2,
            1000,
        )

        a = rng.integers(
            0,
            sample_count,
            size=draw_count,
        )

        b = rng.integers(
            0,
            sample_count,
            size=draw_count,
        )

        low = np.minimum(a, b)
        high = np.maximum(a, b)

        valid = low != high

        for left, right in zip(
            low[valid],
            high[valid],
        ):
            pair_set.add(
                (
                    int(left),
                    int(right),
                )
            )

            if len(pair_set) >= maximum_pairs:
                break

    pairs = np.asarray(
        sorted(pair_set),
        dtype=np.int64,
    )

    return (
        pairs[:, 0],
        pairs[:, 1],
        total_pairs,
    )


# ============================================================================
# Statistics
# ============================================================================


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.size < 2:
        return float("nan")

    if (
        np.std(x) <= 1.0e-15
        or np.std(y) <= 1.0e-15
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def rank_values(
    values: np.ndarray,
) -> np.ndarray:
    """
    Return average ranks with tie handling.

    Implemented locally so scipy is not required.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[order]

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            start + end - 1
        ) * 0.5

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    return ranks


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if len(x) < 2:
        return float("nan")

    return pearson_correlation(
        rank_values(x),
        rank_values(y),
    )


def linear_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    if len(x) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }

    design = np.column_stack(
        (
            x,
            np.ones_like(x),
        )
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    slope = float(
        coefficients[0]
    )

    intercept = float(
        coefficients[1]
    )

    predicted = (
        slope * x
        + intercept
    )

    residual_sum = float(
        np.sum(
            (y - predicted) ** 2
        )
    )

    total_sum = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    if total_sum <= 1.0e-15:
        r_squared = float("nan")
    else:
        r_squared = (
            1.0
            - residual_sum
            / total_sum
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": float(
            r_squared
        ),
    }


def summarize_pair_alignment(
    so3_distance: np.ndarray,
    representation_distance: np.ndarray,
) -> Dict[str, float]:
    fit = linear_fit(
        so3_distance,
        representation_distance,
    )

    return {
        "pairs": int(
            len(so3_distance)
        ),
        "so3_mean_rad": float(
            np.mean(so3_distance)
        ),
        "so3_median_rad": float(
            np.median(so3_distance)
        ),
        "so3_std_rad": float(
            np.std(so3_distance)
        ),
        "representation_distance_mean": float(
            np.mean(
                representation_distance
            )
        ),
        "representation_distance_median": float(
            np.median(
                representation_distance
            )
        ),
        "representation_distance_std": float(
            np.std(
                representation_distance
            )
        ),
        "pearson_r": pearson_correlation(
            so3_distance,
            representation_distance,
        ),
        "spearman_r": spearman_correlation(
            so3_distance,
            representation_distance,
        ),
        "linear_slope": fit["slope"],
        "linear_intercept": (
            fit["intercept"]
        ),
        "linear_r_squared": (
            fit["r_squared"]
        ),
    }


# ============================================================================
# Distance-bin audit
# ============================================================================


def build_distance_bins(
    so3_distance: np.ndarray,
    representation_distance: np.ndarray,
    bin_count: int,
) -> pd.DataFrame:
    quantile_edges = np.quantile(
        so3_distance,
        np.linspace(
            0.0,
            1.0,
            bin_count + 1,
        ),
    )

    # Quantiles can collapse when many rotations have essentially the
    # same separation. Force monotonic unique edges.
    quantile_edges = np.unique(
        quantile_edges
    )

    if len(quantile_edges) < 3:
        raise RuntimeError(
            "SO(3) distance distribution has insufficient variation "
            "for distance-bin analysis."
        )

    rows: List[Dict[str, object]] = []

    actual_bins = (
        len(quantile_edges) - 1
    )

    for bin_index in range(
        actual_bins
    ):
        lower = (
            quantile_edges[
                bin_index
            ]
        )

        upper = (
            quantile_edges[
                bin_index + 1
            ]
        )

        if bin_index == actual_bins - 1:
            mask = (
                (so3_distance >= lower)
                & (so3_distance <= upper)
            )
        else:
            mask = (
                (so3_distance >= lower)
                & (so3_distance < upper)
            )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            continue

        current_so3 = (
            so3_distance[mask]
        )

        current_rep = (
            representation_distance[mask]
        )

        rows.append(
            {
                "bin": bin_index,
                "pair_count": count,
                "so3_lower_rad": float(
                    lower
                ),
                "so3_upper_rad": float(
                    upper
                ),
                "so3_mean_rad": float(
                    np.mean(
                        current_so3
                    )
                ),
                "so3_mean_deg": float(
                    np.degrees(
                        np.mean(
                            current_so3
                        )
                    )
                ),
                "representation_distance_mean": float(
                    np.mean(
                        current_rep
                    )
                ),
                "representation_distance_std": float(
                    np.std(
                        current_rep
                    )
                ),
                "representation_distance_median": float(
                    np.median(
                        current_rep
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Local geometry
# ============================================================================


def build_local_geometry_metrics(
    so3_distance: np.ndarray,
    representation_distance: np.ndarray,
    q_near: float,
    q_far: float,
) -> pd.DataFrame:
    threshold_near = float(
        np.quantile(
            so3_distance,
            q_near,
        )
    )

    threshold_far = float(
        np.quantile(
            so3_distance,
            q_far,
        )
    )

    regions = (
        (
            "near",
            so3_distance <= threshold_near,
        ),
        (
            "medium",
            (
                (so3_distance > threshold_near)
                & (so3_distance <= threshold_far)
            ),
        ),
        (
            "far",
            so3_distance > threshold_far,
        ),
    )

    rows = []

    for name, mask in regions:
        current_so3 = (
            so3_distance[mask]
        )

        current_rep = (
            representation_distance[
                mask
            ]
        )

        metrics = summarize_pair_alignment(
            current_so3,
            current_rep,
        )

        rows.append(
            {
                "region": name,
                "so3_threshold_near_rad": (
                    threshold_near
                ),
                "so3_threshold_far_rad": (
                    threshold_far
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Full distance matrices for neighborhood audit
# ============================================================================


def full_so3_distance_matrix(
    rotations: np.ndarray,
) -> np.ndarray:
    sample_count = (
        rotations.shape[0]
    )

    result = np.empty(
        (
            sample_count,
            sample_count,
        ),
        dtype=np.float64,
    )

    for index in range(
        sample_count
    ):
        relative = (
            rotations[index].T[
                None,
                :,
                :,
            ]
            @ rotations
        )

        trace = np.trace(
            relative,
            axis1=1,
            axis2=2,
        )

        cosine = np.clip(
            (
                trace - 1.0
            ) * 0.5,
            -1.0,
            1.0,
        )

        result[index] = (
            np.arccos(cosine)
        )

    np.fill_diagonal(
        result,
        np.inf,
    )

    return result


def full_representation_distance_matrix(
    representation_normalized: np.ndarray,
) -> np.ndarray:
    cosine = (
        representation_normalized
        @ representation_normalized.T
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    distance = 1.0 - cosine

    np.fill_diagonal(
        distance,
        np.inf,
    )

    return distance


# ============================================================================
# Neighborhood preservation
# ============================================================================


def neighborhood_overlap_metrics(
    so3_matrix: np.ndarray,
    representation_matrix: np.ndarray,
    k_values: Sequence[int],
) -> pd.DataFrame:
    sample_count = (
        so3_matrix.shape[0]
    )

    rows = []

    for requested_k in sorted(
        set(k_values)
    ):
        k = min(
            requested_k,
            sample_count - 1,
        )

        if k <= 0:
            continue

        so3_neighbors = np.argpartition(
            so3_matrix,
            kth=k - 1,
            axis=1,
        )[:, :k]

        representation_neighbors = (
            np.argpartition(
                representation_matrix,
                kth=k - 1,
                axis=1,
            )[:, :k]
        )

        overlaps = np.empty(
            sample_count,
            dtype=np.float64,
        )

        for sample_index in range(
            sample_count
        ):
            intersection = np.intersect1d(
                so3_neighbors[
                    sample_index
                ],
                representation_neighbors[
                    sample_index
                ],
                assume_unique=False,
            )

            overlaps[
                sample_index
            ] = (
                len(intersection)
                / float(k)
            )

        random_expected = (
            k
            / float(
                sample_count - 1
            )
        )

        rows.append(
            {
                "k": requested_k,
                "effective_k": k,
                "samples": sample_count,
                "mean_overlap": float(
                    np.mean(overlaps)
                ),
                "median_overlap": float(
                    np.median(overlaps)
                ),
                "std_overlap": float(
                    np.std(overlaps)
                ),
                "random_expected_overlap": float(
                    random_expected
                ),
                "lift_over_random": float(
                    np.mean(overlaps)
                    / random_expected
                    if random_expected > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Per-sample nearest-neighbor audit
# ============================================================================


def build_nearest_neighbor_audit(
    predictions: pd.DataFrame,
    so3_matrix: np.ndarray,
    representation_matrix: np.ndarray,
) -> pd.DataFrame:
    sample_count = (
        so3_matrix.shape[0]
    )

    so3_neighbor = np.argmin(
        so3_matrix,
        axis=1,
    )

    representation_neighbor = np.argmin(
        representation_matrix,
        axis=1,
    )

    rows: List[
        Dict[str, object]
    ] = []

    for index in range(
        sample_count
    ):
        so3_nn = int(
            so3_neighbor[index]
        )

        rep_nn = int(
            representation_neighbor[
                index
            ]
        )

        row: Dict[
            str,
            object
        ] = {
            "sample_index": index,
            "so3_nearest_index": so3_nn,
            "representation_nearest_index": rep_nn,
            "nearest_neighbor_match": int(
                so3_nn == rep_nn
            ),
            "so3_distance_to_so3_nearest_rad": float(
                so3_matrix[
                    index,
                    so3_nn,
                ]
            ),
            "so3_distance_to_so3_nearest_deg": float(
                np.degrees(
                    so3_matrix[
                        index,
                        so3_nn,
                    ]
                )
            ),
            "representation_distance_to_so3_nearest": float(
                representation_matrix[
                    index,
                    so3_nn,
                ]
            ),
            "representation_distance_to_rep_nearest": float(
                representation_matrix[
                    index,
                    rep_nn,
                ]
            ),
            "so3_distance_to_rep_nearest_rad": float(
                so3_matrix[
                    index,
                    rep_nn,
                ]
            ),
            "so3_distance_to_rep_nearest_deg": float(
                np.degrees(
                    so3_matrix[
                        index,
                        rep_nn,
                    ]
                )
            ),
        }

        for column in FRAME_ID_COLUMNS:
            if column in predictions.columns:
                row[column] = (
                    predictions.iloc[
                        index
                    ][column]
                )

                row[
                    f"so3_nearest_{column}"
                ] = predictions.iloc[
                    so3_nn
                ][column]

                row[
                    f"representation_nearest_{column}"
                ] = predictions.iloc[
                    rep_nn
                ][column]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================


def plot_pair_scatter(
    so3_distance: np.ndarray,
    representation_distance: np.ndarray,
    path: Path,
    maximum_points: int,
    rng: np.random.Generator,
    dpi: int,
) -> None:
    count = len(so3_distance)

    if count > maximum_points:
        selected = rng.choice(
            count,
            size=maximum_points,
            replace=False,
        )

        x = so3_distance[
            selected
        ]

        y = representation_distance[
            selected
        ]
    else:
        x = so3_distance
        y = representation_distance

    metrics = summarize_pair_alignment(
        x,
        y,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 6.5)
    )

    axis.scatter(
        np.degrees(x),
        y,
        s=8,
        alpha=0.20,
    )

    fit = linear_fit(
        x,
        y,
    )

    if np.isfinite(
        fit["slope"]
    ):
        fit_x = np.linspace(
            np.min(x),
            np.max(x),
            200,
        )

        fit_y = (
            fit["slope"]
            * fit_x
            + fit["intercept"]
        )

        axis.plot(
            np.degrees(fit_x),
            fit_y,
            linewidth=2.0,
        )

    annotation = (
        f"pairs = {len(x):,}\n"
        f"Pearson r = {metrics['pearson_r']:.4f}\n"
        f"Spearman r = {metrics['spearman_r']:.4f}\n"
        f"R² = {metrics['linear_r_squared']:.4f}"
    )

    axis.text(
        0.03,
        0.97,
        annotation,
        transform=axis.transAxes,
        va="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.85,
        },
    )

    axis.set_xlabel(
        "Ground-truth SO(3) geodesic distance (deg)"
    )

    axis.set_ylabel(
        "Rotation representation cosine distance"
    )

    axis.set_title(
        "Continuous SO(3) vs learned rotation geometry"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_distance_bins(
    table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.5, 6.0)
    )

    x = table[
        "so3_mean_deg"
    ].to_numpy(
        dtype=np.float64
    )

    y = table[
        "representation_distance_mean"
    ].to_numpy(
        dtype=np.float64
    )

    yerr = table[
        "representation_distance_std"
    ].to_numpy(
        dtype=np.float64
    )

    axis.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.5,
        capsize=3,
    )

    axis.set_xlabel(
        "Mean SO(3) distance in bin (deg)"
    )

    axis.set_ylabel(
        "Mean representation cosine distance"
    )

    axis.set_title(
        "Representation distance across SO(3) separation"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_local_correlations(
    table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.0, 5.5)
    )

    names = table[
        "region"
    ].astype(str).tolist()

    values = table[
        "spearman_r"
    ].to_numpy(
        dtype=np.float64
    )

    positions = np.arange(
        len(names)
    )

    axis.bar(
        positions,
        values,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        names
    )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_ylabel(
        "Spearman correlation"
    )

    axis.set_title(
        "SO(3)-latent alignment by geometric separation"
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_nearest_neighbor_so3_distance(
    table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    correct = table[
        "so3_distance_to_so3_nearest_deg"
    ].to_numpy(
        dtype=np.float64
    )

    latent = table[
        "so3_distance_to_rep_nearest_deg"
    ].to_numpy(
        dtype=np.float64
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 6.0)
    )

    axis.scatter(
        correct,
        latent,
        s=12,
        alpha=0.45,
    )

    maximum = float(
        max(
            np.max(correct),
            np.max(latent),
        )
    )

    axis.plot(
        [0.0, maximum],
        [0.0, maximum],
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_xlabel(
        "SO(3) distance to true SO(3) nearest neighbor (deg)"
    )

    axis.set_ylabel(
        "SO(3) distance to latent nearest neighbor (deg)"
    )

    axis.set_title(
        "Geometric quality of latent nearest neighbors"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


# ============================================================================
# Interpretation
# ============================================================================


def build_interpretation(
    pair_metrics: Mapping[
        str,
        float
    ],
    neighborhood_table: pd.DataFrame,
    local_table: pd.DataFrame,
) -> Dict[str, object]:
    spearman = float(
        pair_metrics[
            "spearman_r"
        ]
    )

    pearson = float(
        pair_metrics[
            "pearson_r"
        ]
    )

    if not np.isfinite(spearman):
        global_status = (
            "UNDETERMINED"
        )
    elif spearman >= 0.70:
        global_status = (
            "STRONG_ALIGNMENT"
        )
    elif spearman >= 0.45:
        global_status = (
            "MODERATE_ALIGNMENT"
        )
    elif spearman >= 0.20:
        global_status = (
            "WEAK_ALIGNMENT"
        )
    else:
        global_status = (
            "POOR_ALIGNMENT"
        )

    neighborhood_status = (
        "UNDETERMINED"
    )

    if len(neighborhood_table):
        lifts = neighborhood_table[
            "lift_over_random"
        ].to_numpy(
            dtype=np.float64
        )

        finite_lifts = lifts[
            np.isfinite(lifts)
        ]

        if len(finite_lifts):
            mean_lift = float(
                np.mean(
                    finite_lifts
                )
            )

            if mean_lift >= 5.0:
                neighborhood_status = (
                    "STRONG_LOCAL_STRUCTURE"
                )
            elif mean_lift >= 2.0:
                neighborhood_status = (
                    "MEANINGFUL_LOCAL_STRUCTURE"
                )
            elif mean_lift > 1.1:
                neighborhood_status = (
                    "WEAK_LOCAL_STRUCTURE"
                )
            else:
                neighborhood_status = (
                    "NEAR_RANDOM_LOCAL_STRUCTURE"
                )
        else:
            mean_lift = float(
                "nan"
            )
    else:
        mean_lift = float(
            "nan"
        )

    local_correlations: Dict[
        str,
        Optional[float]
    ] = {}

    for _, row in (
        local_table.iterrows()
    ):
        local_correlations[
            str(row["region"])
        ] = safe_float(
            row["spearman_r"]
        )

    return {
        "global_geometry_status": (
            global_status
        ),
        "neighborhood_status": (
            neighborhood_status
        ),
        "global_pearson_r": (
            safe_float(
                pearson
            )
        ),
        "global_spearman_r": (
            safe_float(
                spearman
            )
        ),
        "mean_neighborhood_lift_over_random": (
            safe_float(
                mean_lift
            )
        ),
        "local_spearman_correlations": (
            local_correlations
        ),
        "reading_guide": [
            (
                "A positive global Spearman correlation means "
                "larger ground-truth SO(3) separation tends to "
                "produce larger latent separation."
            ),
            (
                "Neighborhood lift above 1.0 means the learned "
                "representation preserves local SO(3) neighbors "
                "better than random chance."
            ),
            (
                "Near-region correlation is particularly important "
                "for odometry because consecutive vehicle rotations "
                "usually occupy a small portion of SO(3)."
            ),
            (
                "Strong prediction accuracy without geometric "
                "alignment would indicate that the decoder can use "
                "the representation even though the bottleneck itself "
                "does not organize continuously by rotation."
            ),
        ],
    }


# ============================================================================
# Console reporting
# ============================================================================


def print_summary(
    sample_count: int,
    representation_dimension: int,
    total_pair_count: int,
    analyzed_pair_count: int,
    pair_metrics: Mapping[
        str,
        float
    ],
    neighborhood_table: pd.DataFrame,
    local_table: pd.DataFrame,
    interpretation: Mapping[
        str,
        object
    ],
) -> None:
    print()
    print("=" * 92)
    print(
        "DeepDCT-VO continuous SO(3) rotation-geometry alignment audit"
    )
    print("=" * 92)

    print(
        f"Samples:                      {sample_count}"
    )

    print(
        "Rotation representation dim:  "
        f"{representation_dimension}"
    )

    print(
        f"Total possible pairs:         {total_pair_count:,}"
    )

    print(
        f"Analyzed pairs:               {analyzed_pair_count:,}"
    )

    print("-" * 92)
    print(
        "Global pairwise geometry"
    )
    print("-" * 92)

    print(
        "Pearson r:                    "
        f"{pair_metrics['pearson_r']:.6f}"
    )

    print(
        "Spearman r:                   "
        f"{pair_metrics['spearman_r']:.6f}"
    )

    print(
        "Linear R^2:                   "
        f"{pair_metrics['linear_r_squared']:.6f}"
    )

    print(
        "Mean SO(3) distance:           "
        f"{np.degrees(pair_metrics['so3_mean_rad']):.6f} deg"
    )

    print(
        "Mean representation distance: "
        f"{pair_metrics['representation_distance_mean']:.6f}"
    )

    print("-" * 92)
    print(
        "SO(3)-nearest-neighbor preservation"
    )
    print("-" * 92)

    if len(neighborhood_table):
        print(
            neighborhood_table[
                [
                    "k",
                    "mean_overlap",
                    "random_expected_overlap",
                    "lift_over_random",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print("-" * 92)
    print(
        "Local/global geometry"
    )
    print("-" * 92)

    print(
        local_table[
            [
                "region",
                "pairs",
                "pearson_r",
                "spearman_r",
                "linear_r_squared",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print("-" * 92)

    print(
        "Global geometry status:        "
        f"{interpretation['global_geometry_status']}"
    )

    print(
        "Neighborhood status:           "
        f"{interpretation['neighborhood_status']}"
    )

    print("=" * 92)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load representation and labels.
    # ------------------------------------------------------------------

    (
        representation,
        rotation_gt_npz,
        representation_key,
    ) = load_npz(
        args.representations
    )

    predictions = load_predictions(
        args.predictions
    )

    if rotation_gt_npz is None:
        rotation_gt = (
            rotation_gt_from_csv(
                predictions
            )
        )

        rotation_gt_source = (
            "frame_predictions.csv"
        )

    else:
        rotation_gt = (
            rotation_gt_npz
        )

        rotation_gt_source = (
            "rotation_representations.npz"
        )

    sample_count = (
        representation.shape[0]
    )

    if len(predictions) != sample_count:
        raise ValueError(
            "Prediction-row count does not match representation count: "
            f"{len(predictions)} vs {sample_count}."
        )

    if (
        rotation_gt.shape[0]
        != sample_count
    ):
        raise ValueError(
            "Ground-truth rotation count does not match representation "
            f"count: {rotation_gt.shape[0]} vs {sample_count}."
        )

    if sample_count < 3:
        raise ValueError(
            "At least three samples are required for "
            "rotation-geometry alignment analysis."
        )

    representation_dimension = int(
        representation.shape[1]
    )

    print("=" * 92)
    print(
        "Loading rotation-geometry audit inputs"
    )
    print("=" * 92)

    print(
        f"Representations:      {args.representations.resolve()}"
    )

    print(
        f"Representation key:   {representation_key}"
    )

    print(
        f"Predictions:          {args.predictions.resolve()}"
    )

    print(
        f"GT rotation source:   {rotation_gt_source}"
    )

    print(
        f"Samples:              {sample_count}"
    )

    print(
        f"Representation shape: {representation.shape}"
    )

    print(
        "Angles in degrees:    "
        f"{args.angles_in_degrees}"
    )

    # ------------------------------------------------------------------
    # Convert GT Euler rotations to SO(3).
    # ------------------------------------------------------------------

    rotation_matrices = (
        euler_xyz_to_rotation_matrix(
            rotation_gt,
            degrees=(
                args.angles_in_degrees
            ),
        )
    )

    representation_normalized = (
        normalized_representation(
            representation
        )
    )

    # ------------------------------------------------------------------
    # Pairwise global geometry.
    # ------------------------------------------------------------------

    (
        pair_i,
        pair_j,
        total_pair_count,
    ) = sampled_pair_indices(
        sample_count=sample_count,
        maximum_pairs=args.max_pairs,
        rng=rng,
    )

    so3_distance = (
        so3_pair_distance(
            rotation_matrices,
            pair_i,
            pair_j,
        )
    )

    representation_distance = (
        representation_pair_distance(
            representation_normalized,
            pair_i,
            pair_j,
        )
    )

    pair_metrics = (
        summarize_pair_alignment(
            so3_distance,
            representation_distance,
        )
    )

    pair_metrics_table = pd.DataFrame(
        [
            {
                "samples": sample_count,
                "representation_dimension": (
                    representation_dimension
                ),
                "total_possible_pairs": (
                    total_pair_count
                ),
                **pair_metrics,
            }
        ]
    )

    pair_metrics_table.to_csv(
        args.output_dir
        / "pairwise_alignment_metrics.csv",
        index=False,
    )

    sampled_pairs = pd.DataFrame(
        {
            "sample_i": pair_i,
            "sample_j": pair_j,
            "so3_distance_rad": (
                so3_distance
            ),
            "so3_distance_deg": (
                np.degrees(
                    so3_distance
                )
            ),
            "representation_cosine_distance": (
                representation_distance
            ),
        }
    )

    sampled_pairs.to_csv(
        args.output_dir
        / "sampled_pair_distances.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # SO(3)-distance bins.
    # ------------------------------------------------------------------

    distance_bins = (
        build_distance_bins(
            so3_distance=so3_distance,
            representation_distance=(
                representation_distance
            ),
            bin_count=args.distance_bins,
        )
    )

    distance_bins.to_csv(
        args.output_dir
        / "distance_bins.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Near / medium / far continuous geometry.
    # ------------------------------------------------------------------

    local_table = (
        build_local_geometry_metrics(
            so3_distance=so3_distance,
            representation_distance=(
                representation_distance
            ),
            q_near=(
                args.local_quantiles[0]
            ),
            q_far=(
                args.local_quantiles[1]
            ),
        )
    )

    local_table.to_csv(
        args.output_dir
        / "local_geometry_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Full matrices for neighborhood preservation.
    #
    # Sequence 10 has ~1200 transitions, so NxN matrices are modest:
    # 1200 x 1200 float64 ~= 11 MB per matrix.
    # ------------------------------------------------------------------

    print()
    print(
        "Computing full SO(3) neighborhood matrix..."
    )

    so3_matrix = (
        full_so3_distance_matrix(
            rotation_matrices
        )
    )

    print(
        "Computing full representation neighborhood matrix..."
    )

    representation_matrix = (
        full_representation_distance_matrix(
            representation_normalized
        )
    )

    neighborhood_table = (
        neighborhood_overlap_metrics(
            so3_matrix=so3_matrix,
            representation_matrix=(
                representation_matrix
            ),
            k_values=args.neighbor_k,
        )
    )

    neighborhood_table.to_csv(
        args.output_dir
        / "neighborhood_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Per-frame nearest-neighbor audit.
    # ------------------------------------------------------------------

    nearest_neighbor_table = (
        build_nearest_neighbor_audit(
            predictions=predictions,
            so3_matrix=so3_matrix,
            representation_matrix=(
                representation_matrix
            ),
        )
    )

    nearest_neighbor_table.to_csv(
        args.output_dir
        / "nearest_neighbor_audit.csv",
        index=False,
    )

    nearest_neighbor_exact_match = float(
        nearest_neighbor_table[
            "nearest_neighbor_match"
        ].mean()
    )

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------

    plot_pair_scatter(
        so3_distance=so3_distance,
        representation_distance=(
            representation_distance
        ),
        path=(
            args.output_dir
            / "so3_vs_representation_distance.png"
        ),
        maximum_points=(
            args.scatter_pairs
        ),
        rng=rng,
        dpi=args.dpi,
    )

    plot_distance_bins(
        table=distance_bins,
        path=(
            args.output_dir
            / "so3_distance_bins.png"
        ),
        dpi=args.dpi,
    )

    plot_local_correlations(
        table=local_table,
        path=(
            args.output_dir
            / "local_geometry_correlations.png"
        ),
        dpi=args.dpi,
    )

    plot_nearest_neighbor_so3_distance(
        table=nearest_neighbor_table,
        path=(
            args.output_dir
            / "nearest_neighbor_so3_distance.png"
        ),
        dpi=args.dpi,
    )

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------

    interpretation = (
        build_interpretation(
            pair_metrics=pair_metrics,
            neighborhood_table=(
                neighborhood_table
            ),
            local_table=local_table,
        )
    )

    neighborhood_records = []

    for _, row in (
        neighborhood_table.iterrows()
    ):
        neighborhood_records.append(
            {
                "k": int(row["k"]),
                "effective_k": int(
                    row["effective_k"]
                ),
                "mean_overlap": safe_float(
                    row["mean_overlap"]
                ),
                "random_expected_overlap": safe_float(
                    row[
                        "random_expected_overlap"
                    ]
                ),
                "lift_over_random": safe_float(
                    row["lift_over_random"]
                ),
            }
        )

    local_records = []

    for _, row in (
        local_table.iterrows()
    ):
        local_records.append(
            {
                "region": str(
                    row["region"]
                ),
                "pairs": int(
                    row["pairs"]
                ),
                "pearson_r": safe_float(
                    row["pearson_r"]
                ),
                "spearman_r": safe_float(
                    row["spearman_r"]
                ),
                "linear_r_squared": safe_float(
                    row[
                        "linear_r_squared"
                    ]
                ),
            }
        )

    summary = {
        "audit": (
            "continuous_so3_rotation_geometry_alignment"
        ),
        "inputs": {
            "representations": str(
                args.representations.resolve()
            ),
            "predictions": str(
                args.predictions.resolve()
            ),
            "representation_key": (
                representation_key
            ),
            "rotation_gt_source": (
                rotation_gt_source
            ),
            "angles_in_degrees": bool(
                args.angles_in_degrees
            ),
            "euler_convention": (
                "extrinsic_xyz_Rz_Ry_Rx"
            ),
        },
        "representation": {
            "samples": sample_count,
            "dimension": (
                representation_dimension
            ),
            "mean_norm": float(
                np.mean(
                    np.linalg.norm(
                        representation,
                        axis=1,
                    )
                )
            ),
            "std_norm": float(
                np.std(
                    np.linalg.norm(
                        representation,
                        axis=1,
                    )
                )
            ),
        },
        "pair_sampling": {
            "total_possible_pairs": int(
                total_pair_count
            ),
            "analyzed_pairs": int(
                len(pair_i)
            ),
            "maximum_pairs": int(
                args.max_pairs
            ),
            "seed": int(
                args.seed
            ),
        },
        "global_pairwise_alignment": {
            key: (
                int(value)
                if key == "pairs"
                else safe_float(value)
            )
            for key, value in (
                pair_metrics.items()
            )
        },
        "neighborhood_alignment": (
            neighborhood_records
        ),
        "nearest_neighbor": {
            "exact_neighbor_match_fraction": (
                nearest_neighbor_exact_match
            ),
            "mean_so3_distance_to_true_nearest_deg": float(
                nearest_neighbor_table[
                    "so3_distance_to_so3_nearest_deg"
                ].mean()
            ),
            "mean_so3_distance_to_latent_nearest_deg": float(
                nearest_neighbor_table[
                    "so3_distance_to_rep_nearest_deg"
                ].mean()
            ),
        },
        "local_geometry": (
            local_records
        ),
        "interpretation": (
            interpretation
        ),
        "outputs": {
            "pairwise_alignment_metrics": (
                "pairwise_alignment_metrics.csv"
            ),
            "distance_bins": (
                "distance_bins.csv"
            ),
            "local_geometry_metrics": (
                "local_geometry_metrics.csv"
            ),
            "neighborhood_metrics": (
                "neighborhood_metrics.csv"
            ),
            "nearest_neighbor_audit": (
                "nearest_neighbor_audit.csv"
            ),
            "sampled_pair_distances": (
                "sampled_pair_distances.csv"
            ),
            "plots": [
                (
                    "so3_vs_representation_distance.png"
                ),
                "so3_distance_bins.png",
                (
                    "local_geometry_correlations.png"
                ),
                (
                    "nearest_neighbor_so3_distance.png"
                ),
            ],
        },
    }

    write_json(
        args.output_dir
        / "summary.json",
        summary,
    )

    print_summary(
        sample_count=sample_count,
        representation_dimension=(
            representation_dimension
        ),
        total_pair_count=(
            total_pair_count
        ),
        analyzed_pair_count=(
            len(pair_i)
        ),
        pair_metrics=pair_metrics,
        neighborhood_table=(
            neighborhood_table
        ),
        local_table=local_table,
        interpretation=(
            interpretation
        ),
    )

    print()
    print(
        "Outputs saved to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()