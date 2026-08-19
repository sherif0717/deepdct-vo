#!/usr/bin/env python3
"""
Conditional rotation-latent alignment audit for DeepDCT-VO.

Purpose
-------
Previous audits established:

1. The exact 14400-D rotation representation can be reconstructed.
2. The full latent covariance geometry varies strongly across sequences.
3. The marginal distribution of the ORIGINAL readout-relevant 3-D
   subspace is comparatively stable.

This script therefore audits the CONDITIONAL relationship:

    latent coordinates -> ground-truth rotation

rather than only the marginal latent distribution.

For the original trained rotation readout:

    r_hat = W h + b

we construct an orthonormal basis Q for:

    span(W.T)

and project the exact representation:

    u = h @ Q

where u has dimension <= 3.

For each KITTI sequence s, fit:

    rotation_gt = u @ A_s + c_s

using ordinary/ridge linear regression.

We then ask:

    Does A_s remain stable across sequences?

Outputs
-------
sequence_mapping_metrics.csv
sequence_mapping_coefficients.csv
sequence_mapping_singular_values.csv
mapping_pairwise_comparison.csv
cross_sequence_transfer.csv
pooled_transfer.csv
summary.json
plots/
    coefficient_frobenius_distance.png
    cross_sequence_vector_rmse.png
    cross_sequence_y_correlation.png

Interpretation
--------------
Stable p(u) but changing A_s implies conditional/domain instability:

    p(rotation | u, sequence)

is sequence dependent even when:

    p(u | sequence)

looks similar.

This is stronger evidence against simple centroid/scale calibration.

Typical command
---------------
python scripts/analyze_rotation_conditional_alignment.py \
    --input-root experiments/semantic_depth_rotation_rep_inputs \
    --checkpoint experiments/semantic_depth_identity_output/best_validation.pt \
    --output-dir experiments/semantic_depth_rotation_conditional_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


AXES = (
    "x",
    "y",
    "z",
)

ALL_SEQUENCES = tuple(
    f"{index:02d}"
    for index in range(11)
)

TRAIN_SEQUENCES = tuple(
    f"{index:02d}"
    for index in range(9)
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit sequence-dependent conditional mappings from "
            "rotation-readout latent coordinates to GT rotation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory containing sequence folders 00 ... 10. "
            "Each folder must contain rotation_representations.npz "
            "and frame_predictions.csv."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Original checkpoint used to define the rotation-readout "
            "subspace."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--representation-file",
        type=str,
        default="rotation_representations.npz",
    )

    parser.add_argument(
        "--representation-key",
        type=str,
        default="rotation_rep",
    )

    parser.add_argument(
        "--prediction-file",
        type=str,
        default="frame_predictions.csv",
    )

    parser.add_argument(
        "--weight-key",
        type=str,
        default="rotation_head.dense.weight",
    )

    parser.add_argument(
        "--bias-key",
        type=str,
        default="rotation_head.dense.bias",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0e-6,
        help=(
            "Small ridge stabilization for local 3-D mappings."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help=(
            "Temporal fraction used to fit each sequence-local mapping. "
            "The remaining tail is used as within-sequence test data."
        ),
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-10,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.input_root.is_dir():
        raise FileNotFoundError(
            f"Missing input root: {args.input_root}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {args.checkpoint}"
        )

    if not (
        0.0
        < args.train_fraction
        < 1.0
    ):
        raise ValueError(
            "--train-fraction must lie strictly between 0 and 1."
        )

    if args.ridge_alpha < 0.0:
        raise ValueError(
            "--ridge-alpha cannot be negative."
        )

    if args.epsilon <= 0.0:
        raise ValueError(
            "--epsilon must be positive."
        )


# ============================================================================
# Data loading
# ============================================================================


def load_representation(
    path: Path,
    key: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing representation file: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if key not in archive:
            raise KeyError(
                f"{path} does not contain {key!r}. "
                f"Available keys: {archive.files}"
            )

        representation = np.asarray(
            archive[key],
            dtype=np.float64,
        )

    if representation.ndim > 2:
        representation = representation.reshape(
            representation.shape[0],
            -1,
        )

    if representation.ndim != 2:
        raise ValueError(
            "Expected rotation representation [N,D], "
            f"received {representation.shape}."
        )

    if not np.all(
        np.isfinite(
            representation
        )
    ):
        raise ValueError(
            f"Non-finite representation in {path}."
        )

    return representation


def load_rotation_gt(
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction CSV: {path}"
        )

    rows: List[
        List[float]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        required = (
            "rotation_gt_x",
            "rotation_gt_y",
            "rotation_gt_z",
        )

        fields = (
            reader.fieldnames
            or []
        )

        missing = [
            field
            for field in required
            if field not in fields
        ]

        if missing:
            raise KeyError(
                f"{path} missing columns: {missing}"
            )

        for row in reader:
            rows.append(
                [
                    float(
                        row[
                            "rotation_gt_x"
                        ]
                    ),
                    float(
                        row[
                            "rotation_gt_y"
                        ]
                    ),
                    float(
                        row[
                            "rotation_gt_z"
                        ]
                    ),
                ]
            )

    result = np.asarray(
        rows,
        dtype=np.float64,
    )

    if (
        result.ndim != 2
        or result.shape[1]
        != 3
    ):
        raise ValueError(
            f"Unexpected GT shape: {result.shape}"
        )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise ValueError(
            f"Non-finite GT rotations: {path}"
        )

    return result


def load_sequence(
    root: Path,
    sequence: str,
    representation_file: str,
    representation_key: str,
    prediction_file: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    h = load_representation(
        root
        / sequence
        / representation_file,
        representation_key,
    )

    y = load_rotation_gt(
        root
        / sequence
        / prediction_file
    )

    if h.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sequence {sequence}: representation/target mismatch "
            f"{h.shape[0]} vs {y.shape[0]}."
        )

    return (
        h,
        y,
    )


# ============================================================================
# Checkpoint readout subspace
# ============================================================================


def normalize_state_dict_keys(
    state_dict: Mapping[
        str,
        Any,
    ],
) -> Dict[
    str,
    Any,
]:
    result: Dict[
        str,
        Any,
    ] = {}

    for key, value in state_dict.items():
        if key.startswith(
            "module."
        ):
            key = key[
                len(
                    "module."
                ):
            ]

        result[
            key
        ] = value

    return result


def load_original_readout(
    checkpoint_path: Path,
    weight_key: str,
    bias_key: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise TypeError(
            "Checkpoint must be a mapping."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    if weight_key not in state_dict:
        raise KeyError(
            f"Missing checkpoint key: {weight_key}"
        )

    if bias_key not in state_dict:
        raise KeyError(
            f"Missing checkpoint key: {bias_key}"
        )

    weight = (
        state_dict[
            weight_key
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    bias = (
        state_dict[
            bias_key
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    if (
        weight.ndim != 2
        or weight.shape[0]
        != 3
    ):
        raise ValueError(
            "Expected rotation readout weight [3,D], "
            f"received {weight.shape}."
        )

    return (
        weight,
        bias,
    )


def build_readout_basis(
    weight: np.ndarray,
) -> np.ndarray:
    """
    Construct an orthonormal basis for span(W.T).

    Returns
    -------
    Q:
        [D, rank], rank <= 3.
    """

    u, singular_values, _ = np.linalg.svd(
        weight.T,
        full_matrices=False,
    )

    tolerance = (
        max(
            weight.shape
        )
        * np.max(
            singular_values
        )
        * np.finfo(
            np.float64
        ).eps
    )

    rank = int(
        np.count_nonzero(
            singular_values
            > tolerance
        )
    )

    if rank <= 0:
        raise RuntimeError(
            "Original rotation readout has zero rank."
        )

    return u[
        :,
        :rank,
    ]


# ============================================================================
# Regression
# ============================================================================


def fit_mapping(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Standardize x using training-domain statistics, then fit:

        y = x_standardized @ coefficients + intercept

    Returns
    -------
    coefficients:
        [R,3]

    intercept:
        [3]

    x_mean:
        [R]

    x_std:
        [R]
    """

    x_mean = np.mean(
        x,
        axis=0,
    )

    x_std = np.std(
        x,
        axis=0,
    )

    x_std = np.where(
        x_std
        > 1.0e-12,
        x_std,
        1.0,
    )

    xs = (
        x
        - x_mean
    ) / x_std

    y_mean = np.mean(
        y,
        axis=0,
    )

    yc = (
        y
        - y_mean
    )

    gram = (
        xs.T
        @ xs
    )

    regularizer = (
        alpha
        * np.eye(
            gram.shape[0],
            dtype=np.float64,
        )
    )

    coefficients = np.linalg.solve(
        gram
        + regularizer,
        xs.T
        @ yc,
    )

    return (
        coefficients,
        y_mean,
        x_mean,
        x_std,
    )


def predict_mapping(
    x: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    xs = (
        x
        - x_mean
    ) / x_std

    return (
        xs
        @ coefficients
        + intercept
    )


# ============================================================================
# Metrics
# ============================================================================


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if (
        np.std(
            a
        )
        <= 1.0e-12
        or np.std(
            b
        )
        <= 1.0e-12
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[
    str,
    Any,
]:
    error = (
        y_pred
        - y_true
    )

    result: Dict[
        str,
        Any,
    ] = {
        "vector_rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
    }

    for index, axis in enumerate(
        AXES
    ):
        gt = y_true[
            :,
            index
        ]

        pred = y_pred[
            :,
            index
        ]

        axis_error = (
            pred
            - gt
        )

        gt_std = float(
            np.std(
                gt
            )
        )

        pred_std = float(
            np.std(
                pred
            )
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

        result[
            f"{axis}_corr"
        ] = safe_correlation(
            gt,
            pred,
        )

        result[
            f"{axis}_std_ratio"
        ] = (
            pred_std
            / gt_std
            if gt_std
            > 1.0e-12
            else float(
                "nan"
            )
        )

    return result


# ============================================================================
# Mapping comparison
# ============================================================================


def matrix_cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> float:
    a_flat = a.reshape(
        -1
    )

    b_flat = b.reshape(
        -1
    )

    denominator = (
        np.linalg.norm(
            a_flat
        )
        * np.linalg.norm(
            b_flat
        )
    )

    if denominator <= epsilon:
        return float(
            "nan"
        )

    return float(
        np.dot(
            a_flat,
            b_flat,
        )
        / denominator
    )


def matrix_angle_degrees(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> float:
    cosine = matrix_cosine_similarity(
        a,
        b,
        epsilon,
    )

    if not math.isfinite(
        cosine
    ):
        return float(
            "nan"
        )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def column_angle_degrees(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> float:
    denominator = (
        np.linalg.norm(
            a
        )
        * np.linalg.norm(
            b
        )
    )

    if denominator <= epsilon:
        return float(
            "nan"
        )

    cosine = float(
        np.dot(
            a,
            b,
        )
        / denominator
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def compare_mapping_pair(
    name_a: str,
    coefficient_a: np.ndarray,
    intercept_a: np.ndarray,
    name_b: str,
    coefficient_b: np.ndarray,
    intercept_b: np.ndarray,
    epsilon: float,
) -> Dict[
    str,
    Any,
]:
    difference = (
        coefficient_b
        - coefficient_a
    )

    result: Dict[
        str,
        Any,
    ] = {
        "mapping_a": name_a,
        "mapping_b": name_b,

        "frobenius_distance": float(
            np.linalg.norm(
                difference,
                ord="fro",
            )
        ),

        "relative_frobenius_distance": float(
            np.linalg.norm(
                difference,
                ord="fro",
            )
            / (
                np.linalg.norm(
                    coefficient_a,
                    ord="fro",
                )
                + epsilon
            )
        ),

        "matrix_cosine_similarity": (
            matrix_cosine_similarity(
                coefficient_a,
                coefficient_b,
                epsilon,
            )
        ),

        "matrix_angle_deg": (
            matrix_angle_degrees(
                coefficient_a,
                coefficient_b,
                epsilon,
            )
        ),

        "intercept_distance": float(
            np.linalg.norm(
                intercept_b
                - intercept_a
            )
        ),
    }

    for axis_index, axis in enumerate(
        AXES
    ):
        result[
            f"{axis}_coefficient_angle_deg"
        ] = column_angle_degrees(
            coefficient_a[
                :,
                axis_index
            ],
            coefficient_b[
                :,
                axis_index
            ],
            epsilon,
        )

        result[
            f"{axis}_coefficient_norm_ratio"
        ] = float(
            np.linalg.norm(
                coefficient_b[
                    :,
                    axis_index
                ]
            )
            / (
                np.linalg.norm(
                    coefficient_a[
                        :,
                        axis_index
                    ]
                )
                + epsilon
            )
        )

    return result


# ============================================================================
# Output
# ============================================================================


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    if not rows:
        return

    fields: List[
        str
    ] = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.floating,
    ):
        value = float(
            value
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(
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
            str(
                key
            ): json_safe(
                item
            )
            for key, item
            in value.items()
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

    return value


# ============================================================================
# Plot helpers
# ============================================================================


def plot_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(
            10,
            8,
        )
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(
            len(
                labels
            )
        )
    )

    axis.set_yticks(
        np.arange(
            len(
                labels
            )
        )
    )

    axis.set_xticklabels(
        labels
    )

    axis.set_yticklabels(
        labels
    )

    axis.set_xlabel(
        "Test sequence"
    )

    axis.set_ylabel(
        "Mapping fit sequence"
    )

    axis.set_title(
        title
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        figure
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    args.input_root = (
        args.input_root
        .expanduser()
        .resolve()
    )

    args.checkpoint = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    validate_args(
        args
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir = (
        args.output_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weight, bias = load_original_readout(
        checkpoint_path=(
            args.checkpoint
        ),
        weight_key=(
            args.weight_key
        ),
        bias_key=(
            args.bias_key
        ),
    )

    basis = build_readout_basis(
        weight
    )

    print("=" * 96)
    print(
        "DeepDCT-VO conditional rotation latent alignment audit"
    )
    print("=" * 96)
    print(
        f"Input root:             {args.input_root}"
    )
    print(
        f"Checkpoint:             {args.checkpoint}"
    )
    print(
        f"Original readout shape: {weight.shape}"
    )
    print(
        f"Readout subspace rank:  {basis.shape[1]}"
    )
    print(
        f"Local train fraction:   {args.train_fraction:.3f}"
    )
    print(
        f"Ridge alpha:            {args.ridge_alpha:g}"
    )
    print("=" * 96)

    # ------------------------------------------------------------------
    # Load exact readout-subspace coordinates.
    # ------------------------------------------------------------------

    coordinates: Dict[
        str,
        np.ndarray,
    ] = {}

    targets: Dict[
        str,
        np.ndarray,
    ] = {}

    for sequence in ALL_SEQUENCES:
        h, y = load_sequence(
            root=args.input_root,
            sequence=sequence,
            representation_file=(
                args.representation_file
            ),
            representation_key=(
                args.representation_key
            ),
            prediction_file=(
                args.prediction_file
            ),
        )

        if h.shape[1] != weight.shape[1]:
            raise ValueError(
                f"Sequence {sequence}: representation D={h.shape[1]} "
                f"but checkpoint expects D={weight.shape[1]}."
            )

        u = (
            h
            @ basis
        )

        coordinates[
            sequence
        ] = u

        targets[
            sequence
        ] = y

        print(
            f"Sequence {sequence}: "
            f"N={h.shape[0]} "
            f"D={h.shape[1]} "
            f"readout_coords={u.shape[1]}"
        )

        del h

    # ------------------------------------------------------------------
    # Fit sequence-local mappings.
    # ------------------------------------------------------------------

    models: Dict[
        str,
        Dict[
            str,
            np.ndarray,
        ],
    ] = {}

    mapping_metric_rows: List[
        Dict[str, Any]
    ] = []

    coefficient_rows: List[
        Dict[str, Any]
    ] = []

    singular_value_rows: List[
        Dict[str, Any]
    ] = []

    for sequence in ALL_SEQUENCES:
        x = coordinates[
            sequence
        ]

        y = targets[
            sequence
        ]

        split = int(
            round(
                x.shape[0]
                * args.train_fraction
            )
        )

        split = max(
            1,
            min(
                split,
                x.shape[0]
                - 1,
            ),
        )

        x_train = x[
            :split
        ]

        y_train = y[
            :split
        ]

        x_test = x[
            split:
        ]

        y_test = y[
            split:
        ]

        (
            coefficient,
            intercept,
            x_mean,
            x_std,
        ) = fit_mapping(
            x_train,
            y_train,
            args.ridge_alpha,
        )

        prediction = predict_mapping(
            x_test,
            coefficient,
            intercept,
            x_mean,
            x_std,
        )

        metrics = prediction_metrics(
            y_test,
            prediction,
        )

        mapping_metric_rows.append(
            {
                "sequence": sequence,
                "train_n": int(
                    x_train.shape[0]
                ),
                "test_n": int(
                    x_test.shape[0]
                ),
                **metrics,
            }
        )

        models[
            sequence
        ] = {
            "coefficient": (
                coefficient
            ),
            "intercept": (
                intercept
            ),
            "x_mean": (
                x_mean
            ),
            "x_std": (
                x_std
            ),
        }

        for latent_index in range(
            coefficient.shape[0]
        ):
            for axis_index, axis in enumerate(
                AXES
            ):
                coefficient_rows.append(
                    {
                        "sequence": sequence,
                        "latent_coordinate": (
                            latent_index
                        ),
                        "target_axis": axis,
                        "coefficient": float(
                            coefficient[
                                latent_index,
                                axis_index,
                            ]
                        ),
                    }
                )

        singular_values = np.linalg.svd(
            coefficient,
            compute_uv=False,
        )

        for index, value in enumerate(
            singular_values
        ):
            singular_value_rows.append(
                {
                    "sequence": sequence,
                    "singular_value_index": (
                        index + 1
                    ),
                    "singular_value": float(
                        value
                    ),
                }
            )

    write_csv(
        args.output_dir
        / "sequence_mapping_metrics.csv",
        mapping_metric_rows,
    )

    write_csv(
        args.output_dir
        / "sequence_mapping_coefficients.csv",
        coefficient_rows,
    )

    write_csv(
        args.output_dir
        / "sequence_mapping_singular_values.csv",
        singular_value_rows,
    )

    # ------------------------------------------------------------------
    # Pairwise mapping comparison.
    # ------------------------------------------------------------------

    mapping_pair_rows: List[
        Dict[str, Any]
    ] = []

    for i, sequence_a in enumerate(
        ALL_SEQUENCES
    ):
        for sequence_b in ALL_SEQUENCES[
            i + 1:
        ]:
            model_a = models[
                sequence_a
            ]

            model_b = models[
                sequence_b
            ]

            mapping_pair_rows.append(
                compare_mapping_pair(
                    name_a=sequence_a,
                    coefficient_a=(
                        model_a[
                            "coefficient"
                        ]
                    ),
                    intercept_a=(
                        model_a[
                            "intercept"
                        ]
                    ),
                    name_b=sequence_b,
                    coefficient_b=(
                        model_b[
                            "coefficient"
                        ]
                    ),
                    intercept_b=(
                        model_b[
                            "intercept"
                        ]
                    ),
                    epsilon=args.epsilon,
                )
            )

    write_csv(
        args.output_dir
        / "mapping_pairwise_comparison.csv",
        mapping_pair_rows,
    )

    # ------------------------------------------------------------------
    # Cross-sequence transfer matrix.
    #
    # Fit using the FIRST train_fraction of source sequence.
    # Evaluate using ALL samples from destination sequence.
    # ------------------------------------------------------------------

    cross_rows: List[
        Dict[str, Any]
    ] = []

    rmse_matrix = np.full(
        (
            len(
                ALL_SEQUENCES
            ),
            len(
                ALL_SEQUENCES
            ),
        ),
        np.nan,
        dtype=np.float64,
    )

    y_corr_matrix = np.full_like(
        rmse_matrix,
        np.nan,
    )

    for source_index, source in enumerate(
        ALL_SEQUENCES
    ):
        model = models[
            source
        ]

        for target_index, target in enumerate(
            ALL_SEQUENCES
        ):
            prediction = predict_mapping(
                coordinates[
                    target
                ],
                model[
                    "coefficient"
                ],
                model[
                    "intercept"
                ],
                model[
                    "x_mean"
                ],
                model[
                    "x_std"
                ],
            )

            metrics = prediction_metrics(
                targets[
                    target
                ],
                prediction,
            )

            row = {
                "fit_sequence": source,
                "test_sequence": target,
                "test_n": int(
                    targets[
                        target
                    ].shape[0]
                ),
                **metrics,
            }

            cross_rows.append(
                row
            )

            rmse_matrix[
                source_index,
                target_index,
            ] = metrics[
                "vector_rmse"
            ]

            y_corr_matrix[
                source_index,
                target_index,
            ] = metrics[
                "y_corr"
            ]

    write_csv(
        args.output_dir
        / "cross_sequence_transfer.csv",
        cross_rows,
    )

    # ------------------------------------------------------------------
    # Pooled 00-08 model.
    #
    # Uses first train_fraction from each sequence 00-08.
    # ------------------------------------------------------------------

    pooled_train_x: List[
        np.ndarray
    ] = []

    pooled_train_y: List[
        np.ndarray
    ] = []

    for sequence in TRAIN_SEQUENCES:
        x = coordinates[
            sequence
        ]

        y = targets[
            sequence
        ]

        split = int(
            round(
                x.shape[0]
                * args.train_fraction
            )
        )

        pooled_train_x.append(
            x[
                :split
            ]
        )

        pooled_train_y.append(
            y[
                :split
            ]
        )

    pooled_x = np.concatenate(
        pooled_train_x,
        axis=0,
    )

    pooled_y = np.concatenate(
        pooled_train_y,
        axis=0,
    )

    (
        pooled_coefficient,
        pooled_intercept,
        pooled_mean,
        pooled_std,
    ) = fit_mapping(
        pooled_x,
        pooled_y,
        args.ridge_alpha,
    )

    pooled_rows: List[
        Dict[str, Any]
    ] = []

    for sequence in ALL_SEQUENCES:
        prediction = predict_mapping(
            coordinates[
                sequence
            ],
            pooled_coefficient,
            pooled_intercept,
            pooled_mean,
            pooled_std,
        )

        metrics = prediction_metrics(
            targets[
                sequence
            ],
            prediction,
        )

        pooled_rows.append(
            {
                "fit_domain": (
                    "pooled_00_08"
                ),
                "test_sequence": (
                    sequence
                ),
                "test_n": int(
                    targets[
                        sequence
                    ].shape[0]
                ),
                **metrics,
            }
        )

    write_csv(
        args.output_dir
        / "pooled_transfer.csv",
        pooled_rows,
    )

    # ------------------------------------------------------------------
    # Compare each sequence mapping against pooled 00-08 mapping.
    # ------------------------------------------------------------------

    pooled_comparison_rows: List[
        Dict[str, Any]
    ] = []

    for sequence in ALL_SEQUENCES:
        model = models[
            sequence
        ]

        pooled_comparison_rows.append(
            compare_mapping_pair(
                name_a=(
                    "pooled_00_08"
                ),
                coefficient_a=(
                    pooled_coefficient
                ),
                intercept_a=(
                    pooled_intercept
                ),
                name_b=(
                    sequence
                ),
                coefficient_b=(
                    model[
                        "coefficient"
                    ]
                ),
                intercept_b=(
                    model[
                        "intercept"
                    ]
                ),
                epsilon=args.epsilon,
            )
        )

    write_csv(
        args.output_dir
        / "pooled_vs_sequence_mapping.csv",
        pooled_comparison_rows,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_matrix(
        matrix=rmse_matrix,
        labels=ALL_SEQUENCES,
        title=(
            "Conditional rotation mapping: "
            "cross-sequence vector RMSE"
        ),
        output_path=(
            plots_dir
            / "cross_sequence_vector_rmse.png"
        ),
    )

    plot_matrix(
        matrix=y_corr_matrix,
        labels=ALL_SEQUENCES,
        title=(
            "Conditional rotation mapping: "
            "cross-sequence Y correlation"
        ),
        output_path=(
            plots_dir
            / "cross_sequence_y_correlation.png"
        ),
    )

    # Pairwise coefficient distance matrix.
    coefficient_distance = np.zeros(
        (
            len(
                ALL_SEQUENCES
            ),
            len(
                ALL_SEQUENCES
            ),
        ),
        dtype=np.float64,
    )

    for i, sequence_a in enumerate(
        ALL_SEQUENCES
    ):
        for j, sequence_b in enumerate(
            ALL_SEQUENCES
        ):
            coefficient_distance[
                i,
                j,
            ] = np.linalg.norm(
                models[
                    sequence_a
                ][
                    "coefficient"
                ]
                - models[
                    sequence_b
                ][
                    "coefficient"
                ],
                ord="fro",
            )

    plot_matrix(
        matrix=coefficient_distance,
        labels=ALL_SEQUENCES,
        title=(
            "Local rotation mapping coefficient "
            "Frobenius distance"
        ),
        output_path=(
            plots_dir
            / "coefficient_frobenius_distance.png"
        ),
    )

    # ------------------------------------------------------------------
    # Console summaries
    # ------------------------------------------------------------------

    print()
    print("=" * 122)
    print(
        "Sequence-local conditional mapping performance"
    )
    print("=" * 122)

    print(
        f"{'Seq':<6}"
        f"{'Train N':>10}"
        f"{'Test N':>10}"
        f"{'RMSE':>14}"
        f"{'x corr':>12}"
        f"{'y corr':>12}"
        f"{'z corr':>12}"
        f"{'y std':>12}"
    )

    print("-" * 122)

    for row in mapping_metric_rows:
        print(
            f"{row['sequence']:<6}"
            f"{row['train_n']:>10d}"
            f"{row['test_n']:>10d}"
            f"{row['vector_rmse']:>14.8f}"
            f"{row['x_corr']:>12.4f}"
            f"{row['y_corr']:>12.4f}"
            f"{row['z_corr']:>12.4f}"
            f"{row['y_std_ratio']:>12.4f}"
        )

    print("=" * 122)

    print()
    print("=" * 122)
    print(
        "Pooled 00-08 conditional mapping transfer"
    )
    print("=" * 122)

    print(
        f"{'Test':<8}"
        f"{'RMSE':>14}"
        f"{'x corr':>12}"
        f"{'y corr':>12}"
        f"{'z corr':>12}"
        f"{'y std':>12}"
    )

    print("-" * 122)

    for row in pooled_rows:
        print(
            f"{row['test_sequence']:<8}"
            f"{row['vector_rmse']:>14.8f}"
            f"{row['x_corr']:>12.4f}"
            f"{row['y_corr']:>12.4f}"
            f"{row['z_corr']:>12.4f}"
            f"{row['y_std_ratio']:>12.4f}"
        )

    print("=" * 122)

    # ------------------------------------------------------------------
    # Focused mappings for 00, 09, 10 relative to pooled model.
    # ------------------------------------------------------------------

    focused = {
        row[
            "mapping_b"
        ]: row
        for row in pooled_comparison_rows
        if row[
            "mapping_b"
        ] in {
            "00",
            "09",
            "10",
        }
    }

    print()
    print("=" * 122)
    print(
        "Mapping difference relative to pooled 00-08 mapping"
    )
    print("=" * 122)

    print(
        f"{'Seq':<8}"
        f"{'Frob dist':>14}"
        f"{'Relative':>14}"
        f"{'Matrix angle':>16}"
        f"{'x angle':>14}"
        f"{'y angle':>14}"
        f"{'z angle':>14}"
    )

    print("-" * 122)

    for sequence in (
        "00",
        "09",
        "10",
    ):
        row = focused[
            sequence
        ]

        print(
            f"{sequence:<8}"
            f"{row['frobenius_distance']:>14.6f}"
            f"{row['relative_frobenius_distance']:>14.6f}"
            f"{row['matrix_angle_deg']:>16.3f}"
            f"{row['x_coefficient_angle_deg']:>14.3f}"
            f"{row['y_coefficient_angle_deg']:>14.3f}"
            f"{row['z_coefficient_angle_deg']:>14.3f}"
        )

    print("=" * 122)

    # ------------------------------------------------------------------
    # JSON summary.
    # ------------------------------------------------------------------

    summary = {
        "input_root": str(
            args.input_root
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "readout_weight_shape": list(
            weight.shape
        ),
        "readout_subspace_rank": int(
            basis.shape[1]
        ),
        "train_fraction": float(
            args.train_fraction
        ),
        "ridge_alpha": float(
            args.ridge_alpha
        ),
        "sequence_local_metrics": (
            mapping_metric_rows
        ),
        "pooled_transfer": (
            pooled_rows
        ),
        "pooled_vs_sequence_mapping": (
            pooled_comparison_rows
        ),
        "mapping_pairwise_comparison": (
            mapping_pair_rows
        ),
    }

    with (
        args.output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(
                summary
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        f"Outputs saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()