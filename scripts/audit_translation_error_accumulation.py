#!/usr/bin/env python3
"""
Audit accumulation of DeepDCT-VO directional-translation prediction error.

Purpose
-------

The DCT round-trip audit has already established that:

    KITTI poses
        <-> DCT directional translation + relative rotation

are geometrically consistent.

This script therefore analyzes the remaining A6 question:

    Why does

        GT rotation + predicted directional translation

    still accumulate substantial trajectory error?

The audit measures:

1. Per-axis GT/prediction statistics.
2. Per-axis MAE, RMSE, signed bias, standard deviation, and correlation.
3. Cumulative signed error in raw DCT coordinates.
4. Error transformed into world coordinates using GT rotation.
5. Cumulative world-frame translation error.
6. Forward-motion scale/bias behavior.
7. Low / medium / high forward-motion regimes.
8. Straight-motion vs turning-motion behavior.
9. Trajectory error checkpoints at 10%, 25%, 50%, 75%, and 100%.
10. Whether accumulation is primarily caused by persistent bias or
    zero-mean residual noise.

No model checkpoint is required.

Expected CSV columns
--------------------

rotation_gt_x
rotation_gt_y
rotation_gt_z

translation_gt_x
translation_gt_y
translation_gt_z

translation_pred_x
translation_pred_y
translation_pred_z

The script assumes the project convention:

    DCT rows = [tx, ty, tz, rx, ry, rz]

and the same inverse-DCT relation already validated by
audit_dct_roundtrip_geometry.py:

    R_rel  = R_i.T @ R_j
    R_half = sqrt(R_rel)

    t_c = R_j.T @ R_half @ delta_t_world

therefore:

    delta_t_world = R_half.T @ R_j @ t_c

Examples
--------

Sequence 10:

    python scripts/audit_translation_error_accumulation.py \
        --sequence 10 \
        --data-root data \
        --frame-predictions \
          experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_10/frame_predictions.csv

Sequence 09:

    python scripts/audit_translation_error_accumulation.py \
        --sequence 09 \
        --data-root data \
        --frame-predictions \
          experiments/track_a/paper_reproduction/\
a6_unseen_00_08_to_09_10/evaluation_sequence_09/frame_predictions.csv

Optional paper scaling:

    --translation-scale-factor 1.007

Outputs
-------

<output-dir>/
    summary.json
    axis_metrics.csv
    motion_regime_metrics.csv
    turn_regime_metrics.csv
    trajectory_checkpoints.csv
    frame_error_accumulation.csv

    cumulative_dct_error.png
    cumulative_world_error.png
    forward_translation_gt_vs_pred.png
    world_position_error.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit DeepDCT-VO A6 translation error accumulation "
            "under ground-truth rotation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="KITTI sequence, normally 09 or 10.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root containing poses/<sequence>.txt.",
    )

    parser.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help=(
            "Explicit KITTI absolute-pose file. "
            "Defaults to data-root/poses/<sequence>.txt."
        ),
    )

    parser.add_argument(
        "--frame-predictions",
        type=Path,
        required=True,
        help="A6 frame_predictions.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "experiments/track_a/paper_reproduction/"
            "a6_translation_error_audit/sequence_<sequence>."
        ),
    )

    parser.add_argument(
        "--translation-scale-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplicative scale applied to predicted directional "
            "translation. Keep 1.0 for the primary audit."
        ),
    )

    parser.add_argument(
        "--euler-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Euler order used by DCT labels.",
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help="Interpret rotation_gt as degrees.",
    )

    parser.add_argument(
        "--low-motion-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Lower GT forward-motion quantile.",
    )

    parser.add_argument(
        "--high-motion-quantile",
        type=float,
        default=2.0 / 3.0,
        help="Upper GT forward-motion quantile.",
    )

    parser.add_argument(
        "--turn-quantile",
        type=float,
        default=0.75,
        help=(
            "GT relative-rotation magnitude quantile above which a "
            "transition is classified as turning."
        ),
    )

    args = parser.parse_args()

    sequence = str(args.sequence).strip()

    if sequence.isdigit():
        sequence = f"{int(sequence):02d}"

    args.sequence = sequence

    if args.pose_file is None:
        args.pose_file = (
            args.data_root
            / "poses"
            / f"{sequence}.txt"
        )

    if args.output_dir is None:
        args.output_dir = (
            Path(
                "experiments/track_a/paper_reproduction/"
                "a6_translation_error_audit"
            )
            / f"sequence_{sequence}"
        )

    if not math.isfinite(
        args.translation_scale_factor
    ):
        raise ValueError(
            "--translation-scale-factor must be finite."
        )

    if args.translation_scale_factor <= 0.0:
        raise ValueError(
            "--translation-scale-factor must be positive."
        )

    if not (
        0.0
        < args.low_motion_quantile
        < args.high_motion_quantile
        < 1.0
    ):
        raise ValueError(
            "Motion quantiles must satisfy "
            "0 < low < high < 1."
        )

    if not (
        0.0
        < args.turn_quantile
        < 1.0
    ):
        raise ValueError(
            "--turn-quantile must lie in (0,1)."
        )

    return args


# ============================================================================
# SO(3)
# ============================================================================


def project_rotation_to_so3(
    rotation: np.ndarray,
) -> np.ndarray:
    u, _, vt = np.linalg.svd(
        np.asarray(
            rotation,
            dtype=np.float64,
        )
    )

    projected = u @ vt

    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt

    return projected


def rotation_matrix_x(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def euler_to_rotation_matrix(
    euler: Sequence[float],
    order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    x, y, z = [
        float(value)
        for value in euler
    ]

    if angles_in_degrees:
        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

    rx = rotation_matrix_x(x)
    ry = rotation_matrix_y(y)
    rz = rotation_matrix_z(z)

    if order == "xyz":
        return rz @ ry @ rx

    if order == "zyx":
        return rx @ ry @ rz

    raise ValueError(
        f"Unsupported Euler order: {order}"
    )


def rotation_square_root(
    rotation: np.ndarray,
) -> np.ndarray:
    rotation = project_rotation_to_so3(
        rotation
    )

    cosine = float(
        np.clip(
            (
                np.trace(rotation)
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    angle = math.acos(cosine)

    if angle < 1.0e-12:
        return np.eye(
            3,
            dtype=np.float64,
        )

    axis = np.asarray(
        [
            rotation[2, 1]
            - rotation[1, 2],
            rotation[0, 2]
            - rotation[2, 0],
            rotation[1, 0]
            - rotation[0, 1],
        ],
        dtype=np.float64,
    )

    denominator = (
        2.0
        * math.sin(angle)
    )

    if abs(denominator) < 1.0e-10:
        # No KITTI consecutive-frame rotation should be near pi,
        # but preserve a safe fallback.
        eigvals, eigvecs = np.linalg.eig(
            rotation
        )

        index = int(
            np.argmin(
                np.abs(
                    eigvals - 1.0
                )
            )
        )

        axis = np.real(
            eigvecs[:, index]
        )

    else:
        axis /= denominator

    axis_norm = float(
        np.linalg.norm(axis)
    )

    if axis_norm < 1.0e-12:
        raise ValueError(
            "Could not recover rotation axis."
        )

    axis /= axis_norm

    half_angle = (
        0.5 * angle
    )

    skew = np.asarray(
        [
            [
                0.0,
                -axis[2],
                axis[1],
            ],
            [
                axis[2],
                0.0,
                -axis[0],
            ],
            [
                -axis[1],
                axis[0],
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    half_rotation = (
        np.eye(
            3,
            dtype=np.float64,
        )
        + math.sin(
            half_angle
        )
        * skew
        + (
            1.0
            - math.cos(
                half_angle
            )
        )
        * (
            skew @ skew
        )
    )

    return project_rotation_to_so3(
        half_rotation
    )


# ============================================================================
# Loading
# ============================================================================


def load_kitti_poses(
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Pose file not found: {path}"
        )

    raw = np.loadtxt(
        path,
        dtype=np.float64,
    )

    if raw.ndim == 1:
        raw = raw.reshape(
            1,
            -1,
        )

    if (
        raw.ndim != 2
        or raw.shape[1] != 12
    ):
        raise ValueError(
            "KITTI pose file must contain "
            "12 values per row."
        )

    poses = np.repeat(
        np.eye(
            4,
            dtype=np.float64,
        )[None, :, :],
        raw.shape[0],
        axis=0,
    )

    poses[:, :3, :4] = (
        raw.reshape(
            -1,
            3,
            4,
        )
    )

    for index in range(
        poses.shape[0]
    ):
        poses[
            index,
            :3,
            :3,
        ] = (
            project_rotation_to_so3(
                poses[
                    index,
                    :3,
                    :3,
                ]
            )
        )

    # Normalize to pose zero.
    first_inverse = (
        np.linalg.inv(
            poses[0]
        )
    )

    for index in range(
        poses.shape[0]
    ):
        poses[index] = (
            first_inverse
            @ poses[index]
        )

    return poses


def load_frame_predictions(
    path: Path,
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    required = [
        "rotation_gt_x",
        "rotation_gt_y",
        "rotation_gt_z",
        "translation_gt_x",
        "translation_gt_y",
        "translation_gt_z",
        "translation_pred_x",
        "translation_pred_y",
        "translation_pred_z",
    ]

    columns = {
        key: []
        for key in required
    }

    frame_prev = []
    frame_curr = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "CSV has no header."
            )

        missing = [
            key
            for key in required
            if key
            not in reader.fieldnames
        ]

        if missing:
            raise KeyError(
                "Missing prediction columns: "
                f"{missing}"
            )

        for row_index, row in enumerate(
            reader
        ):
            for key in required:
                columns[key].append(
                    float(
                        row[key]
                    )
                )

            frame_prev.append(
                int(
                    row.get(
                        "frame_prev",
                        row_index,
                    )
                )
            )

            frame_curr.append(
                int(
                    row.get(
                        "frame_curr",
                        row_index + 1,
                    )
                )
            )

    def stack(
        prefix: str,
    ) -> np.ndarray:
        return np.column_stack(
            [
                columns[
                    f"{prefix}_x"
                ],
                columns[
                    f"{prefix}_y"
                ],
                columns[
                    f"{prefix}_z"
                ],
            ]
        ).astype(
            np.float64,
            copy=False,
        )

    return {
        "rotation_gt": stack(
            "rotation_gt"
        ),
        "translation_gt": stack(
            "translation_gt"
        ),
        "translation_pred": stack(
            "translation_pred"
        ),
        "frame_prev": np.asarray(
            frame_prev,
            dtype=np.int64,
        ),
        "frame_curr": np.asarray(
            frame_curr,
            dtype=np.int64,
        ),
    }


# ============================================================================
# DCT -> world translation
# ============================================================================


def directional_to_world_deltas(
    rotations_gt: np.ndarray,
    translations: np.ndarray,
    euler_order: str,
    angles_in_degrees: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Decode directional translation using GT rotation.

    Returns
    -------
    world_delta:
        [N,3] world-frame displacement increments.

    accumulated_rotation:
        [N+1,3,3] GT rotation trajectory reconstructed from
        the supplied relative Euler rotations.
    """

    rotations_gt = np.asarray(
        rotations_gt,
        dtype=np.float64,
    )

    translations = np.asarray(
        translations,
        dtype=np.float64,
    )

    if (
        rotations_gt.shape
        != translations.shape
    ):
        raise ValueError(
            "rotation_gt and translation "
            "arrays must match."
        )

    count = (
        rotations_gt.shape[0]
    )

    world_delta = np.zeros(
        (count, 3),
        dtype=np.float64,
    )

    accumulated_rotation = np.zeros(
        (
            count + 1,
            3,
            3,
        ),
        dtype=np.float64,
    )

    accumulated_rotation[0] = (
        np.eye(
            3,
            dtype=np.float64,
        )
    )

    for index in range(
        count
    ):
        relative_rotation = (
            euler_to_rotation_matrix(
                rotations_gt[index],
                order=euler_order,
                angles_in_degrees=(
                    angles_in_degrees
                ),
            )
        )

        relative_rotation = (
            project_rotation_to_so3(
                relative_rotation
            )
        )

        current_rotation = (
            accumulated_rotation[
                index
            ]
        )

        next_rotation = (
            project_rotation_to_so3(
                current_rotation
                @ relative_rotation
            )
        )

        half_rotation = (
            rotation_square_root(
                relative_rotation
            )
        )

        world_delta[index] = (
            half_rotation.T
            @ next_rotation
            @ translations[index]
        )

        accumulated_rotation[
            index + 1
        ] = next_rotation

    return (
        world_delta,
        accumulated_rotation,
    )


# ============================================================================
# Statistics
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
        np.std(a) < 1.0e-12
        or np.std(b) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def axis_statistics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    axis_names = (
        "x",
        "y",
        "z",
    )

    output = {}

    for axis_index, axis_name in enumerate(
        axis_names
    ):
        target = (
            gt[:, axis_index]
        )

        estimate = (
            pred[:, axis_index]
        )

        error = (
            estimate - target
        )

        output[axis_name] = {
            "count": int(
                target.shape[0]
            ),
            "gt_mean": float(
                np.mean(target)
            ),
            "gt_std": float(
                np.std(target)
            ),
            "pred_mean": float(
                np.mean(estimate)
            ),
            "pred_std": float(
                np.std(estimate)
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
            "error_std": float(
                np.std(error)
            ),
            "correlation": (
                safe_correlation(
                    target,
                    estimate,
                )
            ),
            "cumulative_signed_error": float(
                np.sum(error)
            ),
        }

    return output


def aggregate_subset_metrics(
    mask: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    count = int(
        np.count_nonzero(mask)
    )

    if count == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias_x": float("nan"),
            "bias_y": float("nan"),
            "bias_z": float("nan"),
            "forward_gt_mean": float("nan"),
            "forward_pred_mean": float("nan"),
            "forward_bias": float("nan"),
            "forward_correlation": float("nan"),
        }

    subset_gt = gt[mask]
    subset_pred = pred[mask]

    error = (
        subset_pred
        - subset_gt
    )

    return {
        "count": count,
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
        "bias_x": float(
            np.mean(
                error[:, 0]
            )
        ),
        "bias_y": float(
            np.mean(
                error[:, 1]
            )
        ),
        "bias_z": float(
            np.mean(
                error[:, 2]
            )
        ),
        "forward_gt_mean": float(
            np.mean(
                subset_gt[:, 2]
            )
        ),
        "forward_pred_mean": float(
            np.mean(
                subset_pred[:, 2]
            )
        ),
        "forward_bias": float(
            np.mean(
                error[:, 2]
            )
        ),
        "forward_correlation": (
            safe_correlation(
                subset_gt[:, 2],
                subset_pred[:, 2],
            )
        ),
    }


# ============================================================================
# Accumulation analysis
# ============================================================================


def cumulative_positions(
    delta: np.ndarray,
) -> np.ndarray:
    positions = np.zeros(
        (
            delta.shape[0] + 1,
            3,
        ),
        dtype=np.float64,
    )

    positions[1:] = np.cumsum(
        delta,
        axis=0,
    )

    return positions


def compute_checkpoints(
    gt_world_delta: np.ndarray,
    pred_world_delta: np.ndarray,
) -> list:
    gt_position = cumulative_positions(
        gt_world_delta
    )

    pred_position = cumulative_positions(
        pred_world_delta
    )

    fractions = (
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
    )

    transition_count = (
        gt_world_delta.shape[0]
    )

    rows = []

    for fraction in fractions:
        transition_index = int(
            round(
                transition_count
                * fraction
            )
        )

        transition_index = min(
            max(
                transition_index,
                1,
            ),
            transition_count,
        )

        gt_p = gt_position[
            transition_index
        ]

        pred_p = pred_position[
            transition_index
        ]

        error = (
            pred_p - gt_p
        )

        rows.append(
            {
                "fraction": fraction,
                "transition": (
                    transition_index
                ),
                "gt_x": float(
                    gt_p[0]
                ),
                "gt_y": float(
                    gt_p[1]
                ),
                "gt_z": float(
                    gt_p[2]
                ),
                "pred_x": float(
                    pred_p[0]
                ),
                "pred_y": float(
                    pred_p[1]
                ),
                "pred_z": float(
                    pred_p[2]
                ),
                "error_x": float(
                    error[0]
                ),
                "error_y": float(
                    error[1]
                ),
                "error_z": float(
                    error[2]
                ),
                "error_norm": float(
                    np.linalg.norm(
                        error
                    )
                ),
            }
        )

    return rows


def bias_noise_decomposition(
    error: np.ndarray,
) -> Dict[str, object]:
    """
    Decompose errors into:

        error_k = mean_error + zero_mean_residual_k

    The cumulative bias component grows deterministically with N.

    This is not a probabilistic uncertainty decomposition; it is a
    diagnostic of how much of final raw-coordinate accumulation can
    be associated with persistent signed bias.
    """

    count = error.shape[0]

    mean_error = np.mean(
        error,
        axis=0,
    )

    zero_mean_residual = (
        error - mean_error
    )

    cumulative_bias = (
        count * mean_error
    )

    cumulative_residual = np.sum(
        zero_mean_residual,
        axis=0,
    )

    total_cumulative = np.sum(
        error,
        axis=0,
    )

    return {
        "mean_error": (
            mean_error.tolist()
        ),
        "cumulative_bias_component": (
            cumulative_bias.tolist()
        ),
        "cumulative_zero_mean_residual": (
            cumulative_residual.tolist()
        ),
        "total_cumulative_error": (
            total_cumulative.tolist()
        ),
        "cumulative_bias_norm": float(
            np.linalg.norm(
                cumulative_bias
            )
        ),
        "total_cumulative_error_norm": float(
            np.linalg.norm(
                total_cumulative
            )
        ),
    }


# ============================================================================
# CSV output
# ============================================================================


def write_dict_rows(
    path: Path,
    rows: list,
) -> None:
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
        writer.writerows(
            rows
        )


def save_axis_metrics(
    path: Path,
    metrics: Mapping[
        str,
        Mapping[
            str,
            float,
        ],
    ],
) -> None:
    rows = []

    for axis, values in (
        metrics.items()
    ):
        row = {
            "axis": axis,
        }

        row.update(
            values
        )

        rows.append(
            row
        )

    write_dict_rows(
        path,
        rows,
    )


# ============================================================================
# Plots
# ============================================================================


def plot_cumulative_dct_error(
    path: Path,
    cumulative_error: np.ndarray,
    sequence: str,
) -> None:
    plt.figure(
        figsize=(11, 6)
    )

    frames = np.arange(
        1,
        cumulative_error.shape[0] + 1,
    )

    plt.plot(
        frames,
        cumulative_error[:, 0],
        label="x",
    )

    plt.plot(
        frames,
        cumulative_error[:, 1],
        label="y",
    )

    plt.plot(
        frames,
        cumulative_error[:, 2],
        label="z",
    )

    plt.xlabel(
        "Transition"
    )

    plt.ylabel(
        "Cumulative signed error [m]"
    )

    plt.title(
        f"Sequence {sequence}: "
        "Cumulative DCT-coordinate translation error"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_cumulative_world_error(
    path: Path,
    cumulative_error: np.ndarray,
    sequence: str,
) -> None:
    plt.figure(
        figsize=(11, 6)
    )

    frames = np.arange(
        1,
        cumulative_error.shape[0] + 1,
    )

    plt.plot(
        frames,
        cumulative_error[:, 0],
        label="world x",
    )

    plt.plot(
        frames,
        cumulative_error[:, 1],
        label="world y",
    )

    plt.plot(
        frames,
        cumulative_error[:, 2],
        label="world z",
    )

    plt.xlabel(
        "Transition"
    )

    plt.ylabel(
        "Cumulative world-frame error [m]"
    )

    plt.title(
        f"Sequence {sequence}: "
        "Cumulative GT-R world-frame translation error"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_forward_translation(
    path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    sequence: str,
) -> None:
    plt.figure(
        figsize=(11, 6)
    )

    frames = np.arange(
        gt.shape[0]
    )

    plt.plot(
        frames,
        gt[:, 2],
        label="GT t_z",
    )

    plt.plot(
        frames,
        pred[:, 2],
        label="Pred t_z",
    )

    plt.xlabel(
        "Transition"
    )

    plt.ylabel(
        "Directional forward translation [m]"
    )

    plt.title(
        f"Sequence {sequence}: "
        "Forward directional translation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_world_position_error(
    path: Path,
    error_norm: np.ndarray,
    sequence: str,
) -> None:
    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        np.arange(
            error_norm.shape[0]
        ),
        error_norm,
    )

    plt.xlabel(
        "Pose index"
    )

    plt.ylabel(
        "GT-R trajectory position error [m]"
    )

    plt.title(
        f"Sequence {sequence}: "
        "Accumulated translation trajectory error"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


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
        "=" * 96
    )
    print(
        "DeepDCT-VO A6 translation-error accumulation audit"
    )
    print(
        "=" * 96
    )
    print(
        f"Sequence:                    {args.sequence}"
    )
    print(
        f"Frame predictions:           {args.frame_predictions}"
    )
    print(
        f"KITTI poses:                 {args.pose_file}"
    )
    print(
        "Translation scale:            "
        f"{args.translation_scale_factor:.9f}"
    )
    print(
        f"Euler order:                 {args.euler_order}"
    )
    print(
        f"Output directory:            {args.output_dir}"
    )
    print(
        "=" * 96
    )

    predictions = (
        load_frame_predictions(
            args.frame_predictions
        )
    )

    poses = load_kitti_poses(
        args.pose_file
    )

    rotation_gt = (
        predictions[
            "rotation_gt"
        ]
    )

    translation_gt = (
        predictions[
            "translation_gt"
        ]
    )

    translation_pred = (
        predictions[
            "translation_pred"
        ]
        * args.translation_scale_factor
    )

    count = (
        translation_gt.shape[0]
    )

    if poses.shape[0] != count + 1:
        raise ValueError(
            "Pose / prediction count mismatch: "
            f"{poses.shape[0]} poses versus "
            f"{count} transitions."
        )

    translation_error = (
        translation_pred
        - translation_gt
    )

    # ========================================================================
    # 1. Per-axis raw DCT statistics
    # ========================================================================

    axis_metrics = (
        axis_statistics(
            translation_gt,
            translation_pred,
        )
    )

    print()
    print(
        "=" * 96
    )
    print(
        "PER-AXIS DIRECTIONAL TRANSLATION STATISTICS"
    )
    print(
        "=" * 96
    )

    print(
        f"{'Axis':<6}"
        f"{'GT mean':>12}"
        f"{'Pred mean':>12}"
        f"{'Bias':>12}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'Corr':>12}"
        f"{'CumErr':>14}"
    )

    print(
        "-" * 96
    )

    for axis in (
        "x",
        "y",
        "z",
    ):
        values = (
            axis_metrics[
                axis
            ]
        )

        print(
            f"{axis:<6}"
            f"{values['gt_mean']:>12.6f}"
            f"{values['pred_mean']:>12.6f}"
            f"{values['bias']:>12.6f}"
            f"{values['mae']:>12.6f}"
            f"{values['rmse']:>12.6f}"
            f"{values['correlation']:>12.4f}"
            f"{values['cumulative_signed_error']:>14.3f}"
        )

    # ========================================================================
    # 2. Forward-motion scale
    # ========================================================================

    gt_z = (
        translation_gt[:, 2]
    )

    pred_z = (
        translation_pred[:, 2]
    )

    valid_ratio = (
        np.abs(gt_z)
        > 1.0e-4
    )

    forward_ratios = (
        pred_z[valid_ratio]
        / gt_z[valid_ratio]
    )

    forward_summary = {
        "valid_ratio_count": int(
            np.count_nonzero(
                valid_ratio
            )
        ),
        "mean_gt_z": float(
            np.mean(gt_z)
        ),
        "mean_pred_z": float(
            np.mean(pred_z)
        ),
        "mean_forward_bias": float(
            np.mean(
                pred_z - gt_z
            )
        ),
        "median_pred_over_gt": float(
            np.median(
                forward_ratios
            )
        ),
        "mean_pred_over_gt": float(
            np.mean(
                forward_ratios
            )
        ),
        "forward_correlation": (
            safe_correlation(
                gt_z,
                pred_z,
            )
        ),
    }

    print()
    print(
        "=" * 96
    )
    print(
        "FORWARD-MOTION DIAGNOSTIC"
    )
    print(
        "=" * 96
    )

    for key, value in (
        forward_summary.items()
    ):
        print(
            f"{key:<34s}: {value}"
        )

    # ========================================================================
    # 3. Decode GT and predicted directional translations using GT rotation.
    # ========================================================================

    (
        gt_world_delta,
        accumulated_gt_rotation,
    ) = directional_to_world_deltas(
        rotation_gt,
        translation_gt,
        euler_order=args.euler_order,
        angles_in_degrees=(
            args.angles_in_degrees
        ),
    )

    (
        pred_world_delta,
        _,
    ) = directional_to_world_deltas(
        rotation_gt,
        translation_pred,
        euler_order=args.euler_order,
        angles_in_degrees=(
            args.angles_in_degrees
        ),
    )

    world_error = (
        pred_world_delta
        - gt_world_delta
    )

    world_axis_metrics = (
        axis_statistics(
            gt_world_delta,
            pred_world_delta,
        )
    )

    gt_world_position = (
        cumulative_positions(
            gt_world_delta
        )
    )

    pred_world_position = (
        cumulative_positions(
            pred_world_delta
        )
    )

    world_position_error = (
        pred_world_position
        - gt_world_position
    )

    world_position_error_norm = (
        np.linalg.norm(
            world_position_error,
            axis=1,
        )
    )

    final_world_error = (
        world_position_error[-1]
    )

    trajectory_summary = {
        "endpoint_error_x": float(
            final_world_error[0]
        ),
        "endpoint_error_y": float(
            final_world_error[1]
        ),
        "endpoint_error_z": float(
            final_world_error[2]
        ),
        "endpoint_error_norm": float(
            np.linalg.norm(
                final_world_error
            )
        ),
        "trajectory_position_rmse": float(
            np.sqrt(
                np.mean(
                    world_position_error_norm
                    ** 2
                )
            )
        ),
        "trajectory_position_mean": float(
            np.mean(
                world_position_error_norm
            )
        ),
        "trajectory_position_max": float(
            np.max(
                world_position_error_norm
            )
        ),
    }

    print()
    print(
        "=" * 96
    )
    print(
        "GT-R WORLD-FRAME ACCUMULATION"
    )
    print(
        "=" * 96
    )

    for key, value in (
        trajectory_summary.items()
    ):
        print(
            f"{key:<34s}: {value:.9f}"
        )

    # ========================================================================
    # 4. Bias vs residual diagnostic
    # ========================================================================

    dct_bias_noise = (
        bias_noise_decomposition(
            translation_error
        )
    )

    world_bias_noise = (
        bias_noise_decomposition(
            world_error
        )
    )

    print()
    print(
        "=" * 96
    )
    print(
        "BIAS / ZERO-MEAN RESIDUAL DECOMPOSITION"
    )
    print(
        "=" * 96
    )

    print(
        "DCT mean error [x,y,z]:       "
        f"{dct_bias_noise['mean_error']}"
    )

    print(
        "DCT N*mean(error):            "
        f"{dct_bias_noise['cumulative_bias_component']}"
    )

    print(
        "World mean error [x,y,z]:     "
        f"{world_bias_noise['mean_error']}"
    )

    print(
        "World N*mean(error):          "
        f"{world_bias_noise['cumulative_bias_component']}"
    )

    # ========================================================================
    # 5. Motion regimes based on GT forward motion.
    # ========================================================================

    low_threshold = float(
        np.quantile(
            gt_z,
            args.low_motion_quantile,
        )
    )

    high_threshold = float(
        np.quantile(
            gt_z,
            args.high_motion_quantile,
        )
    )

    motion_masks = {
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

    motion_rows = []

    print()
    print(
        "=" * 96
    )
    print(
        "FORWARD-MOTION REGIMES"
    )
    print(
        "=" * 96
    )

    print(
        "Thresholds: "
        f"low <= {low_threshold:.6f}, "
        f"medium <= {high_threshold:.6f}, "
        "high > upper threshold"
    )

    for name, mask in (
        motion_masks.items()
    ):
        metrics = (
            aggregate_subset_metrics(
                mask,
                translation_gt,
                translation_pred,
            )
        )

        row = {
            "regime": name,
            **metrics,
        }

        motion_rows.append(
            row
        )

        print(
            f"{name:<8s} "
            f"n={metrics['count']:4d} "
            f"rmse={metrics['rmse']:.6f} "
            f"bias_xyz=("
            f"{metrics['bias_x']:+.6f}, "
            f"{metrics['bias_y']:+.6f}, "
            f"{metrics['bias_z']:+.6f}) "
            f"z_corr={metrics['forward_correlation']:.4f}"
        )

    # ========================================================================
    # 6. Turning versus straight.
    # ========================================================================

    rotation_magnitude = (
        np.linalg.norm(
            rotation_gt,
            axis=1,
        )
    )

    turn_threshold = float(
        np.quantile(
            rotation_magnitude,
            args.turn_quantile,
        )
    )

    turn_masks = {
        "straight_or_mild": (
            rotation_magnitude
            <= turn_threshold
        ),
        "turning": (
            rotation_magnitude
            > turn_threshold
        ),
    }

    turn_rows = []

    print()
    print(
        "=" * 96
    )
    print(
        "TURNING VS STRAIGHT MOTION"
    )
    print(
        "=" * 96
    )

    print(
        "Rotation-magnitude threshold: "
        f"{turn_threshold:.9f}"
    )

    for name, mask in (
        turn_masks.items()
    ):
        metrics = (
            aggregate_subset_metrics(
                mask,
                translation_gt,
                translation_pred,
            )
        )

        row = {
            "regime": name,
            "rotation_threshold": (
                turn_threshold
            ),
            **metrics,
        }

        turn_rows.append(
            row
        )

        print(
            f"{name:<18s} "
            f"n={metrics['count']:4d} "
            f"rmse={metrics['rmse']:.6f} "
            f"bias_xyz=("
            f"{metrics['bias_x']:+.6f}, "
            f"{metrics['bias_y']:+.6f}, "
            f"{metrics['bias_z']:+.6f})"
        )

    # ========================================================================
    # 7. Checkpoints
    # ========================================================================

    checkpoint_rows = (
        compute_checkpoints(
            gt_world_delta,
            pred_world_delta,
        )
    )

    print()
    print(
        "=" * 96
    )
    print(
        "TRAJECTORY ERROR CHECKPOINTS"
    )
    print(
        "=" * 96
    )

    print(
        f"{'Progress':>10}"
        f"{'Frame':>10}"
        f"{'Err X':>12}"
        f"{'Err Y':>12}"
        f"{'Err Z':>12}"
        f"{'Norm':>12}"
    )

    print(
        "-" * 68
    )

    for row in checkpoint_rows:
        print(
            f"{100.0 * row['fraction']:>9.0f}%"
            f"{row['transition']:>10d}"
            f"{row['error_x']:>12.3f}"
            f"{row['error_y']:>12.3f}"
            f"{row['error_z']:>12.3f}"
            f"{row['error_norm']:>12.3f}"
        )

    # ========================================================================
    # 8. Per-frame accumulation table
    # ========================================================================

    cumulative_dct_error = (
        np.cumsum(
            translation_error,
            axis=0,
        )
    )

    cumulative_world_error = (
        np.cumsum(
            world_error,
            axis=0,
        )
    )

    frame_rows = []

    for index in range(
        count
    ):
        frame_rows.append(
            {
                "transition_index": (
                    index
                ),
                "frame_prev": int(
                    predictions[
                        "frame_prev"
                    ][index]
                ),
                "frame_curr": int(
                    predictions[
                        "frame_curr"
                    ][index]
                ),

                "gt_tx": float(
                    translation_gt[
                        index,
                        0,
                    ]
                ),
                "gt_ty": float(
                    translation_gt[
                        index,
                        1,
                    ]
                ),
                "gt_tz": float(
                    translation_gt[
                        index,
                        2,
                    ]
                ),

                "pred_tx": float(
                    translation_pred[
                        index,
                        0,
                    ]
                ),
                "pred_ty": float(
                    translation_pred[
                        index,
                        1,
                    ]
                ),
                "pred_tz": float(
                    translation_pred[
                        index,
                        2,
                    ]
                ),

                "error_tx": float(
                    translation_error[
                        index,
                        0,
                    ]
                ),
                "error_ty": float(
                    translation_error[
                        index,
                        1,
                    ]
                ),
                "error_tz": float(
                    translation_error[
                        index,
                        2,
                    ]
                ),

                "cum_error_tx": float(
                    cumulative_dct_error[
                        index,
                        0,
                    ]
                ),
                "cum_error_ty": float(
                    cumulative_dct_error[
                        index,
                        1,
                    ]
                ),
                "cum_error_tz": float(
                    cumulative_dct_error[
                        index,
                        2,
                    ]
                ),

                "world_error_x": float(
                    world_error[
                        index,
                        0,
                    ]
                ),
                "world_error_y": float(
                    world_error[
                        index,
                        1,
                    ]
                ),
                "world_error_z": float(
                    world_error[
                        index,
                        2,
                    ]
                ),

                "cum_world_error_x": float(
                    cumulative_world_error[
                        index,
                        0,
                    ]
                ),
                "cum_world_error_y": float(
                    cumulative_world_error[
                        index,
                        1,
                    ]
                ),
                "cum_world_error_z": float(
                    cumulative_world_error[
                        index,
                        2,
                    ]
                ),

                "trajectory_error_norm": float(
                    world_position_error_norm[
                        index + 1
                    ]
                ),

                "gt_rotation_magnitude": float(
                    rotation_magnitude[
                        index
                    ]
                ),
            }
        )

    # ========================================================================
    # Save outputs
    # ========================================================================

    save_axis_metrics(
        args.output_dir
        / "axis_metrics.csv",
        axis_metrics,
    )

    save_axis_metrics(
        args.output_dir
        / "world_axis_metrics.csv",
        world_axis_metrics,
    )

    write_dict_rows(
        args.output_dir
        / "motion_regime_metrics.csv",
        motion_rows,
    )

    write_dict_rows(
        args.output_dir
        / "turn_regime_metrics.csv",
        turn_rows,
    )

    write_dict_rows(
        args.output_dir
        / "trajectory_checkpoints.csv",
        checkpoint_rows,
    )

    write_dict_rows(
        args.output_dir
        / "frame_error_accumulation.csv",
        frame_rows,
    )

    plot_cumulative_dct_error(
        args.output_dir
        / "cumulative_dct_error.png",
        cumulative_dct_error,
        args.sequence,
    )

    plot_cumulative_world_error(
        args.output_dir
        / "cumulative_world_error.png",
        cumulative_world_error,
        args.sequence,
    )

    plot_forward_translation(
        args.output_dir
        / "forward_translation_gt_vs_pred.png",
        translation_gt,
        translation_pred,
        args.sequence,
    )

    plot_world_position_error(
        args.output_dir
        / "world_position_error.png",
        world_position_error_norm,
        args.sequence,
    )

    summary = {
        "sequence": (
            args.sequence
        ),
        "frame_predictions": str(
            args.frame_predictions
        ),
        "pose_file": str(
            args.pose_file
        ),
        "num_transitions": int(
            count
        ),
        "translation_scale_factor": float(
            args.translation_scale_factor
        ),
        "euler_order": (
            args.euler_order
        ),
        "angles_in_degrees": (
            args.angles_in_degrees
        ),

        "directional_axis_metrics": (
            axis_metrics
        ),

        "world_axis_metrics": (
            world_axis_metrics
        ),

        "forward_summary": (
            forward_summary
        ),

        "trajectory_summary": (
            trajectory_summary
        ),

        "bias_noise_decomposition": {
            "directional_dct": (
                dct_bias_noise
            ),
            "world": (
                world_bias_noise
            ),
        },

        "motion_regime_thresholds": {
            "low": (
                low_threshold
            ),
            "high": (
                high_threshold
            ),
        },

        "motion_regimes": {
            row["regime"]: row
            for row in motion_rows
        },

        "turn_threshold": (
            turn_threshold
        ),

        "turn_regimes": {
            row["regime"]: row
            for row in turn_rows
        },

        "trajectory_checkpoints": (
            checkpoint_rows
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
        "=" * 96
    )
    print(
        "AUDIT COMPLETE"
    )
    print(
        "=" * 96
    )
    print(
        "Primary quantities to inspect:"
    )
    print(
        "  1. directional z-axis bias"
    )
    print(
        "  2. cumulative signed z error"
    )
    print(
        "  3. world-frame endpoint error components"
    )
    print(
        "  4. bias across low/medium/high motion regimes"
    )
    print(
        "  5. straight vs turning error"
    )
    print()
    print(
        f"Outputs: {args.output_dir}"
    )
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()