#!/usr/bin/env python3
"""
Audit DeepDCT-VO DCT geometry independently of the neural network.

Primary question
----------------

Does

    GT KITTI pose
        -> DCT directional translation + relative rotation
        -> inverse DCT reconstruction

recover the original KITTI trajectory?

This script performs several independent checks:

G1-A
    KITTI absolute poses -> ordinary relative SE(3) -> absolute trajectory.

G1-B
    KITTI absolute poses -> generated DCT labels -> inverse DCT ->
    absolute trajectory.

G1-C
    Existing *_dct.txt labels -> inverse DCT -> absolute trajectory.

G1-D
    Compare existing DCT labels against DCT labels regenerated directly
    from the KITTI absolute poses.

Optional G2
    If --frame-predictions is supplied, reconstruct:

        GT R   + GT directional t
        GT R   + predicted directional t
        Pred R + GT directional t
        Pred R + predicted directional t

    This separates Model-R trajectory error from Model-T trajectory error.

Expected interpretation
-----------------------

The decisive G1 result is the reconstruction from EXISTING DCT labels:

    existing DCT GT t + existing DCT GT R
        -> inverse DCT
        -> KITTI trajectory

If the dataset labels and inverse-DCT implementation use exactly the same
geometry/convention, trajectory error should be very small.

Example
-------

Sequence 10:

    python scripts/audit_dct_roundtrip_geometry.py \
        --sequence 10 \
        --data-root data

With A6 frame predictions:

    python scripts/audit_dct_roundtrip_geometry.py \
        --sequence 10 \
        --data-root data \
        --frame-predictions \
          experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/\
evaluation_sequence_10/frame_predictions.csv

Outputs
-------

<output-dir>/
    summary.json
    regenerated_dct.txt
    trajectory_kitti_gt.txt
    trajectory_relative_se3_roundtrip.txt
    trajectory_regenerated_dct_roundtrip.txt
    trajectory_existing_dct_roundtrip.txt
    trajectory_g1_xz.png

When --frame-predictions is supplied:
    trajectory_gt_r_gt_t.txt
    trajectory_gt_r_pred_t.txt
    trajectory_pred_r_gt_t.txt
    trajectory_pred_r_pred_t.txt
    trajectory_g2_xz.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DeepDCT-VO DCT round-trip geometry.",
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
        help=(
            "Dataset root containing poses/<seq>.txt and "
            "out_csv/<seq>_dct.txt."
        ),
    )

    parser.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help=(
            "Explicit KITTI absolute pose file. "
            "Defaults to data-root/poses/<sequence>.txt."
        ),
    )

    parser.add_argument(
        "--dct-file",
        type=Path,
        default=None,
        help=(
            "Explicit existing DCT label file. "
            "Defaults to data-root/out_csv/<sequence>_dct.txt."
        ),
    )

    parser.add_argument(
        "--frame-predictions",
        type=Path,
        default=None,
        help=(
            "Optional A6 frame_predictions.csv. When supplied, "
            "run the four-way G2 GT-R/Pred-R decomposition."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Audit output directory. Defaults to "
            "experiments/track_a/paper_reproduction/"
            "a6_geometry_audit/sequence_<sequence>."
        ),
    )

    parser.add_argument(
        "--euler-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Euler convention matching the evaluator/DCT labels.",
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help="Interpret DCT rotation labels as degrees instead of radians.",
    )

    parser.add_argument(
        "--translation-scale-factor",
        type=float,
        default=1.0,
        help=(
            "Optional scale for predicted translations in G2 only. "
            "Leave 1.0 for the geometry audit."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if the existing-DCT GT round trip exceeds "
            "the configured translation tolerance."
        ),
    )

    parser.add_argument(
        "--translation-tolerance",
        type=float,
        default=1.0e-3,
        help=(
            "Maximum allowed endpoint error in metres for --strict. "
            "Use this only after confirming how the source DCT labels "
            "were numerically generated."
        ),
    )

    args = parser.parse_args()

    sequence = str(args.sequence).strip()
    if sequence.isdigit():
        sequence = f"{int(sequence):02d}"

    args.sequence = sequence

    if args.pose_file is None:
        args.pose_file = (
            args.data_root / "poses" / f"{sequence}.txt"
        )

    if args.dct_file is None:
        args.dct_file = (
            args.data_root / "out_csv" / f"{sequence}_dct.txt"
        )

    if args.output_dir is None:
        args.output_dir = (
            Path(
                "experiments/track_a/paper_reproduction/"
                "a6_geometry_audit"
            )
            / f"sequence_{sequence}"
        )

    if not math.isfinite(args.translation_scale_factor):
        raise ValueError(
            "--translation-scale-factor must be finite."
        )

    if args.translation_scale_factor <= 0.0:
        raise ValueError(
            "--translation-scale-factor must be positive."
        )

    if args.translation_tolerance < 0.0:
        raise ValueError(
            "--translation-tolerance cannot be negative."
        )

    return args


# ============================================================================
# Basic SO(3) utilities
# ============================================================================


def project_rotation_to_so3(
    rotation: np.ndarray,
) -> np.ndarray:
    """Project a nearly rotational matrix onto SO(3)."""

    rotation = np.asarray(rotation, dtype=np.float64)

    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt

    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt

    return projected


def rotation_matrix_x(angle: float) -> np.ndarray:
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


def rotation_matrix_y(angle: float) -> np.ndarray:
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


def rotation_matrix_z(angle: float) -> np.ndarray:
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
    order: str = "xyz",
    angles_in_degrees: bool = False,
) -> np.ndarray:
    """
    Match evaluate_deepdct_vo.py.

    xyz:
        R = Rz @ Ry @ Rx

    zyx:
        R = Rx @ Ry @ Rz
    """

    x, y, z = [float(value) for value in euler]

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
        f"Unsupported Euler order: {order!r}"
    )


def rotation_matrix_to_euler_xyz(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Inverse of:

        R = Rz(z) @ Ry(y) @ Rx(x)

    Return [x, y, z].
    """

    rotation = project_rotation_to_so3(rotation)

    sy = -float(rotation[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))

    y = math.asin(sy)

    cy = math.cos(y)

    if abs(cy) > 1.0e-9:
        x = math.atan2(
            rotation[2, 1],
            rotation[2, 2],
        )
        z = math.atan2(
            rotation[1, 0],
            rotation[0, 0],
        )
    else:
        # Gimbal-lock fallback. KITTI relative rotations are small,
        # so this branch should rarely, if ever, be used.
        x = 0.0

        if sy > 0:
            z = math.atan2(
                -rotation[0, 1],
                rotation[1, 1],
            )
        else:
            z = math.atan2(
                rotation[0, 1],
                rotation[1, 1],
            )

    return np.asarray(
        [x, y, z],
        dtype=np.float64,
    )


def rotation_matrix_to_euler_zyx(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Inverse of:

        R = Rx(x) @ Ry(y) @ Rz(z)

    Return [x, y, z].
    """

    rotation = project_rotation_to_so3(rotation)

    sy = float(rotation[0, 2])
    sy = float(np.clip(sy, -1.0, 1.0))

    y = math.asin(sy)
    cy = math.cos(y)

    if abs(cy) > 1.0e-9:
        x = math.atan2(
            -rotation[1, 2],
            rotation[2, 2],
        )
        z = math.atan2(
            -rotation[0, 1],
            rotation[0, 0],
        )
    else:
        x = math.atan2(
            rotation[2, 1],
            rotation[1, 1],
        )
        z = 0.0

    return np.asarray(
        [x, y, z],
        dtype=np.float64,
    )


def rotation_matrix_to_euler(
    rotation: np.ndarray,
    order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    if order == "xyz":
        euler = rotation_matrix_to_euler_xyz(rotation)

    elif order == "zyx":
        euler = rotation_matrix_to_euler_zyx(rotation)

    else:
        raise ValueError(
            f"Unsupported Euler order: {order!r}"
        )

    if angles_in_degrees:
        euler = np.degrees(euler)

    return euler


def rotation_square_root(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Principal SO(3) square root.

    Convert to axis-angle and divide the principal angle by two.
    """

    rotation = project_rotation_to_so3(rotation)

    cosine = float(
        np.clip(
            (np.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    angle = math.acos(cosine)

    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)

    if abs(math.pi - angle) > 1.0e-7:
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )

        axis /= 2.0 * math.sin(angle)

        norm = float(np.linalg.norm(axis))

        if norm < 1.0e-12:
            raise ValueError(
                "Could not recover SO(3) rotation axis."
            )

        axis /= norm

    else:
        # Stable recovery near pi.
        diagonal = np.diag(rotation)

        axis = np.sqrt(
            np.maximum(
                (diagonal + 1.0) / 2.0,
                0.0,
            )
        )

        largest = int(np.argmax(axis))

        if axis[largest] < 1.0e-12:
            raise ValueError(
                "Could not recover axis near pi rotation."
            )

        if largest == 0:
            axis[1] = math.copysign(
                axis[1],
                rotation[0, 1] + rotation[1, 0],
            )
            axis[2] = math.copysign(
                axis[2],
                rotation[0, 2] + rotation[2, 0],
            )

        elif largest == 1:
            axis[0] = math.copysign(
                axis[0],
                rotation[0, 1] + rotation[1, 0],
            )
            axis[2] = math.copysign(
                axis[2],
                rotation[1, 2] + rotation[2, 1],
            )

        else:
            axis[0] = math.copysign(
                axis[0],
                rotation[0, 2] + rotation[2, 0],
            )
            axis[1] = math.copysign(
                axis[1],
                rotation[1, 2] + rotation[2, 1],
            )

        axis /= np.linalg.norm(axis)

    half_angle = 0.5 * angle

    k = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )

    half_rotation = (
        np.eye(3, dtype=np.float64)
        + math.sin(half_angle) * k
        + (1.0 - math.cos(half_angle))
        * (k @ k)
    )

    return project_rotation_to_so3(
        half_rotation
    )


def rotation_angle_degrees(
    rotation: np.ndarray,
) -> float:
    cosine = float(
        np.clip(
            (np.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    return math.degrees(math.acos(cosine))


# ============================================================================
# Loading
# ============================================================================


def load_kitti_poses(
    path: Path,
) -> np.ndarray:
    """Load KITTI 12-value 3x4 absolute poses as [N,4,4]."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Pose file not found: {path}"
        )

    raw = np.loadtxt(
        path,
        dtype=np.float64,
    )

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.ndim != 2 or raw.shape[1] != 12:
        raise ValueError(
            "KITTI pose file must contain 12 values per row. "
            f"Received shape {raw.shape} from {path}."
        )

    poses = np.repeat(
        np.eye(4, dtype=np.float64)[None, ...],
        raw.shape[0],
        axis=0,
    )

    poses[:, :3, :4] = raw.reshape(
        -1,
        3,
        4,
    )

    for index in range(poses.shape[0]):
        poses[index, :3, :3] = (
            project_rotation_to_so3(
                poses[index, :3, :3]
            )
        )

    return poses


def load_dct_labels(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load existing DeepDCT labels.

    Project convention:
        tx ty tz rx ry rz

    Dataset therefore uses:
        translation = row[:3]
        rotation    = row[3:6]
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"DCT file not found: {path}"
        )

    rows = []

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
                    # Allow one textual header.
                    continue

                raise ValueError(
                    f"Non-numeric DCT row at "
                    f"{path}:{line_number}."
                ) from error

            if len(values) < 6:
                raise ValueError(
                    f"DCT row {line_number} in {path} "
                    f"contains {len(values)} values; "
                    "expected at least six."
                )

            rows.append(
                values[-6:]
            )

    if not rows:
        raise ValueError(
            f"No numeric labels found in {path}."
        )

    labels = np.asarray(
        rows,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(labels)):
        raise ValueError(
            f"DCT labels contain non-finite values: {path}"
        )

    translations = labels[:, :3]
    rotations = labels[:, 3:6]

    return translations, rotations


def load_frame_predictions(
    path: Path,
) -> Dict[str, np.ndarray]:
    """Load the columns needed for the four G2 combinations."""

    if not path.is_file():
        raise FileNotFoundError(
            f"frame_predictions.csv not found: {path}"
        )

    required = (
        "rotation_gt_x",
        "rotation_gt_y",
        "rotation_gt_z",
        "rotation_pred_x",
        "rotation_pred_y",
        "rotation_pred_z",
        "translation_gt_x",
        "translation_gt_y",
        "translation_gt_z",
        "translation_pred_x",
        "translation_pred_y",
        "translation_pred_z",
    )

    columns = {
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
                f"No CSV header found in {path}."
            )

        missing = [
            name
            for name in required
            if name not in reader.fieldnames
        ]

        if missing:
            raise KeyError(
                "frame_predictions.csv is missing "
                f"required columns: {missing}"
            )

        for row in reader:
            for name in required:
                columns[name].append(
                    float(row[name])
                )

    def stack(
        prefix: str,
    ) -> np.ndarray:
        return np.column_stack(
            [
                columns[f"{prefix}_x"],
                columns[f"{prefix}_y"],
                columns[f"{prefix}_z"],
            ]
        ).astype(
            np.float64,
            copy=False,
        )

    return {
        "rotation_gt": stack(
            "rotation_gt"
        ),
        "rotation_pred": stack(
            "rotation_pred"
        ),
        "translation_gt": stack(
            "translation_gt"
        ),
        "translation_pred": stack(
            "translation_pred"
        ),
    }


# ============================================================================
# Geometry
# ============================================================================


def normalize_trajectory_to_first_pose(
    poses: np.ndarray,
) -> np.ndarray:
    """Express all KITTI poses relative to pose 0."""

    first_inverse = np.linalg.inv(
        poses[0]
    )

    normalized = np.empty_like(
        poses,
        dtype=np.float64,
    )

    for index in range(poses.shape[0]):
        normalized[index] = (
            first_inverse @ poses[index]
        )

        normalized[index, :3, :3] = (
            project_rotation_to_so3(
                normalized[index, :3, :3]
            )
        )

    return normalized


def absolute_to_relative_transforms(
    poses: np.ndarray,
) -> np.ndarray:
    """Compute T_i^-1 T_j for consecutive absolute poses."""

    relative = np.zeros(
        (poses.shape[0] - 1, 4, 4),
        dtype=np.float64,
    )

    for index in range(
        poses.shape[0] - 1
    ):
        relative[index] = (
            np.linalg.inv(poses[index])
            @ poses[index + 1]
        )

        relative[index, :3, :3] = (
            project_rotation_to_so3(
                relative[index, :3, :3]
            )
        )

    return relative


def integrate_relative_transforms(
    relative: np.ndarray,
) -> np.ndarray:
    """Ordinary SE(3) reconstruction from identity."""

    trajectory = np.zeros(
        (relative.shape[0] + 1, 4, 4),
        dtype=np.float64,
    )

    trajectory[0] = np.eye(
        4,
        dtype=np.float64,
    )

    for index in range(
        relative.shape[0]
    ):
        trajectory[index + 1] = (
            trajectory[index]
            @ relative[index]
        )

        trajectory[
            index + 1,
            :3,
            :3,
        ] = project_rotation_to_so3(
            trajectory[
                index + 1,
                :3,
                :3,
            ]
        )

    return trajectory


def generate_dct_from_absolute_poses(
    poses: np.ndarray,
    euler_order: str,
    angles_in_degrees: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the DCT representation implied by the current evaluator.

    For consecutive i -> j:

        R_rel  = R_i.T @ R_j
        R_half = sqrt(R_rel)
        t_c    = R_j.T @ R_half @ (t_j - t_i)
    """

    num_transitions = (
        poses.shape[0] - 1
    )

    rotations = np.zeros(
        (num_transitions, 3),
        dtype=np.float64,
    )

    translations = np.zeros(
        (num_transitions, 3),
        dtype=np.float64,
    )

    for index in range(
        num_transitions
    ):
        pose_i = poses[index]
        pose_j = poses[index + 1]

        r_i = project_rotation_to_so3(
            pose_i[:3, :3]
        )

        r_j = project_rotation_to_so3(
            pose_j[:3, :3]
        )

        t_i = pose_i[:3, 3]
        t_j = pose_j[:3, 3]

        relative_rotation = (
            r_i.T @ r_j
        )

        relative_rotation = (
            project_rotation_to_so3(
                relative_rotation
            )
        )

        relative_half = (
            rotation_square_root(
                relative_rotation
            )
        )

        delta_t_world = (
            t_j - t_i
        )

        directional_translation = (
            r_j.T
            @ relative_half
            @ delta_t_world
        )

        euler = rotation_matrix_to_euler(
            relative_rotation,
            order=euler_order,
            angles_in_degrees=angles_in_degrees,
        )

        translations[index] = (
            directional_translation
        )

        rotations[index] = euler

    return translations, rotations


def integrate_dct(
    rotations: np.ndarray,
    translations: np.ndarray,
    euler_order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    """
    Inverse-DCT trajectory integration.

    This intentionally mirrors the current
    evaluate_deepdct_vo.py integrate_relative_poses() logic.

    Given:

        R_rel = R_i.T @ R_j
        R_half = sqrt(R_rel)
        t_c = R_j.T @ R_half @ delta_t_world

    invert:

        R_j = R_i @ R_rel

        delta_t_world
            = R_half.T @ R_j @ t_c

        relative translation in camera-i coordinates
            = R_i.T @ delta_t_world

        T_j = T_i @ T_i_j
    """

    rotations = np.asarray(
        rotations,
        dtype=np.float64,
    )

    translations = np.asarray(
        translations,
        dtype=np.float64,
    )

    if rotations.shape != translations.shape:
        raise ValueError(
            "rotations and translations must have "
            f"matching shapes, got {rotations.shape} "
            f"and {translations.shape}."
        )

    if (
        rotations.ndim != 2
        or rotations.shape[1] != 3
    ):
        raise ValueError(
            "DCT arrays must have shape [N,3]."
        )

    if not np.all(np.isfinite(rotations)):
        raise ValueError(
            "Rotation array contains non-finite values."
        )

    if not np.all(
        np.isfinite(translations)
    ):
        raise ValueError(
            "Translation array contains non-finite values."
        )

    trajectory = np.zeros(
        (rotations.shape[0] + 1, 4, 4),
        dtype=np.float64,
    )

    trajectory[0] = np.eye(
        4,
        dtype=np.float64,
    )

    for index in range(
        rotations.shape[0]
    ):
        relative_rotation = (
            euler_to_rotation_matrix(
                rotations[index],
                order=euler_order,
                angles_in_degrees=angles_in_degrees,
            )
        )

        relative_rotation = (
            project_rotation_to_so3(
                relative_rotation
            )
        )

        current_rotation = (
            project_rotation_to_so3(
                trajectory[
                    index,
                    :3,
                    :3,
                ]
            )
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

        directional_translation = (
            translations[index]
        )

        delta_t_world = (
            half_rotation.T
            @ next_rotation
            @ directional_translation
        )

        relative_translation_i = (
            current_rotation.T
            @ delta_t_world
        )

        relative_transform = np.eye(
            4,
            dtype=np.float64,
        )

        relative_transform[
            :3,
            :3,
        ] = relative_rotation

        relative_transform[
            :3,
            3,
        ] = relative_translation_i

        trajectory[index + 1] = (
            trajectory[index]
            @ relative_transform
        )

        trajectory[
            index + 1,
            :3,
            :3,
        ] = project_rotation_to_so3(
            trajectory[
                index + 1,
                :3,
                :3,
            ]
        )

    return trajectory


# ============================================================================
# Metrics
# ============================================================================


def trajectory_translation_errors(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> np.ndarray:
    if reference.shape != estimate.shape:
        raise ValueError(
            "Trajectory shapes do not match: "
            f"{reference.shape} vs {estimate.shape}."
        )

    return np.linalg.norm(
        estimate[:, :3, 3]
        - reference[:, :3, 3],
        axis=1,
    )


def trajectory_rotation_errors_degrees(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> np.ndarray:
    errors = np.zeros(
        reference.shape[0],
        dtype=np.float64,
    )

    for index in range(
        reference.shape[0]
    ):
        r_ref = reference[
            index,
            :3,
            :3,
        ]

        r_est = estimate[
            index,
            :3,
            :3,
        ]

        error_rotation = (
            r_ref.T @ r_est
        )

        errors[index] = (
            rotation_angle_degrees(
                error_rotation
            )
        )

    return errors


def summarize_trajectory(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> Dict[str, float]:
    translation_error = (
        trajectory_translation_errors(
            reference,
            estimate,
        )
    )

    rotation_error = (
        trajectory_rotation_errors_degrees(
            reference,
            estimate,
        )
    )

    endpoint_error = float(
        translation_error[-1]
    )

    return {
        "translation_rmse_m": float(
            np.sqrt(
                np.mean(
                    translation_error ** 2
                )
            )
        ),
        "translation_mean_m": float(
            np.mean(
                translation_error
            )
        ),
        "translation_median_m": float(
            np.median(
                translation_error
            )
        ),
        "translation_max_m": float(
            np.max(
                translation_error
            )
        ),
        "endpoint_error_m": endpoint_error,
        "rotation_rmse_deg": float(
            np.sqrt(
                np.mean(
                    rotation_error ** 2
                )
            )
        ),
        "rotation_mean_deg": float(
            np.mean(
                rotation_error
            )
        ),
        "rotation_max_deg": float(
            np.max(
                rotation_error
            )
        ),
    }


def compare_label_arrays(
    existing_t: np.ndarray,
    existing_r: np.ndarray,
    regenerated_t: np.ndarray,
    regenerated_r: np.ndarray,
) -> Dict[str, object]:
    count = min(
        existing_t.shape[0],
        regenerated_t.shape[0],
    )

    t_error = (
        existing_t[:count]
        - regenerated_t[:count]
    )

    r_error = (
        existing_r[:count]
        - regenerated_r[:count]
    )

    return {
        "compared_transitions": int(
            count
        ),
        "existing_transition_count": int(
            existing_t.shape[0]
        ),
        "regenerated_transition_count": int(
            regenerated_t.shape[0]
        ),
        "translation_rmse_per_component": float(
            np.sqrt(
                np.mean(
                    t_error ** 2
                )
            )
        ),
        "translation_mae_per_component": float(
            np.mean(
                np.abs(t_error)
            )
        ),
        "translation_axis_rmse": (
            np.sqrt(
                np.mean(
                    t_error ** 2,
                    axis=0,
                )
            )
            .tolist()
        ),
        "rotation_rmse_per_component": float(
            np.sqrt(
                np.mean(
                    r_error ** 2
                )
            )
        ),
        "rotation_mae_per_component": float(
            np.mean(
                np.abs(r_error)
            )
        ),
        "rotation_axis_rmse": (
            np.sqrt(
                np.mean(
                    r_error ** 2,
                    axis=0,
                )
            )
            .tolist()
        ),
    }


# ============================================================================
# Output
# ============================================================================


def save_trajectory(
    path: Path,
    trajectory: np.ndarray,
) -> None:
    rows = trajectory[:, :3, :4].reshape(
        trajectory.shape[0],
        12,
    )

    np.savetxt(
        path,
        rows,
        fmt="%.12e",
    )


def save_regenerated_dct(
    path: Path,
    translations: np.ndarray,
    rotations: np.ndarray,
) -> None:
    labels = np.concatenate(
        [
            translations,
            rotations,
        ],
        axis=1,
    )

    np.savetxt(
        path,
        labels,
        fmt="%.10f",
    )


def plot_g1(
    path: Path,
    kitti_gt: np.ndarray,
    relative_roundtrip: np.ndarray,
    regenerated_dct_roundtrip: np.ndarray,
    existing_dct_roundtrip: np.ndarray,
    sequence: str,
) -> None:
    plt.figure(
        figsize=(10, 9)
    )

    plt.plot(
        kitti_gt[:, 0, 3],
        kitti_gt[:, 2, 3],
        label="KITTI GT",
        linewidth=2.0,
    )

    plt.plot(
        relative_roundtrip[:, 0, 3],
        relative_roundtrip[:, 2, 3],
        label="SE(3) round trip",
        linestyle="--",
    )

    plt.plot(
        regenerated_dct_roundtrip[:, 0, 3],
        regenerated_dct_roundtrip[:, 2, 3],
        label="Regenerated DCT round trip",
        linestyle=":",
    )

    plt.plot(
        existing_dct_roundtrip[:, 0, 3],
        existing_dct_roundtrip[:, 2, 3],
        label="Existing *_dct.txt round trip",
        linestyle="-.",
    )

    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title(
        f"Sequence {sequence}: DCT geometry round-trip audit"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def plot_g2(
    path: Path,
    reference: np.ndarray,
    trajectories: Mapping[
        str,
        np.ndarray,
    ],
    sequence: str,
) -> None:
    plt.figure(
        figsize=(10, 9)
    )

    plt.plot(
        reference[:, 0, 3],
        reference[:, 2, 3],
        label="KITTI GT",
        linewidth=2.5,
    )

    labels = {
        "gt_r_gt_t": "GT R + GT t",
        "gt_r_pred_t": "GT R + Pred t",
        "pred_r_gt_t": "Pred R + GT t",
        "pred_r_pred_t": "Pred R + Pred t",
    }

    for name, trajectory in (
        trajectories.items()
    ):
        plt.plot(
            trajectory[:, 0, 3],
            trajectory[:, 2, 3],
            label=labels.get(
                name,
                name,
            ),
        )

    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title(
        f"Sequence {sequence}: "
        "GT-R / Pred-R trajectory decomposition"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def print_metric_block(
    title: str,
    metrics: Mapping[str, float],
) -> None:
    print("-" * 88)
    print(title)
    print("-" * 88)

    for key, value in metrics.items():
        print(
            f"{key:<34s}: {value:.12g}"
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

    print("=" * 88)
    print(
        "DeepDCT-VO A6 DCT round-trip geometry audit"
    )
    print("=" * 88)
    print(
        f"Sequence:                     {args.sequence}"
    )
    print(
        f"KITTI pose file:              {args.pose_file}"
    )
    print(
        f"Existing DCT label file:      {args.dct_file}"
    )
    print(
        f"Euler order:                  {args.euler_order}"
    )
    print(
        "Angles in degrees:           "
        f"{args.angles_in_degrees}"
    )
    print(
        f"Output directory:             {args.output_dir}"
    )

    if args.frame_predictions is not None:
        print(
            "Frame predictions:             "
            f"{args.frame_predictions}"
        )

    print("=" * 88)

    # ----------------------------------------------------------------
    # Load ground truth.
    # ----------------------------------------------------------------
    raw_kitti_poses = load_kitti_poses(
        args.pose_file
    )

    kitti_gt = (
        normalize_trajectory_to_first_pose(
            raw_kitti_poses
        )
    )

    existing_t, existing_r = (
        load_dct_labels(
            args.dct_file
        )
    )

    expected_transitions = (
        kitti_gt.shape[0] - 1
    )

    if existing_t.shape[0] != expected_transitions:
        raise ValueError(
            "Pose/DCT transition-count mismatch. "
            f"Pose file has {kitti_gt.shape[0]} poses "
            f"and therefore {expected_transitions} "
            f"transitions, but DCT file has "
            f"{existing_t.shape[0]} rows."
        )

    print(
        f"Absolute KITTI poses:         {kitti_gt.shape[0]}"
    )
    print(
        f"DCT transitions:              {existing_t.shape[0]}"
    )

    # ================================================================
    # G1-A:
    # absolute -> relative SE(3) -> absolute
    # ================================================================
    ordinary_relative = (
        absolute_to_relative_transforms(
            kitti_gt
        )
    )

    relative_roundtrip = (
        integrate_relative_transforms(
            ordinary_relative
        )
    )

    metrics_relative = summarize_trajectory(
        kitti_gt,
        relative_roundtrip,
    )

    # ================================================================
    # G1-B:
    # absolute -> regenerated DCT -> inverse DCT -> absolute
    # ================================================================
    regenerated_t, regenerated_r = (
        generate_dct_from_absolute_poses(
            kitti_gt,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )
    )

    regenerated_dct_roundtrip = (
        integrate_dct(
            regenerated_r,
            regenerated_t,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )
    )

    metrics_regenerated_dct = (
        summarize_trajectory(
            kitti_gt,
            regenerated_dct_roundtrip,
        )
    )

    # ================================================================
    # G1-C:
    # existing *_dct.txt -> inverse DCT -> absolute
    # ================================================================
    existing_dct_roundtrip = (
        integrate_dct(
            existing_r,
            existing_t,
            euler_order=args.euler_order,
            angles_in_degrees=args.angles_in_degrees,
        )
    )

    metrics_existing_dct = (
        summarize_trajectory(
            kitti_gt,
            existing_dct_roundtrip,
        )
    )

    # ================================================================
    # G1-D:
    # Compare original DCT file against regenerated DCT values.
    # ================================================================
    label_comparison = (
        compare_label_arrays(
            existing_t,
            existing_r,
            regenerated_t,
            regenerated_r,
        )
    )

    print()
    print("=" * 88)
    print("G1 — GEOMETRY ROUND-TRIP RESULTS")
    print("=" * 88)

    print_metric_block(
        "G1-A: KITTI -> relative SE(3) -> KITTI",
        metrics_relative,
    )

    print_metric_block(
        "G1-B: KITTI -> regenerated DCT -> inverse DCT",
        metrics_regenerated_dct,
    )

    print_metric_block(
        "G1-C: existing *_dct.txt -> inverse DCT -> KITTI",
        metrics_existing_dct,
    )

    print("-" * 88)
    print(
        "G1-D: Existing DCT vs DCT regenerated from KITTI poses"
    )
    print("-" * 88)

    print(
        "translation RMSE/component:    "
        f"{label_comparison['translation_rmse_per_component']:.12g}"
    )

    print(
        "rotation RMSE/component:       "
        f"{label_comparison['rotation_rmse_per_component']:.12g}"
    )

    print(
        "translation axis RMSE:         "
        f"{label_comparison['translation_axis_rmse']}"
    )

    print(
        "rotation axis RMSE:            "
        f"{label_comparison['rotation_axis_rmse']}"
    )

    # ----------------------------------------------------------------
    # Save G1 outputs.
    # ----------------------------------------------------------------
    save_trajectory(
        args.output_dir
        / "trajectory_kitti_gt.txt",
        kitti_gt,
    )

    save_trajectory(
        args.output_dir
        / "trajectory_relative_se3_roundtrip.txt",
        relative_roundtrip,
    )

    save_trajectory(
        args.output_dir
        / "trajectory_regenerated_dct_roundtrip.txt",
        regenerated_dct_roundtrip,
    )

    save_trajectory(
        args.output_dir
        / "trajectory_existing_dct_roundtrip.txt",
        existing_dct_roundtrip,
    )

    save_regenerated_dct(
        args.output_dir
        / "regenerated_dct.txt",
        regenerated_t,
        regenerated_r,
    )

    plot_g1(
        args.output_dir
        / "trajectory_g1_xz.png",
        kitti_gt=kitti_gt,
        relative_roundtrip=relative_roundtrip,
        regenerated_dct_roundtrip=(
            regenerated_dct_roundtrip
        ),
        existing_dct_roundtrip=(
            existing_dct_roundtrip
        ),
        sequence=args.sequence,
    )

    # ================================================================
    # Optional G2:
    # four-way rotation/translation source decomposition.
    # ================================================================
    g2_metrics: Optional[
        Dict[str, Dict[str, float]]
    ] = None

    if args.frame_predictions is not None:
        predictions = (
            load_frame_predictions(
                args.frame_predictions
            )
        )

        count = predictions[
            "rotation_gt"
        ].shape[0]

        if count != expected_transitions:
            raise ValueError(
                "frame_predictions transition count "
                "does not match KITTI pose count: "
                f"{count} vs {expected_transitions}."
            )

        # The CSV ground truth should agree with
        # the existing training labels.
        csv_t_gt_error = (
            predictions["translation_gt"]
            - existing_t
        )

        csv_r_gt_error = (
            predictions["rotation_gt"]
            - existing_r
        )

        print()
        print("=" * 88)
        print("G2 — INPUT CONSISTENCY")
        print("=" * 88)
        print(
            "CSV GT translation vs DCT RMSE: "
            f"{np.sqrt(np.mean(csv_t_gt_error ** 2)):.12g}"
        )
        print(
            "CSV GT rotation vs DCT RMSE:    "
            f"{np.sqrt(np.mean(csv_r_gt_error ** 2)):.12g}"
        )

        predicted_t = (
            predictions[
                "translation_pred"
            ]
            * args.translation_scale_factor
        )

        trajectories = {
            "gt_r_gt_t": integrate_dct(
                predictions["rotation_gt"],
                predictions["translation_gt"],
                euler_order=args.euler_order,
                angles_in_degrees=args.angles_in_degrees,
            ),

            "gt_r_pred_t": integrate_dct(
                predictions["rotation_gt"],
                predicted_t,
                euler_order=args.euler_order,
                angles_in_degrees=args.angles_in_degrees,
            ),

            "pred_r_gt_t": integrate_dct(
                predictions["rotation_pred"],
                predictions["translation_gt"],
                euler_order=args.euler_order,
                angles_in_degrees=args.angles_in_degrees,
            ),

            "pred_r_pred_t": integrate_dct(
                predictions["rotation_pred"],
                predicted_t,
                euler_order=args.euler_order,
                angles_in_degrees=args.angles_in_degrees,
            ),
        }

        g2_metrics = {}

        print()
        print("=" * 88)
        print(
            "G2 — ROTATION / TRANSLATION SOURCE DECOMPOSITION"
        )
        print("=" * 88)
        print(
            "Predicted translation scale:   "
            f"{args.translation_scale_factor:.9f}"
        )

        for name, trajectory in (
            trajectories.items()
        ):
            metrics = summarize_trajectory(
                kitti_gt,
                trajectory,
            )

            g2_metrics[name] = metrics

            pretty_name = {
                "gt_r_gt_t": (
                    "GT R + GT directional t"
                ),
                "gt_r_pred_t": (
                    "GT R + predicted directional t"
                ),
                "pred_r_gt_t": (
                    "Pred R + GT directional t"
                ),
                "pred_r_pred_t": (
                    "Pred R + predicted directional t"
                ),
            }[name]

            print_metric_block(
                pretty_name,
                metrics,
            )

            save_trajectory(
                args.output_dir
                / f"trajectory_{name}.txt",
                trajectory,
            )

        plot_g2(
            args.output_dir
            / "trajectory_g2_xz.png",
            reference=kitti_gt,
            trajectories=trajectories,
            sequence=args.sequence,
        )

    # ----------------------------------------------------------------
    # Interpretation / gate.
    # ----------------------------------------------------------------
    existing_endpoint = (
        metrics_existing_dct[
            "endpoint_error_m"
        ]
    )

    generated_endpoint = (
        metrics_regenerated_dct[
            "endpoint_error_m"
        ]
    )

    print()
    print("=" * 88)
    print("AUDIT INTERPRETATION")
    print("=" * 88)

    print(
        "Regenerated-DCT endpoint error: "
        f"{generated_endpoint:.12g} m"
    )

    print(
        "Existing-DCT endpoint error:    "
        f"{existing_endpoint:.12g} m"
    )

    if generated_endpoint < 1.0e-6:
        print(
            "[PASS] Forward DCT and inverse DCT are "
            "internally self-consistent."
        )
    else:
        print(
            "[FAIL] Even regenerated DCT does not round-trip "
            "to KITTI GT. Inspect the inverse-DCT equations, "
            "Euler convention, or pose convention."
        )

    if existing_endpoint < args.translation_tolerance:
        print(
            "[PASS] Existing *_dct.txt labels are compatible "
            "with the audited inverse-DCT reconstruction."
        )
    else:
        print(
            "[ATTENTION] Existing *_dct.txt labels do not "
            "round-trip to KITTI GT within the requested "
            f"{args.translation_tolerance:g} m tolerance."
        )
        print(
            "            Compare G1-B against G1-C and G1-D. "
            "If G1-B passes while G1-C fails, the stored DCT "
            "labels were generated with a different convention "
            "than the current evaluator."
        )

    # ----------------------------------------------------------------
    # JSON summary.
    # ----------------------------------------------------------------
    summary = {
        "sequence": args.sequence,
        "pose_file": str(
            args.pose_file
        ),
        "dct_file": str(
            args.dct_file
        ),
        "frame_predictions": (
            str(args.frame_predictions)
            if args.frame_predictions
            is not None
            else None
        ),
        "euler_order": args.euler_order,
        "angles_in_degrees": (
            args.angles_in_degrees
        ),
        "translation_scale_factor": (
            args.translation_scale_factor
        ),
        "num_poses": int(
            kitti_gt.shape[0]
        ),
        "num_transitions": int(
            existing_t.shape[0]
        ),
        "g1": {
            "relative_se3_roundtrip": (
                metrics_relative
            ),
            "regenerated_dct_roundtrip": (
                metrics_regenerated_dct
            ),
            "existing_dct_roundtrip": (
                metrics_existing_dct
            ),
            "existing_vs_regenerated_labels": (
                label_comparison
            ),
        },
        "g2": g2_metrics,
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
        f"Outputs saved to: {args.output_dir}"
    )
    print("=" * 88)

    if (
        args.strict
        and existing_endpoint
        > args.translation_tolerance
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()