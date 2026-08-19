#!/usr/bin/env python3
"""
Verify relative-pose coordinate-frame, rotation, DCT, and trajectory conventions.

This script is intentionally independent of the training/evaluation code. It is
designed to answer four questions:

1. Which relative-transform direction and composition order reproduce KITTI poses?
2. Does the Euler convention round-trip to the original relative rotations?
3. Does the configured DCT translation encode/decode pair round-trip correctly?
4. Which interpretation of predicted relative poses best agrees with ground truth?

Expected KITTI absolute pose format
-----------------------------------
Each line contains 12 floats representing a row-major 3x4 matrix:

    T_W_C = [R_W_C | t_W_C]

The default relative transform is:

    T_Ci_Cj = inv(T_W_Ci) @ T_W_Cj

and the default trajectory update is:

    T_W_Cj = T_W_Ci @ T_Ci_Cj

DCT convention
--------------
The default implementation follows:

    t_c = inv(R_rel) @ inv(R_ref) @ R_c_sqrt @ t_w

and:

    t_w = inv(R_c_sqrt) @ R_ref @ R_rel @ t_c

Because R_ref and R_c_sqrt are project-specific, provide them with
--r-ref-* and --r-c-sqrt-* arguments, or adapt build_dct_context().

Examples
--------
Relative-pose convention:

    python scripts/debug/verify_relative_pose_frames.py \
        --pose-file data/poses/10.txt \
        --stage relative \
        --output-dir experiments/frame_verification/sequence_10

Euler convention:

    python scripts/debug/verify_relative_pose_frames.py \
        --pose-file data/poses/10.txt \
        --stage rotation \
        --euler-order xyz \
        --output-dir experiments/frame_verification/sequence_10

DCT round trip with constant matrices:

    python scripts/debug/verify_relative_pose_frames.py \
        --pose-file data/poses/10.txt \
        --dct-file data/out_csv/10_dct.txt \
        --stage dct-roundtrip \
        --r-ref-npy path/to/R_ref.npy \
        --r-c-sqrt-npy path/to/R_C_sqrt.npy \
        --output-dir experiments/frame_verification/sequence_10

Ground-truth DCT reconstruction:

    python scripts/debug/verify_relative_pose_frames.py \
        --pose-file data/poses/10.txt \
        --dct-file data/out_csv/10_dct.txt \
        --stage reconstruct-ground-truth \
        --dct-translation-cols 0,1,2 \
        --dct-rotation-cols 3, 4, 5 \
        --output-dir experiments/frame_verification/sequence_10

Prediction hypotheses:

    python scripts/debug/verify_relative_pose_frames.py \
        --pose-file data/poses/10.txt \
        --prediction-file experiments/frame_verification/sequence_10/raw_predictions.csv \
        --stage prediction-hypotheses \
        --prediction-rotation-cols rotation_raw_x,rotation_raw_y,rotation_raw_z \
        --prediction-translation-cols translation_decoded_x,translation_decoded_y,translation_decoded_z \
        --output-dir experiments/frame_verification/sequence_10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation
from deepdct.geometry.se3 import project_to_so3, rotation_sqrt

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


EPS = 1e-12
IDENTITY_3 = np.eye(3, dtype=np.float64)
IDENTITY_4 = np.eye(4, dtype=np.float64)

Column = Union[int, str]

def add_bool_argument(parser, name, default=False, help=None):
    """
    Python 3.8 replacement for argparse.BooleanOptionalAction.
    Creates both:

        --foo
        --no-foo
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        name,
        dest=name.lstrip("-").replace("-", "_"),
        action="store_true",
        help=help,
    )
    group.add_argument(
        "--no-" + name.lstrip("-"),
        dest=name.lstrip("-").replace("-", "_"),
        action="store_false",
    )
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


@dataclass(frozen=True)
class PoseMetrics:
    translation_rmse: float
    translation_mean: float
    translation_median: float
    translation_max: float
    rotation_rmse_deg: float
    rotation_mean_deg: float
    rotation_median_deg: float
    rotation_max_deg: float
    endpoint_translation_error: float
    endpoint_rotation_error_deg: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "translation_rmse": self.translation_rmse,
            "translation_mean": self.translation_mean,
            "translation_median": self.translation_median,
            "translation_max": self.translation_max,
            "rotation_rmse_deg": self.rotation_rmse_deg,
            "rotation_mean_deg": self.rotation_mean_deg,
            "rotation_median_deg": self.rotation_median_deg,
            "rotation_max_deg": self.rotation_max_deg,
            "endpoint_translation_error": self.endpoint_translation_error,
            "endpoint_rotation_error_deg": self.endpoint_rotation_error_deg,
        }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Verify DeepDCT-VO relative-pose coordinate-frame conventions.",
    )

    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--dct-file", type=Path)
    parser.add_argument("--prediction-file", type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "relative",
            "rotation",
            "rotation-labels",
            "dct-roundtrip",
            "reconstruct-ground-truth",
            "teacher-forcing",
            "prediction-hypotheses",
            "verify-prediction-reconstruction",
            "all",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/frame_verification"),
    )
    parser.add_argument("--num-samples", type=int, default=None)
    add_bool_argument(
        parser,
        "--normalize-first-pose",
        default=True,
    )

    parser.add_argument(
        "--relative-direction",
        choices=("forward", "inverse"),
        default="forward",
        help=(
            "forward: inv(T_i) @ T_{i+1}; "
            "inverse: inv(T_{i+1}) @ T_i"
        ),
    )
    parser.add_argument(
        "--composition-rule",
        choices=("right", "left", "right-inverse", "left-inverse"),
        default="right",
    )

    parser.add_argument(
        "--euler-order",
        choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"),
        default="xyz",
    )
    parser.add_argument(
        "--euler-convention",
        choices=("extrinsic", "intrinsic"),
        default="extrinsic",
    )
    add_bool_argument(
        parser,
        "--angles-in-degrees",
        default=False,
    )

    parser.add_argument(
        "--dct-translation-cols",
        default="0,1,2",
        help="Directional translation columns: tc_x,tc_y,tc_z.",
    )

    parser.add_argument(
        "--dct-rotation-cols",
        default="3,4,5",
        help="Euler rotation columns: roll,pitch,yaw.",
    )

    parser.add_argument(
        "--dct-direct-translation-cols",
        default=None,
        help=(
            "Optional three columns containing pre-DCT translation. "
            "When absent, the script uses the relative-transform translation."
        ),
    )

    parser.add_argument(
        "--prediction-rotation-cols",
        default="rotation_raw_x,rotation_raw_y,rotation_raw_z",
    )
    parser.add_argument(
        "--prediction-translation-cols",
        default="translation_decoded_x,translation_decoded_y,translation_decoded_z",
    )
    parser.add_argument(
        "--prediction-dct-translation-cols",
        default="translation_dct_raw_x,translation_dct_raw_y,translation_dct_raw_z",
    )
    parser.add_argument(
        "--prediction-gt-rotation-cols",
        default="rotation_gt_x,rotation_gt_y,rotation_gt_z",
    )
    parser.add_argument(
        "--prediction-gt-dct-translation-cols",
        default="translation_dct_gt_x,translation_dct_gt_y,translation_dct_gt_z",
    )

    matrix_group = parser.add_argument_group("DCT matrices")

    add_bool_argument(
        parser,
        "--save-plots",
        default=True,
    )
    parser.add_argument(
        "--print-first",
        type=int,
        default=20,
        help="Number of per-step diagnostics to write.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Return a non-zero exit code when a deterministic validation fails.",
    )
    parser.add_argument(
        "--predicted-trajectory-file",
        type=Path,
        help=(
            "KITTI-format predicted trajectory exported by "
            "scripts/evaluate_deepdct_vo.py."
        ),
    )

    return parser.parse_args()


def ensure_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        raise ValueError(f"{name} contains non-finite values at {bad[:10].tolist()}.")


def load_kitti_poses(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Pose file does not exist: {path}")

    rows = np.loadtxt(path, dtype=np.float64)

    if rows.ndim == 1:
        rows = rows[None, :]

    if rows.shape[1] != 12:
        raise ValueError(
            f"Expected 12 values per KITTI pose row, found shape {rows.shape}."
        )

    poses = np.repeat(
        IDENTITY_4[None, :, :],
        rows.shape[0],
        axis=0,
    )
    poses[:, :3, :] = rows.reshape(-1, 3, 4)

    ensure_finite("KITTI poses", poses)

    raw_rotations = poses[:, :3, :3].copy()

    orthogonality_errors = np.asarray(
        [
            np.linalg.norm(rotation.T @ rotation - IDENTITY_3)
            for rotation in raw_rotations
        ],
        dtype=np.float64,
    )

    determinant_errors = np.abs(
        np.linalg.det(raw_rotations) - 1.0
    )

    print()
    print("Raw KITTI rotation quality")
    print("==========================")
    print(
        "Mean orthogonality error: "
        f"{np.mean(orthogonality_errors):.6e}"
    )
    print(
        "Maximum orthogonality error: "
        f"{np.max(orthogonality_errors):.6e}"
    )
    print(
        "Mean determinant error: "
        f"{np.mean(determinant_errors):.6e}"
    )
    print(
        "Maximum determinant error: "
        f"{np.max(determinant_errors):.6e}"
    )

    for index in range(len(poses)):
        poses[index, :3, :3] = project_rotation_to_so3(
            poses[index, :3, :3]
        )

    return poses

def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    """
    Project a nearly rotational 3x3 matrix onto the closest valid SO(3)
    rotation matrix using SVD.
    """
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt

    # Ensure a proper rotation with determinant +1 rather than a reflection.
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt

    return projected


def normalize_poses(poses: np.ndarray) -> np.ndarray:
    t0_inv = invert_transform(poses[0])
    return np.einsum("ij,njk->nik", t0_inv, poses)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def invert_transforms(transforms: np.ndarray) -> np.ndarray:
    return np.stack([invert_transform(t) for t in transforms], axis=0)


def rotation_quality(rotations: np.ndarray) -> Dict[str, float]:
    orth_errors = np.linalg.norm(
        np.einsum("nji,njk->nik", rotations, rotations) - IDENTITY_3,
        axis=(1, 2),
    )
    determinant_errors = np.abs(np.linalg.det(rotations) - 1.0)
    return {
        "orthogonality_error_mean": float(np.mean(orth_errors)),
        "orthogonality_error_max": float(np.max(orth_errors)),
        "determinant_error_mean": float(np.mean(determinant_errors)),
        "determinant_error_max": float(np.max(determinant_errors)),
    }


def derive_relative_poses(
    absolute_poses: np.ndarray,
    direction: str = "forward",
) -> np.ndarray:
    relative: List[np.ndarray] = []
    for current, nxt in zip(absolute_poses[:-1], absolute_poses[1:]):
        if direction == "forward":
            relative.append(invert_transform(current) @ nxt)
        elif direction == "inverse":
            relative.append(invert_transform(nxt) @ current)
        else:
            raise ValueError(f"Unsupported relative direction: {direction}")
    return np.stack(relative, axis=0)


def compose_step(global_pose: np.ndarray, relative_pose: np.ndarray, rule: str) -> np.ndarray:
    if rule == "right":
        return global_pose @ relative_pose
    if rule == "left":
        return relative_pose @ global_pose
    if rule == "right-inverse":
        return global_pose @ invert_transform(relative_pose)
    if rule == "left-inverse":
        return invert_transform(relative_pose) @ global_pose
    raise ValueError(f"Unsupported composition rule: {rule}")


def integrate_relative_poses(
    relative_poses: np.ndarray,
    rule: str = "right",
    initial_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    current = np.array(
        IDENTITY_4 if initial_pose is None else initial_pose,
        dtype=np.float64,
        copy=True,
    )
    trajectory = [current.copy()]
    for relative in relative_poses:
        current = compose_step(current, relative, rule)
        trajectory.append(current.copy())
    return np.stack(trajectory, axis=0)


def rotation_geodesic_deg(
    rotation_a: np.ndarray,
    rotation_b: np.ndarray,
) -> float:
    rotation_a = project_rotation_to_so3(rotation_a)
    rotation_b = project_rotation_to_so3(rotation_b)

    delta = rotation_a.T @ rotation_b
    delta = project_rotation_to_so3(delta)

    cosine = float(
        np.clip(
            (np.trace(delta) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    return math.degrees(math.acos(cosine))


def trajectory_metrics(reference: np.ndarray, estimate: np.ndarray) -> PoseMetrics:
    count = min(len(reference), len(estimate))
    if count == 0:
        raise ValueError("Cannot compare empty trajectories.")

    reference = reference[:count]
    estimate = estimate[:count]

    translation_errors = np.linalg.norm(
        estimate[:, :3, 3] - reference[:, :3, 3],
        axis=1,
    )
    rotation_errors = np.asarray(
        [
            rotation_geodesic_deg(a[:3, :3], b[:3, :3])
            for a, b in zip(reference, estimate)
        ],
        dtype=np.float64,
    )

    return PoseMetrics(
        translation_rmse=float(np.sqrt(np.mean(translation_errors ** 2))),
        translation_mean=float(np.mean(translation_errors)),
        translation_median=float(np.median(translation_errors)),
        translation_max=float(np.max(translation_errors)),
        rotation_rmse_deg=float(np.sqrt(np.mean(rotation_errors ** 2))),
        rotation_mean_deg=float(np.mean(rotation_errors)),
        rotation_median_deg=float(np.median(rotation_errors)),
        rotation_max_deg=float(np.max(rotation_errors)),
        endpoint_translation_error=float(translation_errors[-1]),
        endpoint_rotation_error_deg=float(rotation_errors[-1]),
    )


def axis_rotation(axis: str, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    if axis == "x":
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )
    if axis == "y":
        return np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )
    if axis == "z":
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported axis: {axis}")

def wrap_angle_radians(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_angle_difference(
    angle_a: np.ndarray,
    angle_b: np.ndarray,
    degrees: bool,
) -> np.ndarray:
    """
    Return the shortest signed difference angle_a - angle_b.
    """
    if degrees:
        difference = np.deg2rad(angle_a - angle_b)
        return np.rad2deg(wrap_angle_radians(difference))

    return wrap_angle_radians(angle_a - angle_b)


def euler_to_matrix(
    angles: Sequence[float],
    order: str,
    convention: str,
    degrees: bool,
) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected three Euler angles, found shape {values.shape}.")
    if degrees:
        values = np.deg2rad(values)

    elemental = [axis_rotation(axis, angle) for axis, angle in zip(order, values)]

    result = np.eye(3, dtype=np.float64)
    if convention == "extrinsic":
        for rotation in elemental:
            result = rotation @ result
    elif convention == "intrinsic":
        for rotation in elemental:
            result = result @ rotation
    else:
        raise ValueError(f"Unsupported Euler convention: {convention}")
    return result


def matrix_to_euler_xyz_extrinsic(rotation: np.ndarray) -> np.ndarray:
    """
    Inverse of euler_to_matrix(..., order='xyz', convention='extrinsic').

    The forward matrix is Rz @ Ry @ Rx.
    """
    sy = -rotation[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    y = math.asin(sy)
    cy = math.cos(y)

    if abs(cy) > 1e-9:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = 0.0
        z = math.atan2(-rotation[0, 1], rotation[1, 1])

    return np.asarray([x, y, z], dtype=np.float64)


def matrix_to_euler_zyx_extrinsic(rotation: np.ndarray) -> np.ndarray:
    """
    Inverse of euler_to_matrix(..., order='zyx', convention='extrinsic').

    The forward matrix is Rx @ Ry @ Rz.
    """
    sy = rotation[0, 2]
    sy = float(np.clip(sy, -1.0, 1.0))
    y = math.asin(sy)
    cy = math.cos(y)

    if abs(cy) > 1e-9:
        z = math.atan2(-rotation[0, 1], rotation[0, 0])
        x = math.atan2(-rotation[1, 2], rotation[2, 2])
    else:
        z = 0.0
        x = math.atan2(rotation[2, 1], rotation[1, 1])

    return np.asarray([z, y, x], dtype=np.float64)


def matrix_to_euler(
    rotation: np.ndarray,
    order: str,
    convention: str,
    degrees: bool,
) -> np.ndarray:
    """
    Matrix-to-Euler support for the two most common project conventions.

    Other orders are still accepted for label-to-matrix conversion, but the
    independent matrix-to-Euler round-trip stage requires xyz/extrinsic or
    zyx/extrinsic. This avoids silently using an incorrect generic formula.
    """
    if convention != "extrinsic":
        raise NotImplementedError(
            "Independent matrix-to-Euler round trip currently supports "
            "extrinsic conventions only."
        )

    if order == "xyz":
        values = matrix_to_euler_xyz_extrinsic(rotation)
    elif order == "zyx":
        values = matrix_to_euler_zyx_extrinsic(rotation)
    else:
        raise NotImplementedError(
            "Independent matrix-to-Euler round trip currently supports "
            "orders xyz and zyx only. Label reconstruction still supports "
            "all CLI-listed orders."
        )

    return np.rad2deg(values) if degrees else values


def transforms_from_rotation_translation(
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    if len(rotations) != len(translations):
        raise ValueError("Rotation and translation row counts differ.")
    transforms = np.repeat(IDENTITY_4[None, :, :], len(rotations), axis=0)
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = translations
    return transforms

def stage_rotation_labels(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    if args.dct_file is None:
        raise ValueError(
            "--dct-file is required for stage 'rotation-labels'."
        )

    relative_gt = derive_relative_poses(
        absolute_poses,
        args.relative_direction,
    )

    table, names = load_table(args.dct_file)

    label_angles = select_columns(
        table,
        names,
        parse_columns(args.dct_rotation_cols),
        "DCT rotation labels",
    )

    count = min(len(relative_gt), len(label_angles))
    relative_gt = relative_gt[:count]
    label_angles = label_angles[:count]

    recomputed_angles = rotations_to_angles(
        args,
        relative_gt[:, :3, :3],
    )

    angle_difference = wrapped_angle_difference(
        label_angles,
        recomputed_angles,
        args.angles_in_degrees,
    )

    absolute_angle_error = np.abs(angle_difference)

    # More reliable than direct Euler subtraction:
    # reconstruct each label rotation and compare matrices.
    label_rotations = rotations_from_angles(
        args,
        label_angles,
    )

    geodesic_errors = np.asarray(
        [
            rotation_geodesic_deg(
                relative_gt[index, :3, :3],
                label_rotations[index],
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )

    save_csv(
        args.output_dir / "rotation_label_comparison.csv",
        [
            "step",
            "recomputed_a0",
            "recomputed_a1",
            "recomputed_a2",
            "label_a0",
            "label_a1",
            "label_a2",
            "wrapped_error_a0",
            "wrapped_error_a1",
            "wrapped_error_a2",
            "geodesic_error_deg",
        ],
        (
            (
                index,
                *recomputed_angles[index].tolist(),
                *label_angles[index].tolist(),
                *angle_difference[index].tolist(),
                geodesic_errors[index],
            )
            for index in range(count)
        ),
    )

    component_names = list(args.euler_order)

    result = {
        "rows": int(count),
        "euler_order": args.euler_order,
        "euler_convention": args.euler_convention,
        "angles_in_degrees": bool(args.angles_in_degrees),
        "angle_component_errors": {
            component_names[index]: {
                "mean_absolute": float(
                    np.mean(absolute_angle_error[:, index])
                ),
                "median_absolute": float(
                    np.median(absolute_angle_error[:, index])
                ),
                "max_absolute": float(
                    np.max(absolute_angle_error[:, index])
                ),
            }
            for index in range(3)
        },
        "matrix_geodesic_error_deg": {
            "mean": float(np.mean(geodesic_errors)),
            "median": float(np.median(geodesic_errors)),
            "rmse": float(
                np.sqrt(np.mean(geodesic_errors ** 2))
            ),
            "max": float(np.max(geodesic_errors)),
        },
    }

    passed = bool(
        np.max(geodesic_errors) < 1e-4
    )
    result["passed"] = passed

    print()
    print("Euler label comparison")
    print("======================")
    print(f"Rows:                {count}")
    print(
        "Convention:          "
        f"{args.euler_order}/{args.euler_convention}"
    )
    print(
        "Units:               "
        f"{'degrees' if args.angles_in_degrees else 'radians'}"
    )

    for index, name in enumerate(component_names):
        print(
            f"Mean |{name}| error:      "
            f"{np.mean(absolute_angle_error[:, index]):.6e}"
        )
        print(
            f"Maximum |{name}| error:   "
            f"{np.max(absolute_angle_error[:, index]):.6e}"
        )

    print(
        "Mean geodesic error: "
        f"{np.mean(geodesic_errors):.6e} deg"
    )
    print(
        "Max geodesic error:  "
        f"{np.max(geodesic_errors):.6e} deg"
    )
    print(
        "Rotation-label status: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return result, passed


def parse_columns(specification: Optional[str]) -> Optional[Tuple[Column, Column, Column]]:
    if specification is None:
        return None

    parts = [part.strip() for part in specification.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"Expected exactly three comma-separated columns, got: {specification!r}"
        )

    parsed: List[Column] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError:
            parsed.append(part)
    return parsed[0], parsed[1], parsed[2]


def detect_delimiter(path: Path) -> Optional[str]:
    with path.open("r", encoding="utf-8") as handle:
        sample = handle.read(4096)

    if "," in sample:
        return ","
    if "\t" in sample:
        return "\t"
    return None


def first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise ValueError(f"No data rows found in {path}.")


def has_header(path: Path, delimiter: Optional[str]) -> bool:
    line = first_nonempty_line(path)
    tokens = line.split(delimiter) if delimiter else line.split()
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return True
    return False


def load_table(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    delimiter = detect_delimiter(path)
    header_present = has_header(path, delimiter)

    if header_present:
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            names=True,
            dtype=np.float64,
            encoding="utf-8",
            comments="#",
        )
        if data.dtype.names is None:
            raise ValueError(f"Could not parse a header from {path}.")
        names = list(data.dtype.names)
        if data.shape == ():
            data = np.asarray([data], dtype=data.dtype)
        matrix = np.column_stack([data[name] for name in names])
        return np.asarray(matrix, dtype=np.float64), names

    matrix = np.loadtxt(path, dtype=np.float64, delimiter=delimiter, comments="#")
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    return matrix, None


def select_columns(
    matrix: np.ndarray,
    names: Optional[Sequence[str]],
    columns: Sequence[Column],
    label: str,
) -> np.ndarray:
    indices: List[int] = []
    for column in columns:
        if isinstance(column, int):
            index = column
        else:
            if names is None:
                raise ValueError(
                    f"{label} requests named column {column!r}, "
                    "but the input file has no header."
                )
            try:
                index = list(names).index(column)
            except ValueError as exc:
                raise ValueError(
                    f"{label} column {column!r} not found. Available columns: {names}"
                ) from exc

        if index < 0 or index >= matrix.shape[1]:
            raise IndexError(
                f"{label} column index {index} is outside [0, {matrix.shape[1] - 1}]."
            )
        indices.append(index)

    selected = matrix[:, indices]
    ensure_finite(label, selected)
    return selected

def encode_directional_translation(
    delta_t_world: np.ndarray,
    rotation_i: np.ndarray,
    rotation_j: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the translation label generated by scripts/dct_labels.py.

    Parameters
    ----------
    delta_t_world:
        World-frame displacement tj - ti.
    rotation_i:
        Camera orientation Ri from KITTI absolute pose i.
    rotation_j:
        Camera orientation Rj from KITTI absolute pose j.

    Returns
    -------
    tc:
        Directional translation stored in columns 0,1,2 of *_dct.txt.
    """
    rotation_i = project_to_so3(rotation_i)
    rotation_j = project_to_so3(rotation_j)

    rotation_relative = project_to_so3(
        rotation_i.T @ rotation_j
    )
    rotation_half = rotation_sqrt(rotation_relative)

    return rotation_j.T @ rotation_half @ delta_t_world


def decode_directional_translation(
    tc: np.ndarray,
    ri: np.ndarray,
    rj: np.ndarray,
) -> np.ndarray:
    rc = project_to_so3(ri.T @ rj)
    rc_half = rotation_sqrt(rc)

    delta_t_world = rc_half.T @ rj @ tc
    return delta_t_world

def decode_relative_translation_current_frame(
    tc: np.ndarray,
    ri: np.ndarray,
    rj: np.ndarray,
) -> np.ndarray:
    rc = project_to_so3(ri.T @ rj)
    rc_half = rotation_sqrt(rc)

    delta_t_world = rc_half.T @ rj @ tc
    t_relative_ci = ri.T @ delta_t_world

    return t_relative_ci


def encode_directional_translation_batch(
    delta_t_world: np.ndarray,
    rotations_i: np.ndarray,
    rotations_j: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            encode_directional_translation(delta_t, ri, rj)
            for delta_t, ri, rj in zip(
                delta_t_world,
                rotations_i,
                rotations_j,
            )
        ],
        axis=0,
    )


def decode_directional_translation_batch(
    translations_directional: np.ndarray,
    rotations_i: np.ndarray,
    rotations_j: np.ndarray,
) -> np.ndarray:
    """
    Decode a batch of directional-translation labels into world-frame
    displacements.

    For each transition i -> j, this applies:

        Rc = Ri.T @ Rj
        Rc_half = rotation_sqrt(Rc)
        delta_t_world = Rc_half.T @ Rj @ tc

    Parameters
    ----------
    translations_directional:
        Array with shape (N, 3), containing tc values from columns 0,1,2
        of the DCT label file.
    rotations_i:
        Array with shape (N, 3, 3), containing the absolute camera
        orientations at frame i.
    rotations_j:
        Array with shape (N, 3, 3), containing the absolute camera
        orientations at frame j.

    Returns
    -------
    np.ndarray
        Array with shape (N, 3), containing tj - ti in the KITTI
        world/reference frame.
    """
    translations_directional = np.asarray(
        translations_directional,
        dtype=np.float64,
    )
    rotations_i = np.asarray(rotations_i, dtype=np.float64)
    rotations_j = np.asarray(rotations_j, dtype=np.float64)

    if translations_directional.ndim != 2:
        raise ValueError(
            "translations_directional must have shape (N, 3); "
            f"found {translations_directional.shape}."
        )

    if translations_directional.shape[1] != 3:
        raise ValueError(
            "translations_directional must have exactly three columns; "
            f"found {translations_directional.shape}."
        )

    if rotations_i.ndim != 3 or rotations_i.shape[1:] != (3, 3):
        raise ValueError(
            "rotations_i must have shape (N, 3, 3); "
            f"found {rotations_i.shape}."
        )

    if rotations_j.ndim != 3 or rotations_j.shape[1:] != (3, 3):
        raise ValueError(
            "rotations_j must have shape (N, 3, 3); "
            f"found {rotations_j.shape}."
        )

    row_count = len(translations_directional)

    if len(rotations_i) != row_count:
        raise ValueError(
            "translations_directional and rotations_i have different "
            f"row counts: {row_count} and {len(rotations_i)}."
        )

    if len(rotations_j) != row_count:
        raise ValueError(
            "translations_directional and rotations_j have different "
            f"row counts: {row_count} and {len(rotations_j)}."
        )

    if row_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    ensure_finite(
        "directional translations",
        translations_directional,
    )
    ensure_finite("rotations_i", rotations_i)
    ensure_finite("rotations_j", rotations_j)

    decoded = np.stack(
        [
            decode_directional_translation(tc, ri, rj)
            for tc, ri, rj in zip(
                translations_directional,
                rotations_i,
                rotations_j,
            )
        ],
        axis=0,
    )

    ensure_finite("decoded world displacements", decoded)
    return decoded


def decode_relative_translation_current_frame_batch(
    translations_directional: np.ndarray,
    rotations_i: np.ndarray,
    rotations_j: np.ndarray,
) -> np.ndarray:
    """
    Decode a batch of directional-translation labels into the translation
    components of forward relative transforms:

        T_Ci_Cj = inv(T_W_Ci) @ T_W_Cj

    The directional label is first decoded into the world displacement:

        delta_t_world = tj - ti

    It is then represented in the current camera frame:

        t_Ci_Cj = Ri.T @ delta_t_world

    Parameters
    ----------
    translations_directional:
        Array with shape (N, 3), containing tc values from columns 0,1,2
        of the DCT label file.
    rotations_i:
        Array with shape (N, 3, 3), containing the absolute camera
        orientations at frame i.
    rotations_j:
        Array with shape (N, 3, 3), containing the absolute camera
        orientations at frame j.

    Returns
    -------
    np.ndarray
        Array with shape (N, 3), containing relative translations
        expressed in frame Ci. These values can be used as the translation
        components of relative transforms integrated with:

            T_next = T_current @ T_relative
    """
    translations_directional = np.asarray(
        translations_directional,
        dtype=np.float64,
    )
    rotations_i = np.asarray(rotations_i, dtype=np.float64)
    rotations_j = np.asarray(rotations_j, dtype=np.float64)

    if translations_directional.ndim != 2:
        raise ValueError(
            "translations_directional must have shape (N, 3); "
            f"found {translations_directional.shape}."
        )

    if translations_directional.shape[1] != 3:
        raise ValueError(
            "translations_directional must have exactly three columns; "
            f"found {translations_directional.shape}."
        )

    if rotations_i.ndim != 3 or rotations_i.shape[1:] != (3, 3):
        raise ValueError(
            "rotations_i must have shape (N, 3, 3); "
            f"found {rotations_i.shape}."
        )

    if rotations_j.ndim != 3 or rotations_j.shape[1:] != (3, 3):
        raise ValueError(
            "rotations_j must have shape (N, 3, 3); "
            f"found {rotations_j.shape}."
        )

    row_count = len(translations_directional)

    if len(rotations_i) != row_count:
        raise ValueError(
            "translations_directional and rotations_i have different "
            f"row counts: {row_count} and {len(rotations_i)}."
        )

    if len(rotations_j) != row_count:
        raise ValueError(
            "translations_directional and rotations_j have different "
            f"row counts: {row_count} and {len(rotations_j)}."
        )

    if row_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    ensure_finite(
        "directional translations",
        translations_directional,
    )
    ensure_finite("rotations_i", rotations_i)
    ensure_finite("rotations_j", rotations_j)

    decoded = np.stack(
        [
            decode_relative_translation_current_frame(tc, ri, rj)
            for tc, ri, rj in zip(
                translations_directional,
                rotations_i,
                rotations_j,
            )
        ],
        axis=0,
    )

    ensure_finite(
        "decoded current-frame relative translations",
        decoded,
    )
    return decoded



def limit_rows(array: np.ndarray, count: Optional[int]) -> np.ndarray:
    if count is None:
        return array
    if count <= 0:
        raise ValueError("--num-samples must be positive.")
    return array[:count]


def align_relative_count(
    relative_gt: np.ndarray,
    *arrays: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    count = min([len(relative_gt)] + [len(array) for array in arrays])
    if count == 0:
        raise ValueError("No overlapping relative-pose rows are available.")
    return (relative_gt[:count],) + tuple(array[:count] for array in arrays)


def json_default(value):
    """Convert NumPy objects into JSON-compatible Python values."""
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=json_default,
        )


def save_pose_file(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = poses[:, :3, :].reshape(len(poses), 12)
    np.savetxt(path, rows, fmt="%.12e")


def save_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_trajectory(
    output_path: Path,
    trajectories: Mapping[str, np.ndarray],
    plane: str = "xz",
) -> None:
    if plt is None:
        raise RuntimeError(
            "Matplotlib is not installed. Re-run with --no-save-plots "
            "or install matplotlib."
        )

    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in axis_map:
        raise ValueError(f"Unsupported plot plane: {plane}")
    first_axis, second_axis = axis_map[plane]

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111)

    for label, poses in trajectories.items():
        positions = poses[:, :3, 3]
        axes.plot(
            positions[:, first_axis],
            positions[:, second_axis],
            label=label,
            linewidth=1.4,
        )

    axes.set_xlabel(plane[0])
    axes.set_ylabel(plane[1])
    axes.set_title(f"Trajectory comparison ({plane.upper()} plane)")
    axes.axis("equal")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_metric_table(title: str, entries: Mapping[str, PoseMetrics]) -> None:
    print()
    print(title)
    print("=" * len(title))
    print(
        f"{'candidate':<24}"
        f"{'trans RMSE':>14}"
        f"{'trans max':>14}"
        f"{'rot RMSE deg':>16}"
        f"{'rot max deg':>14}"
        f"{'endpoint m':>14}"
    )
    for name, metrics in entries.items():
        print(
            f"{name:<24}"
            f"{metrics.translation_rmse:>14.6e}"
            f"{metrics.translation_max:>14.6e}"
            f"{metrics.rotation_rmse_deg:>16.6e}"
            f"{metrics.rotation_max_deg:>14.6e}"
            f"{metrics.endpoint_translation_error:>14.6e}"
        )


def stage_relative(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    forward = derive_relative_poses(absolute_poses, "forward")
    inverse = derive_relative_poses(absolute_poses, "inverse")

    inverse_consistency = np.max(
        np.abs(inverse - invert_transforms(forward))
    )

    candidates = {
        "forward/right": integrate_relative_poses(forward, "right"),
        "forward/left": integrate_relative_poses(forward, "left"),
        "forward/right-inverse": integrate_relative_poses(forward, "right-inverse"),
        "forward/left-inverse": integrate_relative_poses(forward, "left-inverse"),
        "inverse/right": integrate_relative_poses(inverse, "right"),
        "inverse/left": integrate_relative_poses(inverse, "left"),
        "inverse/right-inverse": integrate_relative_poses(inverse, "right-inverse"),
        "inverse/left-inverse": integrate_relative_poses(inverse, "left-inverse"),
    }

    metrics = {
        name: trajectory_metrics(absolute_poses, trajectory)
        for name, trajectory in candidates.items()
    }
    print_metric_table("Relative transform/composition candidates", metrics)

    preferred_order = [
        "forward/right",
        "inverse/right-inverse",
        "forward/left",
        "forward/right-inverse",
        "forward/left-inverse",
        "inverse/right",
        "inverse/left",
        "inverse/left-inverse",
    ]

    minimum_rmse = min(
        metric.translation_rmse
        for metric in metrics.values()
    )

    tie_tolerance = max(
        1e-9,
        minimum_rmse * 1e-6,
    )

    equivalent_best = {
        name
        for name, metric in metrics.items()
        if abs(metric.translation_rmse - minimum_rmse) <= tie_tolerance
    }

    best_name = next(
        name
        for name in preferred_order
        if name in equivalent_best
    )
    best = metrics[best_name]

    result: Dict[str, object] = {
        "rotation_quality": rotation_quality(absolute_poses[:, :3, :3]),
        "inverse_consistency_max_abs": float(inverse_consistency),
        "best_candidate": best_name,
        "candidates": {name: value.as_dict() for name, value in metrics.items()},
    }

    save_pose_file(args.output_dir / "gt_normalized.txt", absolute_poses)
    for name, trajectory in candidates.items():
        safe_name = name.replace("/", "_")
        save_pose_file(args.output_dir / f"relative_reconstruction_{safe_name}.txt", trajectory)

    if args.save_plots:
        plot_trajectory(
            args.output_dir / "01_gt_absolute_vs_gt_relative_reconstructed_xz.png",
            {
                "ground truth": absolute_poses,
                "best reconstruction": candidates[best_name],
            },
            plane="xz",
        )
        plot_trajectory(
            args.output_dir / "01_gt_absolute_vs_gt_relative_reconstructed_xy.png",
            {
                "ground truth": absolute_poses,
                "best reconstruction": candidates[best_name],
            },
            plane="xy",
        )

    passed = bool(
        best.translation_max < 1e-6
        and best.rotation_max_deg < 1e-4
        and inverse_consistency < 1e-8
    )
    result["passed"] = passed
    print(f"\nBest candidate: {best_name}")
    print(f"Relative-stage status: {'PASS' if passed else 'FAIL'}")
    return result, passed


def stage_rotation(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    relative = derive_relative_poses(absolute_poses, args.relative_direction)

    errors: List[float] = []
    extracted_angles: List[np.ndarray] = []
    for transform in relative:
        original = transform[:3, :3]
        angles = matrix_to_euler(
            original,
            args.euler_order,
            args.euler_convention,
            args.angles_in_degrees,
        )
        reconstructed = euler_to_matrix(
            angles,
            args.euler_order,
            args.euler_convention,
            args.angles_in_degrees,
        )
        errors.append(rotation_geodesic_deg(original, reconstructed))
        extracted_angles.append(angles)

    error_array = np.asarray(errors)
    angle_array = np.stack(extracted_angles, axis=0)

    result = {
        "euler_order": args.euler_order,
        "euler_convention": args.euler_convention,
        "angles_in_degrees": args.angles_in_degrees,
        "rotation_roundtrip_error_deg": {
            "mean": float(np.mean(error_array)),
            "median": float(np.median(error_array)),
            "max": float(np.max(error_array)),
        },
    }

    np.savetxt(
        args.output_dir / "rotation_roundtrip_angles.txt",
        angle_array,
        fmt="%.12e",
    )
    save_csv(
        args.output_dir / "rotation_roundtrip_errors.csv",
        ["step", "error_deg"],
        ((index, value) for index, value in enumerate(error_array)),
    )

    passed = bool(float(np.max(error_array)) < 1e-4)
    result["passed"] = passed

    print("\nEuler round trip")
    print("================")
    print(f"Order:             {args.euler_order}")
    print(f"Convention:        {args.euler_convention}")
    print(f"Mean error:        {np.mean(error_array):.6e} deg")
    print(f"Maximum error:     {np.max(error_array):.6e} deg")
    print(f"Rotation status:   {'PASS' if passed else 'FAIL'}")
    return result, passed


def load_dct_components(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if args.dct_file is None:
        raise ValueError("--dct-file is required for this stage.")

    matrix, names = load_table(args.dct_file)
    rotations = select_columns(
        matrix,
        names,
        parse_columns(args.dct_rotation_cols),
        "DCT rotation",
    )
    translations_dct = select_columns(
        matrix,
        names,
        parse_columns(args.dct_translation_cols),
        "DCT translation",
    )

    direct_columns = parse_columns(args.dct_direct_translation_cols)
    direct = (
        select_columns(matrix, names, direct_columns, "DCT direct translation")
        if direct_columns is not None
        else None
    )
    return rotations, translations_dct, direct


def rotations_from_angles(
    args: argparse.Namespace,
    angles: np.ndarray,
) -> np.ndarray:
    sequence = args.euler_order

    if args.euler_convention == "intrinsic":
        sequence = sequence.upper()
    else:
        sequence = sequence.lower()

    return Rotation.from_euler(
        sequence,
        angles,
        degrees=args.angles_in_degrees,
    ).as_matrix()

def rotations_to_angles(
    args: argparse.Namespace,
    rotations: np.ndarray,
) -> np.ndarray:
    sequence = args.euler_order

    if args.euler_convention == "intrinsic":
        sequence = sequence.upper()
    else:
        sequence = sequence.lower()

    return Rotation.from_matrix(rotations).as_euler(
        sequence,
        degrees=args.angles_in_degrees,
    )



def stage_dct_roundtrip(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    """
    Verify the directional-translation labels generated by dct_labels.py.

    The label generator defines:

        delta_t_world = t_j - t_i
        R_c = R_i.T @ R_j
        R_c_half = rotation_sqrt(R_c)
        t_c = R_j.T @ R_c_half @ delta_t_world

    This stage checks:

        1. Re-encoding delta_t_world reproduces the stored t_c label.
        2. Decoding the stored t_c recovers delta_t_world.
        3. Converting the decoded world displacement into frame C_i
           reproduces the translation component of:

               T_Ci_Cj = inv(T_W_Ci) @ T_W_Cj
    """
    if args.dct_file is None:
        raise ValueError(
            "--dct-file is required for stage 'dct-roundtrip'."
        )

    # ---------------------------------------------------------
    # Load directional-translation labels.
    #
    # Correct dct_labels.py column layout:
    #   columns 0,1,2 -> tc_x, tc_y, tc_z
    #   columns 3,4,5 -> roll, pitch, yaw
    # ---------------------------------------------------------
    table, names = load_table(args.dct_file)

    translations_directional_file = select_columns(
        table,
        names,
        parse_columns(args.dct_translation_cols),
        "directional translation labels",
    )

    rotation_angles_file = select_columns(
        table,
        names,
        parse_columns(args.dct_rotation_cols),
        "Euler rotation labels",
    )

    # ---------------------------------------------------------
    # Ground-truth absolute-pose components.
    # ---------------------------------------------------------
    rotations_i = absolute_poses[:-1, :3, :3]
    rotations_j = absolute_poses[1:, :3, :3]

    positions_i = absolute_poses[:-1, :3, 3]
    positions_j = absolute_poses[1:, :3, 3]

    delta_t_world_gt = positions_j - positions_i

    relative_gt = derive_relative_poses(
        absolute_poses,
        args.relative_direction,
    )

    # ---------------------------------------------------------
    # Align all arrays to the same number of transitions.
    # ---------------------------------------------------------
    count = min(
        len(relative_gt),
        len(rotations_i),
        len(rotations_j),
        len(delta_t_world_gt),
        len(translations_directional_file),
        len(rotation_angles_file),
    )

    if count == 0:
        raise ValueError(
            "No overlapping directional-translation rows are available."
        )

    relative_gt = relative_gt[:count]
    rotations_i = rotations_i[:count]
    rotations_j = rotations_j[:count]
    delta_t_world_gt = delta_t_world_gt[:count]
    translations_directional_file = (
        translations_directional_file[:count]
    )
    rotation_angles_file = rotation_angles_file[:count]

    # ---------------------------------------------------------
    # Reconstruct relative rotations from the stored Euler labels.
    #
    # This independently verifies that directional-translation
    # decoding uses the same rotation labels as the dataset.
    # ---------------------------------------------------------
    relative_rotations_from_labels = rotations_from_angles(
        args,
        rotation_angles_file,
    )

    relative_rotation_errors_deg = np.asarray(
        [
            rotation_geodesic_deg(
                relative_gt[index, :3, :3],
                relative_rotations_from_labels[index],
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Encode ground-truth world displacement into directional
    # translation and compare against the stored file values.
    # ---------------------------------------------------------
    translations_directional_encoded = (
        encode_directional_translation_batch(
            delta_t_world_gt,
            rotations_i,
            rotations_j,
        )
    )

    encode_vs_file_error = np.linalg.norm(
        translations_directional_encoded
        - translations_directional_file,
        axis=1,
    )

    # ---------------------------------------------------------
    # Decode stored directional translation back into world-frame
    # displacement.
    # ---------------------------------------------------------
    delta_t_world_decoded = (
        decode_directional_translation_batch(
            translations_directional_file,
            rotations_i,
            rotations_j,
        )
    )

    decoded_world_error = np.linalg.norm(
        delta_t_world_decoded - delta_t_world_gt,
        axis=1,
    )

    # ---------------------------------------------------------
    # Full algebraic round trip:
    #
    #   delta_t_world
    #       -> encoded tc
    #       -> decoded delta_t_world
    # ---------------------------------------------------------
    delta_t_world_roundtrip = (
        decode_directional_translation_batch(
            translations_directional_encoded,
            rotations_i,
            rotations_j,
        )
    )

    algebraic_roundtrip_error = np.linalg.norm(
        delta_t_world_roundtrip - delta_t_world_gt,
        axis=1,
    )

    # ---------------------------------------------------------
    # Convert decoded directional translations into the
    # translation component of T_Ci_Cj.
    # ---------------------------------------------------------
    relative_translations_decoded = (
        decode_relative_translation_current_frame_batch(
            translations_directional_file,
            rotations_i,
            rotations_j,
        )
    )

    relative_translations_gt = relative_gt[:, :3, 3]

    decoded_relative_error = np.linalg.norm(
        relative_translations_decoded
        - relative_translations_gt,
        axis=1,
    )

    # ---------------------------------------------------------
    # Save detailed per-transition diagnostics.
    # ---------------------------------------------------------
    save_csv(
        args.output_dir / "directional_translation_roundtrip.csv",
        [
            "step",
            "file_tc_x",
            "file_tc_y",
            "file_tc_z",
            "encoded_tc_x",
            "encoded_tc_y",
            "encoded_tc_z",
            "gt_world_dx",
            "gt_world_dy",
            "gt_world_dz",
            "decoded_world_dx",
            "decoded_world_dy",
            "decoded_world_dz",
            "gt_relative_tx",
            "gt_relative_ty",
            "gt_relative_tz",
            "decoded_relative_tx",
            "decoded_relative_ty",
            "decoded_relative_tz",
            "encode_vs_file_l2",
            "decoded_world_l2",
            "algebraic_roundtrip_l2",
            "decoded_relative_l2",
            "rotation_label_geodesic_error_deg",
        ],
        (
            (
                index,
                *translations_directional_file[index].tolist(),
                *translations_directional_encoded[index].tolist(),
                *delta_t_world_gt[index].tolist(),
                *delta_t_world_decoded[index].tolist(),
                *relative_translations_gt[index].tolist(),
                *relative_translations_decoded[index].tolist(),
                encode_vs_file_error[index],
                decoded_world_error[index],
                algebraic_roundtrip_error[index],
                decoded_relative_error[index],
                relative_rotation_errors_deg[index],
            )
            for index in range(count)
        ),
    )

    # ---------------------------------------------------------
    # Machine-readable summary.
    # ---------------------------------------------------------
    result: Dict[str, object] = {
        "rows": int(count),
        "column_mapping": {
            "directional_translation": list(
                parse_columns(args.dct_translation_cols)
            ),
            "euler_rotation": list(
                parse_columns(args.dct_rotation_cols)
            ),
        },
        "encode_vs_file_directional_translation_l2": (
            summarize_vector(encode_vs_file_error)
        ),
        "decoded_world_displacement_l2": (
            summarize_vector(decoded_world_error)
        ),
        "algebraic_encode_decode_roundtrip_l2": (
            summarize_vector(algebraic_roundtrip_error)
        ),
        "decoded_relative_translation_l2": (
            summarize_vector(decoded_relative_error)
        ),
        "rotation_label_geodesic_error_deg": (
            summarize_vector(relative_rotation_errors_deg)
        ),
    }

    # ---------------------------------------------------------
    # Deterministic pass thresholds.
    #
    # The label file is written with finite decimal precision, so
    # file-comparison thresholds are looser than the pure algebraic
    # round-trip threshold.
    # ---------------------------------------------------------
    passed = bool(
        float(np.max(encode_vs_file_error)) < 1e-8
        and float(np.max(decoded_world_error)) < 1e-8
        and float(np.max(algebraic_roundtrip_error)) < 1e-10
        and float(np.max(decoded_relative_error)) < 1e-8
        and float(np.max(relative_rotation_errors_deg)) < 1e-4
    )

    result["passed"] = passed

    # ---------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------
    print()
    print("Directional translation round trip")
    print("==================================")
    print(f"Rows:                              {count}")
    print(
        "Encode/file mean:                  "
        f"{np.mean(encode_vs_file_error):.6e}"
    )
    print(
        "Encode/file maximum:               "
        f"{np.max(encode_vs_file_error):.6e}"
    )
    print(
        "Decoded world mean:                "
        f"{np.mean(decoded_world_error):.6e}"
    )
    print(
        "Decoded world maximum:             "
        f"{np.max(decoded_world_error):.6e}"
    )
    print(
        "Algebraic round-trip mean:         "
        f"{np.mean(algebraic_roundtrip_error):.6e}"
    )
    print(
        "Algebraic round-trip maximum:      "
        f"{np.max(algebraic_roundtrip_error):.6e}"
    )
    print(
        "Decoded relative mean:             "
        f"{np.mean(decoded_relative_error):.6e}"
    )
    print(
        "Decoded relative maximum:          "
        f"{np.max(decoded_relative_error):.6e}"
    )
    print(
        "Rotation-label maximum error:      "
        f"{np.max(relative_rotation_errors_deg):.6e} deg"
    )
    print(
        "Directional translation status:    "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return result, passed




def summarize_vector(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "max": float(np.max(values)),
    }


def stage_reconstruct_ground_truth(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    """
    Reconstruct the KITTI trajectory from the stored ground-truth labels.

    The DCT label layout is:

        columns 0,1,2 -> directional translation tc
        columns 3,4,5 -> roll, pitch, yaw

    The label generator defines:

        Rc = Ri.T @ Rj
        Rc_half = rotation_sqrt(Rc)
        tc = Rj.T @ Rc_half @ (tj - ti)

    This stage:

        1. Loads stored Euler and directional-translation labels.
        2. Reconstructs each relative rotation Rc from Euler angles.
        3. Decodes tc into the translation component of T_Ci_Cj.
        4. Builds the relative SE(3) transforms.
        5. Integrates them using the configured composition rule.
        6. Compares the reconstructed trajectory against KITTI poses.
    """
    if args.dct_file is None:
        raise ValueError(
            "--dct-file is required for stage "
            "'reconstruct-ground-truth'."
        )

    # ---------------------------------------------------------
    # Load the stored labels.
    # ---------------------------------------------------------
    table, names = load_table(args.dct_file)

    translations_directional = select_columns(
        table,
        names,
        parse_columns(args.dct_translation_cols),
        "ground-truth directional translation",
    )

    rotation_angles = select_columns(
        table,
        names,
        parse_columns(args.dct_rotation_cols),
        "ground-truth Euler rotation",
    )

    # ---------------------------------------------------------
    # Ground-truth absolute pose components.
    # ---------------------------------------------------------
    rotations_i = absolute_poses[:-1, :3, :3]
    rotations_j = absolute_poses[1:, :3, :3]

    relative_gt = derive_relative_poses(
        absolute_poses,
        args.relative_direction,
    )

    # ---------------------------------------------------------
    # Align all arrays to the same transition count.
    # ---------------------------------------------------------
    count = min(
        len(relative_gt),
        len(rotations_i),
        len(rotations_j),
        len(translations_directional),
        len(rotation_angles),
    )

    if count == 0:
        raise ValueError(
            "No overlapping ground-truth label rows are available."
        )

    relative_gt = relative_gt[:count]
    rotations_i = rotations_i[:count]
    rotations_j = rotations_j[:count]
    translations_directional = (
        translations_directional[:count]
    )
    rotation_angles = rotation_angles[:count]

    # ---------------------------------------------------------
    # Reconstruct relative rotations from stored Euler labels.
    # ---------------------------------------------------------
    relative_rotations_decoded = rotations_from_angles(
        args,
        rotation_angles,
    )

    # ---------------------------------------------------------
    # Decode stored directional translations into the
    # translation components of T_Ci_Cj.
    #
    # This uses the absolute Ri and Rj because the directional
    # translation encoding is defined with both orientations.
    # ---------------------------------------------------------
    relative_translations_decoded = (
        decode_relative_translation_current_frame_batch(
            translations_directional,
            rotations_i,
            rotations_j,
        )
    )

    # ---------------------------------------------------------
    # Assemble and integrate the decoded relative transforms.
    # ---------------------------------------------------------
    reconstructed_relative = (
        transforms_from_rotation_translation(
            relative_rotations_decoded,
            relative_translations_decoded,
        )
    )

    reconstructed_trajectory = integrate_relative_poses(
        reconstructed_relative,
        args.composition_rule,
    )

    reference_trajectory = absolute_poses[
        : len(reconstructed_trajectory)
    ]

    metrics = trajectory_metrics(
        reference_trajectory,
        reconstructed_trajectory,
    )

    # ---------------------------------------------------------
    # Per-step comparison against directly derived GT relative
    # transforms.
    # ---------------------------------------------------------
    relative_translation_errors = np.linalg.norm(
        reconstructed_relative[:, :3, 3]
        - relative_gt[:, :3, 3],
        axis=1,
    )

    relative_rotation_errors_deg = np.asarray(
        [
            rotation_geodesic_deg(
                relative_gt[index, :3, :3],
                reconstructed_relative[index, :3, :3],
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )

    relative_transform_max_abs_errors = np.asarray(
        [
            np.max(
                np.abs(
                    reconstructed_relative[index]
                    - relative_gt[index]
                )
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Save reconstructed poses and diagnostics.
    # ---------------------------------------------------------
    save_pose_file(
        args.output_dir
        / "gt_directional_labels_reconstructed_trajectory.txt",
        reconstructed_trajectory,
    )

    save_pose_file(
        args.output_dir
        / "gt_directional_labels_reconstructed_relative.txt",
        reconstructed_relative,
    )

    save_csv(
        args.output_dir
        / "gt_directional_label_reconstruction_errors.csv",
        [
            "step",
            "translation_error",
            "rotation_error_deg",
            "relative_transform_max_abs_error",
            "gt_tx",
            "gt_ty",
            "gt_tz",
            "decoded_tx",
            "decoded_ty",
            "decoded_tz",
        ],
        (
            (
                index,
                relative_translation_errors[index],
                relative_rotation_errors_deg[index],
                relative_transform_max_abs_errors[index],
                *relative_gt[index, :3, 3].tolist(),
                *reconstructed_relative[index, :3, 3].tolist(),
            )
            for index in range(count)
        ),
    )

    if args.save_plots:
        plot_trajectory(
            args.output_dir
            / "02_gt_absolute_vs_directional_gt_reconstructed_xz.png",
            {
                "ground truth": reference_trajectory,
                "decoded GT labels": reconstructed_trajectory,
            },
            plane="xz",
        )

        plot_trajectory(
            args.output_dir
            / "02_gt_absolute_vs_directional_gt_reconstructed_xy.png",
            {
                "ground truth": reference_trajectory,
                "decoded GT labels": reconstructed_trajectory,
            },
            plane="xy",
        )

    # ---------------------------------------------------------
    # Machine-readable summary.
    # ---------------------------------------------------------
    result: Dict[str, object] = {
        "rows": int(count),
        "column_mapping": {
            "directional_translation": list(
                parse_columns(args.dct_translation_cols)
            ),
            "euler_rotation": list(
                parse_columns(args.dct_rotation_cols)
            ),
        },
        "trajectory_metrics": metrics.as_dict(),
        "per_step_translation_error": summarize_vector(
            relative_translation_errors
        ),
        "per_step_rotation_error_deg": summarize_vector(
            relative_rotation_errors_deg
        ),
        "per_step_relative_transform_max_abs_error": (
            summarize_vector(
                relative_transform_max_abs_errors
            )
        ),
    }

    # ---------------------------------------------------------
    # Deterministic pass thresholds.
    #
    # These are intentionally slightly looser than the pure DCT
    # round-trip thresholds because this stage also includes:
    #
    #   - finite-precision label files,
    #   - Euler reconstruction,
    #   - trajectory accumulation.
    # ---------------------------------------------------------
    passed = bool(
        metrics.translation_rmse < 1e-6
        and metrics.endpoint_translation_error < 1e-6
        and metrics.rotation_rmse_deg < 1e-4
        and metrics.endpoint_rotation_error_deg < 1e-4
        and float(np.max(relative_translation_errors)) < 1e-8
        and float(np.max(relative_rotation_errors_deg)) < 1e-4
        and float(np.max(relative_transform_max_abs_errors)) < 1e-8
    )

    result["passed"] = passed

    # ---------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------
    print_metric_table(
        "Ground-truth directional-label reconstruction",
        {
            "decoded GT labels": metrics,
        },
    )

    print()
    print("Per-step reconstruction")
    print("=======================")
    print(
        "Translation error mean:             "
        f"{np.mean(relative_translation_errors):.6e}"
    )
    print(
        "Translation error maximum:          "
        f"{np.max(relative_translation_errors):.6e}"
    )
    print(
        "Rotation error mean:                "
        f"{np.mean(relative_rotation_errors_deg):.6e} deg"
    )
    print(
        "Rotation error maximum:             "
        f"{np.max(relative_rotation_errors_deg):.6e} deg"
    )
    print(
        "Relative-transform max abs error:   "
        f"{np.max(relative_transform_max_abs_errors):.6e}"
    )
    print(
        "Ground-truth reconstruction status: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return result, passed

def load_prediction_components(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    if args.prediction_file is None:
        raise ValueError("--prediction-file is required for this stage.")
    return load_table(args.prediction_file)


def try_select_columns(
    matrix: np.ndarray,
    names: Optional[Sequence[str]],
    specification: str,
    label: str,
) -> Optional[np.ndarray]:
    try:
        return select_columns(
            matrix,
            names,
            parse_columns(specification),
            label,
        )
    except (ValueError, IndexError):
        return None


def translation_cosines(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    numerator = np.sum(predicted * ground_truth, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(ground_truth, axis=1)
    result = np.full(len(predicted), np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return result


def norm_ratios(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(ground_truth, axis=1)
    result = np.full(len(predicted), np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = np.linalg.norm(predicted[valid], axis=1) / denominator[valid]
    return result


def finite_summary(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    summary = summarize_vector(finite)
    summary["count"] = int(finite.size)
    return summary


def write_step_diagnostics(
    args: argparse.Namespace,
    gt_relative: np.ndarray,
    predicted_rotations: np.ndarray,
    predicted_translations: np.ndarray,
) -> None:
    count = min(args.print_first, len(gt_relative))
    gt_translations = gt_relative[:count, :3, 3]
    predicted_translations = predicted_translations[:count]
    cosines = translation_cosines(predicted_translations, gt_translations)
    ratios = norm_ratios(predicted_translations, gt_translations)

    rows = []
    for index in range(count):
        gt_rotation = gt_relative[index, :3, :3]
        predicted_rotation = predicted_rotations[index]
        rows.append(
            (
                index,
                *gt_translations[index].tolist(),
                *predicted_translations[index].tolist(),
                np.linalg.norm(gt_translations[index]),
                np.linalg.norm(predicted_translations[index]),
                cosines[index],
                ratios[index],
                rotation_geodesic_deg(gt_rotation, predicted_rotation),
            )
        )

    save_csv(
        args.output_dir / "first_steps_diagnostics.csv",
        [
            "step",
            "gt_tx",
            "gt_ty",
            "gt_tz",
            "pred_tx",
            "pred_ty",
            "pred_tz",
            "gt_step_norm",
            "pred_step_norm",
            "translation_cosine",
            "translation_norm_ratio",
            "rotation_error_deg",
        ],
        rows,
    )


def stage_teacher_forcing(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    """
    Evaluate four rotation/translation source combinations:

        GT_R__GT_T
        PRED_R__GT_T
        GT_R__PRED_T
        PRED_R__PRED_T

    Rotation values are relative Euler angles.

    Translation values are directional translations tc, not ordinary
    relative-frame translations.

    Directional translation decoding depends on the current and next absolute
    orientations:

        Rc = Ri.T @ Rj
        delta_t_world = sqrt(Rc).T @ Rj @ tc
        t_relative_ci = Ri.T @ delta_t_world

    Therefore, modes using predicted rotation recursively propagate the
    predicted absolute orientation:

        Rj_pred = Ri_pred @ Rc_pred

    No ground-truth future orientation is used in predicted-rotation modes.
    """
    if args.prediction_file is None:
        raise ValueError(
            "--prediction-file is required for stage 'teacher-forcing'."
        )

    # ---------------------------------------------------------
    # Load prediction table.
    # ---------------------------------------------------------
    matrix, names = load_prediction_components(args)

    predicted_rotation_angles = select_columns(
        matrix,
        names,
        parse_columns(args.prediction_rotation_cols),
        "predicted relative Euler rotation",
    )

    predicted_directional_translation = select_columns(
        matrix,
        names,
        parse_columns(args.prediction_dct_translation_cols),
        "predicted directional translation",
    )

    ground_truth_rotation_angles = select_columns(
        matrix,
        names,
        parse_columns(args.prediction_gt_rotation_cols),
        "ground-truth relative Euler rotation",
    )

    ground_truth_directional_translation = select_columns(
        matrix,
        names,
        parse_columns(args.prediction_gt_dct_translation_cols),
        "ground-truth directional translation",
    )

    # ---------------------------------------------------------
    # Convert Euler labels/predictions into relative rotations.
    # ---------------------------------------------------------
    predicted_relative_rotations = rotations_from_angles(
        args,
        predicted_rotation_angles,
    )

    ground_truth_relative_rotations = rotations_from_angles(
        args,
        ground_truth_rotation_angles,
    )

    directly_derived_relative_gt = derive_relative_poses(
        absolute_poses,
        args.relative_direction,
    )

    # ---------------------------------------------------------
    # Align all sources.
    # ---------------------------------------------------------
    count = min(
        len(directly_derived_relative_gt),
        len(predicted_relative_rotations),
        len(predicted_directional_translation),
        len(ground_truth_relative_rotations),
        len(ground_truth_directional_translation),
    )

    if count == 0:
        raise ValueError(
            "No overlapping rows are available for teacher forcing."
        )

    directly_derived_relative_gt = directly_derived_relative_gt[:count]
    predicted_relative_rotations = (
        predicted_relative_rotations[:count]
    )
    predicted_directional_translation = (
        predicted_directional_translation[:count]
    )
    ground_truth_relative_rotations = (
        ground_truth_relative_rotations[:count]
    )
    ground_truth_directional_translation = (
        ground_truth_directional_translation[:count]
    )

    reference_trajectory = absolute_poses[: count + 1]

    # ---------------------------------------------------------
    # Roll out one teacher-forcing mode.
    # ---------------------------------------------------------
    def reconstruct_mode(
        mode_name: str,
        relative_rotations: np.ndarray,
        directional_translations: np.ndarray,
        use_ground_truth_absolute_orientation: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct relative transforms and their integrated trajectory.

        For GT-rotation modes, Ri and Rj are taken from the normalized KITTI
        poses.

        For predicted-rotation modes, orientation is propagated recursively
        from identity using the predicted relative rotations.
        """
        reconstructed_relative: List[np.ndarray] = []

        # Normalized trajectories begin at identity.
        current_predicted_orientation = np.eye(
            3,
            dtype=np.float64,
        )

        for index in range(count):
            relative_rotation = project_to_so3(
                relative_rotations[index]
            )

            if use_ground_truth_absolute_orientation:
                rotation_i = reference_trajectory[
                    index, :3, :3
                ]
                rotation_j = reference_trajectory[
                    index + 1, :3, :3
                ]
            else:
                rotation_i = current_predicted_orientation
                rotation_j = project_to_so3(
                    rotation_i @ relative_rotation
                )

            relative_translation = (
                decode_relative_translation_current_frame(
                    directional_translations[index],
                    rotation_i,
                    rotation_j,
                )
            )

            relative_transform = np.eye(
                4,
                dtype=np.float64,
            )
            relative_transform[:3, :3] = relative_rotation
            relative_transform[:3, 3] = relative_translation

            reconstructed_relative.append(relative_transform)

            if not use_ground_truth_absolute_orientation:
                current_predicted_orientation = rotation_j

        relative_array = np.stack(
            reconstructed_relative,
            axis=0,
        )

        trajectory = integrate_relative_poses(
            relative_array,
            args.composition_rule,
        )

        save_pose_file(
            args.output_dir
            / f"teacher_forcing_{mode_name}_relative.txt",
            relative_array,
        )

        save_pose_file(
            args.output_dir
            / f"teacher_forcing_{mode_name}_trajectory.txt",
            trajectory,
        )

        return relative_array, trajectory

    # ---------------------------------------------------------
    # Four source combinations.
    # ---------------------------------------------------------
    mode_inputs = {
        "GT_R__GT_T": {
            "rotations": ground_truth_relative_rotations,
            "translations": ground_truth_directional_translation,
            "use_gt_absolute_orientation": True,
        },
        "PRED_R__GT_T": {
            "rotations": predicted_relative_rotations,
            "translations": ground_truth_directional_translation,
            "use_gt_absolute_orientation": False,
        },
        "GT_R__PRED_T": {
            "rotations": ground_truth_relative_rotations,
            "translations": predicted_directional_translation,
            "use_gt_absolute_orientation": True,
        },
        "PRED_R__PRED_T": {
            "rotations": predicted_relative_rotations,
            "translations": predicted_directional_translation,
            "use_gt_absolute_orientation": False,
        },
    }

    relative_results: Dict[str, np.ndarray] = {}
    trajectories: Dict[str, np.ndarray] = {}
    metrics: Dict[str, PoseMetrics] = {}

    for mode_name, mode in mode_inputs.items():
        relative_array, trajectory = reconstruct_mode(
            mode_name=mode_name,
            relative_rotations=mode["rotations"],
            directional_translations=mode["translations"],
            use_ground_truth_absolute_orientation=(
                mode["use_gt_absolute_orientation"]
            ),
        )

        relative_results[mode_name] = relative_array
        trajectories[mode_name] = trajectory
        metrics[mode_name] = trajectory_metrics(
            reference_trajectory,
            trajectory,
        )

    # ---------------------------------------------------------
    # Per-step diagnostics for each mode.
    # ---------------------------------------------------------
    diagnostic_rows = []

    for index in range(count):
        gt_relative = directly_derived_relative_gt[index]

        for mode_name in mode_inputs:
            reconstructed = relative_results[mode_name][index]

            translation_error = np.linalg.norm(
                reconstructed[:3, 3]
                - gt_relative[:3, 3]
            )

            rotation_error_deg = rotation_geodesic_deg(
                gt_relative[:3, :3],
                reconstructed[:3, :3],
            )

            gt_translation = gt_relative[:3, 3]
            reconstructed_translation = reconstructed[:3, 3]

            denominator = (
                np.linalg.norm(gt_translation)
                * np.linalg.norm(reconstructed_translation)
            )

            if denominator > EPS:
                cosine = float(
                    np.dot(
                        gt_translation,
                        reconstructed_translation,
                    )
                    / denominator
                )
            else:
                cosine = float("nan")

            if np.linalg.norm(gt_translation) > EPS:
                norm_ratio = float(
                    np.linalg.norm(reconstructed_translation)
                    / np.linalg.norm(gt_translation)
                )
            else:
                norm_ratio = float("nan")

            diagnostic_rows.append(
                (
                    index,
                    mode_name,
                    translation_error,
                    rotation_error_deg,
                    cosine,
                    norm_ratio,
                    *gt_translation.tolist(),
                    *reconstructed_translation.tolist(),
                )
            )

    save_csv(
        args.output_dir
        / "teacher_forcing_step_diagnostics.csv",
        [
            "step",
            "mode",
            "translation_error",
            "rotation_error_deg",
            "translation_cosine",
            "translation_norm_ratio",
            "gt_tx",
            "gt_ty",
            "gt_tz",
            "reconstructed_tx",
            "reconstructed_ty",
            "reconstructed_tz",
        ],
        diagnostic_rows,
    )

    # ---------------------------------------------------------
    # Plots.
    # ---------------------------------------------------------
    if args.save_plots:
        plot_trajectory(
            args.output_dir
            / "teacher_forcing_modes_xz.png",
            {
                "ground truth": reference_trajectory,
                "GT R + GT T": trajectories["GT_R__GT_T"],
                "Pred R + GT T": trajectories["PRED_R__GT_T"],
                "GT R + Pred T": trajectories["GT_R__PRED_T"],
                "Pred R + Pred T": trajectories["PRED_R__PRED_T"],
            },
            plane="xz",
        )

        plot_trajectory(
            args.output_dir
            / "teacher_forcing_modes_xy.png",
            {
                "ground truth": reference_trajectory,
                "GT R + GT T": trajectories["GT_R__GT_T"],
                "Pred R + GT T": trajectories["PRED_R__GT_T"],
                "GT R + Pred T": trajectories["GT_R__PRED_T"],
                "Pred R + Pred T": trajectories["PRED_R__PRED_T"],
            },
            plane="xy",
        )

    # ---------------------------------------------------------
    # Summary and deterministic GT/GT gate.
    # ---------------------------------------------------------
    print_metric_table(
        "Teacher-forcing modes",
        metrics,
    )

    gt_gt_relative_translation_errors = np.linalg.norm(
        relative_results["GT_R__GT_T"][:, :3, 3]
        - directly_derived_relative_gt[:, :3, 3],
        axis=1,
    )

    gt_gt_relative_rotation_errors = np.asarray(
        [
            rotation_geodesic_deg(
                directly_derived_relative_gt[index, :3, :3],
                relative_results["GT_R__GT_T"][
                    index, :3, :3
                ],
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )

    gt_gt_metrics = metrics["GT_R__GT_T"]

    passed = bool(
        gt_gt_metrics.translation_rmse < 1e-5
        and gt_gt_metrics.endpoint_translation_error < 1e-5
        and gt_gt_metrics.rotation_rmse_deg < 1e-4
        and gt_gt_metrics.endpoint_rotation_error_deg < 1e-4
        and float(
            np.max(gt_gt_relative_translation_errors)
        ) < 1e-8
        and float(
            np.max(gt_gt_relative_rotation_errors)
        ) < 1e-4
    )

    result: Dict[str, object] = {
        "rows": int(count),
        "modes": {
            name: metric.as_dict()
            for name, metric in metrics.items()
        },
        "gt_gt_per_step_translation_error": summarize_vector(
            gt_gt_relative_translation_errors
        ),
        "gt_gt_per_step_rotation_error_deg": summarize_vector(
            gt_gt_relative_rotation_errors
        ),
        "passed": passed,
        "interpretation": {
            "GT_R__GT_T": (
                "Deterministic geometry and evaluation-path check."
            ),
            "PRED_R__GT_T": (
                "Effect of predicted rotation while holding directional "
                "translation labels fixed."
            ),
            "GT_R__PRED_T": (
                "Effect of predicted directional translation while holding "
                "rotation fixed."
            ),
            "PRED_R__PRED_T": (
                "Actual coupled prediction behavior."
            ),
        },
    }

    print()
    print("Teacher-forcing interpretation")
    print("==============================")
    print(
        "GT rotation + GT translation:       "
        f"ATE-like RMSE {metrics['GT_R__GT_T'].translation_rmse:.6e}"
    )
    print(
        "Pred rotation + GT translation:     "
        f"ATE-like RMSE {metrics['PRED_R__GT_T'].translation_rmse:.6e}"
    )
    print(
        "GT rotation + Pred translation:     "
        f"ATE-like RMSE {metrics['GT_R__PRED_T'].translation_rmse:.6e}"
    )
    print(
        "Pred rotation + Pred translation:   "
        f"ATE-like RMSE {metrics['PRED_R__PRED_T'].translation_rmse:.6e}"
    )
    print(
        "GT/GT deterministic status:         "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return result, passed


def hypothesis_translations(
    name: str,
    predicted_translations: np.ndarray,
    predicted_rotations: np.ndarray,
    gt_absolute: np.ndarray,
) -> np.ndarray:
    if name == "as-is":
        return predicted_translations
    if name == "negated":
        return -predicted_translations
    if name == "next-to-current":
        return np.einsum("nij,nj->ni", predicted_rotations, predicted_translations)
    if name == "current-to-next":
        return np.einsum(
            "nji,nj->ni",
            predicted_rotations,
            predicted_translations,
        )
    if name == "world-to-current":
        current_world_rotations = gt_absolute[:-1, :3, :3]
        return np.einsum(
            "nji,nj->ni",
            current_world_rotations,
            predicted_translations,
        )
    raise ValueError(f"Unsupported translation hypothesis: {name}")


def axis_hypotheses() -> Dict[str, np.ndarray]:
    return {
        "xyz": np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        "-x_y_z": np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        "x_-y_z": np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64),
        "x_y_-z": np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
        "x_z_y": np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
        "z_y_x": np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    }


def stage_prediction_hypotheses(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    matrix, names = load_prediction_components(args)

    pred_angles = select_columns(
        matrix,
        names,
        parse_columns(args.prediction_rotation_cols),
        "predicted rotation",
    )
    predicted_rotations = rotations_from_angles(args, pred_angles)

    predicted_translations = try_select_columns(
        matrix,
        names,
        args.prediction_translation_cols,
        "predicted decoded translation",
    )

    if predicted_translations is None:
        predicted_dct = select_columns(
            matrix,
            names,
            parse_columns(args.prediction_dct_translation_cols),
            "predicted DCT translation",
        )
        context = build_dct_context(args)
        predicted_translations = decode_dct_batch(
            predicted_dct,
            predicted_rotations,
            context,
        )

    relative_gt = derive_relative_poses(absolute_poses, args.relative_direction)
    relative_gt, predicted_rotations, predicted_translations = align_relative_count(
        relative_gt,
        predicted_rotations,
        predicted_translations,
    )
    absolute_reference = absolute_poses[: len(relative_gt) + 1]

    candidates: Dict[str, np.ndarray] = {}

    base_translation_modes = (
        "as-is",
        "negated",
        "next-to-current",
        "current-to-next",
        "world-to-current",
    )

    for translation_mode in base_translation_modes:
        translations = hypothesis_translations(
            translation_mode,
            predicted_translations,
            predicted_rotations,
            absolute_reference,
        )

        relative = transforms_from_rotation_translation(
            predicted_rotations,
            translations,
        )
        candidates[f"{translation_mode}/right"] = integrate_relative_poses(
            relative,
            "right",
        )
        candidates[f"{translation_mode}/right-inverse"] = integrate_relative_poses(
            relative,
            "right-inverse",
        )

    for axis_name, axis_matrix in axis_hypotheses().items():
        translations = np.einsum(
            "ij,nj->ni",
            axis_matrix,
            predicted_translations,
        )
        rotations = np.einsum(
            "ij,njk,kl->nil",
            axis_matrix,
            predicted_rotations,
            axis_matrix.T,
        )
        relative = transforms_from_rotation_translation(rotations, translations)
        candidates[f"axis-{axis_name}/right"] = integrate_relative_poses(
            relative,
            "right",
        )

    metrics = {
        name: trajectory_metrics(absolute_reference, trajectory)
        for name, trajectory in candidates.items()
    }
    print_metric_table("Prediction-frame hypotheses", metrics)

    best_name = min(metrics, key=lambda key: metrics[key].translation_rmse)
    best_trajectory = candidates[best_name]

    gt_translations = relative_gt[:, :3, 3]
    cosines = translation_cosines(predicted_translations, gt_translations)
    ratios = norm_ratios(predicted_translations, gt_translations)

    write_step_diagnostics(
        args,
        relative_gt,
        predicted_rotations,
        predicted_translations,
    )

    for name, trajectory in candidates.items():
        save_pose_file(
            args.output_dir / f"prediction_hypothesis_{name.replace('/', '_')}.txt",
            trajectory,
        )

    if args.save_plots:
        plot_trajectory(
            args.output_dir / "10_candidate_frame_hypotheses_xz.png",
            {
                "ground truth": absolute_reference,
                "as-is/right": candidates["as-is/right"],
                "as-is/right-inverse": candidates["as-is/right-inverse"],
                f"best: {best_name}": best_trajectory,
            },
            plane="xz",
        )
        plot_trajectory(
            args.output_dir / "10_candidate_frame_hypotheses_xy.png",
            {
                "ground truth": absolute_reference,
                "as-is/right": candidates["as-is/right"],
                "as-is/right-inverse": candidates["as-is/right-inverse"],
                f"best: {best_name}": best_trajectory,
            },
            plane="xy",
        )

    result = {
        "best_hypothesis": best_name,
        "translation_cosine": finite_summary(cosines),
        "translation_norm_ratio": finite_summary(ratios),
        "hypotheses": {name: value.as_dict() for name, value in metrics.items()},
        "passed": True,
        "note": (
            "This exploratory stage does not assert a pass/fail threshold. "
            "Select a convention only after deterministic GT checks pass."
        ),
    }

    print(f"\nBest prediction hypothesis by translation RMSE: {best_name}")
    print(
        "Raw decoded translation cosine mean: "
        f"{result['translation_cosine'].get('mean', float('nan')):.6f}"
    )
    return result, True

def stage_verify_prediction_reconstruction(
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    """
    Compare the evaluator-exported predicted trajectory against:

    1. The evaluator's current reconstruction:
           raw predicted directional translation tc is inserted directly
           into the relative SE(3) transform.

    2. The corrected reconstruction:
           tc is decoded into the current-camera-frame relative translation
           before constructing the relative SE(3) transform.

    The expected audit outcome is:

        evaluator-style reconstruction
            matches predicted_trajectory.txt

        corrected directional reconstruction
            differs from predicted_trajectory.txt

    If so, this proves that evaluate_deepdct_vo.py exported and evaluated
    raw directional translations as ordinary relative translations.
    """
    if args.prediction_file is None:
        raise ValueError(
            "--prediction-file is required for stage "
            "'verify-prediction-reconstruction'."
        )

    if args.predicted_trajectory_file is None:
        raise ValueError(
            "--predicted-trajectory-file is required for stage "
            "'verify-prediction-reconstruction'."
        )

    # ---------------------------------------------------------
    # Load raw evaluator predictions.
    # ---------------------------------------------------------
    table, names = load_prediction_components(args)

    predicted_rotation_angles = select_columns(
        table,
        names,
        parse_columns(args.prediction_rotation_cols),
        "predicted relative Euler rotation",
    )

    predicted_directional_translations = select_columns(
        table,
        names,
        parse_columns(args.prediction_dct_translation_cols),
        "predicted directional translation",
    )

    predicted_relative_rotations = rotations_from_angles(
        args,
        predicted_rotation_angles,
    )

    # ---------------------------------------------------------
    # Load the evaluator-exported trajectory.
    # ---------------------------------------------------------
    exported_trajectory = load_kitti_poses(
        args.predicted_trajectory_file
    )

    # The evaluator starts at identity. Normalize defensively in case
    # the exported file was transformed after evaluation.
    exported_trajectory = normalize_poses(exported_trajectory)

    transition_count = min(
        len(predicted_relative_rotations),
        len(predicted_directional_translations),
        len(exported_trajectory) - 1,
    )

    if transition_count <= 0:
        raise ValueError(
            "No overlapping prediction transitions are available."
        )

    predicted_relative_rotations = (
        predicted_relative_rotations[:transition_count]
    )
    predicted_directional_translations = (
        predicted_directional_translations[:transition_count]
    )
    exported_trajectory = exported_trajectory[
        : transition_count + 1
    ]

    # ---------------------------------------------------------
    # Reconstruction A: reproduce the evaluator's current logic.
    #
    # evaluate_deepdct_vo.py currently does:
    #
    #     transform[:3, 3] = translation_pred
    #
    # even though translation_pred is directional translation tc.
    # ---------------------------------------------------------
    evaluator_style_relative = (
        transforms_from_rotation_translation(
            predicted_relative_rotations,
            predicted_directional_translations,
        )
    )

    evaluator_style_trajectory = integrate_relative_poses(
        evaluator_style_relative,
        args.composition_rule,
    )

    # ---------------------------------------------------------
    # Reconstruction B: correctly decode directional translation.
    #
    # Predicted absolute orientation is propagated recursively:
    #
    #     R_j_pred = R_i_pred @ R_rel_pred
    #
    # Then tc is converted to the translation component of:
    #
    #     T_Ci_Cj = inv(T_W_Ci) @ T_W_Cj
    # ---------------------------------------------------------
    corrected_relative_transforms: List[np.ndarray] = []

    current_predicted_rotation = np.eye(
        3,
        dtype=np.float64,
    )

    for index in range(transition_count):
        relative_rotation = project_to_so3(
            predicted_relative_rotations[index]
        )

        next_predicted_rotation = project_to_so3(
            current_predicted_rotation @ relative_rotation
        )

        relative_translation = (
            decode_relative_translation_current_frame(
                predicted_directional_translations[index],
                current_predicted_rotation,
                next_predicted_rotation,
            )
        )

        relative_transform = np.eye(
            4,
            dtype=np.float64,
        )
        relative_transform[:3, :3] = relative_rotation
        relative_transform[:3, 3] = relative_translation

        corrected_relative_transforms.append(
            relative_transform
        )

        current_predicted_rotation = next_predicted_rotation

    corrected_relative = np.stack(
        corrected_relative_transforms,
        axis=0,
    )

    corrected_trajectory = integrate_relative_poses(
        corrected_relative,
        args.composition_rule,
    )

    # ---------------------------------------------------------
    # Extract increments from the exported trajectory.
    # ---------------------------------------------------------
    exported_relative = derive_relative_poses(
        exported_trajectory,
        direction="forward",
    )

    # ---------------------------------------------------------
    # Pose-wise trajectory comparisons.
    # ---------------------------------------------------------
    evaluator_translation_differences = np.linalg.norm(
        evaluator_style_trajectory[:, :3, 3]
        - exported_trajectory[:, :3, 3],
        axis=1,
    )

    corrected_translation_differences = np.linalg.norm(
        corrected_trajectory[:, :3, 3]
        - exported_trajectory[:, :3, 3],
        axis=1,
    )

    evaluator_rotation_differences = np.asarray(
        [
            rotation_geodesic_deg(
                evaluator_style_trajectory[index, :3, :3],
                exported_trajectory[index, :3, :3],
            )
            for index in range(len(exported_trajectory))
        ],
        dtype=np.float64,
    )

    corrected_rotation_differences = np.asarray(
        [
            rotation_geodesic_deg(
                corrected_trajectory[index, :3, :3],
                exported_trajectory[index, :3, :3],
            )
            for index in range(len(exported_trajectory))
        ],
        dtype=np.float64,
    )

    evaluator_transform_differences = np.max(
        np.abs(
            evaluator_style_trajectory
            - exported_trajectory
        ),
        axis=(1, 2),
    )

    corrected_transform_differences = np.max(
        np.abs(
            corrected_trajectory
            - exported_trajectory
        ),
        axis=(1, 2),
    )

    # ---------------------------------------------------------
    # Increment-wise comparisons.
    # ---------------------------------------------------------
    evaluator_increment_translation_differences = np.linalg.norm(
        evaluator_style_relative[:, :3, 3]
        - exported_relative[:, :3, 3],
        axis=1,
    )

    corrected_increment_translation_differences = np.linalg.norm(
        corrected_relative[:, :3, 3]
        - exported_relative[:, :3, 3],
        axis=1,
    )

    evaluator_increment_rotation_differences = np.asarray(
        [
            rotation_geodesic_deg(
                evaluator_style_relative[index, :3, :3],
                exported_relative[index, :3, :3],
            )
            for index in range(transition_count)
        ],
        dtype=np.float64,
    )

    corrected_increment_rotation_differences = np.asarray(
        [
            rotation_geodesic_deg(
                corrected_relative[index, :3, :3],
                exported_relative[index, :3, :3],
            )
            for index in range(transition_count)
        ],
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # First mismatching pose for each reconstruction.
    # ---------------------------------------------------------
    tolerance = 1e-8

    evaluator_bad_indices = np.flatnonzero(
        evaluator_transform_differences > tolerance
    )

    corrected_bad_indices = np.flatnonzero(
        corrected_transform_differences > tolerance
    )

    first_evaluator_mismatch = (
        int(evaluator_bad_indices[0])
        if evaluator_bad_indices.size > 0
        else None
    )

    first_corrected_mismatch = (
        int(corrected_bad_indices[0])
        if corrected_bad_indices.size > 0
        else None
    )

    # ---------------------------------------------------------
    # Detailed CSV.
    # ---------------------------------------------------------
    save_csv(
        args.output_dir
        / "prediction_reconstruction_comparison.csv",
        [
            "pose_index",
            "exported_x",
            "exported_y",
            "exported_z",
            "evaluator_style_x",
            "evaluator_style_y",
            "evaluator_style_z",
            "corrected_x",
            "corrected_y",
            "corrected_z",
            "evaluator_translation_difference",
            "corrected_translation_difference",
            "evaluator_rotation_difference_deg",
            "corrected_rotation_difference_deg",
            "evaluator_transform_max_abs_difference",
            "corrected_transform_max_abs_difference",
        ],
        (
            (
                index,
                *exported_trajectory[index, :3, 3].tolist(),
                *evaluator_style_trajectory[
                    index, :3, 3
                ].tolist(),
                *corrected_trajectory[
                    index, :3, 3
                ].tolist(),
                evaluator_translation_differences[index],
                corrected_translation_differences[index],
                evaluator_rotation_differences[index],
                corrected_rotation_differences[index],
                evaluator_transform_differences[index],
                corrected_transform_differences[index],
            )
            for index in range(len(exported_trajectory))
        ),
    )

    save_csv(
        args.output_dir
        / "prediction_increment_comparison.csv",
        [
            "step",
            "raw_tc_x",
            "raw_tc_y",
            "raw_tc_z",
            "exported_relative_tx",
            "exported_relative_ty",
            "exported_relative_tz",
            "evaluator_relative_tx",
            "evaluator_relative_ty",
            "evaluator_relative_tz",
            "corrected_relative_tx",
            "corrected_relative_ty",
            "corrected_relative_tz",
            "evaluator_translation_difference",
            "corrected_translation_difference",
            "evaluator_rotation_difference_deg",
            "corrected_rotation_difference_deg",
        ],
        (
            (
                index,
                *predicted_directional_translations[index].tolist(),
                *exported_relative[index, :3, 3].tolist(),
                *evaluator_style_relative[
                    index, :3, 3
                ].tolist(),
                *corrected_relative[
                    index, :3, 3
                ].tolist(),
                evaluator_increment_translation_differences[index],
                corrected_increment_translation_differences[index],
                evaluator_increment_rotation_differences[index],
                corrected_increment_rotation_differences[index],
            )
            for index in range(transition_count)
        ),
    )

    save_pose_file(
        args.output_dir
        / "evaluator_style_reconstructed_trajectory.txt",
        evaluator_style_trajectory,
    )

    save_pose_file(
        args.output_dir
        / "corrected_directional_reconstructed_trajectory.txt",
        corrected_trajectory,
    )

    if args.save_plots:
        plot_trajectory(
            args.output_dir
            / "prediction_reconstruction_comparison_xz.png",
            {
                "evaluator export": exported_trajectory,
                "evaluator-style reconstruction": (
                    evaluator_style_trajectory
                ),
                "corrected directional reconstruction": (
                    corrected_trajectory
                ),
            },
            plane="xz",
        )

        plot_trajectory(
            args.output_dir
            / "prediction_reconstruction_comparison_xy.png",
            {
                "evaluator export": exported_trajectory,
                "evaluator-style reconstruction": (
                    evaluator_style_trajectory
                ),
                "corrected directional reconstruction": (
                    corrected_trajectory
                ),
            },
            plane="xy",
        )

    evaluator_matches_export = bool(
        float(np.max(evaluator_transform_differences))
        < tolerance
    )

    corrected_matches_export = bool(
        float(np.max(corrected_transform_differences))
        < tolerance
    )

    result: Dict[str, object] = {
        "transitions": int(transition_count),
        "comparison_tolerance": float(tolerance),
        "evaluator_style_vs_export": {
            "translation_difference": summarize_vector(
                evaluator_translation_differences
            ),
            "rotation_difference_deg": summarize_vector(
                evaluator_rotation_differences
            ),
            "transform_max_abs_difference": summarize_vector(
                evaluator_transform_differences
            ),
            "first_mismatch_pose": first_evaluator_mismatch,
            "matches_export": evaluator_matches_export,
        },
        "corrected_directional_vs_export": {
            "translation_difference": summarize_vector(
                corrected_translation_differences
            ),
            "rotation_difference_deg": summarize_vector(
                corrected_rotation_differences
            ),
            "transform_max_abs_difference": summarize_vector(
                corrected_transform_differences
            ),
            "first_mismatch_pose": first_corrected_mismatch,
            "matches_export": corrected_matches_export,
        },
        "increment_comparison": {
            "evaluator_translation_difference": summarize_vector(
                evaluator_increment_translation_differences
            ),
            "corrected_translation_difference": summarize_vector(
                corrected_increment_translation_differences
            ),
            "evaluator_rotation_difference_deg": summarize_vector(
                evaluator_increment_rotation_differences
            ),
            "corrected_rotation_difference_deg": summarize_vector(
                corrected_increment_rotation_differences
            ),
        },
    }

    # This stage passes when the exported trajectory is reproduced by
    # the evaluator's current implementation. That confirms the audit
    # diagnosis. It does not mean the implementation is geometrically
    # correct.
    passed = bool(
        corrected_matches_export
        and not evaluator_matches_export
    )
    result["passed"] = passed

    print()
    print("Prediction reconstruction audit")
    print("===============================")
    print(f"Transitions:                         {transition_count}")
    print(
        "Evaluator-style/export max diff:    "
        f"{np.max(evaluator_transform_differences):.6e}"
    )
    print(
        "Corrected/export max diff:          "
        f"{np.max(corrected_transform_differences):.6e}"
    )
    print(
        "First evaluator-style mismatch:     "
        f"{first_evaluator_mismatch}"
    )
    print(
        "First corrected-path mismatch:      "
        f"{first_corrected_mismatch}"
    )
    print(
        "Evaluator-style matches export:     "
        f"{evaluator_matches_export}"
    )
    print(
        "Corrected decode matches export:    "
        f"{corrected_matches_export}"
    )

    if corrected_matches_export and not evaluator_matches_export:
        print()
        print(
            "AUDIT FINDING: evaluate_deepdct_vo.py exported the "
            "trajectory obtained by decoding directional translation "
            "before constructing and composing relative SE(3) poses."
        )
    elif evaluator_matches_export and not corrected_matches_export:
        print()
        print(
            "AUDIT FINDING: evaluate_deepdct_vo.py still exported the "
            "trajectory obtained by inserting directional translation "
            "tc directly into the relative SE(3) transform."
        )

    print(
        "Prediction reconstruction status:   "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return result, passed


def run_stage(
    stage: str,
    args: argparse.Namespace,
    absolute_poses: np.ndarray,
) -> Tuple[Dict[str, object], bool]:
    if stage == "relative":
        return stage_relative(args, absolute_poses)
    if stage == "rotation":
        return stage_rotation(args, absolute_poses)
    if stage == "dct-roundtrip":
        return stage_dct_roundtrip(args, absolute_poses)
    if stage == "reconstruct-ground-truth":
        return stage_reconstruct_ground_truth(args, absolute_poses)
    if stage == "teacher-forcing":
        return stage_teacher_forcing(args, absolute_poses)
    if stage == "prediction-hypotheses":
        return stage_prediction_hypotheses(args, absolute_poses)
    if stage == "rotation-labels":
        return stage_rotation_labels(args, absolute_poses)
    if stage == "verify-prediction-reconstruction":
        return stage_verify_prediction_reconstruction(
            args,
            absolute_poses,
        )
    raise ValueError(f"Unknown stage: {stage}")



def validate_required_files(stage: str, args: argparse.Namespace) -> None:
    if stage == "verify-prediction-reconstruction":
        if args.prediction_file is None:
            raise ValueError(
                "--prediction-file is required for stage "
                "'verify-prediction-reconstruction'."
            )

        if args.predicted_trajectory_file is None:
            raise ValueError(
                "--predicted-trajectory-file is required for stage "
                "'verify-prediction-reconstruction'."
            )
    if stage in ("dct-roundtrip", "reconstruct-ground-truth") and args.dct_file is None:
        raise ValueError(f"--dct-file is required for stage {stage!r}.")
    if stage in ("teacher-forcing", "prediction-hypotheses") and args.prediction_file is None:
        raise ValueError(f"--prediction-file is required for stage {stage!r}.")
    if stage in (
        "dct-roundtrip",
        "reconstruct-ground-truth",
        "rotation-labels",
    ) and args.dct_file is None:
        raise ValueError(
            f"--dct-file is required for stage {stage!r}."
        )



def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    absolute_poses = load_kitti_poses(args.pose_file)
    if args.normalize_first_pose:
        absolute_poses = normalize_poses(absolute_poses)

    if args.num_samples is not None:
        # num-samples counts relative transitions, so retain N+1 absolute poses.
        absolute_poses = absolute_poses[: args.num_samples + 1]

    print("=" * 80)
    print("Relative-pose coordinate-frame verification")
    print("=" * 80)
    print(f"Pose file:               {args.pose_file}")
    print(f"Absolute poses:          {len(absolute_poses)}")
    print(f"Relative transitions:    {max(0, len(absolute_poses) - 1)}")
    print(f"Normalize first pose:    {args.normalize_first_pose}")
    print(f"Euler convention:        {args.euler_order}/{args.euler_convention}")
    print(f"Relative direction:      {args.relative_direction}")
    print(f"Composition rule:        {args.composition_rule}")
    print(f"Output directory:        {args.output_dir}")

    stages = (
        [
            "relative",
            "rotation",
            "dct-roundtrip",
            "reconstruct-ground-truth",
            "teacher-forcing",
            "prediction-hypotheses",
            "rotation-labels",
            "verify-prediction-reconstruction",
        ]
        if args.stage == "all"
        else [args.stage]
    )

    aggregate: Dict[str, object] = {
        "configuration": {
            "pose_file": str(args.pose_file),
            "dct_file": str(args.dct_file) if args.dct_file else None,
            "prediction_file": (
                str(args.prediction_file) if args.prediction_file else None
            ),
            "absolute_pose_count": len(absolute_poses),
            "relative_pose_count": max(0, len(absolute_poses) - 1),
            "normalize_first_pose": args.normalize_first_pose,
            "relative_direction": args.relative_direction,
            "composition_rule": args.composition_rule,
            "euler_order": args.euler_order,
            "euler_convention": args.euler_convention,
            "angles_in_degrees": args.angles_in_degrees,
        },
        "stages": {},
    }

    all_passed = True
    for stage in stages:
        validate_required_files(stage, args)
        try:
            result, passed = run_stage(stage, args, absolute_poses)
        except NotImplementedError as exc:
            print(f"\nStage {stage!r} is not implemented for this convention: {exc}")
            aggregate["stages"][stage] = {
                "passed": False,
                "error": str(exc),
            }
            all_passed = False
            continue
        aggregate["stages"][stage] = result
        all_passed = all_passed and passed

    aggregate["all_deterministic_checks_passed"] = all_passed
    summary_path = args.output_dir / "verification_summary.json"
    save_json(summary_path, aggregate)

    print()
    print("=" * 80)
    print(f"Summary written to: {summary_path}")
    print(f"Overall deterministic status: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 80)

    if args.fail_on_threshold and not all_passed:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)