#!/usr/bin/env python3
"""
Audit DeepDCT-VO / KITTI rotation-axis and Euler-angle conventions.

Purpose
-------
Determine empirically which Euler component corresponds to vehicle
heading/yaw for the KITTI pose convention used by DeepDCT-VO.

The audit is GT-only. It does NOT depend on model predictions.

For each consecutive pair of KITTI poses:

    T_i = [R_i | t_i]
    T_j = [R_j | t_j]

we compute the relative camera rotation:

    R_rel = R_i.T @ R_j

assuming KITTI poses map camera coordinates into the world frame.

We then:

1. Convert R_rel -> Euler xyz using scipy's lowercase "xyz",
   i.e. EXTRINSIC xyz rotations.

2. Reconstruct R_rel from those Euler angles and measure the SO(3)
   round-trip error.

3. Compute physical vehicle heading from the camera forward vector:

       f_world = R_world_camera @ [0, 0, 1]

   projected onto the world x-z ground plane.

4. Compare physical pairwise heading change against:
       Euler x
       Euler y
       Euler z

5. Independently calculate a current-camera-frame turning angle from
   the transformed next-camera forward vector:

       f_next_in_current = R_rel @ [0, 0, 1]

       turn_angle = atan2(f_x, f_z)

   This should be strongly related to rotation about camera y.

Expected KITTI camera convention
--------------------------------
    x : right
    y : down
    z : forward

Therefore ordinary vehicle left/right yaw should primarily correspond
to rotation about camera y, not camera z.

Typical command
---------------

python scripts/audit_rotation_axis_convention.py \
    --poses data/poses/10.txt \
    --sequence 10 \
    --output-dir experiments/rotation_axis_convention_audit/sequence_10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise ImportError(
        "This audit requires scipy.\n"
        "Install it with:\n"
        "  pip install scipy"
    ) from exc


AXIS_NAMES = (
    "x",
    "y",
    "z",
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit KITTI / DeepDCT-VO rotation axis and "
            "Euler convention."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help=(
            "KITTI absolute pose file. Each row must contain "
            "the flattened 3x4 camera-to-world pose matrix."
        ),
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default="unknown",
        help="Sequence label used only in output/reporting.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--euler-order",
        type=str,
        default="xyz",
        choices=("xyz",),
        help=(
            "DeepDCT-VO convention being audited. Lowercase scipy "
            "'xyz' means extrinsic xyz."
        ),
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help=(
            "Write per-frame Euler/heading values in degrees. "
            "Summary statistics are always also reported in degrees."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Utilities
# ============================================================================


def wrap_angle_radians(
    angle: np.ndarray,
) -> np.ndarray:
    return (
        angle
        + np.pi
    ) % (
        2.0
        * np.pi
    ) - np.pi


def safe_corr(
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
        a.size < 2
        or b.size < 2
        or np.std(a) <= 1.0e-15
        or np.std(b) <= 1.0e-15
    ):
        return float("nan")

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def rms(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def rotation_angle_radians(
    rotation_matrix: np.ndarray,
) -> float:
    """
    Geodesic SO(3) angle from a proper rotation matrix.
    """

    trace_value = float(
        np.trace(
            rotation_matrix
        )
    )

    cosine = (
        trace_value
        - 1.0
    ) / 2.0

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.arccos(
            cosine
        )
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
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        value = float(value)

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
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================================
# Pose loading
# ============================================================================


def load_kitti_poses(
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Pose file does not exist: {path}"
        )

    raw = np.loadtxt(
        path,
        dtype=np.float64,
    )

    if raw.ndim == 1:
        raw = raw[
            None,
            :
        ]

    if (
        raw.ndim != 2
        or raw.shape[1] != 12
    ):
        raise ValueError(
            "Expected KITTI pose rows containing 12 values "
            "(flattened 3x4 matrices), received "
            f"{raw.shape}."
        )

    transforms = np.repeat(
        np.eye(
            4,
            dtype=np.float64,
        )[None, :, :],
        raw.shape[0],
        axis=0,
    )

    transforms[
        :,
        :3,
        :4,
    ] = raw.reshape(
        -1,
        3,
        4,
    )

    return transforms


# ============================================================================
# Rotation checks
# ============================================================================


def audit_absolute_rotations(
    transforms: np.ndarray,
) -> Dict[str, float]:
    orthogonality_errors: List[
        float
    ] = []

    determinant_errors: List[
        float
    ] = []

    identity = np.eye(
        3,
        dtype=np.float64,
    )

    for transform in transforms:
        rotation = transform[
            :3,
            :3,
        ]

        orthogonality_errors.append(
            float(
                np.linalg.norm(
                    rotation.T
                    @ rotation
                    - identity,
                    ord="fro",
                )
            )
        )

        determinant_errors.append(
            abs(
                float(
                    np.linalg.det(
                        rotation
                    )
                )
                - 1.0
            )
        )

    return {
        "mean_orthogonality_error": float(
            np.mean(
                orthogonality_errors
            )
        ),
        "max_orthogonality_error": float(
            np.max(
                orthogonality_errors
            )
        ),
        "mean_determinant_error": float(
            np.mean(
                determinant_errors
            )
        ),
        "max_determinant_error": float(
            np.max(
                determinant_errors
            )
        ),
    }


def compute_relative_rotations(
    transforms: np.ndarray,
) -> np.ndarray:
    relative: List[
        np.ndarray
    ] = []

    for index in range(
        transforms.shape[0] - 1
    ):
        current_rotation = transforms[
            index,
            :3,
            :3,
        ]

        next_rotation = transforms[
            index + 1,
            :3,
            :3,
        ]

        # KITTI absolute poses are treated as camera -> world.
        #
        # Therefore the rotation taking vectors from the NEXT
        # camera orientation relative to the CURRENT camera is:
        #
        #     R_rel = R_i^T R_j
        #
        relative_rotation = (
            current_rotation.T
            @ next_rotation
        )

        relative.append(
            relative_rotation
        )

    return np.stack(
        relative,
        axis=0,
    )


def euler_round_trip_audit(
    relative_rotations: np.ndarray,
    order: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    euler = Rotation.from_matrix(
        relative_rotations
    ).as_euler(
        order,
        degrees=False,
    )

    reconstructed = Rotation.from_euler(
        order,
        euler,
        degrees=False,
    ).as_matrix()

    errors: List[
        float
    ] = []

    for original, recovered in zip(
        relative_rotations,
        reconstructed,
    ):
        residual = (
            original.T
            @ recovered
        )

        errors.append(
            rotation_angle_radians(
                residual
            )
        )

    return (
        euler,
        np.asarray(
            errors,
            dtype=np.float64,
        ),
    )


# ============================================================================
# Physical heading calculations
# ============================================================================


def absolute_forward_headings(
    transforms: np.ndarray,
) -> np.ndarray:
    """
    Calculate camera-forward heading in the world x-z plane.

    KITTI camera forward axis:
        +z

    Pose rotation:
        camera -> world

    Therefore:
        f_world = R_wc @ [0, 0, 1]

    Heading convention here:
        atan2(world_x, world_z)

    Positive/negative sign is less important than identifying which
    relative Euler coordinate tracks it.
    """

    camera_forward = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )

    headings: List[
        float
    ] = []

    for transform in transforms:
        rotation = transform[
            :3,
            :3,
        ]

        forward_world = (
            rotation
            @ camera_forward
        )

        heading = math.atan2(
            float(
                forward_world[0]
            ),
            float(
                forward_world[2]
            ),
        )

        headings.append(
            heading
        )

    return np.asarray(
        headings,
        dtype=np.float64,
    )


def pairwise_world_heading_change(
    transforms: np.ndarray,
) -> np.ndarray:
    """
    Signed heading change measured directly from consecutive
    camera forward directions in the world x-z plane.

    This is intentionally independent of Euler decomposition.
    """

    headings = absolute_forward_headings(
        transforms
    )

    return wrap_angle_radians(
        np.diff(
            headings
        )
    )


def relative_forward_turn_angle(
    relative_rotations: np.ndarray,
) -> np.ndarray:
    """
    Independent current-camera-frame turning measure.

    Apply R_rel to the forward vector [0,0,1]:

        f_next = R_rel @ z_hat

    Project onto current camera x-z plane and calculate:

        atan2(f_next_x, f_next_z)

    For predominantly planar KITTI vehicle motion this should
    correspond closely, up to sign/convention, to camera-y yaw.
    """

    camera_forward = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )

    angles: List[
        float
    ] = []

    for relative_rotation in relative_rotations:
        next_forward = (
            relative_rotation
            @ camera_forward
        )

        angle = math.atan2(
            float(
                next_forward[0]
            ),
            float(
                next_forward[2]
            ),
        )

        angles.append(
            angle
        )

    return np.asarray(
        angles,
        dtype=np.float64,
    )


# ============================================================================
# Statistical comparison
# ============================================================================


def axis_comparison(
    euler: np.ndarray,
    reference: np.ndarray,
) -> List[
    Dict[str, float]
]:
    rows: List[
        Dict[str, float]
    ] = []

    for axis_index, axis_name in enumerate(
        AXIS_NAMES
    ):
        values = euler[
            :,
            axis_index,
        ]

        correlation = safe_corr(
            values,
            reference,
        )

        # Fit:
        #
        #     reference ~= slope * Euler_axis + intercept
        #
        design = np.column_stack(
            [
                values,
                np.ones_like(
                    values
                ),
            ]
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            reference,
            rcond=None,
        )

        slope = float(
            coefficients[0]
        )

        intercept = float(
            coefficients[1]
        )

        fitted = (
            slope
            * values
            + intercept
        )

        fit_rmse = rms(
            fitted
            - reference
        )

        rows.append(
            {
                "axis": axis_name,
                "correlation": correlation,
                "absolute_correlation": abs(
                    correlation
                )
                if math.isfinite(
                    correlation
                )
                else float(
                    "nan"
                ),
                "linear_slope": slope,
                "linear_intercept_rad": intercept,
                "fit_rmse_rad": fit_rmse,
                "fit_rmse_deg": float(
                    np.degrees(
                        fit_rmse
                    )
                ),
                "axis_std_rad": float(
                    np.std(
                        values
                    )
                ),
                "axis_std_deg": float(
                    np.degrees(
                        np.std(
                            values
                        )
                    )
                ),
            }
        )

    return rows


def best_axis(
    rows: List[
        Dict[str, float]
    ],
) -> Dict[str, float]:
    finite_rows = [
        row
        for row in rows
        if math.isfinite(
            row[
                "absolute_correlation"
            ]
        )
    ]

    if not finite_rows:
        raise RuntimeError(
            "Unable to determine best-correlated rotation axis."
        )

    return max(
        finite_rows,
        key=lambda row: row[
            "absolute_correlation"
        ],
    )


# ============================================================================
# Outputs
# ============================================================================


def write_csv(
    path: Path,
    rows: List[
        Dict[str, Any]
    ],
) -> None:
    if not rows:
        return

    fieldnames: List[
        str
    ] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def plot_axis_vs_heading(
    euler: np.ndarray,
    heading_change: np.ndarray,
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(
            10,
            6,
        )
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    frame_pairs = np.arange(
        euler.shape[0]
    )

    axis.plot(
        frame_pairs,
        np.degrees(
            heading_change
        ),
        label="physical heading change",
        linewidth=1.3,
    )

    for axis_index, axis_name in enumerate(
        AXIS_NAMES
    ):
        axis.plot(
            frame_pairs,
            np.degrees(
                euler[
                    :,
                    axis_index,
                ]
            ),
            label=f"Euler {axis_name}",
            alpha=0.75,
        )

    axis.set_xlabel(
        "Frame-pair index"
    )

    axis.set_ylabel(
        "Angle (deg)"
    )

    axis.set_title(
        "Physical heading change vs relative Euler components"
    )

    axis.legend()

    axis.grid(
        True
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        figure
    )


def plot_heading_scatter(
    euler: np.ndarray,
    heading_change: np.ndarray,
    output_dir: Path,
) -> None:
    for axis_index, axis_name in enumerate(
        AXIS_NAMES
    ):
        figure = plt.figure(
            figsize=(
                6,
                6,
            )
        )

        axis = figure.add_subplot(
            1,
            1,
            1,
        )

        axis.scatter(
            np.degrees(
                euler[
                    :,
                    axis_index,
                ]
            ),
            np.degrees(
                heading_change
            ),
            s=7,
            alpha=0.4,
        )

        corr = safe_corr(
            euler[
                :,
                axis_index,
            ],
            heading_change,
        )

        axis.set_xlabel(
            f"Relative Euler {axis_name} (deg)"
        )

        axis.set_ylabel(
            "Physical heading change (deg)"
        )

        axis.set_title(
            f"Heading vs Euler {axis_name}: r={corr:+.4f}"
        )

        axis.grid(
            True
        )

        figure.tight_layout()

        figure.savefig(
            output_dir
            / f"heading_vs_euler_{axis_name}.png",
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

    args.poses = (
        args.poses
        .expanduser()
        .resolve()
    )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
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

    transforms = load_kitti_poses(
        args.poses
    )

    rotation_quality = (
        audit_absolute_rotations(
            transforms
        )
    )

    relative_rotations = (
        compute_relative_rotations(
            transforms
        )
    )

    euler, round_trip_error = (
        euler_round_trip_audit(
            relative_rotations,
            args.euler_order,
        )
    )

    world_heading_change = (
        pairwise_world_heading_change(
            transforms
        )
    )

    camera_turn_angle = (
        relative_forward_turn_angle(
            relative_rotations
        )
    )

    if not (
        euler.shape[0]
        == world_heading_change.shape[0]
        == camera_turn_angle.shape[0]
    ):
        raise RuntimeError(
            "Internal transition-count mismatch."
        )

    world_rows = axis_comparison(
        euler,
        world_heading_change,
    )

    camera_rows = axis_comparison(
        euler,
        camera_turn_angle,
    )

    best_world = best_axis(
        world_rows
    )

    best_camera = best_axis(
        camera_rows
    )

    heading_agreement_corr = safe_corr(
        world_heading_change,
        camera_turn_angle,
    )

    # ------------------------------------------------------------------
    # Per-frame export.
    # ------------------------------------------------------------------

    frame_rows: List[
        Dict[str, Any]
    ] = []

    scale = (
        180.0
        / np.pi
        if args.angles_in_degrees
        else 1.0
    )

    unit = (
        "deg"
        if args.angles_in_degrees
        else "rad"
    )

    for index in range(
        euler.shape[0]
    ):
        frame_rows.append(
            {
                "transition_index": index,
                "frame_prev": index,
                "frame_curr": index + 1,
                f"euler_x_{unit}": float(
                    euler[
                        index,
                        0,
                    ]
                    * scale
                ),
                f"euler_y_{unit}": float(
                    euler[
                        index,
                        1,
                    ]
                    * scale
                ),
                f"euler_z_{unit}": float(
                    euler[
                        index,
                        2,
                    ]
                    * scale
                ),
                f"world_heading_change_{unit}": float(
                    world_heading_change[
                        index
                    ]
                    * scale
                ),
                f"camera_forward_turn_{unit}": float(
                    camera_turn_angle[
                        index
                    ]
                    * scale
                ),
                "round_trip_so3_error_deg": float(
                    np.degrees(
                        round_trip_error[
                            index
                        ]
                    )
                ),
            }
        )

    write_csv(
        args.output_dir
        / "per_transition_axis_audit.csv",
        frame_rows,
    )

    write_csv(
        args.output_dir
        / "world_heading_axis_correlations.csv",
        world_rows,
    )

    write_csv(
        args.output_dir
        / "camera_turn_axis_correlations.csv",
        camera_rows,
    )

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------

    plot_axis_vs_heading(
        euler=euler,
        heading_change=world_heading_change,
        output_path=(
            plots_dir
            / "physical_heading_vs_euler_components.png"
        ),
    )

    plot_heading_scatter(
        euler=euler,
        heading_change=world_heading_change,
        output_dir=plots_dir,
    )

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------

    world_best_axis = str(
        best_world[
            "axis"
        ]
    )

    camera_best_axis = str(
        best_camera[
            "axis"
        ]
    )

    if (
        world_best_axis == "y"
        and camera_best_axis == "y"
    ):
        interpretation = (
            "PASS: vehicle heading/turn is most strongly associated "
            "with Euler y. A diagnostic that treats Euler z as "
            "heading is using the wrong semantic axis. This does not "
            "by itself indicate a training-label or pose-reconstruction "
            "axis swap."
        )

    elif (
        world_best_axis
        == camera_best_axis
    ):
        interpretation = (
            "CHECK: both independent heading measures identify Euler "
            f"{world_best_axis} rather than y. Inspect the pose-frame "
            "and Euler-convention assumptions before modifying any "
            "training or evaluation code."
        )

    else:
        interpretation = (
            "WARNING: world-frame heading and camera-frame turning "
            "identify different Euler axes. This requires a deeper "
            "coordinate-frame/convention audit before changing the "
            "diagnostic script."
        )

    # ------------------------------------------------------------------
    # JSON summary.
    # ------------------------------------------------------------------

    summary = {
        "sequence": args.sequence,
        "pose_file": str(
            args.poses
        ),
        "absolute_poses": int(
            transforms.shape[0]
        ),
        "relative_transitions": int(
            relative_rotations.shape[0]
        ),
        "assumed_pose_mapping": (
            "camera_to_world"
        ),
        "camera_axes": {
            "x": "right",
            "y": "down",
            "z": "forward",
        },
        "euler_order": (
            args.euler_order
        ),
        "scipy_euler_semantics": (
            "lowercase xyz = extrinsic xyz"
        ),
        "absolute_rotation_quality": (
            rotation_quality
        ),
        "round_trip": {
            "mean_so3_error_deg": float(
                np.degrees(
                    np.mean(
                        round_trip_error
                    )
                )
            ),
            "max_so3_error_deg": float(
                np.degrees(
                    np.max(
                        round_trip_error
                    )
                )
            ),
        },
        "heading_measure_agreement": {
            "world_vs_camera_turn_corr": (
                heading_agreement_corr
            ),
        },
        "world_heading_axis_comparison": (
            world_rows
        ),
        "camera_turn_axis_comparison": (
            camera_rows
        ),
        "best_axis_world_heading": (
            best_world
        ),
        "best_axis_camera_turn": (
            best_camera
        ),
        "interpretation": (
            interpretation
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

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------

    print()
    print("=" * 96)
    print(
        "DeepDCT-VO / KITTI rotation axis convention audit"
    )
    print("=" * 96)

    print(
        f"Sequence:                 {args.sequence}"
    )

    print(
        f"Pose file:                {args.poses}"
    )

    print(
        f"Absolute poses:           {transforms.shape[0]}"
    )

    print(
        f"Relative transitions:     {relative_rotations.shape[0]}"
    )

    print(
        "Pose interpretation:      camera -> world"
    )

    print(
        "Camera coordinates:       x=right, y=down, z=forward"
    )

    print(
        f"Euler convention:         extrinsic {args.euler_order}"
    )

    print("=" * 96)

    print()
    print("=" * 96)
    print(
        "Raw KITTI rotation-matrix quality"
    )
    print("=" * 96)

    print(
        "Mean orthogonality error: "
        f"{rotation_quality['mean_orthogonality_error']:.12e}"
    )

    print(
        "Maximum orthogonality:    "
        f"{rotation_quality['max_orthogonality_error']:.12e}"
    )

    print(
        "Mean determinant error:   "
        f"{rotation_quality['mean_determinant_error']:.12e}"
    )

    print(
        "Maximum determinant:      "
        f"{rotation_quality['max_determinant_error']:.12e}"
    )

    print("=" * 96)

    print()
    print("=" * 96)
    print(
        "Euler round-trip reconstruction"
    )
    print("=" * 96)

    print(
        "Mean SO(3) reconstruction error: "
        f"{np.degrees(np.mean(round_trip_error)):.12e} deg"
    )

    print(
        "Maximum SO(3) reconstruction:    "
        f"{np.degrees(np.max(round_trip_error)):.12e} deg"
    )

    print("=" * 96)

    print()
    print("=" * 112)
    print(
        "Physical WORLD heading change vs relative Euler components"
    )
    print("=" * 112)

    print(
        f"{'Axis':<8}"
        f"{'Std deg':>14}"
        f"{'Correlation':>16}"
        f"{'|Correlation|':>16}"
        f"{'Slope':>14}"
        f"{'Fit RMSE deg':>16}"
    )

    print("-" * 112)

    for row in world_rows:
        print(
            f"{row['axis']:<8}"
            f"{row['axis_std_deg']:>14.6f}"
            f"{row['correlation']:>16.6f}"
            f"{row['absolute_correlation']:>16.6f}"
            f"{row['linear_slope']:>14.6f}"
            f"{row['fit_rmse_deg']:>16.6f}"
        )

    print("=" * 112)

    print()
    print("=" * 112)
    print(
        "CURRENT-camera forward-turn angle vs relative Euler components"
    )
    print("=" * 112)

    print(
        f"{'Axis':<8}"
        f"{'Std deg':>14}"
        f"{'Correlation':>16}"
        f"{'|Correlation|':>16}"
        f"{'Slope':>14}"
        f"{'Fit RMSE deg':>16}"
    )

    print("-" * 112)

    for row in camera_rows:
        print(
            f"{row['axis']:<8}"
            f"{row['axis_std_deg']:>14.6f}"
            f"{row['correlation']:>16.6f}"
            f"{row['absolute_correlation']:>16.6f}"
            f"{row['linear_slope']:>14.6f}"
            f"{row['fit_rmse_deg']:>16.6f}"
        )

    print("=" * 112)

    print()
    print("=" * 96)
    print(
        "Axis identification"
    )
    print("=" * 96)

    print(
        "World-heading best axis:  "
        f"{world_best_axis} "
        f"(corr={best_world['correlation']:+.6f})"
    )

    print(
        "Camera-turn best axis:    "
        f"{camera_best_axis} "
        f"(corr={best_camera['correlation']:+.6f})"
    )

    print(
        "World/camera heading agreement: "
        f"{heading_agreement_corr:+.6f}"
    )

    print()
    print(
        interpretation
    )

    print("=" * 96)

    print()
    print(
        f"Outputs saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()