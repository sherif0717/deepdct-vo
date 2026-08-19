#!/usr/bin/env python3
"""
Rotation-error audit for DeepDCT-VO.

Purpose
-------
Diagnose the rotation failure responsible for global trajectory divergence.

The script analyzes:

1. Per-axis relative rotation prediction:
       x / y / z
   including:
       - bias
       - MAE
       - RMSE
       - correlation
       - predicted / GT standard-deviation ratio

2. Geodesic SO(3) relative-rotation error:
       R_error = R_gt^T R_pred

3. Cumulative orientation drift:
       R_global[k+1] = R_global[k] R_relative[k]

4. Heading / yaw accumulation:
       especially the z Euler component for xyz convention.

5. Turn-regime behavior:
       straight / moderate / strong
   defined from training-independent quantiles of |GT z rotation|
   within the analyzed sequence.

6. Counterfactual trajectory:
       Predicted rotation + GT translation
   using the current evaluate_deepdct_vo.py trajectory reconstruction.

Expected input
--------------
Evaluator-generated frame_predictions.csv containing:

    rotation_gt_x
    rotation_gt_y
    rotation_gt_z

    rotation_pred_x
    rotation_pred_y
    rotation_pred_z

    translation_gt_x
    translation_gt_y
    translation_gt_z

Optional:

    frame_prev
    frame_curr
    sequence

Example
-------
python scripts/analyze_rotation_error.py \
    --predictions \
        experiments/semantic_depth_pooled_translation_head/evaluation_sequence_10/frame_predictions.csv \
    --output-dir \
        experiments/semantic_depth_pooled_translation_head/rotation_audit \
    --euler-order xyz

Notes
-----
- Euler angles are assumed to be radians unless --angles-in-degrees is used.
- For Euler order xyz, the z component is treated as the primary
  yaw / heading diagnostic.
- Geodesic SO(3) error is convention-independent and should be preferred
  over raw Euler subtraction when interpreting total rotation error.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError as error:
    raise ImportError(
        "This script requires scipy. Install it with:\n"
        "    pip install scipy"
    ) from error


# ============================================================================
# Constants
# ============================================================================

AXES = ("x", "y", "z")

ROTATION_GT_COLUMNS = (
    "rotation_gt_x",
    "rotation_gt_y",
    "rotation_gt_z",
)

ROTATION_PRED_COLUMNS = (
    "rotation_pred_x",
    "rotation_pred_y",
    "rotation_pred_z",
)

TRANSLATION_GT_COLUMNS = (
    "translation_gt_x",
    "translation_gt_y",
    "translation_gt_z",
)

REQUIRED_COLUMNS = (
    *ROTATION_GT_COLUMNS,
    *ROTATION_PRED_COLUMNS,
    *TRANSLATION_GT_COLUMNS,
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit DeepDCT-VO rotation predictions and accumulated "
            "orientation drift."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Evaluator-generated frame_predictions.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for rotation-audit outputs.",
    )

    parser.add_argument(
        "--euler-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help=(
            "Euler order used by the evaluator. "
            "For xyz, z is treated as the heading/yaw diagnostic."
        ),
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help=(
            "Interpret CSV rotation values as degrees. "
            "Normally leave disabled for DeepDCT-VO."
        ),
    )

    parser.add_argument(
        "--turn-quantiles",
        type=float,
        nargs=2,
        metavar=("LOW_Q", "HIGH_Q"),
        default=(1.0 / 3.0, 2.0 / 3.0),
        help=(
            "Quantiles of absolute GT heading rotation defining "
            "straight, moderate, and strong turn regimes."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved plot resolution.",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.predictions.is_file():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {args.predictions}"
        )

    low_q, high_q = args.turn_quantiles

    if not (
        0.0
        < low_q
        < high_q
        < 1.0
    ):
        raise ValueError(
            "--turn-quantiles must satisfy "
            "0 < LOW_Q < HIGH_Q < 1."
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )


# ============================================================================
# Evaluator import
# ============================================================================


def find_repo_root() -> Path:
    script_path = Path(
        __file__
    ).resolve()

    candidate = (
        script_path.parent.parent
    )

    if (
        (candidate / "deepdct").is_dir()
        and (
            candidate
            / "scripts"
            / "evaluate_deepdct_vo.py"
        ).is_file()
    ):
        return candidate

    current = Path.cwd().resolve()

    if (
        (current / "deepdct").is_dir()
        and (
            current
            / "scripts"
            / "evaluate_deepdct_vo.py"
        ).is_file()
    ):
        return current

    raise FileNotFoundError(
        "Could not locate repository root."
    )


def load_evaluator_module(
    repo_root: Path,
) -> Any:
    evaluator_path = (
        repo_root
        / "scripts"
        / "evaluate_deepdct_vo.py"
    )

    spec = importlib.util.spec_from_file_location(
        "deepdct_rotation_audit_evaluator",
        str(evaluator_path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Could not load {evaluator_path}."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    required = (
        "integrate_relative_poses",
        "compute_trajectory_metrics",
        "save_kitti_trajectory",
    )

    missing = [
        name
        for name in required
        if not hasattr(
            module,
            name,
        )
    ]

    if missing:
        raise AttributeError(
            "evaluate_deepdct_vo.py is missing required helpers: "
            f"{missing}."
        )

    return module


# ============================================================================
# CSV loading
# ============================================================================


def parse_float(
    row: Mapping[str, str],
    column: str,
    row_number: int,
) -> float:
    try:
        value = float(
            row[column]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Could not parse {column!r} "
            f"at CSV row {row_number}."
        ) from error

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"Non-finite {column!r} "
            f"at CSV row {row_number}."
        )

    return value


def load_predictions(
    path: Path,
) -> Tuple[
    List[Dict[str, str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    metadata: List[
        Dict[str, str]
    ] = []

    rotation_gt: List[
        List[float]
    ] = []

    rotation_pred: List[
        List[float]
    ] = []

    translation_gt: List[
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

        fieldnames = (
            reader.fieldnames
            if reader.fieldnames
            is not None
            else []
        )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]

        if missing:
            raise KeyError(
                "Prediction CSV is missing columns: "
                f"{missing}."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            metadata.append(
                dict(
                    row
                )
            )

            rotation_gt.append(
                [
                    parse_float(
                        row,
                        column,
                        row_number,
                    )
                    for column
                    in ROTATION_GT_COLUMNS
                ]
            )

            rotation_pred.append(
                [
                    parse_float(
                        row,
                        column,
                        row_number,
                    )
                    for column
                    in ROTATION_PRED_COLUMNS
                ]
            )

            translation_gt.append(
                [
                    parse_float(
                        row,
                        column,
                        row_number,
                    )
                    for column
                    in TRANSLATION_GT_COLUMNS
                ]
            )

    if not metadata:
        raise RuntimeError(
            "Prediction CSV contains no frames."
        )

    return (
        metadata,
        np.asarray(
            rotation_gt,
            dtype=np.float64,
        ),
        np.asarray(
            rotation_pred,
            dtype=np.float64,
        ),
        np.asarray(
            translation_gt,
            dtype=np.float64,
        ),
    )


# ============================================================================
# Utilities
# ============================================================================


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if (
        np.std(
            x
        )
        < 1.0e-12
        or np.std(
            y
        )
        < 1.0e-12
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    if not rows:
        return

    fieldnames: List[
        str
    ] = []

    for row in rows:
        for key in row.keys():
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

        for row in rows:
            writer.writerow(
                row
            )


def json_ready(
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
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_ready(
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
            json_ready(
                item
            )
            for item in value
        ]

    return value


# ============================================================================
# Euler / SO(3) conversion
# ============================================================================


def euler_to_rotation_matrices(
    euler: np.ndarray,
    *,
    order: str,
    degrees: bool,
) -> np.ndarray:
    rotations = Rotation.from_euler(
        order,
        euler,
        degrees=degrees,
    )

    return rotations.as_matrix()


def rotation_angle_degrees(
    matrices: np.ndarray,
) -> np.ndarray:
    rotations = Rotation.from_matrix(
        matrices
    )

    return np.degrees(
        rotations.magnitude()
    )


# ============================================================================
# Relative rotation diagnostics
# ============================================================================


def compute_relative_diagnostics(
    rotation_gt: np.ndarray,
    rotation_pred: np.ndarray,
    *,
    order: str,
    degrees: bool,
) -> Dict[str, np.ndarray]:
    gt_matrices = (
        euler_to_rotation_matrices(
            rotation_gt,
            order=order,
            degrees=degrees,
        )
    )

    pred_matrices = (
        euler_to_rotation_matrices(
            rotation_pred,
            order=order,
            degrees=degrees,
        )
    )

    # Rotation transforming GT relative orientation into predicted
    # relative orientation.
    error_matrices = (
        np.transpose(
            gt_matrices,
            axes=(0, 2, 1),
        )
        @ pred_matrices
    )

    geodesic_error_deg = (
        rotation_angle_degrees(
            error_matrices
        )
    )

    error_rotvec = (
        Rotation.from_matrix(
            error_matrices
        ).as_rotvec()
    )

    error_rotvec_deg = (
        np.degrees(
            error_rotvec
        )
    )

    if degrees:
        gt_euler_deg = (
            rotation_gt.copy()
        )

        pred_euler_deg = (
            rotation_pred.copy()
        )
    else:
        gt_euler_deg = (
            np.degrees(
                rotation_gt
            )
        )

        pred_euler_deg = (
            np.degrees(
                rotation_pred
            )
        )

    raw_euler_error_deg = (
        pred_euler_deg
        - gt_euler_deg
    )

    return {
        "gt_matrices": gt_matrices,
        "pred_matrices": pred_matrices,
        "error_matrices": error_matrices,
        "geodesic_error_deg": (
            geodesic_error_deg
        ),
        "error_rotvec_deg": (
            error_rotvec_deg
        ),
        "gt_euler_deg": (
            gt_euler_deg
        ),
        "pred_euler_deg": (
            pred_euler_deg
        ),
        "raw_euler_error_deg": (
            raw_euler_error_deg
        ),
    }


# ============================================================================
# Per-axis statistics
# ============================================================================


def build_axis_statistics(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
) -> List[
    Dict[str, Any]
]:
    gt = diagnostics[
        "gt_euler_deg"
    ]

    pred = diagnostics[
        "pred_euler_deg"
    ]

    error = (
        pred - gt
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for axis_index, axis_name in enumerate(
        AXES
    ):
        gt_axis = gt[
            :,
            axis_index
        ]

        pred_axis = pred[
            :,
            axis_index
        ]

        error_axis = error[
            :,
            axis_index
        ]

        gt_std = float(
            np.std(
                gt_axis
            )
        )

        pred_std = float(
            np.std(
                pred_axis
            )
        )

        rows.append(
            {
                "axis": axis_name,
                "samples": int(
                    gt.shape[0]
                ),
                "gt_mean_deg": float(
                    np.mean(
                        gt_axis
                    )
                ),
                "pred_mean_deg": float(
                    np.mean(
                        pred_axis
                    )
                ),
                "bias_deg": float(
                    np.mean(
                        error_axis
                    )
                ),
                "mae_deg": float(
                    np.mean(
                        np.abs(
                            error_axis
                        )
                    )
                ),
                "rmse_deg": float(
                    np.sqrt(
                        np.mean(
                            error_axis ** 2
                        )
                    )
                ),
                "gt_std_deg": gt_std,
                "pred_std_deg": (
                    pred_std
                ),
                "std_ratio": (
                    pred_std
                    / gt_std
                    if gt_std
                    > 1.0e-12
                    else float(
                        "nan"
                    )
                ),
                "correlation": safe_corr(
                    pred_axis,
                    gt_axis,
                ),
            }
        )

    return rows


# ============================================================================
# Turn regimes
# ============================================================================


def derive_turn_regimes(
    gt_heading_deg: np.ndarray,
    quantiles: Tuple[
        float,
        float,
    ],
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    absolute_heading = (
        np.abs(
            gt_heading_deg
        )
    )

    low_threshold = float(
        np.quantile(
            absolute_heading,
            quantiles[0],
        )
    )

    high_threshold = float(
        np.quantile(
            absolute_heading,
            quantiles[1],
        )
    )

    labels = np.full(
        absolute_heading.shape,
        "moderate",
        dtype=object,
    )

    labels[
        absolute_heading
        <= low_threshold
    ] = "straight"

    labels[
        absolute_heading
        > high_threshold
    ] = "strong"

    return (
        labels,
        low_threshold,
        high_threshold,
    )


def build_regime_statistics(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    regimes: np.ndarray,
) -> List[
    Dict[str, Any]
]:
    gt = diagnostics[
        "gt_euler_deg"
    ]

    pred = diagnostics[
        "pred_euler_deg"
    ]

    geodesic = diagnostics[
        "geodesic_error_deg"
    ]

    rows: List[
        Dict[str, Any]
    ] = []

    masks = {
        "all": np.ones(
            regimes.shape[0],
            dtype=bool,
        ),
        "straight": (
            regimes == "straight"
        ),
        "moderate": (
            regimes == "moderate"
        ),
        "strong": (
            regimes == "strong"
        ),
    }

    for regime_name, mask in (
        masks.items()
    ):
        if not np.any(
            mask
        ):
            continue

        gt_subset = gt[
            mask
        ]

        pred_subset = pred[
            mask
        ]

        error_subset = (
            pred_subset
            - gt_subset
        )

        row: Dict[
            str,
            Any,
        ] = {
            "regime": regime_name,
            "samples": int(
                np.count_nonzero(
                    mask
                )
            ),
            "geodesic_error_mean_deg": float(
                np.mean(
                    geodesic[
                        mask
                    ]
                )
            ),
            "geodesic_error_rmse_deg": float(
                np.sqrt(
                    np.mean(
                        geodesic[
                            mask
                        ]
                        ** 2
                    )
                )
            ),
        }

        for axis_index, axis_name in enumerate(
            AXES
        ):
            gt_axis = (
                gt_subset[
                    :,
                    axis_index
                ]
            )

            pred_axis = (
                pred_subset[
                    :,
                    axis_index
                ]
            )

            error_axis = (
                error_subset[
                    :,
                    axis_index
                ]
            )

            row[
                f"{axis_name}_bias_deg"
            ] = float(
                np.mean(
                    error_axis
                )
            )

            row[
                f"{axis_name}_rmse_deg"
            ] = float(
                np.sqrt(
                    np.mean(
                        error_axis ** 2
                    )
                )
            )

            row[
                f"{axis_name}_corr"
            ] = safe_corr(
                pred_axis,
                gt_axis,
            )

        rows.append(
            row
        )

    return rows


# ============================================================================
# Cumulative orientation analysis
# ============================================================================


def integrate_rotations(
    relative_rotations: np.ndarray,
) -> np.ndarray:
    number_of_steps = (
        relative_rotations.shape[0]
    )

    cumulative = np.repeat(
        np.eye(
            3,
            dtype=np.float64,
        )[None, :, :],
        number_of_steps + 1,
        axis=0,
    )

    for index in range(
        number_of_steps
    ):
        cumulative[
            index + 1
        ] = (
            cumulative[
                index
            ]
            @ relative_rotations[
                index
            ]
        )

    return cumulative


def cumulative_orientation_diagnostics(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    *,
    order: str,
) -> Dict[str, np.ndarray]:
    cumulative_gt = (
        integrate_rotations(
            diagnostics[
                "gt_matrices"
            ]
        )
    )

    cumulative_pred = (
        integrate_rotations(
            diagnostics[
                "pred_matrices"
            ]
        )
    )

    cumulative_error = (
        np.transpose(
            cumulative_gt,
            axes=(0, 2, 1),
        )
        @ cumulative_pred
    )

    cumulative_geodesic_error = (
        rotation_angle_degrees(
            cumulative_error
        )
    )

    cumulative_gt_euler = (
        Rotation.from_matrix(
            cumulative_gt
        ).as_euler(
            order,
            degrees=True,
        )
    )

    cumulative_pred_euler = (
        Rotation.from_matrix(
            cumulative_pred
        ).as_euler(
            order,
            degrees=True,
        )
    )

    # Euler values wrap at +/-180. For accumulated heading we
    # unwrap the z component after converting to radians.
    gt_heading_unwrapped = np.degrees(
        np.unwrap(
            np.radians(
                cumulative_gt_euler[
                    :,
                    2
                ]
            )
        )
    )

    pred_heading_unwrapped = np.degrees(
        np.unwrap(
            np.radians(
                cumulative_pred_euler[
                    :,
                    2
                ]
            )
        )
    )

    heading_error = (
        pred_heading_unwrapped
        - gt_heading_unwrapped
    )

    return {
        "cumulative_gt": (
            cumulative_gt
        ),
        "cumulative_pred": (
            cumulative_pred
        ),
        "cumulative_error": (
            cumulative_error
        ),
        "cumulative_geodesic_error_deg": (
            cumulative_geodesic_error
        ),
        "cumulative_gt_euler_deg": (
            cumulative_gt_euler
        ),
        "cumulative_pred_euler_deg": (
            cumulative_pred_euler
        ),
        "gt_heading_unwrapped_deg": (
            gt_heading_unwrapped
        ),
        "pred_heading_unwrapped_deg": (
            pred_heading_unwrapped
        ),
        "heading_error_deg": (
            heading_error
        ),
    }


# ============================================================================
# Frame-level output
# ============================================================================


def build_frame_rows(
    metadata: Sequence[
        Mapping[str, str]
    ],
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    cumulative: Mapping[
        str,
        np.ndarray
    ],
    regimes: np.ndarray,
) -> List[
    Dict[str, Any]
]:
    rows: List[
        Dict[str, Any]
    ] = []

    gt = diagnostics[
        "gt_euler_deg"
    ]

    pred = diagnostics[
        "pred_euler_deg"
    ]

    error = (
        pred - gt
    )

    rotvec_error = (
        diagnostics[
            "error_rotvec_deg"
        ]
    )

    for index in range(
        gt.shape[0]
    ):
        row: Dict[
            str,
            Any,
        ] = {
            "index": index,
            "sequence": metadata[
                index
            ].get(
                "sequence",
                "",
            ),
            "frame_prev": metadata[
                index
            ].get(
                "frame_prev",
                "",
            ),
            "frame_curr": metadata[
                index
            ].get(
                "frame_curr",
                "",
            ),
            "turn_regime": str(
                regimes[
                    index
                ]
            ),
            "relative_geodesic_error_deg": float(
                diagnostics[
                    "geodesic_error_deg"
                ][
                    index
                ]
            ),
            "cumulative_geodesic_error_deg": float(
                cumulative[
                    "cumulative_geodesic_error_deg"
                ][
                    index + 1
                ]
            ),
            "cumulative_heading_error_deg": float(
                cumulative[
                    "heading_error_deg"
                ][
                    index + 1
                ]
            ),
        }

        for axis_index, axis_name in enumerate(
            AXES
        ):
            row[
                f"rotation_gt_{axis_name}_deg"
            ] = float(
                gt[
                    index,
                    axis_index,
                ]
            )

            row[
                f"rotation_pred_{axis_name}_deg"
            ] = float(
                pred[
                    index,
                    axis_index,
                ]
            )

            row[
                f"rotation_error_{axis_name}_deg"
            ] = float(
                error[
                    index,
                    axis_index,
                ]
            )

            row[
                f"so3_error_rotvec_{axis_name}_deg"
            ] = float(
                rotvec_error[
                    index,
                    axis_index,
                ]
            )

        rows.append(
            row
        )

    return rows


# ============================================================================
# Counterfactual trajectory
# ============================================================================


def run_rotation_counterfactual(
    evaluator: Any,
    rotation_gt: np.ndarray,
    rotation_pred: np.ndarray,
    translation_gt: np.ndarray,
    *,
    euler_order: str,
    angles_in_degrees: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    gt_trajectory = (
        evaluator.integrate_relative_poses(
            rotations=rotation_gt,
            translations=translation_gt,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        )
    )

    pred_rotation_gt_translation = (
        evaluator.integrate_relative_poses(
            rotations=rotation_pred,
            translations=translation_gt,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        )
    )

    metrics = (
        evaluator.compute_trajectory_metrics(
            ground_truth_trajectory=(
                gt_trajectory
            ),
            predicted_trajectory=(
                pred_rotation_gt_translation
            ),
        )
    )

    if hasattr(
        metrics,
        "__dict__",
    ):
        metric_dict = dict(
            metrics.__dict__
        )
    elif isinstance(
        metrics,
        Mapping,
    ):
        metric_dict = dict(
            metrics
        )
    else:
        metric_dict = {}

        for key in (
            "ate_rmse",
            "rpe_translation_rmse",
            "rpe_rotation_rmse_degrees",
            "endpoint_error",
            "endpoint_error_percent",
            "translational_drift_percent",
            "rotational_drift_degrees_per_100m",
        ):
            if hasattr(
                metrics,
                key,
            ):
                metric_dict[
                    key
                ] = getattr(
                    metrics,
                    key,
                )

    return (
        gt_trajectory,
        pred_rotation_gt_translation,
        metric_dict,
    )


# ============================================================================
# Plotting
# ============================================================================


def plot_relative_axes(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    output_dir: Path,
    dpi: int,
) -> None:
    gt = diagnostics[
        "gt_euler_deg"
    ]

    pred = diagnostics[
        "pred_euler_deg"
    ]

    x = np.arange(
        gt.shape[0]
    )

    for axis_index, axis_name in enumerate(
        AXES
    ):
        figure = plt.figure(
            figsize=(11.0, 5.5)
        )

        axis = figure.add_subplot(
            1,
            1,
            1,
        )

        axis.plot(
            x,
            gt[
                :,
                axis_index
            ],
            label="GT",
            linewidth=1.0,
        )

        axis.plot(
            x,
            pred[
                :,
                axis_index
            ],
            label="Prediction",
            linewidth=1.0,
        )

        axis.set_xlabel(
            "Transition index"
        )

        axis.set_ylabel(
            f"Relative rotation {axis_name} (deg)"
        )

        axis.set_title(
            f"Relative rotation: {axis_name}-axis"
        )

        axis.grid(
            True
        )

        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_dir
            / f"relative_rotation_{axis_name}.png",
            dpi=dpi,
        )

        plt.close(
            figure
        )


def plot_axis_error(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    output_path: Path,
    dpi: int,
) -> None:
    error = (
        diagnostics[
            "raw_euler_error_deg"
        ]
    )

    x = np.arange(
        error.shape[0]
    )

    figure = plt.figure(
        figsize=(11.0, 6.0)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for axis_index, axis_name in enumerate(
        AXES
    ):
        axis.plot(
            x,
            error[
                :,
                axis_index
            ],
            label=axis_name,
            linewidth=0.9,
        )

    axis.axhline(
        0.0,
        linestyle="--",
    )

    axis.set_xlabel(
        "Transition index"
    )

    axis.set_ylabel(
        "Pred - GT rotation (deg)"
    )

    axis.set_title(
        "Framewise Euler rotation error"
    )

    axis.grid(
        True
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
    )

    plt.close(
        figure
    )


def plot_geodesic_error(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    output_path: Path,
    dpi: int,
) -> None:
    values = diagnostics[
        "geodesic_error_deg"
    ]

    figure = plt.figure(
        figsize=(11.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        np.arange(
            values.shape[0]
        ),
        values,
        linewidth=0.9,
    )

    axis.set_xlabel(
        "Transition index"
    )

    axis.set_ylabel(
        "SO(3) error (deg)"
    )

    axis.set_title(
        "Relative rotation geodesic error"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
    )

    plt.close(
        figure
    )


def plot_cumulative_orientation(
    cumulative: Mapping[
        str,
        np.ndarray
    ],
    output_dir: Path,
    dpi: int,
) -> None:
    geodesic = cumulative[
        "cumulative_geodesic_error_deg"
    ]

    heading_gt = cumulative[
        "gt_heading_unwrapped_deg"
    ]

    heading_pred = cumulative[
        "pred_heading_unwrapped_deg"
    ]

    heading_error = cumulative[
        "heading_error_deg"
    ]

    x = np.arange(
        geodesic.shape[0]
    )

    figure = plt.figure(
        figsize=(11.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        x,
        geodesic,
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Frame"
    )

    axis.set_ylabel(
        "Cumulative SO(3) error (deg)"
    )

    axis.set_title(
        "Accumulated orientation error"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "cumulative_geodesic_error.png",
        dpi=dpi,
    )

    plt.close(
        figure
    )

    figure = plt.figure(
        figsize=(11.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        x,
        heading_gt,
        label="GT heading",
    )

    axis.plot(
        x,
        heading_pred,
        label="Pred heading",
    )

    axis.set_xlabel(
        "Frame"
    )

    axis.set_ylabel(
        "Unwrapped heading (deg)"
    )

    axis.set_title(
        "Accumulated heading"
    )

    axis.grid(
        True
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "cumulative_heading_gt_vs_pred.png",
        dpi=dpi,
    )

    plt.close(
        figure
    )

    figure = plt.figure(
        figsize=(11.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        x,
        heading_error,
        linewidth=1.0,
    )

    axis.axhline(
        0.0,
        linestyle="--",
    )

    axis.set_xlabel(
        "Frame"
    )

    axis.set_ylabel(
        "Pred - GT heading (deg)"
    )

    axis.set_title(
        "Accumulated heading error"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "cumulative_heading_error.png",
        dpi=dpi,
    )

    plt.close(
        figure
    )


def plot_rotation_only_trajectory(
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    gt_positions = (
        gt_trajectory[
            :,
            :3,
            3,
        ]
    )

    pred_positions = (
        pred_trajectory[
            :,
            :3,
            3,
        ]
    )

    figure = plt.figure(
        figsize=(9.0, 7.0)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        gt_positions[
            :,
            0
        ],
        gt_positions[
            :,
            1
        ],
        label="GT R + GT t",
    )

    axis.plot(
        pred_positions[
            :,
            0
        ],
        pred_positions[
            :,
            1
        ],
        label="Pred R + GT t",
    )

    axis.set_xlabel(
        "X"
    )

    axis.set_ylabel(
        "Y"
    )

    axis.set_title(
        "Rotation-only trajectory attribution"
    )

    axis.axis(
        "equal"
    )

    axis.grid(
        True
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
    )

    plt.close(
        figure
    )


# ============================================================================
# Console reporting
# ============================================================================


def print_axis_table(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    print()
    print("=" * 108)
    print("Per-axis relative rotation statistics")
    print("=" * 108)

    print(
        f"{'Axis':<8}"
        f"{'Bias deg':>14}"
        f"{'MAE deg':>14}"
        f"{'RMSE deg':>14}"
        f"{'GT std':>14}"
        f"{'Pred std':>14}"
        f"{'Std ratio':>14}"
        f"{'Corr':>14}"
    )

    print(
        "-" * 108
    )

    for row in rows:
        print(
            f"{row['axis']:<8}"
            f"{row['bias_deg']:>14.6f}"
            f"{row['mae_deg']:>14.6f}"
            f"{row['rmse_deg']:>14.6f}"
            f"{row['gt_std_deg']:>14.6f}"
            f"{row['pred_std_deg']:>14.6f}"
            f"{row['std_ratio']:>14.4f}"
            f"{row['correlation']:>14.4f}"
        )

    print(
        "=" * 108
    )


def print_regime_table(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    print()
    print("=" * 116)
    print("Turn-regime rotation diagnostics")
    print("=" * 116)

    print(
        f"{'Regime':<12}"
        f"{'N':>8}"
        f"{'SO3 RMSE':>14}"
        f"{'x RMSE':>14}"
        f"{'y RMSE':>14}"
        f"{'z RMSE':>14}"
        f"{'z Bias':>14}"
        f"{'z Corr':>14}"
    )

    print(
        "-" * 116
    )

    for row in rows:
        print(
            f"{row['regime']:<12}"
            f"{row['samples']:>8}"
            f"{row['geodesic_error_rmse_deg']:>14.6f}"
            f"{row['x_rmse_deg']:>14.6f}"
            f"{row['y_rmse_deg']:>14.6f}"
            f"{row['z_rmse_deg']:>14.6f}"
            f"{row['z_bias_deg']:>14.6f}"
            f"{row['z_corr']:>14.4f}"
        )

    print(
        "=" * 116
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    args.predictions = (
        args.predictions
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

    trajectories_dir = (
        args.output_dir
        / "trajectories"
    )

    trajectories_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repo_root = (
        find_repo_root()
    )

    evaluator = (
        load_evaluator_module(
            repo_root
        )
    )

    (
        metadata,
        rotation_gt,
        rotation_pred,
        translation_gt,
    ) = load_predictions(
        args.predictions
    )

    num_samples = int(
        rotation_gt.shape[0]
    )

    print("=" * 88)
    print("DeepDCT-VO rotation prediction audit")
    print("=" * 88)
    print(
        f"Predictions:       {args.predictions}"
    )
    print(
        f"Transitions:       {num_samples}"
    )
    print(
        f"Euler order:       {args.euler_order}"
    )
    print(
        f"Angles in degrees: {args.angles_in_degrees}"
    )
    print("=" * 88)

    # ------------------------------------------------------------------
    # Relative rotation diagnostics
    # ------------------------------------------------------------------

    diagnostics = (
        compute_relative_diagnostics(
            rotation_gt=rotation_gt,
            rotation_pred=rotation_pred,
            order=args.euler_order,
            degrees=args.angles_in_degrees,
        )
    )

    axis_rows = (
        build_axis_statistics(
            diagnostics
        )
    )

    # For xyz convention, z is the heading/yaw component.
    heading_axis_index = 2

    gt_heading_deg = (
        diagnostics[
            "gt_euler_deg"
        ][
            :,
            heading_axis_index
        ]
    )

    (
        regimes,
        straight_threshold,
        strong_threshold,
    ) = derive_turn_regimes(
        gt_heading_deg=gt_heading_deg,
        quantiles=tuple(
            args.turn_quantiles
        ),
    )

    regime_rows = (
        build_regime_statistics(
            diagnostics=diagnostics,
            regimes=regimes,
        )
    )

    # ------------------------------------------------------------------
    # Cumulative orientation
    # ------------------------------------------------------------------

    cumulative = (
        cumulative_orientation_diagnostics(
            diagnostics=diagnostics,
            order=args.euler_order,
        )
    )

    frame_rows = (
        build_frame_rows(
            metadata=metadata,
            diagnostics=diagnostics,
            cumulative=cumulative,
            regimes=regimes,
        )
    )

    # ------------------------------------------------------------------
    # Rotation-only trajectory counterfactual
    # ------------------------------------------------------------------

    (
        gt_trajectory,
        pred_r_gt_t_trajectory,
        trajectory_metrics,
    ) = run_rotation_counterfactual(
        evaluator=evaluator,
        rotation_gt=rotation_gt,
        rotation_pred=rotation_pred,
        translation_gt=translation_gt,
        euler_order=args.euler_order,
        angles_in_degrees=(
            args.angles_in_degrees
        ),
    )

    evaluator.save_kitti_trajectory(
        trajectories_dir
        / "GT_R_GT_t.txt",
        gt_trajectory,
    )

    evaluator.save_kitti_trajectory(
        trajectories_dir
        / "Pred_R_GT_t.txt",
        pred_r_gt_t_trajectory,
    )

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------

    write_csv(
        args.output_dir
        / "rotation_axis_statistics.csv",
        axis_rows,
    )

    write_csv(
        args.output_dir
        / "rotation_regime_statistics.csv",
        regime_rows,
    )

    write_csv(
        args.output_dir
        / "frame_rotation_errors.csv",
        frame_rows,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_relative_axes(
        diagnostics=diagnostics,
        output_dir=plots_dir,
        dpi=args.dpi,
    )

    plot_axis_error(
        diagnostics=diagnostics,
        output_path=(
            plots_dir
            / "relative_rotation_error_axes.png"
        ),
        dpi=args.dpi,
    )

    plot_geodesic_error(
        diagnostics=diagnostics,
        output_path=(
            plots_dir
            / "relative_geodesic_error.png"
        ),
        dpi=args.dpi,
    )

    plot_cumulative_orientation(
        cumulative=cumulative,
        output_dir=plots_dir,
        dpi=args.dpi,
    )

    plot_rotation_only_trajectory(
        gt_trajectory=gt_trajectory,
        pred_trajectory=(
            pred_r_gt_t_trajectory
        ),
        output_path=(
            plots_dir
            / "rotation_only_trajectory_xy.png"
        ),
        dpi=args.dpi,
    )

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------

    relative_geodesic = (
        diagnostics[
            "geodesic_error_deg"
        ]
    )

    cumulative_geodesic = (
        cumulative[
            "cumulative_geodesic_error_deg"
        ]
    )

    heading_error = (
        cumulative[
            "heading_error_deg"
        ]
    )

    final_heading_error = float(
        heading_error[
            -1
        ]
    )

    final_geodesic_error = float(
        cumulative_geodesic[
            -1
        ]
    )

    maximum_cumulative_geodesic = float(
        np.max(
            cumulative_geodesic
        )
    )

    mean_relative_geodesic = float(
        np.mean(
            relative_geodesic
        )
    )

    rmse_relative_geodesic = float(
        np.sqrt(
            np.mean(
                relative_geodesic ** 2
            )
        )
    )

    summary = {
        "predictions": str(
            args.predictions
        ),
        "transitions": (
            num_samples
        ),
        "euler_order": (
            args.euler_order
        ),
        "angles_in_degrees": (
            args.angles_in_degrees
        ),
        "heading_axis": (
            "z"
        ),
        "relative_rotation": {
            "mean_geodesic_error_deg": (
                mean_relative_geodesic
            ),
            "rmse_geodesic_error_deg": (
                rmse_relative_geodesic
            ),
            "axis_statistics": (
                axis_rows
            ),
        },
        "turn_regimes": {
            "quantiles": list(
                args.turn_quantiles
            ),
            "straight_threshold_abs_gt_z_deg": (
                straight_threshold
            ),
            "strong_threshold_abs_gt_z_deg": (
                strong_threshold
            ),
            "statistics": (
                regime_rows
            ),
        },
        "cumulative_orientation": {
            "final_geodesic_error_deg": (
                final_geodesic_error
            ),
            "maximum_geodesic_error_deg": (
                maximum_cumulative_geodesic
            ),
            "final_heading_error_deg": (
                final_heading_error
            ),
            "heading_error_mean_deg": float(
                np.mean(
                    heading_error
                )
            ),
            "heading_error_rmse_deg": float(
                np.sqrt(
                    np.mean(
                        heading_error ** 2
                    )
                )
            ),
        },
        "pred_rotation_gt_translation_trajectory": (
            trajectory_metrics
        ),
        "interpretation": {
            "primary_question": (
                "Which relative-rotation component produces "
                "the accumulated global orientation failure?"
            ),
            "caution": (
                "Raw Euler-axis errors are convention-dependent. "
                "Use SO(3) geodesic error for total rotation quality, "
                "and use z-axis Euler diagnostics specifically for "
                "heading interpretation under xyz convention."
            ),
        },
    }

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_ready(
                summary
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print_axis_table(
        axis_rows
    )

    print_regime_table(
        regime_rows
    )

    print()
    print("=" * 88)
    print("Cumulative orientation diagnostics")
    print("=" * 88)
    print(
        "Relative SO(3) mean error:     "
        f"{mean_relative_geodesic:.6f} deg"
    )
    print(
        "Relative SO(3) RMSE:           "
        f"{rmse_relative_geodesic:.6f} deg"
    )
    print(
        "Final cumulative SO(3) error:  "
        f"{final_geodesic_error:.6f} deg"
    )
    print(
        "Maximum cumulative SO(3):      "
        f"{maximum_cumulative_geodesic:.6f} deg"
    )
    print(
        "Final cumulative heading error:"
        f" {final_heading_error:+.6f} deg"
    )
    print(
        "Heading error RMSE:             "
        f"{np.sqrt(np.mean(heading_error ** 2)):.6f} deg"
    )
    print("=" * 88)

    if trajectory_metrics:
        print()
        print("=" * 88)
        print("Predicted rotation + GT translation trajectory")
        print("=" * 88)

        for key, value in (
            trajectory_metrics.items()
        ):
            if isinstance(
                value,
                (int, float, np.number),
            ):
                print(
                    f"{key:<40} {float(value):.6f}"
                )

        print("=" * 88)

    print()
    print("=" * 88)
    print("Rotation audit complete")
    print("=" * 88)
    print(
        f"Output directory: {args.output_dir}"
    )
    print(
        f"Summary:          {summary_path}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()