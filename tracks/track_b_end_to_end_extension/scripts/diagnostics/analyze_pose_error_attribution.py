#!/usr/bin/env python3
"""
Pose-error attribution audit for DeepDCT-VO.

Purpose
-------
Determine how much trajectory divergence is attributable to:

    1. predicted rotation + predicted translation
    2. GT rotation        + predicted translation
    3. predicted rotation + GT translation
    4. GT rotation        + GT translation

The script also analyzes directional-translation magnitude and direction:

    scale_ratio_i =
        ||t_pred_i|| / (||t_gt_i|| + eps)

    direction_error_i =
        acos(
            dot(t_pred_i, t_gt_i)
            /
            (||t_pred_i|| ||t_gt_i|| + eps)
        )

and summarizes these quantities overall and by low/medium/high
forward-motion regimes derived from GT t_z quantiles.

IMPORTANT
---------
This is an integration attribution audit.

"GT rotation + predicted translation" does NOT re-run Model T using
ground-truth rotation conditioning. It takes the already-exported predicted
directional translation and changes only the rotation stream used for
trajectory reconstruction.

This isolates trajectory-integration consequences of the two predicted pose
components without changing the neural-network forward pass.

The script imports trajectory functions directly from:

    scripts/evaluate_deepdct_vo.py

so it uses the same corrected inverse-DCT directional-translation decoding,
Euler convention, trajectory composition, and trajectory metrics as the
current evaluator.

Expected input
--------------
An evaluator-generated frame_predictions.csv containing at least:

    rotation_gt_x
    rotation_gt_y
    rotation_gt_z

    rotation_pred_x
    rotation_pred_y
    rotation_pred_z

    translation_gt_x
    translation_gt_y
    translation_gt_z

    translation_pred_x
    translation_pred_y
    translation_pred_z

Optional metadata columns:

    sequence
    frame_prev
    frame_curr

Example
-------
python scripts/analyze_pose_error_attribution.py \\
    --predictions \\
        experiments/semantic_depth_pooled_translation_head/evaluation_sequence_10/frame_predictions.csv \\
    --output-dir \\
        experiments/semantic_depth_pooled_translation_head/pose_error_attribution \\
    --euler-order xyz \\
    --regime-quantiles 0.25 0.75
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Constants
# ============================================================================

REQUIRED_COLUMNS = (
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

TRANSLATION_PRED_COLUMNS = (
    "translation_pred_x",
    "translation_pred_y",
    "translation_pred_z",
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute DeepDCT-VO trajectory error to rotation and "
            "translation components."
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
        help="Directory for attribution outputs.",
    )

    parser.add_argument(
        "--euler-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help=(
            "Euler convention. Must match evaluate_deepdct_vo.py "
            "for the source evaluation."
        ),
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help=(
            "Interpret exported relative Euler rotations as degrees. "
            "Normally leave disabled for DeepDCT-VO."
        ),
    )

    parser.add_argument(
        "--regime-quantiles",
        type=float,
        nargs=2,
        metavar=("LOW_Q", "HIGH_Q"),
        default=(0.25, 0.75),
        help=(
            "GT t_z quantiles defining low, medium, and high "
            "forward-motion regimes."
        ),
    )

    parser.add_argument(
        "--scale-epsilon",
        type=float,
        default=1.0e-8,
        help="Numerical floor for translation norm ratios.",
    )

    parser.add_argument(
        "--minimum-direction-norm",
        type=float,
        default=1.0e-5,
        help=(
            "Frames with either translation norm below this value "
            "are excluded from translation-direction angle statistics."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Plot resolution.",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.predictions.is_file():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {args.predictions}"
        )

    low_q, high_q = args.regime_quantiles

    if not (
        0.0
        < low_q
        < high_q
        < 1.0
    ):
        raise ValueError(
            "--regime-quantiles must satisfy "
            "0 < LOW_Q < HIGH_Q < 1."
        )

    if args.scale_epsilon <= 0.0:
        raise ValueError(
            "--scale-epsilon must be positive."
        )

    if args.minimum_direction_norm < 0.0:
        raise ValueError(
            "--minimum-direction-norm cannot be negative."
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
        "Could not find repository root containing deepdct/ and "
        "scripts/evaluate_deepdct_vo.py."
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
        "deepdct_pose_attribution_evaluator",
        str(evaluator_path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            "Could not create module specification for "
            f"{evaluator_path}."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Dataclasses and some module-level machinery expect the imported
    # module to exist in sys.modules while it is being executed.
    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    required_functions = (
        "integrate_relative_poses",
        "compute_trajectory_metrics",
        "save_kitti_trajectory",
    )

    missing = [
        name
        for name in required_functions
        if not hasattr(
            module,
            name,
        )
    ]

    if missing:
        raise AttributeError(
            "Current evaluate_deepdct_vo.py is missing required "
            f"trajectory helpers: {missing}."
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
            f"Could not parse {column!r} at CSV row {row_number}."
        ) from error

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"Non-finite value in {column!r} at CSV row {row_number}."
        )

    return value


def load_frame_predictions(
    path: Path,
) -> Tuple[
    List[Dict[str, str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    metadata_rows: List[
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

    translation_pred: List[
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
            if reader.fieldnames is not None
            else []
        )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]

        if missing_columns:
            raise KeyError(
                "Prediction CSV is missing required columns: "
                f"{missing_columns}."
            )

        for row_index, row in enumerate(
            reader,
            start=2,
        ):
            metadata_rows.append(
                dict(
                    row
                )
            )

            rotation_gt.append(
                [
                    parse_float(
                        row,
                        column,
                        row_index,
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
                        row_index,
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
                        row_index,
                    )
                    for column
                    in TRANSLATION_GT_COLUMNS
                ]
            )

            translation_pred.append(
                [
                    parse_float(
                        row,
                        column,
                        row_index,
                    )
                    for column
                    in TRANSLATION_PRED_COLUMNS
                ]
            )

    if not metadata_rows:
        raise RuntimeError(
            f"Prediction CSV is empty: {path}"
        )

    arrays = (
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
        np.asarray(
            translation_pred,
            dtype=np.float64,
        ),
    )

    for array in arrays:
        if (
            array.ndim != 2
            or array.shape[1] != 3
        ):
            raise RuntimeError(
                "Expected pose arrays with shape [N, 3], "
                f"received {array.shape}."
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "Pose arrays contain NaN or infinity."
            )

    sample_counts = {
        array.shape[0]
        for array in arrays
    }

    if len(
        sample_counts
    ) != 1:
        raise RuntimeError(
            "Pose arrays contain different numbers of samples."
        )

    return (
        metadata_rows,
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
    )


# ============================================================================
# Generic helpers
# ============================================================================


def safe_float(
    value: Any,
) -> Any:
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


def json_ready(
    value: Any,
) -> Any:
    if is_dataclass(
        value
    ):
        return json_ready(
            asdict(
                value
            )
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.floating,
    ):
        return safe_float(
            float(
                value
            )
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

    if isinstance(
        value,
        float,
    ):
        return safe_float(
            value
        )

    return value


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                {
                    key: safe_float(
                        value
                    )
                    for key, value
                    in row.items()
                }
            )


def descriptive_statistics(
    values: np.ndarray,
) -> Dict[str, Any]:
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(
        -1
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "rmse": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    return {
        "count": int(
            values.size
        ),
        "mean": float(
            np.mean(
                values
            )
        ),
        "std": float(
            np.std(
                values
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        ),
        "min": float(
            np.min(
                values
            )
        ),
        "q25": float(
            np.quantile(
                values,
                0.25,
            )
        ),
        "median": float(
            np.median(
                values
            )
        ),
        "q75": float(
            np.quantile(
                values,
                0.75,
            )
        ),
        "max": float(
            np.max(
                values
            )
        ),
    }


# ============================================================================
# Translation diagnostics
# ============================================================================


def translation_diagnostics(
    translation_gt: np.ndarray,
    translation_pred: np.ndarray,
    *,
    epsilon: float,
    minimum_direction_norm: float,
) -> Dict[str, np.ndarray]:
    gt_norm = np.linalg.norm(
        translation_gt,
        axis=1,
    )

    pred_norm = np.linalg.norm(
        translation_pred,
        axis=1,
    )

    scale_ratio = (
        pred_norm
        / np.maximum(
            gt_norm,
            epsilon,
        )
    )

    norm_error = (
        pred_norm
        - gt_norm
    )

    vector_error = np.linalg.norm(
        translation_pred
        - translation_gt,
        axis=1,
    )

    direction_error = np.full(
        translation_gt.shape[0],
        np.nan,
        dtype=np.float64,
    )

    valid_direction = (
        (gt_norm >= minimum_direction_norm)
        & (
            pred_norm
            >= minimum_direction_norm
        )
    )

    if np.any(
        valid_direction
    ):
        numerator = np.sum(
            translation_gt[
                valid_direction
            ]
            * translation_pred[
                valid_direction
            ],
            axis=1,
        )

        denominator = (
            gt_norm[
                valid_direction
            ]
            * pred_norm[
                valid_direction
            ]
        )

        cosine = (
            numerator
            / np.maximum(
                denominator,
                epsilon,
            )
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        direction_error[
            valid_direction
        ] = np.degrees(
            np.arccos(
                cosine
            )
        )

    return {
        "gt_norm": gt_norm,
        "pred_norm": pred_norm,
        "scale_ratio": scale_ratio,
        "norm_error": norm_error,
        "vector_error": vector_error,
        "direction_error_deg": direction_error,
        "gt_tz": translation_gt[
            :,
            2
        ],
        "pred_tz": translation_pred[
            :,
            2
        ],
    }


def derive_regimes(
    gt_tz: np.ndarray,
    quantiles: Tuple[
        float,
        float,
    ],
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    low_q, high_q = (
        quantiles
    )

    low_threshold = float(
        np.quantile(
            gt_tz,
            low_q,
        )
    )

    high_threshold = float(
        np.quantile(
            gt_tz,
            high_q,
        )
    )

    if not (
        low_threshold
        < high_threshold
    ):
        raise RuntimeError(
            "Forward-motion regime thresholds are not "
            "strictly ordered."
        )

    labels = np.full(
        gt_tz.shape,
        "medium",
        dtype=object,
    )

    labels[
        gt_tz
        <= low_threshold
    ] = "low"

    labels[
        gt_tz
        > high_threshold
    ] = "high"

    return (
        labels,
        low_threshold,
        high_threshold,
    )


def build_translation_summary_rows(
    diagnostics: Mapping[
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

    group_masks = {
        "all": np.ones(
            regimes.shape[0],
            dtype=bool,
        ),
        "low": (
            regimes == "low"
        ),
        "medium": (
            regimes == "medium"
        ),
        "high": (
            regimes == "high"
        ),
    }

    for group_name, mask in (
        group_masks.items()
    ):
        if not np.any(
            mask
        ):
            continue

        scale = diagnostics[
            "scale_ratio"
        ][
            mask
        ]

        direction = diagnostics[
            "direction_error_deg"
        ][
            mask
        ]

        gt_norm = diagnostics[
            "gt_norm"
        ][
            mask
        ]

        pred_norm = diagnostics[
            "pred_norm"
        ][
            mask
        ]

        gt_tz = diagnostics[
            "gt_tz"
        ][
            mask
        ]

        pred_tz = diagnostics[
            "pred_tz"
        ][
            mask
        ]

        vector_error = diagnostics[
            "vector_error"
        ][
            mask
        ]

        finite_direction = direction[
            np.isfinite(
                direction
            )
        ]

        if (
            np.std(
                gt_tz
            )
            > 1.0e-12
            and np.std(
                pred_tz
            )
            > 1.0e-12
        ):
            tz_corr = float(
                np.corrcoef(
                    gt_tz,
                    pred_tz,
                )[0, 1]
            )
        else:
            tz_corr = float(
                "nan"
            )

        rows.append(
            {
                "regime": group_name,
                "samples": int(
                    np.count_nonzero(
                        mask
                    )
                ),
                "gt_norm_mean": float(
                    np.mean(
                        gt_norm
                    )
                ),
                "pred_norm_mean": float(
                    np.mean(
                        pred_norm
                    )
                ),
                "scale_ratio_mean": float(
                    np.mean(
                        scale
                    )
                ),
                "scale_ratio_median": float(
                    np.median(
                        scale
                    )
                ),
                "scale_ratio_std": float(
                    np.std(
                        scale
                    )
                ),
                "scale_underprediction_fraction": float(
                    np.mean(
                        scale < 1.0
                    )
                ),
                "scale_overprediction_fraction": float(
                    np.mean(
                        scale > 1.0
                    )
                ),
                "translation_vector_error_mean": float(
                    np.mean(
                        vector_error
                    )
                ),
                "translation_vector_error_rmse": float(
                    np.sqrt(
                        np.mean(
                            vector_error ** 2
                        )
                    )
                ),
                "direction_error_mean_deg": (
                    float(
                        np.mean(
                            finite_direction
                        )
                    )
                    if finite_direction.size
                    else float(
                        "nan"
                    )
                ),
                "direction_error_median_deg": (
                    float(
                        np.median(
                            finite_direction
                        )
                    )
                    if finite_direction.size
                    else float(
                        "nan"
                    )
                ),
                "direction_error_rmse_deg": (
                    float(
                        np.sqrt(
                            np.mean(
                                finite_direction ** 2
                            )
                        )
                    )
                    if finite_direction.size
                    else float(
                        "nan"
                    )
                ),
                "gt_tz_mean": float(
                    np.mean(
                        gt_tz
                    )
                ),
                "pred_tz_mean": float(
                    np.mean(
                        pred_tz
                    )
                ),
                "tz_bias": float(
                    np.mean(
                        pred_tz
                        - gt_tz
                    )
                ),
                "tz_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                pred_tz
                                - gt_tz
                            )
                            ** 2
                        )
                    )
                ),
                "tz_corr": tz_corr,
                "gt_tz_std": float(
                    np.std(
                        gt_tz
                    )
                ),
                "pred_tz_std": float(
                    np.std(
                        pred_tz
                    )
                ),
            }
        )

    return rows


def build_frame_rows(
    metadata_rows: Sequence[
        Mapping[str, str]
    ],
    diagnostics: Mapping[
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

    for index in range(
        len(
            metadata_rows
        )
    ):
        metadata = (
            metadata_rows[
                index
            ]
        )

        rows.append(
            {
                "index": index,
                "sequence": metadata.get(
                    "sequence",
                    "",
                ),
                "frame_prev": metadata.get(
                    "frame_prev",
                    "",
                ),
                "frame_curr": metadata.get(
                    "frame_curr",
                    "",
                ),
                "regime": str(
                    regimes[
                        index
                    ]
                ),
                "gt_tz": float(
                    diagnostics[
                        "gt_tz"
                    ][
                        index
                    ]
                ),
                "pred_tz": float(
                    diagnostics[
                        "pred_tz"
                    ][
                        index
                    ]
                ),
                "gt_translation_norm": float(
                    diagnostics[
                        "gt_norm"
                    ][
                        index
                    ]
                ),
                "pred_translation_norm": float(
                    diagnostics[
                        "pred_norm"
                    ][
                        index
                    ]
                ),
                "scale_ratio": float(
                    diagnostics[
                        "scale_ratio"
                    ][
                        index
                    ]
                ),
                "norm_error": float(
                    diagnostics[
                        "norm_error"
                    ][
                        index
                    ]
                ),
                "translation_vector_error": float(
                    diagnostics[
                        "vector_error"
                    ][
                        index
                    ]
                ),
                "translation_direction_error_deg": (
                    float(
                        diagnostics[
                            "direction_error_deg"
                        ][
                            index
                        ]
                    )
                    if np.isfinite(
                        diagnostics[
                            "direction_error_deg"
                        ][
                            index
                        ]
                    )
                    else None
                ),
            }
        )

    return rows


# ============================================================================
# Trajectory attribution
# ============================================================================


def integrate_condition(
    evaluator: Any,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    euler_order: str,
    angles_in_degrees: bool,
) -> np.ndarray:
    return evaluator.integrate_relative_poses(
        rotations=rotations,
        translations=translations,
        euler_order=euler_order,
        angles_in_degrees=angles_in_degrees,
    )


def metrics_to_dict(
    metrics: Any,
) -> Dict[str, Any]:
    if is_dataclass(
        metrics
    ):
        return {
            key: safe_float(
                value
            )
            for key, value
            in asdict(
                metrics
            ).items()
        }

    if isinstance(
        metrics,
        Mapping,
    ):
        return {
            str(
                key
            ): safe_float(
                value
            )
            for key, value
            in metrics.items()
        }

    raise TypeError(
        "Unsupported trajectory metric object: "
        f"{type(metrics).__name__}."
    )


def run_trajectory_attribution(
    evaluator: Any,
    rotation_gt: np.ndarray,
    rotation_pred: np.ndarray,
    translation_gt: np.ndarray,
    translation_pred: np.ndarray,
    *,
    euler_order: str,
    angles_in_degrees: bool,
) -> Tuple[
    Dict[str, np.ndarray],
    List[Dict[str, Any]],
]:
    trajectories = {
        "GT_R_GT_t": integrate_condition(
            evaluator=evaluator,
            rotations=rotation_gt,
            translations=translation_gt,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        ),
        "Pred_R_Pred_t": integrate_condition(
            evaluator=evaluator,
            rotations=rotation_pred,
            translations=translation_pred,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        ),
        "GT_R_Pred_t": integrate_condition(
            evaluator=evaluator,
            rotations=rotation_gt,
            translations=translation_pred,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        ),
        "Pred_R_GT_t": integrate_condition(
            evaluator=evaluator,
            rotations=rotation_pred,
            translations=translation_gt,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        ),
    }

    ground_truth = (
        trajectories[
            "GT_R_GT_t"
        ]
    )

    rows: List[
        Dict[str, Any]
    ] = []

    descriptions = {
        "Pred_R_Pred_t": (
            "Observed model trajectory"
        ),
        "GT_R_Pred_t": (
            "Rotation error removed; predicted translation retained"
        ),
        "Pred_R_GT_t": (
            "Translation error removed; predicted rotation retained"
        ),
        "GT_R_GT_t": (
            "Ground-truth reconstruction control"
        ),
    }

    order = (
        "Pred_R_Pred_t",
        "GT_R_Pred_t",
        "Pred_R_GT_t",
        "GT_R_GT_t",
    )

    for condition in order:
        metrics = (
            evaluator.compute_trajectory_metrics(
                ground_truth_trajectory=ground_truth,
                predicted_trajectory=(
                    trajectories[
                        condition
                    ]
                ),
            )
        )

        row = {
            "condition": condition,
            "description": descriptions[
                condition
            ],
        }

        row.update(
            metrics_to_dict(
                metrics
            )
        )

        rows.append(
            row
        )

    return (
        trajectories,
        rows,
    )


# ============================================================================
# Attribution interpretation
# ============================================================================


def finite_metric(
    row: Mapping[str, Any],
    key: str,
) -> float:
    value = row.get(
        key
    )

    if value is None:
        return float(
            "nan"
        )

    return float(
        value
    )


def build_attribution_summary(
    metric_rows: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[str, Any]:
    by_condition = {
        str(
            row[
                "condition"
            ]
        ): row
        for row in metric_rows
    }

    observed = by_condition[
        "Pred_R_Pred_t"
    ]

    no_rotation_error = (
        by_condition[
            "GT_R_Pred_t"
        ]
    )

    no_translation_error = (
        by_condition[
            "Pred_R_GT_t"
        ]
    )

    observed_ate = finite_metric(
        observed,
        "ate_rmse",
    )

    gt_r_pred_t_ate = finite_metric(
        no_rotation_error,
        "ate_rmse",
    )

    pred_r_gt_t_ate = finite_metric(
        no_translation_error,
        "ate_rmse",
    )

    observed_drift = finite_metric(
        observed,
        "translational_drift_percent",
    )

    gt_r_pred_t_drift = finite_metric(
        no_rotation_error,
        "translational_drift_percent",
    )

    pred_r_gt_t_drift = finite_metric(
        no_translation_error,
        "translational_drift_percent",
    )

    rotation_ate_reduction = (
        observed_ate
        - gt_r_pred_t_ate
    )

    translation_ate_reduction = (
        observed_ate
        - pred_r_gt_t_ate
    )

    rotation_drift_reduction = (
        observed_drift
        - gt_r_pred_t_drift
    )

    translation_drift_reduction = (
        observed_drift
        - pred_r_gt_t_drift
    )

    if (
        translation_ate_reduction
        > rotation_ate_reduction
    ):
        larger_ate_contributor = (
            "translation"
        )
    elif (
        rotation_ate_reduction
        > translation_ate_reduction
    ):
        larger_ate_contributor = (
            "rotation"
        )
    else:
        larger_ate_contributor = (
            "approximately_equal"
        )

    return {
        "observed_ate_rmse": observed_ate,
        "observed_translation_drift_percent": (
            observed_drift
        ),
        "gt_rotation_pred_translation_ate_rmse": (
            gt_r_pred_t_ate
        ),
        "pred_rotation_gt_translation_ate_rmse": (
            pred_r_gt_t_ate
        ),
        "ate_reduction_when_rotation_error_removed": (
            rotation_ate_reduction
        ),
        "ate_reduction_when_translation_error_removed": (
            translation_ate_reduction
        ),
        "drift_reduction_when_rotation_error_removed": (
            rotation_drift_reduction
        ),
        "drift_reduction_when_translation_error_removed": (
            translation_drift_reduction
        ),
        "larger_ate_contributor_by_counterfactual_reduction": (
            larger_ate_contributor
        ),
        "caution": (
            "Rotation and translation effects are coupled under "
            "SE(3) integration. Counterfactual reductions are "
            "diagnostic contributions, not additive causal percentages."
        ),
    }


# ============================================================================
# Plotting
# ============================================================================


def plot_all_trajectories(
    trajectories: Mapping[
        str,
        np.ndarray
    ],
    *,
    axis_a: int,
    axis_b: int,
    axis_a_label: str,
    axis_b_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    figure = plt.figure(
        figsize=(9.0, 7.0)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    labels = {
        "GT_R_GT_t": "GT R + GT t",
        "Pred_R_Pred_t": "Pred R + Pred t",
        "GT_R_Pred_t": "GT R + Pred t",
        "Pred_R_GT_t": "Pred R + GT t",
    }

    for key in (
        "GT_R_GT_t",
        "Pred_R_Pred_t",
        "GT_R_Pred_t",
        "Pred_R_GT_t",
    ):
        positions = (
            trajectories[
                key
            ][
                :,
                :3,
                3,
            ]
        )

        axis.plot(
            positions[
                :,
                axis_a
            ],
            positions[
                :,
                axis_b
            ],
            label=labels[
                key
            ],
        )

    axis.set_xlabel(
        axis_a_label
    )

    axis.set_ylabel(
        axis_b_label
    )

    axis.set_title(
        "Pose-error attribution trajectories"
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


def plot_scale_ratio(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    regimes: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    scale_ratio = diagnostics[
        "scale_ratio"
    ]

    figure = plt.figure(
        figsize=(10.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    x = np.arange(
        scale_ratio.shape[0]
    )

    axis.plot(
        x,
        scale_ratio,
        linewidth=0.9,
        label="||t_pred|| / ||t_gt||",
    )

    axis.axhline(
        1.0,
        linestyle="--",
        label="Ideal scale",
    )

    axis.set_xlabel(
        "Transition index"
    )

    axis.set_ylabel(
        "Translation scale ratio"
    )

    axis.set_title(
        "Framewise translation magnitude ratio"
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


def plot_direction_error(
    diagnostics: Mapping[
        str,
        np.ndarray
    ],
    output_path: Path,
    dpi: int,
) -> None:
    direction_error = (
        diagnostics[
            "direction_error_deg"
        ]
    )

    figure = plt.figure(
        figsize=(10.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        np.arange(
            direction_error.shape[0]
        ),
        direction_error,
        linewidth=0.9,
    )

    axis.set_xlabel(
        "Transition index"
    )

    axis.set_ylabel(
        "Direction error (deg)"
    )

    axis.set_title(
        "Framewise translation-direction error"
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


# ============================================================================
# Console summary
# ============================================================================


def print_trajectory_table(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    print()
    print("=" * 112)
    print("Counterfactual trajectory attribution")
    print("=" * 112)

    print(
        f"{'Condition':<20}"
        f"{'ATE RMSE':>14}"
        f"{'RPE trans':>14}"
        f"{'RPE rot deg':>14}"
        f"{'Endpoint':>14}"
        f"{'Drift %':>14}"
    )

    print(
        "-" * 112
    )

    for row in rows:
        print(
            f"{str(row['condition']):<20}"
            f"{float(row['ate_rmse']):>14.6f}"
            f"{float(row['rpe_translation_rmse']):>14.6f}"
            f"{float(row['rpe_rotation_rmse_degrees']):>14.6f}"
            f"{float(row['endpoint_error']):>14.6f}"
            f"{float(row['translational_drift_percent']):>14.3f}"
        )

    print(
        "=" * 112
    )


def print_translation_table(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    print()
    print("=" * 120)
    print("Translation magnitude / direction diagnostics")
    print("=" * 120)

    print(
        f"{'Regime':<10}"
        f"{'N':>8}"
        f"{'GT norm':>12}"
        f"{'Pred norm':>12}"
        f"{'Scale med':>12}"
        f"{'Under %':>12}"
        f"{'Dir err':>12}"
        f"{'z RMSE':>12}"
        f"{'z corr':>12}"
    )

    print(
        "-" * 120
    )

    for row in rows:
        print(
            f"{str(row['regime']):<10}"
            f"{int(row['samples']):>8}"
            f"{float(row['gt_norm_mean']):>12.5f}"
            f"{float(row['pred_norm_mean']):>12.5f}"
            f"{float(row['scale_ratio_median']):>12.5f}"
            f"{100.0 * float(row['scale_underprediction_fraction']):>12.2f}"
            f"{float(row['direction_error_mean_deg']):>12.3f}"
            f"{float(row['tz_rmse']):>12.5f}"
            f"{float(row['tz_corr']):>12.4f}"
        )

    print(
        "=" * 120
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
        metadata_rows,
        rotation_gt,
        rotation_pred,
        translation_gt,
        translation_pred,
    ) = load_frame_predictions(
        args.predictions
    )

    num_samples = int(
        rotation_gt.shape[0]
    )

    print("=" * 88)
    print("DeepDCT-VO pose-error attribution audit")
    print("=" * 88)
    print(
        f"Predictions:          {args.predictions}"
    )
    print(
        f"Transitions:          {num_samples}"
    )
    print(
        f"Euler order:          {args.euler_order}"
    )
    print(
        f"Angles in degrees:    {args.angles_in_degrees}"
    )
    print(
        "Trajectory decoder:   "
        "current evaluate_deepdct_vo.py"
    )
    print("=" * 88)

    # ------------------------------------------------------------------
    # Translation magnitude/direction analysis
    # ------------------------------------------------------------------

    diagnostics = (
        translation_diagnostics(
            translation_gt=translation_gt,
            translation_pred=translation_pred,
            epsilon=args.scale_epsilon,
            minimum_direction_norm=(
                args.minimum_direction_norm
            ),
        )
    )

    (
        regimes,
        low_threshold,
        high_threshold,
    ) = derive_regimes(
        gt_tz=diagnostics[
            "gt_tz"
        ],
        quantiles=tuple(
            args.regime_quantiles
        ),
    )

    translation_summary_rows = (
        build_translation_summary_rows(
            diagnostics=diagnostics,
            regimes=regimes,
        )
    )

    frame_rows = (
        build_frame_rows(
            metadata_rows=metadata_rows,
            diagnostics=diagnostics,
            regimes=regimes,
        )
    )

    write_csv(
        args.output_dir
        / "translation_summary.csv",
        translation_summary_rows,
    )

    write_csv(
        args.output_dir
        / "frame_translation_diagnostics.csv",
        frame_rows,
    )

    # ------------------------------------------------------------------
    # Four trajectory counterfactuals
    # ------------------------------------------------------------------

    (
        trajectories,
        trajectory_metric_rows,
    ) = run_trajectory_attribution(
        evaluator=evaluator,
        rotation_gt=rotation_gt,
        rotation_pred=rotation_pred,
        translation_gt=translation_gt,
        translation_pred=translation_pred,
        euler_order=args.euler_order,
        angles_in_degrees=(
            args.angles_in_degrees
        ),
    )

    write_csv(
        args.output_dir
        / "trajectory_attribution.csv",
        trajectory_metric_rows,
    )

    for name, trajectory in (
        trajectories.items()
    ):
        evaluator.save_kitti_trajectory(
            trajectories_dir
            / f"{name}.txt",
            trajectory,
        )

    attribution = (
        build_attribution_summary(
            trajectory_metric_rows
        )
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_all_trajectories(
        trajectories=trajectories,
        axis_a=0,
        axis_b=1,
        axis_a_label="X",
        axis_b_label="Y",
        output_path=(
            plots_dir
            / "trajectory_attribution_xy.png"
        ),
        dpi=args.dpi,
    )

    plot_all_trajectories(
        trajectories=trajectories,
        axis_a=0,
        axis_b=2,
        axis_a_label="X",
        axis_b_label="Z",
        output_path=(
            plots_dir
            / "trajectory_attribution_xz.png"
        ),
        dpi=args.dpi,
    )

    plot_scale_ratio(
        diagnostics=diagnostics,
        regimes=regimes,
        output_path=(
            plots_dir
            / "translation_scale_ratio.png"
        ),
        dpi=args.dpi,
    )

    plot_direction_error(
        diagnostics=diagnostics,
        output_path=(
            plots_dir
            / "translation_direction_error.png"
        ),
        dpi=args.dpi,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    overall_translation = next(
        row
        for row
        in translation_summary_rows
        if row[
            "regime"
        ] == "all"
    )

    summary = {
        "predictions": str(
            args.predictions
        ),
        "transitions": (
            num_samples
        ),
        "trajectory_decoder": (
            str(
                repo_root
                / "scripts"
                / "evaluate_deepdct_vo.py"
            )
        ),
        "euler_order": (
            args.euler_order
        ),
        "angles_in_degrees": (
            args.angles_in_degrees
        ),
        "regime_definition": {
            "quantiles": list(
                args.regime_quantiles
            ),
            "low_threshold_gt_tz": (
                low_threshold
            ),
            "high_threshold_gt_tz": (
                high_threshold
            ),
            "rule": {
                "low": (
                    f"t_z <= {low_threshold}"
                ),
                "medium": (
                    f"{low_threshold} < t_z <= "
                    f"{high_threshold}"
                ),
                "high": (
                    f"t_z > {high_threshold}"
                ),
            },
        },
        "translation_overall": (
            overall_translation
        ),
        "translation_by_regime": (
            translation_summary_rows
        ),
        "trajectory_conditions": (
            trajectory_metric_rows
        ),
        "attribution": (
            attribution
        ),
        "interpretation_caution": (
            "Counterfactual trajectories isolate integration effects "
            "of exported rotation and translation streams. They do not "
            "re-run the translation network under alternative rotation "
            "conditioning. Rotation and translation contributions are "
            "coupled and should not be interpreted as additive percentages."
        ),
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

    print_translation_table(
        translation_summary_rows
    )

    print_trajectory_table(
        trajectory_metric_rows
    )

    print()
    print("Attribution reductions")
    print("-" * 88)
    print(
        "ATE reduction with GT rotation:       "
        f"{attribution['ate_reduction_when_rotation_error_removed']:.6f}"
    )
    print(
        "ATE reduction with GT translation:    "
        f"{attribution['ate_reduction_when_translation_error_removed']:.6f}"
    )
    print(
        "Drift reduction with GT rotation:     "
        f"{attribution['drift_reduction_when_rotation_error_removed']:.3f} pp"
    )
    print(
        "Drift reduction with GT translation:  "
        f"{attribution['drift_reduction_when_translation_error_removed']:.3f} pp"
    )
    print(
        "Larger ATE contributor:               "
        f"{attribution['larger_ate_contributor_by_counterfactual_reduction']}"
    )
    print("-" * 88)

    print()
    print("=" * 88)
    print("Pose-error attribution audit complete")
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