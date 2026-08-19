#!/usr/bin/env python3
"""
Audit the DeepDCT-VO rotation representation/readout.

Purpose
-------
Given:

    rotation_representations.npz
        rotation_rep: [N, D]

    frame_predictions.csv
        rotation_gt_{x,y,z}
        rotation_pred_{x,y,z}

    checkpoint
        rotation_head.dense.weight: [3, D]
        rotation_head.dense.bias:   [3]

reconstruct:

    reconstructed_rotation = h @ W.T + b

and verify that this exactly reproduces the evaluator-exported rotation
predictions.

This is a control experiment for the rotation representation audit.

If reconstruction PASSes:
    - the exported h is genuinely the representation consumed by the
      trained rotation readout;
    - any failure in the exported rotation output can be attributed to
      the combination of h and the learned W,b mapping rather than an
      extraction mismatch.

The script additionally reports:
    - reconstruction max absolute error / RMSE;
    - per-axis GT vs exported prediction statistics;
    - per-axis GT vs reconstructed prediction statistics;
    - output bias, RMSE, correlation, std ratio;
    - learned weight norms and pairwise weight-vector angles;
    - contribution standard deviation h @ W.T before bias;
    - prediction mean and bias contribution;
    - optional baseline comparison against a constant GT-mean predictor.

Important
---------
The representation file and frame_predictions.csv MUST have been generated
from the same checkpoint supplied with --checkpoint.

Example
-------
python scripts/audit_rotation_representation_readout.py \\
    --checkpoint experiments/semantic_depth_identity_output/best_validation.pt \\
    --representation experiments/semantic_depth_rotation_rep_inputs/10/rotation_representations.npz \\
    --predictions experiments/semantic_depth_rotation_rep_inputs/10/frame_predictions.csv \\
    --output-dir experiments/semantic_depth_rotation_readout_audit/10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch


# ============================================================================
# Constants
# ============================================================================

AXES = ("x", "y", "z")

GT_COLUMNS = (
    "rotation_gt_x",
    "rotation_gt_y",
    "rotation_gt_z",
)

PRED_COLUMNS = (
    "rotation_pred_x",
    "rotation_pred_y",
    "rotation_pred_z",
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the learned DeepDCT-VO rotation representation/readout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Exact checkpoint used to generate the representation "
            "and predictions."
        ),
    )

    parser.add_argument(
        "--representation",
        type=Path,
        required=True,
        help="rotation_representations.npz.",
    )

    parser.add_argument(
        "--representation-key",
        type=str,
        default="rotation_rep",
        help="Array key inside rotation_representations.npz.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Matching evaluator frame_predictions.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Audit output directory.",
    )

    parser.add_argument(
        "--weight-key",
        type=str,
        default="rotation_head.dense.weight",
        help="Rotation readout weight key in model_state_dict.",
    )

    parser.add_argument(
        "--bias-key",
        type=str,
        default="rotation_head.dense.bias",
        help="Rotation readout bias key in model_state_dict.",
    )

    parser.add_argument(
        "--pass-tolerance",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum allowed absolute reconstruction difference "
            "for PASS."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    for path, description in (
        (
            args.checkpoint,
            "checkpoint",
        ),
        (
            args.representation,
            "representation file",
        ),
        (
            args.predictions,
            "prediction CSV",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{description} does not exist: {path}"
            )

    if args.pass_tolerance <= 0.0:
        raise ValueError(
            "--pass-tolerance must be positive."
        )


# ============================================================================
# Checkpoint loading
# ============================================================================


def load_checkpoint(
    path: Path,
) -> Mapping[str, Any]:
    checkpoint = torch.load(
        path,
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

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing model_state_dict."
        )

    return checkpoint


def normalize_state_dict_keys(
    state_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Remove a leading 'module.' prefix if the checkpoint came from
    DataParallel / DistributedDataParallel.
    """

    normalized: Dict[
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

        normalized[
            key
        ] = value

    return normalized


def load_rotation_readout(
    checkpoint: Mapping[str, Any],
    weight_key: str,
    bias_key: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    state_dict = checkpoint[
        "model_state_dict"
    ]

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise TypeError(
            "model_state_dict must be a mapping."
        )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    if weight_key not in state_dict:
        candidates = [
            key
            for key in state_dict
            if (
                "rotation_head"
                in key
                and "weight"
                in key
            )
        ]

        raise KeyError(
            f"Could not find {weight_key!r}.\n"
            f"Rotation weight candidates: {candidates}"
        )

    if bias_key not in state_dict:
        candidates = [
            key
            for key in state_dict
            if (
                "rotation_head"
                in key
                and "bias"
                in key
            )
        ]

        raise KeyError(
            f"Could not find {bias_key!r}.\n"
            f"Rotation bias candidates: {candidates}"
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
        or weight.shape[0] != 3
    ):
        raise ValueError(
            "Expected rotation readout weight shape [3,D], "
            f"received {weight.shape}."
        )

    if bias.shape != (
        3,
    ):
        raise ValueError(
            "Expected rotation readout bias shape [3], "
            f"received {bias.shape}."
        )

    return weight, bias


# ============================================================================
# Input loading
# ============================================================================


def load_representation(
    path: Path,
    key: str,
) -> np.ndarray:
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
            archive[
                key
            ],
            dtype=np.float64,
        )

    if representation.ndim > 2:
        representation = representation.reshape(
            representation.shape[0],
            -1,
        )

    if representation.ndim != 2:
        raise ValueError(
            "Rotation representation must have shape [N,D], "
            f"received {representation.shape}."
        )

    if not np.all(
        np.isfinite(
            representation
        )
    ):
        raise ValueError(
            "Rotation representation contains non-finite values."
        )

    return representation


def load_predictions(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[Dict[str, str]],
]:
    gt_rows: List[
        List[float]
    ] = []

    pred_rows: List[
        List[float]
    ] = []

    metadata_rows: List[
        Dict[str, str]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fields = (
            reader.fieldnames
            if reader.fieldnames
            is not None
            else []
        )

        required = (
            *GT_COLUMNS,
            *PRED_COLUMNS,
        )

        missing = [
            column
            for column in required
            if column not in fields
        ]

        if missing:
            raise KeyError(
                "Prediction CSV is missing required columns: "
                f"{missing}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                gt = [
                    float(
                        row[column]
                    )
                    for column
                    in GT_COLUMNS
                ]

                pred = [
                    float(
                        row[column]
                    )
                    for column
                    in PRED_COLUMNS
                ]

            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    f"Could not parse CSV row {row_number}."
                ) from error

            gt_rows.append(
                gt
            )

            pred_rows.append(
                pred
            )

            metadata_rows.append(
                dict(
                    row
                )
            )

    gt_array = np.asarray(
        gt_rows,
        dtype=np.float64,
    )

    pred_array = np.asarray(
        pred_rows,
        dtype=np.float64,
    )

    if (
        gt_array.ndim != 2
        or gt_array.shape[1] != 3
    ):
        raise ValueError(
            f"Unexpected GT shape: {gt_array.shape}."
        )

    if pred_array.shape != gt_array.shape:
        raise ValueError(
            "GT/prediction shape mismatch: "
            f"{gt_array.shape} vs {pred_array.shape}."
        )

    if not np.all(
        np.isfinite(
            gt_array
        )
    ):
        raise ValueError(
            "GT rotations contain non-finite values."
        )

    if not np.all(
        np.isfinite(
            pred_array
        )
    ):
        raise ValueError(
            "Predicted rotations contain non-finite values."
        )

    return (
        gt_array,
        pred_array,
        metadata_rows,
    )


# ============================================================================
# Statistics
# ============================================================================


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if (
        np.std(
            x
        )
        <= 1.0e-12
        or np.std(
            y
        )
        <= 1.0e-12
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


def axis_statistics(
    gt: np.ndarray,
    prediction: np.ndarray,
) -> List[
    Dict[str, float]
]:
    rows: List[
        Dict[str, float]
    ] = []

    for index, axis in enumerate(
        AXES
    ):
        gt_axis = gt[
            :,
            index
        ]

        pred_axis = prediction[
            :,
            index
        ]

        error = (
            pred_axis
            - gt_axis
        )

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
                "axis": axis,
                "gt_mean": float(
                    np.mean(
                        gt_axis
                    )
                ),
                "pred_mean": float(
                    np.mean(
                        pred_axis
                    )
                ),
                "bias": float(
                    np.mean(
                        error
                    )
                ),
                "mae": float(
                    np.mean(
                        np.abs(
                            error
                        )
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            error ** 2
                        )
                    )
                ),
                "gt_std": gt_std,
                "pred_std": pred_std,
                "std_ratio": (
                    pred_std
                    / gt_std
                    if gt_std > 1.0e-12
                    else float(
                        "nan"
                    )
                ),
                "correlation": safe_correlation(
                    gt_axis,
                    pred_axis,
                ),
            }
        )

    return rows


def weight_statistics(
    weight: np.ndarray,
    bias: np.ndarray,
    representation: np.ndarray,
) -> List[
    Dict[str, float]
]:
    centered_representation = (
        representation
        - np.mean(
            representation,
            axis=0,
        )
    )

    linear_centered_output = (
        centered_representation
        @ weight.T
    )

    rows: List[
        Dict[str, float]
    ] = []

    for axis_index, axis in enumerate(
        AXES
    ):
        row_weight = weight[
            axis_index
        ]

        rows.append(
            {
                "axis": axis,
                "weight_l1": float(
                    np.sum(
                        np.abs(
                            row_weight
                        )
                    )
                ),
                "weight_l2": float(
                    np.linalg.norm(
                        row_weight
                    )
                ),
                "weight_max_abs": float(
                    np.max(
                        np.abs(
                            row_weight
                        )
                    )
                ),
                "bias": float(
                    bias[
                        axis_index
                    ]
                ),
                "centered_linear_output_std": float(
                    np.std(
                        linear_centered_output[
                            :,
                            axis_index
                        ]
                    )
                ),
                "centered_linear_output_mean_abs": float(
                    np.mean(
                        np.abs(
                            linear_centered_output[
                                :,
                                axis_index
                            ]
                        )
                    )
                ),
            }
        )

    return rows


def pairwise_weight_angles(
    weight: np.ndarray,
) -> List[
    Dict[str, float]
]:
    rows: List[
        Dict[str, float]
    ] = []

    pairs = (
        (0, 1),
        (0, 2),
        (1, 2),
    )

    for first, second in pairs:
        a = weight[
            first
        ]

        b = weight[
            second
        ]

        denominator = (
            np.linalg.norm(
                a
            )
            * np.linalg.norm(
                b
            )
        )

        if denominator <= 1.0e-12:
            cosine = float(
                "nan"
            )

            angle = float(
                "nan"
            )

        else:
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

            angle = float(
                np.degrees(
                    np.arccos(
                        cosine
                    )
                )
            )

        rows.append(
            {
                "axis_a": AXES[
                    first
                ],
                "axis_b": AXES[
                    second
                ],
                "cosine_similarity": cosine,
                "angle_deg": angle,
            }
        )

    return rows


def constant_mean_baseline(
    gt: np.ndarray,
) -> Dict[str, Any]:
    mean = np.mean(
        gt,
        axis=0,
    )

    prediction = np.repeat(
        mean[
            None,
            :
        ],
        gt.shape[0],
        axis=0,
    )

    error = (
        prediction
        - gt
    )

    return {
        "mean_prediction": (
            mean.tolist()
        ),
        "vector_rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "axis_statistics": (
            axis_statistics(
                gt,
                prediction,
            )
        ),
    }


# ============================================================================
# File output
# ============================================================================


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
            str(
                key
            ): json_ready(
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


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    args.checkpoint = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    args.representation = (
        args.representation
        .expanduser()
        .resolve()
    )

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

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    weight, bias = load_rotation_readout(
        checkpoint=checkpoint,
        weight_key=args.weight_key,
        bias_key=args.bias_key,
    )

    representation = load_representation(
        args.representation,
        args.representation_key,
    )

    (
        rotation_gt,
        rotation_exported,
        metadata_rows,
    ) = load_predictions(
        args.predictions
    )

    # ----------------------------------------------------------
    # Shape checks
    # ----------------------------------------------------------

    if representation.shape[0] != rotation_gt.shape[0]:
        raise ValueError(
            "Representation/prediction sample count mismatch: "
            f"{representation.shape[0]} vs "
            f"{rotation_gt.shape[0]}."
        )

    if representation.shape[1] != weight.shape[1]:
        raise ValueError(
            "Representation/readout feature dimension mismatch: "
            f"h has D={representation.shape[1]}, "
            f"W expects D={weight.shape[1]}."
        )

    # ----------------------------------------------------------
    # Exact learned-readout reconstruction
    # ----------------------------------------------------------

    linear_output = (
        representation
        @ weight.T
    )

    rotation_reconstructed = (
        linear_output
        + bias[
            None,
            :
        ]
    )

    reconstruction_difference = (
        rotation_reconstructed
        - rotation_exported
    )

    max_abs_difference = float(
        np.max(
            np.abs(
                reconstruction_difference
            )
        )
    )

    mean_abs_difference = float(
        np.mean(
            np.abs(
                reconstruction_difference
            )
        )
    )

    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                reconstruction_difference ** 2
            )
        )
    )

    row_l2_difference = np.linalg.norm(
        reconstruction_difference,
        axis=1,
    )

    matching_rows_mask = np.all(
        np.abs(
            reconstruction_difference
        )
        <= args.pass_tolerance,
        axis=1,
    )

    matching_rows = int(
        np.count_nonzero(
            matching_rows_mask
        )
    )

    first_mismatch = None

    mismatch_indices = np.flatnonzero(
        ~matching_rows_mask
    )

    if mismatch_indices.size:
        first_mismatch = int(
            mismatch_indices[
                0
            ]
        )

    audit_pass = bool(
        max_abs_difference
        <= args.pass_tolerance
    )

    # ----------------------------------------------------------
    # Readout behavior
    # ----------------------------------------------------------

    exported_axis_stats = axis_statistics(
        rotation_gt,
        rotation_exported,
    )

    reconstructed_axis_stats = axis_statistics(
        rotation_gt,
        rotation_reconstructed,
    )

    weight_rows = weight_statistics(
        weight=weight,
        bias=bias,
        representation=representation,
    )

    weight_angle_rows = pairwise_weight_angles(
        weight
    )

    constant_baseline = constant_mean_baseline(
        rotation_gt
    )

    model_vector_rmse = float(
        np.sqrt(
            np.mean(
                (
                    rotation_exported
                    - rotation_gt
                )
                ** 2
            )
        )
    )

    # ----------------------------------------------------------
    # Frame-wise reconstruction output
    # ----------------------------------------------------------

    frame_rows: List[
        Dict[str, Any]
    ] = []

    for index in range(
        representation.shape[0]
    ):
        metadata = metadata_rows[
            index
        ]

        row: Dict[
            str,
            Any,
        ] = {
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
            "row_l2_reconstruction_difference": float(
                row_l2_difference[
                    index
                ]
            ),
        }

        for axis_index, axis in enumerate(
            AXES
        ):
            row[
                f"rotation_gt_{axis}"
            ] = float(
                rotation_gt[
                    index,
                    axis_index,
                ]
            )

            row[
                f"rotation_exported_{axis}"
            ] = float(
                rotation_exported[
                    index,
                    axis_index,
                ]
            )

            row[
                f"rotation_reconstructed_{axis}"
            ] = float(
                rotation_reconstructed[
                    index,
                    axis_index,
                ]
            )

            row[
                f"reconstruction_difference_{axis}"
            ] = float(
                reconstruction_difference[
                    index,
                    axis_index,
                ]
            )

            row[
                f"linear_contribution_{axis}"
            ] = float(
                linear_output[
                    index,
                    axis_index,
                ]
            )

        frame_rows.append(
            row
        )

    write_csv(
        args.output_dir
        / "frame_reconstruction.csv",
        frame_rows,
    )

    write_csv(
        args.output_dir
        / "rotation_axis_statistics.csv",
        exported_axis_stats,
    )

    write_csv(
        args.output_dir
        / "rotation_readout_weight_statistics.csv",
        weight_rows,
    )

    write_csv(
        args.output_dir
        / "rotation_readout_weight_angles.csv",
        weight_angle_rows,
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    summary = {
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "representation": str(
            args.representation
        ),
        "representation_key": (
            args.representation_key
        ),
        "predictions": str(
            args.predictions
        ),
        "samples": int(
            representation.shape[0]
        ),
        "representation_dimension": int(
            representation.shape[1]
        ),
        "weight_shape": list(
            weight.shape
        ),
        "bias_shape": list(
            bias.shape
        ),
        "reconstruction": {
            "status": (
                "PASS"
                if audit_pass
                else "FAIL"
            ),
            "pass_tolerance": float(
                args.pass_tolerance
            ),
            "max_absolute_difference": (
                max_abs_difference
            ),
            "mean_absolute_difference": (
                mean_abs_difference
            ),
            "rmse": reconstruction_rmse,
            "maximum_row_l2_difference": float(
                np.max(
                    row_l2_difference
                )
            ),
            "matching_rows": (
                matching_rows
            ),
            "total_rows": int(
                representation.shape[0]
            ),
            "first_mismatch_index": (
                first_mismatch
            ),
        },
        "exported_prediction_statistics": (
            exported_axis_stats
        ),
        "reconstructed_prediction_statistics": (
            reconstructed_axis_stats
        ),
        "readout_weight_statistics": (
            weight_rows
        ),
        "readout_weight_angles": (
            weight_angle_rows
        ),
        "constant_gt_mean_baseline": (
            constant_baseline
        ),
        "model_vector_rmse": (
            model_vector_rmse
        ),
        "interpretation": {
            "if_pass": (
                "The saved rotation_rep is confirmed to be the exact "
                "representation consumed by rotation_head.dense. "
                "The learned W,b readout reproduces the evaluator "
                "rotation predictions."
            ),
            "if_fail": (
                "Do not interpret readout statistics until the "
                "representation extraction/checkpoint mismatch is resolved."
            ),
        },
    }

    with (
        args.output_dir
        / "summary.json"
    ).open(
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

    # ----------------------------------------------------------
    # Console report
    # ----------------------------------------------------------

    print()
    print("=" * 88)
    print("Rotation Representation / Readout Reconstruction Audit")
    print("=" * 88)

    print(
        f"Checkpoint:             {args.checkpoint}"
    )

    print(
        f"Checkpoint epoch:       {checkpoint.get('epoch')}"
    )

    print(
        f"Representation:         {args.representation}"
    )

    print(
        f"Representation key:     {args.representation_key}"
    )

    print(
        f"Representation shape:   {representation.shape}"
    )

    print(
        f"Dense weight key:       {args.weight_key}"
    )

    print(
        f"Dense bias key:         {args.bias_key}"
    )

    print(
        f"Dense weight shape:     {weight.shape}"
    )

    print(
        f"Samples:                {representation.shape[0]}"
    )

    print("-" * 88)

    print(
        f"Audit status:                    "
        f"{'PASS' if audit_pass else 'FAIL'}"
    )

    print(
        f"Maximum absolute difference:     "
        f"{max_abs_difference:.12e}"
    )

    print(
        f"Mean absolute difference:        "
        f"{mean_abs_difference:.12e}"
    )

    print(
        f"RMSE across all elements:        "
        f"{reconstruction_rmse:.12e}"
    )

    print(
        f"Maximum row L2 difference:       "
        f"{np.max(row_l2_difference):.12e}"
    )

    print(
        f"Matching rows:                   "
        f"{matching_rows}/{representation.shape[0]}"
    )

    print(
        f"First mismatch index:            "
        f"{first_mismatch}"
    )

    print()
    print("=" * 88)
    print("Learned rotation readout quality")
    print("=" * 88)

    print(
        f"{'Axis':<8}"
        f"{'Bias':>14}"
        f"{'RMSE':>14}"
        f"{'GT std':>14}"
        f"{'Pred std':>14}"
        f"{'Std ratio':>14}"
        f"{'Corr':>14}"
    )

    print("-" * 88)

    for row in exported_axis_stats:
        print(
            f"{row['axis']:<8}"
            f"{row['bias']:>14.8f}"
            f"{row['rmse']:>14.8f}"
            f"{row['gt_std']:>14.8f}"
            f"{row['pred_std']:>14.8f}"
            f"{row['std_ratio']:>14.4f}"
            f"{row['correlation']:>14.4f}"
        )

    print("=" * 88)

    print()
    print("=" * 88)
    print("Learned weight statistics")
    print("=" * 88)

    print(
        f"{'Axis':<8}"
        f"{'||W||2':>14}"
        f"{'Max |W|':>14}"
        f"{'Bias':>14}"
        f"{'Linear std':>14}"
    )

    print("-" * 88)

    for row in weight_rows:
        print(
            f"{row['axis']:<8}"
            f"{row['weight_l2']:>14.8f}"
            f"{row['weight_max_abs']:>14.8f}"
            f"{row['bias']:>14.8f}"
            f"{row['centered_linear_output_std']:>14.8f}"
        )

    print("=" * 88)

    print()
    print(
        "Model vector RMSE:              "
        f"{model_vector_rmse:.9f}"
    )

    print(
        "Constant GT-mean baseline RMSE: "
        f"{constant_baseline['vector_rmse']:.9f}"
    )

    print()
    print(
        f"Outputs saved to: {args.output_dir}"
    )

    if not audit_pass:
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()