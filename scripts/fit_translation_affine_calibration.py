"""Fit per-axis affine translation calibration on a validation sequence.

Fits:

    ground_truth = scale * prediction + offset

The resulting parameters must be fitted on validation data only and then
applied unchanged to held-out test predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-axis affine translation calibration."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Validation frame_predictions.csv.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output calibration JSON.",
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help=(
            "Optional ridge regularization on the scale term. "
            "Normally leave at zero for the initial audit."
        ),
    )

    return parser.parse_args()


def fit_affine(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    ridge: float,
) -> Dict[str, float]:
    """Fit ground_truth = scale * prediction + offset."""

    if prediction.ndim != 1 or ground_truth.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have equal shape.")

    design = np.column_stack(
        [
            prediction,
            np.ones_like(prediction),
        ]
    )

    regularizer = np.array(
        [
            [ridge, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    normal_matrix = design.T @ design + regularizer
    normal_vector = design.T @ ground_truth

    try:
        parameters = np.linalg.solve(
            normal_matrix,
            normal_vector,
        )
    except np.linalg.LinAlgError:
        parameters = np.linalg.lstsq(
            design,
            ground_truth,
            rcond=None,
        )[0]

    scale = float(parameters[0])
    offset = float(parameters[1])

    calibrated = scale * prediction + offset
    residual = calibrated - ground_truth

    return {
        "scale": scale,
        "offset": offset,
        "calibration_mse": float(np.mean(residual ** 2)),
        "calibration_rmse": float(
            np.sqrt(np.mean(residual ** 2))
        ),
        "calibration_mae": float(
            np.mean(np.abs(residual))
        ),
        "sample_count": int(prediction.size),
    }


def main() -> None:
    args = parse_args()

    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)

    if args.ridge < 0:
        raise ValueError("--ridge cannot be negative.")

    dataframe = pd.read_csv(args.input_csv)

    calibration: Dict[str, object] = {
        "model": "ground_truth = scale * prediction + offset",
        "fit_source": str(args.input_csv.resolve()),
        "ridge": args.ridge,
        "axes": {},
    }

    for axis in AXES:
        prediction_column = f"translation_pred_{axis}"
        ground_truth_column = f"translation_gt_{axis}"

        missing = [
            column
            for column in (
                prediction_column,
                ground_truth_column,
            )
            if column not in dataframe.columns
        ]

        if missing:
            raise KeyError(
                f"Missing columns: {missing}"
            )

        prediction = dataframe[
            prediction_column
        ].to_numpy(dtype=np.float64)

        ground_truth = dataframe[
            ground_truth_column
        ].to_numpy(dtype=np.float64)

        valid = (
            np.isfinite(prediction)
            & np.isfinite(ground_truth)
        )

        if valid.sum() < 2:
            raise ValueError(
                f"Too few valid samples for axis {axis}."
            )

        calibration["axes"][axis] = fit_affine(
            prediction=prediction[valid],
            ground_truth=ground_truth[valid],
            ridge=args.ridge,
        )

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration,
            file,
            indent=2,
            sort_keys=True,
        )

    print("=" * 72)
    print("Affine calibration fitted")
    print("=" * 72)

    for axis in AXES:
        values = calibration["axes"][axis]

        print(
            f"{axis}: "
            f"scale={values['scale']:.9f}, "
            f"offset={values['offset']:.9f}, "
            f"RMSE={values['calibration_rmse']:.9f}"
        )

    print(f"Saved: {args.output_json.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()