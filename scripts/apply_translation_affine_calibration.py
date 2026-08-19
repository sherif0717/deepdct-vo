"""Apply fixed affine translation calibration to prediction CSV columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply affine calibration to translation predictions."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--calibration-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--replace-predictions",
        action="store_true",
        help=(
            "Replace translation_pred_x/y/z. Otherwise add "
            "translation_pred_calibrated_x/y/z columns."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataframe = pd.read_csv(args.input_csv)

    with args.calibration_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(file)

    for axis in AXES:
        source_column = f"translation_pred_{axis}"

        if source_column not in dataframe.columns:
            raise KeyError(
                f"Missing column: {source_column}"
            )

        values = dataframe[
            source_column
        ].to_numpy(dtype=np.float64)

        parameters = calibration["axes"][axis]

        calibrated = (
            float(parameters["scale"]) * values
            + float(parameters["offset"])
        )

        if args.replace_predictions:
            destination_column = source_column
        else:
            destination_column = (
                f"translation_pred_calibrated_{axis}"
            )

        dataframe[destination_column] = calibrated

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output_csv,
        index=False,
    )

    print(
        "Saved calibrated predictions:",
        args.output_csv.resolve(),
    )


if __name__ == "__main__":
    main()