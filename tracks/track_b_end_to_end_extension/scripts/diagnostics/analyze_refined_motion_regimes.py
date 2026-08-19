"""Refined DeepDCT directional-translation regime analysis.

Analyzes KITTI DeepDCT label files using fixed tc_z motion intervals:

    very_low:    tc_z < 0.25
    low:         0.25 <= tc_z < 0.50
    medium_low:  0.50 <= tc_z < 0.75
    medium:      0.75 <= tc_z < 1.00
    high:        1.00 <= tc_z < 1.25
    very_high:   1.25 <= tc_z < 1.50
    extreme:     tc_z >= 1.50

Default split:

    train:       00-08
    validation:  09
    test:        10

Expected label ordering:

    tc_x tc_y tc_z rotation_x rotation_y rotation_z

Outputs
-------
refined_motion_regime_analysis/
├── summary.json
├── global_split_regime_statistics.csv
├── per_sequence_regime_statistics.csv
├── per_sequence_statistics.csv
├── frame_regime_assignments.csv
└── plots/
    ├── regime_fraction_by_split.png
    ├── regime_count_by_split.png
    ├── regime_fraction_by_sequence.png
    ├── tc_z_distribution_by_split.png
    └── tc_z_distribution_by_sequence.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRAIN_SEQUENCES = (
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
)

DEFAULT_VALIDATION_SEQUENCES = ("09",)
DEFAULT_TEST_SEQUENCES = ("10",)

REGIME_NAMES = (
    "very_low",
    "low",
    "medium_low",
    "medium",
    "high",
    "very_high",
    "extreme",
)

# Intervals:
#
# (-inf, 0.25)
# [0.25, 0.50)
# [0.50, 0.75)
# [0.75, 1.00)
# [1.00, 1.25)
# [1.25, 1.50)
# [1.50, +inf)
REGIME_BOUNDARIES = np.asarray(
    [
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze refined DeepDCT tc_z motion regimes across "
            "KITTI train, validation, and test splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root containing out_csv/.",
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=list(DEFAULT_TRAIN_SEQUENCES),
        help="Training KITTI sequences.",
    )

    parser.add_argument(
        "--validation-sequences",
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEQUENCES),
        help="Validation KITTI sequences.",
    )

    parser.add_argument(
        "--test-sequences",
        nargs="+",
        default=list(DEFAULT_TEST_SEQUENCES),
        help="Held-out test KITTI sequences.",
    )

    parser.add_argument(
        "--label-pattern",
        type=str,
        default="{sequence}_dct.txt",
        help="Label filename pattern below out_csv/.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("refined_motion_regime_analysis"),
        help="Analysis output directory.",
    )

    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=80,
        help="Number of shared histogram bins.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved plot resolution.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line configuration."""

    if args.histogram_bins <= 0:
        raise ValueError(
            "--histogram-bins must be positive."
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )

    all_sequences = (
        list(args.train_sequences)
        + list(args.validation_sequences)
        + list(args.test_sequences)
    )

    normalized = [
        str(sequence).zfill(2)
        for sequence in all_sequences
    ]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "A sequence appears in more than one split."
        )


def read_numeric_labels(
    path: Path,
) -> np.ndarray:
    """Read a DeepDCT label file as [N, 6].

    Supports ordinary whitespace-separated .txt files and optionally
    comma-separated rows. If an optional frame-index column exists,
    the final six numeric columns are retained.
    """

    rows: List[List[float]] = []

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

            tokens = line.replace(
                ",",
                " ",
            ).split()

            try:
                numeric = [
                    float(token)
                    for token in tokens
                ]
            except ValueError:
                # Allow a single header before numeric data.
                if not rows:
                    continue

                raise ValueError(
                    f"Non-numeric value in {path} "
                    f"at line {line_number}: {line!r}"
                )

            if len(numeric) < 6:
                raise ValueError(
                    f"Expected at least six numeric values in "
                    f"{path} line {line_number}; received "
                    f"{len(numeric)}."
                )

            rows.append(
                numeric[-6:]
            )

    if not rows:
        raise ValueError(
            f"No numeric DCT labels found in {path}."
        )

    labels = np.asarray(
        rows,
        dtype=np.float64,
    )

    if labels.shape[1] != 6:
        raise ValueError(
            f"Expected [N, 6], received {labels.shape} "
            f"from {path}."
        )

    if not np.isfinite(labels).all():
        raise ValueError(
            f"Non-finite DCT labels found in {path}."
        )

    return labels


def classify_regime(
    tc_z: np.ndarray,
) -> np.ndarray:
    """Assign refined tc_z regime indices.

    np.digitize with right=False gives:

        x < 0.25       -> 0
        0.25 <= x < .5 -> 1
        ...
        x >= 1.50      -> 6
    """

    indices = np.digitize(
        tc_z,
        REGIME_BOUNDARIES,
        right=False,
    )

    if np.any(indices < 0):
        raise RuntimeError(
            "Invalid negative regime index."
        )

    if np.any(indices >= len(REGIME_NAMES)):
        raise RuntimeError(
            "Invalid regime index outside configured range."
        )

    return indices.astype(
        np.int64,
        copy=False,
    )


def load_split(
    *,
    data_root: Path,
    sequences: Sequence[str],
    split_name: str,
    label_pattern: str,
) -> pd.DataFrame:
    """Load labels for one experimental split."""

    label_root = (
        data_root
        / "out_csv"
    )

    frames: List[pd.DataFrame] = []

    for sequence_value in sequences:
        sequence = str(
            sequence_value
        ).zfill(2)

        path = (
            label_root
            / label_pattern.format(
                sequence=sequence
            )
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing DCT label file: {path}"
            )

        labels = read_numeric_labels(
            path
        )

        directional_translation = (
            labels[:, 0:3]
        )

        rotation = (
            labels[:, 3:6]
        )

        tc_norm = np.linalg.norm(
            directional_translation,
            axis=1,
        )

        tc_z = directional_translation[
            :,
            2,
        ]

        regime_index = classify_regime(
            tc_z
        )

        frame = pd.DataFrame(
            {
                "split": split_name,
                "sequence": sequence,
                "transition_index": np.arange(
                    labels.shape[0],
                    dtype=np.int64,
                ),
                "tc_x": directional_translation[:, 0],
                "tc_y": directional_translation[:, 1],
                "tc_z": tc_z,
                "tc_norm": tc_norm,
                "rotation_x": rotation[:, 0],
                "rotation_y": rotation[:, 1],
                "rotation_z": rotation[:, 2],
                "regime_index": regime_index,
                "regime_name": [
                    REGIME_NAMES[index]
                    for index in regime_index
                ],
                "label_path": str(path),
            }
        )

        frames.append(
            frame
        )

    if not frames:
        raise ValueError(
            f"Split {split_name!r} contains no sequences."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_all_data(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Load train, validation, and test labels."""

    frames = [
        load_split(
            data_root=args.data_root,
            sequences=args.train_sequences,
            split_name="train",
            label_pattern=args.label_pattern,
        ),
        load_split(
            data_root=args.data_root,
            sequences=args.validation_sequences,
            split_name="validation",
            label_pattern=args.label_pattern,
        ),
        load_split(
            data_root=args.data_root,
            sequences=args.test_sequences,
            split_name="test",
            label_pattern=args.label_pattern,
        ),
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def safe_std(
    values: np.ndarray,
) -> float:
    """Return sample standard deviation."""

    if values.size <= 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def summarize_values(
    values: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """Return common descriptive statistics."""

    return {
        f"{prefix}_mean": float(
            np.mean(values)
        ),
        f"{prefix}_standard_deviation": safe_std(
            values
        ),
        f"{prefix}_minimum": float(
            np.min(values)
        ),
        f"{prefix}_percentile_25": float(
            np.percentile(
                values,
                25.0,
            )
        ),
        f"{prefix}_median": float(
            np.median(values)
        ),
        f"{prefix}_percentile_75": float(
            np.percentile(
                values,
                75.0,
            )
        ),
        f"{prefix}_maximum": float(
            np.max(values)
        ),
    }


def compute_split_regime_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute regime counts and statistics for each split."""

    rows: List[Dict[str, object]] = []

    split_order = (
        "train",
        "validation",
        "test",
    )

    for split in split_order:
        split_frame = dataframe[
            dataframe["split"] == split
        ]

        total_count = len(
            split_frame
        )

        for regime_index, regime_name in enumerate(
            REGIME_NAMES
        ):
            subset = split_frame[
                split_frame["regime_name"]
                == regime_name
            ]

            count = len(
                subset
            )

            row: Dict[str, object] = {
                "split": split,
                "regime_index": regime_index,
                "regime_name": regime_name,
                "count": count,
                "fraction": (
                    count / total_count
                    if total_count
                    else float("nan")
                ),
            }

            if count:
                tc_z = subset[
                    "tc_z"
                ].to_numpy(
                    dtype=np.float64
                )

                tc_norm = subset[
                    "tc_norm"
                ].to_numpy(
                    dtype=np.float64
                )

                row.update(
                    summarize_values(
                        tc_z,
                        "tc_z",
                    )
                )

                row["tc_norm_mean"] = float(
                    np.mean(tc_norm)
                )

                row["tc_norm_median"] = float(
                    np.median(tc_norm)
                )
            else:
                for key in (
                    "tc_z_mean",
                    "tc_z_standard_deviation",
                    "tc_z_minimum",
                    "tc_z_percentile_25",
                    "tc_z_median",
                    "tc_z_percentile_75",
                    "tc_z_maximum",
                    "tc_norm_mean",
                    "tc_norm_median",
                ):
                    row[key] = float(
                        "nan"
                    )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def compute_per_sequence_regime_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute one row per sequence and regime."""

    rows: List[Dict[str, object]] = []

    grouped = dataframe.groupby(
        [
            "split",
            "sequence",
        ],
        sort=True,
    )

    for (
        split,
        sequence,
    ), sequence_frame in grouped:

        total_count = len(
            sequence_frame
        )

        for regime_index, regime_name in enumerate(
            REGIME_NAMES
        ):
            subset = sequence_frame[
                sequence_frame["regime_name"]
                == regime_name
            ]

            count = len(
                subset
            )

            row: Dict[str, object] = {
                "split": split,
                "sequence": sequence,
                "regime_index": regime_index,
                "regime_name": regime_name,
                "count": count,
                "fraction": (
                    count / total_count
                    if total_count
                    else float("nan")
                ),
            }

            if count:
                tc_z = subset[
                    "tc_z"
                ].to_numpy(
                    dtype=np.float64
                )

                row.update(
                    summarize_values(
                        tc_z,
                        "tc_z",
                    )
                )
            else:
                for key in (
                    "tc_z_mean",
                    "tc_z_standard_deviation",
                    "tc_z_minimum",
                    "tc_z_percentile_25",
                    "tc_z_median",
                    "tc_z_percentile_75",
                    "tc_z_maximum",
                ):
                    row[key] = float(
                        "nan"
                    )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def compute_per_sequence_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute overall tc_z statistics for every sequence."""

    rows: List[Dict[str, object]] = []

    grouped = dataframe.groupby(
        [
            "split",
            "sequence",
        ],
        sort=True,
    )

    for (
        split,
        sequence,
    ), group in grouped:

        tc_z = group[
            "tc_z"
        ].to_numpy(
            dtype=np.float64
        )

        tc_norm = group[
            "tc_norm"
        ].to_numpy(
            dtype=np.float64
        )

        row: Dict[str, object] = {
            "split": split,
            "sequence": sequence,
            "transitions": int(
                len(group)
            ),
        }

        row.update(
            summarize_values(
                tc_z,
                "tc_z",
            )
        )

        if (
            np.std(tc_z) > 0.0
            and np.std(tc_norm) > 0.0
        ):
            correlation = float(
                np.corrcoef(
                    tc_z,
                    tc_norm,
                )[0, 1]
            )
        else:
            correlation = float(
                "nan"
            )

        row[
            "tc_z_norm_correlation"
        ] = correlation

        for regime_name in REGIME_NAMES:
            regime_count = int(
                np.count_nonzero(
                    group[
                        "regime_name"
                    ].to_numpy()
                    == regime_name
                )
            )

            row[
                f"{regime_name}_count"
            ] = regime_count

            row[
                f"{regime_name}_fraction"
            ] = float(
                regime_count
                / len(group)
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def save_figure(
    figure: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    """Save and close a Matplotlib figure."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_regime_fraction_by_split(
    statistics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot normalized regime composition for train/val/test."""

    split_order = (
        "train",
        "validation",
        "test",
    )

    positions = np.arange(
        len(split_order)
    )

    bottom = np.zeros(
        len(split_order),
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        figsize=(11, 7)
    )

    for regime_name in REGIME_NAMES:
        values = []

        for split in split_order:
            row = statistics[
                (statistics["split"] == split)
                & (
                    statistics["regime_name"]
                    == regime_name
                )
            ]

            values.append(
                100.0
                * float(
                    row.iloc[0]["fraction"]
                )
            )

        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        axes.bar(
            positions,
            values_array,
            bottom=bottom,
            label=regime_name,
        )

        bottom += values_array

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        split_order
    )

    axes.set_ylim(
        0.0,
        100.0,
    )

    axes.set_xlabel(
        "Dataset split"
    )

    axes.set_ylabel(
        "Transitions (%)"
    )

    axes.set_title(
        "Refined DeepDCT tc_z motion-regime composition by split"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.legend(
        ncol=2,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_regime_count_by_split(
    statistics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot absolute regime counts for train/val/test."""

    split_order = (
        "train",
        "validation",
        "test",
    )

    positions = np.arange(
        len(split_order)
    )

    bottom = np.zeros(
        len(split_order),
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        figsize=(11, 7)
    )

    for regime_name in REGIME_NAMES:
        counts = []

        for split in split_order:
            row = statistics[
                (statistics["split"] == split)
                & (
                    statistics["regime_name"]
                    == regime_name
                )
            ]

            counts.append(
                int(
                    row.iloc[0][
                        "count"
                    ]
                )
            )

        counts_array = np.asarray(
            counts,
            dtype=np.float64,
        )

        axes.bar(
            positions,
            counts_array,
            bottom=bottom,
            label=regime_name,
        )

        bottom += counts_array

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        split_order
    )

    axes.set_xlabel(
        "Dataset split"
    )

    axes.set_ylabel(
        "Number of transitions"
    )

    axes.set_title(
        "Refined DeepDCT tc_z regime counts by split"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.legend(
        ncol=2,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_regime_fraction_by_sequence(
    per_sequence: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot normalized motion-regime composition by sequence."""

    sequence_frame = per_sequence.sort_values(
        [
            "split",
            "sequence",
        ]
    )

    labels = [
        (
            f"{row.sequence}\n"
            f"{row.split}"
        )
        for row in sequence_frame.itertuples()
    ]

    positions = np.arange(
        len(sequence_frame)
    )

    bottom = np.zeros(
        len(sequence_frame),
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        figsize=(16, 8)
    )

    for regime_name in REGIME_NAMES:
        values = (
            sequence_frame[
                f"{regime_name}_fraction"
            ].to_numpy(
                dtype=np.float64
            )
            * 100.0
        )

        axes.bar(
            positions,
            values,
            bottom=bottom,
            label=regime_name,
        )

        bottom += values

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        labels
    )

    axes.set_ylim(
        0.0,
        100.0,
    )

    axes.set_xlabel(
        "KITTI sequence and split"
    )

    axes.set_ylabel(
        "Transitions (%)"
    )

    axes.set_title(
        "Refined tc_z motion-regime fraction by KITTI sequence"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.legend(
        ncol=2,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def draw_regime_boundaries(
    axes: plt.Axes,
) -> None:
    """Draw the six shared tc_z boundaries."""

    for boundary in REGIME_BOUNDARIES:
        axes.axvline(
            float(boundary),
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )


def shared_histogram_edges(
    dataframe: pd.DataFrame,
    bin_count: int,
) -> np.ndarray:
    """Create histogram edges shared across every group."""

    minimum = float(
        dataframe["tc_z"].min()
    )

    maximum = float(
        dataframe["tc_z"].max()
    )

    if math.isclose(
        minimum,
        maximum,
    ):
        maximum = minimum + 1.0e-6

    return np.linspace(
        minimum,
        maximum,
        bin_count + 1,
    )


def plot_tc_z_distribution_by_split(
    dataframe: pd.DataFrame,
    histogram_bins: int,
    path: Path,
    dpi: int,
) -> None:
    """Plot normalized tc_z density for train/validation/test."""

    edges = shared_histogram_edges(
        dataframe,
        histogram_bins,
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    figure, axes = plt.subplots(
        figsize=(14, 8)
    )

    for split in (
        "train",
        "validation",
        "test",
    ):
        values = dataframe.loc[
            dataframe["split"] == split,
            "tc_z",
        ].to_numpy(
            dtype=np.float64
        )

        density, _ = np.histogram(
            values,
            bins=edges,
            density=True,
        )

        axes.plot(
            centers,
            density,
            linewidth=1.4,
            label=(
                f"{split} "
                f"(n={len(values)})"
            ),
        )

    draw_regime_boundaries(
        axes
    )

    axes.set_xlabel(
        "DeepDCT directional translation tc_z"
    )

    axes.set_ylabel(
        "Probability density"
    )

    axes.set_title(
        "DeepDCT tc_z distribution by dataset split"
    )

    axes.grid(
        alpha=0.3,
    )

    axes.legend()

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_tc_z_distribution_by_sequence(
    dataframe: pd.DataFrame,
    histogram_bins: int,
    path: Path,
    dpi: int,
) -> None:
    """Plot normalized tc_z distributions for sequences 00-10."""

    edges = shared_histogram_edges(
        dataframe,
        histogram_bins,
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    figure, axes = plt.subplots(
        figsize=(16, 9)
    )

    grouped = dataframe.groupby(
        [
            "split",
            "sequence",
        ],
        sort=True,
    )

    for (
        split,
        sequence,
    ), group in grouped:

        values = group[
            "tc_z"
        ].to_numpy(
            dtype=np.float64
        )

        density, _ = np.histogram(
            values,
            bins=edges,
            density=True,
        )

        axes.plot(
            centers,
            density,
            linewidth=1.1,
            label=(
                f"{sequence} ({split})"
            ),
        )

    draw_regime_boundaries(
        axes
    )

    axes.set_xlabel(
        "DeepDCT directional translation tc_z"
    )

    axes.set_ylabel(
        "Probability density"
    )

    axes.set_title(
        "DeepDCT tc_z distribution by KITTI sequence"
    )

    axes.grid(
        alpha=0.3,
    )

    axes.legend(
        ncol=2,
        fontsize=8,
    )

    save_figure(
        figure,
        path,
        dpi,
    )


def json_safe(
    value: object,
) -> object:
    """Convert NumPy values and nonfinite floats for strict JSON."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
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

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        numeric = float(
            value
        )

        if not math.isfinite(
            numeric
        ):
            return None

        return numeric

    return value


def build_summary(
    dataframe: pd.DataFrame,
    split_statistics: pd.DataFrame,
    per_sequence_statistics: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Build analysis metadata and headline statistics."""

    split_summaries: Dict[str, object] = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        subset = dataframe[
            dataframe["split"] == split
        ]

        tc_z = subset[
            "tc_z"
        ].to_numpy(
            dtype=np.float64
        )

        tc_norm = subset[
            "tc_norm"
        ].to_numpy(
            dtype=np.float64
        )

        correlation = (
            float(
                np.corrcoef(
                    tc_z,
                    tc_norm,
                )[0, 1]
            )
            if (
                np.std(tc_z) > 0
                and np.std(tc_norm) > 0
            )
            else float("nan")
        )

        split_summaries[
            split
        ] = {
            "transition_count": int(
                len(subset)
            ),
            **summarize_values(
                tc_z,
                "tc_z",
            ),
            "tc_z_vs_norm_correlation": correlation,
            "mean_abs_tc_norm_minus_tc_z": float(
                np.mean(
                    np.abs(
                        tc_norm
                        - tc_z
                    )
                )
            ),
        }

    return {
        "data_root": str(
            args.data_root.resolve()
        ),
        "train_sequences": [
            str(value).zfill(2)
            for value in args.train_sequences
        ],
        "validation_sequences": [
            str(value).zfill(2)
            for value in args.validation_sequences
        ],
        "test_sequences": [
            str(value).zfill(2)
            for value in args.test_sequences
        ],
        "label_order": [
            "tc_x",
            "tc_y",
            "tc_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
        ],
        "regimes": [
            {
                "name": "very_low",
                "condition": "tc_z < 0.25",
            },
            {
                "name": "low",
                "condition": (
                    "0.25 <= tc_z < 0.50"
                ),
            },
            {
                "name": "medium_low",
                "condition": (
                    "0.50 <= tc_z < 0.75"
                ),
            },
            {
                "name": "medium",
                "condition": (
                    "0.75 <= tc_z < 1.00"
                ),
            },
            {
                "name": "high",
                "condition": (
                    "1.00 <= tc_z < 1.25"
                ),
            },
            {
                "name": "very_high",
                "condition": (
                    "1.25 <= tc_z < 1.50"
                ),
            },
            {
                "name": "extreme",
                "condition": (
                    "tc_z >= 1.50"
                ),
            },
        ],
        "split_summary": split_summaries,
        "global_split_regime_statistics": (
            split_statistics.to_dict(
                orient="records"
            )
        ),
        "per_sequence_statistics": (
            per_sequence_statistics.to_dict(
                orient="records"
            )
        ),
    }


def print_summary(
    statistics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Print split-by-regime fractions."""

    print()
    print("=" * 122)
    print(
        "Refined DeepDCT tc_z motion-regime analysis"
    )
    print("=" * 122)

    print(
        f"{'Split':<13}"
        f"{'N':>9}"
        + "".join(
            f"{name:>14}"
            for name in REGIME_NAMES
        )
    )

    print("-" * 122)

    for split in (
        "train",
        "validation",
        "test",
    ):
        rows = statistics[
            statistics["split"]
            == split
        ].sort_values(
            "regime_index"
        )

        total = int(
            rows["count"].sum()
        )

        fractions = {
            row.regime_name: (
                100.0
                * float(
                    row.fraction
                )
            )
            for row in rows.itertuples()
        }

        print(
            f"{split:<13}"
            f"{total:>9d}"
            + "".join(
                f"{fractions[name]:>13.2f}%"
                for name in REGIME_NAMES
            )
        )

    print("-" * 122)
    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )
    print("=" * 122)


def main() -> None:
    """Run refined motion-regime analysis."""

    args = parse_args()
    validate_args(args)

    output_dir = (
        args.output_dir
    )

    plots_dir = (
        output_dir
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_all_data(
        args
    )

    split_statistics = (
        compute_split_regime_statistics(
            dataframe
        )
    )

    per_sequence_regime = (
        compute_per_sequence_regime_statistics(
            dataframe
        )
    )

    per_sequence = (
        compute_per_sequence_statistics(
            dataframe
        )
    )

    dataframe.to_csv(
        output_dir
        / "frame_regime_assignments.csv",
        index=False,
    )

    split_statistics.to_csv(
        output_dir
        / "global_split_regime_statistics.csv",
        index=False,
    )

    per_sequence_regime.to_csv(
        output_dir
        / "per_sequence_regime_statistics.csv",
        index=False,
    )

    per_sequence.to_csv(
        output_dir
        / "per_sequence_statistics.csv",
        index=False,
    )

    plot_regime_fraction_by_split(
        statistics=split_statistics,
        path=(
            plots_dir
            / "regime_fraction_by_split.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_count_by_split(
        statistics=split_statistics,
        path=(
            plots_dir
            / "regime_count_by_split.png"
        ),
        dpi=args.dpi,
    )

    plot_regime_fraction_by_sequence(
        per_sequence=per_sequence,
        path=(
            plots_dir
            / "regime_fraction_by_sequence.png"
        ),
        dpi=args.dpi,
    )

    plot_tc_z_distribution_by_split(
        dataframe=dataframe,
        histogram_bins=(
            args.histogram_bins
        ),
        path=(
            plots_dir
            / "tc_z_distribution_by_split.png"
        ),
        dpi=args.dpi,
    )

    plot_tc_z_distribution_by_sequence(
        dataframe=dataframe,
        histogram_bins=(
            args.histogram_bins
        ),
        path=(
            plots_dir
            / "tc_z_distribution_by_sequence.png"
        ),
        dpi=args.dpi,
    )

    summary = build_summary(
        dataframe=dataframe,
        split_statistics=(
            split_statistics
        ),
        per_sequence_statistics=(
            per_sequence
        ),
        args=args,
    )

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                summary
            ),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    print_summary(
        statistics=split_statistics,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()