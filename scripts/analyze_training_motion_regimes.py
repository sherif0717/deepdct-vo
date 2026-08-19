"""Analyze DeepDCT directional-translation motion regimes in KITTI 00-08.

The script reads DeepDCT supervision files directly from:

    data/out_csv/<sequence>_dct.txt

Expected label ordering
-----------------------

    tc_x tc_y tc_z rotation_x rotation_y rotation_z

The first three columns therefore represent directional translation.

By default, the script classifies tc_z into:

    low_motion:       tc_z < 0.5
    medium_motion:    0.5 <= tc_z < 1.0
    high_motion:      tc_z >= 1.0

These boundaries intentionally match the fixed-regime analysis previously
used for KITTI sequence 10.

Generated outputs
-----------------

training_motion_regime_analysis/
├── summary.json
├── per_sequence_statistics.csv
├── global_regime_statistics.csv
├── frame_regime_assignments.csv
└── plots/
    ├── global_regime_distribution.png
    ├── per_sequence_regime_distribution.png
    ├── per_sequence_regime_fraction.png
    └── tc_z_distribution_by_sequence.png

Example
-------

    python3 scripts/analyze_training_motion_regimes.py \
        --data-root data \
        --sequences 00 01 02 03 04 05 06 07 08 \
        --output-dir training_motion_regime_analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEQUENCES = (
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

REGIME_NAMES = (
    "low_motion",
    "medium_motion",
    "high_motion",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze DeepDCT directional-translation motion-regime "
            "distribution across KITTI training sequences."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root containing out_csv/<sequence>_dct.txt."
        ),
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help="KITTI sequences included in the analysis.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_motion_regime_analysis"),
        help="Analysis output directory.",
    )

    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.5,
        help=(
            "Upper boundary of the low-motion tc_z regime."
        ),
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=1.0,
        help=(
            "Lower boundary of the high-motion tc_z regime."
        ),
    )

    parser.add_argument(
        "--label-pattern",
        type=str,
        default="{sequence}_dct.txt",
        help="Filename pattern under data/out_csv.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved plot resolution.",
    )

    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=60,
        help=(
            "Number of shared bins for tc_z distribution plots."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments."""

    if args.low_threshold >= args.high_threshold:
        raise ValueError(
            "--low-threshold must be smaller than --high-threshold."
        )

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    if args.histogram_bins <= 0:
        raise ValueError(
            "--histogram-bins must be positive."
        )

    if not args.sequences:
        raise ValueError(
            "At least one sequence must be supplied."
        )


def read_numeric_rows(path: Path) -> np.ndarray:
    """Read numeric whitespace- or comma-separated label rows.

    The DeepDCT text files used in this project normally contain six
    whitespace-separated numeric values per row. A small amount of
    flexibility is retained for comma-separated files.
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

            if not line:
                continue

            if line.startswith("#"):
                continue

            tokens = line.replace(
                ",",
                " ",
            ).split()

            try:
                values = [
                    float(token)
                    for token in tokens
                ]
            except ValueError:
                # Permit one nonnumeric header row if present.
                if not rows:
                    continue

                raise ValueError(
                    f"Non-numeric value in {path} "
                    f"at line {line_number}: {line!r}"
                )

            if len(values) < 6:
                raise ValueError(
                    f"Expected at least six numeric columns in {path} "
                    f"at line {line_number}; received {len(values)}."
                )

            # If an optional frame index precedes the six labels,
            # retain the final six columns.
            rows.append(
                values[-6:]
            )

    if not rows:
        raise ValueError(
            f"No numeric label rows found in {path}."
        )

    labels = np.asarray(
        rows,
        dtype=np.float64,
    )

    if labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError(
            f"Expected label array [N, 6]; received "
            f"{labels.shape} from {path}."
        )

    if not np.isfinite(labels).all():
        raise ValueError(
            f"Non-finite values found in {path}."
        )

    return labels


def classify_regime(
    tc_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Return regime indices 0=low, 1=medium, 2=high."""

    regimes = np.full(
        tc_z.shape,
        -1,
        dtype=np.int64,
    )

    regimes[
        tc_z < low_threshold
    ] = 0

    regimes[
        (tc_z >= low_threshold)
        & (tc_z < high_threshold)
    ] = 1

    regimes[
        tc_z >= high_threshold
    ] = 2

    if np.any(regimes < 0):
        raise RuntimeError(
            "Failed to classify one or more tc_z values."
        )

    return regimes


def load_training_labels(
    data_root: Path,
    sequences: Sequence[str],
    label_pattern: str,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    """Load all requested training labels into one table."""

    label_root = data_root / "out_csv"

    frames: List[pd.DataFrame] = []

    for sequence in sequences:
        sequence = str(sequence).zfill(2)

        path = (
            label_root
            / label_pattern.format(
                sequence=sequence
            )
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing label file for sequence {sequence}: {path}"
            )

        labels = read_numeric_rows(
            path
        )

        tc = labels[:, 0:3]
        rotation = labels[:, 3:6]

        tc_norm = np.linalg.norm(
            tc,
            axis=1,
        )

        regime_index = classify_regime(
            tc_z=tc[:, 2],
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

        dataframe = pd.DataFrame(
            {
                "sequence": sequence,
                "transition_index": np.arange(
                    labels.shape[0],
                    dtype=np.int64,
                ),
                "tc_x": tc[:, 0],
                "tc_y": tc[:, 1],
                "tc_z": tc[:, 2],
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
            dataframe
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def safe_std(
    values: np.ndarray,
) -> float:
    """Sample standard deviation."""

    if values.size <= 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def compute_per_sequence_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute tc_z and regime statistics per sequence."""

    rows: List[Dict[str, object]] = []

    for sequence, group in dataframe.groupby(
        "sequence",
        sort=True,
    ):
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

        count = len(group)

        regime_counts = (
            group["regime_name"]
            .value_counts()
            .to_dict()
        )

        row: Dict[str, object] = {
            "sequence": sequence,
            "transitions": count,
            "tc_z_mean": float(
                np.mean(tc_z)
            ),
            "tc_z_standard_deviation": safe_std(
                tc_z
            ),
            "tc_z_minimum": float(
                np.min(tc_z)
            ),
            "tc_z_percentile_25": float(
                np.percentile(
                    tc_z,
                    25.0,
                )
            ),
            "tc_z_median": float(
                np.median(tc_z)
            ),
            "tc_z_percentile_75": float(
                np.percentile(
                    tc_z,
                    75.0,
                )
            ),
            "tc_z_maximum": float(
                np.max(tc_z)
            ),
            "tc_norm_mean": float(
                np.mean(tc_norm)
            ),
            "tc_norm_median": float(
                np.median(tc_norm)
            ),
            "tc_z_norm_correlation": (
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
            ),
        }

        for regime_name in REGIME_NAMES:
            regime_count = int(
                regime_counts.get(
                    regime_name,
                    0,
                )
            )

            row[
                f"{regime_name}_count"
            ] = regime_count

            row[
                f"{regime_name}_fraction"
            ] = (
                regime_count
                / count
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def compute_global_regime_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pooled statistics for each motion regime."""

    rows: List[Dict[str, object]] = []

    total_count = len(dataframe)

    for regime_index, regime_name in enumerate(
        REGIME_NAMES
    ):
        group = dataframe[
            dataframe["regime_name"]
            == regime_name
        ]

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

        if tc_z.size == 0:
            row = {
                "regime_index": regime_index,
                "regime_name": regime_name,
                "count": 0,
                "fraction": 0.0,
                "tc_z_mean": None,
                "tc_z_standard_deviation": None,
                "tc_z_minimum": None,
                "tc_z_median": None,
                "tc_z_maximum": None,
                "tc_norm_mean": None,
                "tc_norm_median": None,
            }

        else:
            row = {
                "regime_index": regime_index,
                "regime_name": regime_name,
                "count": int(
                    tc_z.size
                ),
                "fraction": float(
                    tc_z.size
                    / total_count
                ),
                "tc_z_mean": float(
                    np.mean(tc_z)
                ),
                "tc_z_standard_deviation": safe_std(
                    tc_z
                ),
                "tc_z_minimum": float(
                    np.min(tc_z)
                ),
                "tc_z_median": float(
                    np.median(tc_z)
                ),
                "tc_z_maximum": float(
                    np.max(tc_z)
                ),
                "tc_norm_mean": float(
                    np.mean(tc_norm)
                ),
                "tc_norm_median": float(
                    np.median(tc_norm)
                ),
            }

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
    """Save and close one figure."""

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


def plot_global_regime_distribution(
    global_statistics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot total transition count in each regime."""

    labels = global_statistics[
        "regime_name"
    ].tolist()

    counts = global_statistics[
        "count"
    ].to_numpy()

    fractions = global_statistics[
        "fraction"
    ].to_numpy()

    positions = np.arange(
        len(labels)
    )

    figure, axes = plt.subplots(
        figsize=(9, 6)
    )

    bars = axes.bar(
        positions,
        counts,
    )

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        labels
    )

    axes.set_xlabel(
        "Motion regime"
    )

    axes.set_ylabel(
        "Number of transitions"
    )

    axes.set_title(
        "KITTI 00-08 DeepDCT motion-regime distribution"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, count, fraction in zip(
        bars,
        counts,
        fractions,
    ):
        axes.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            bar.get_height(),
            (
                f"{int(count)}\n"
                f"{100.0 * fraction:.1f}%"
            ),
            ha="center",
            va="bottom",
        )

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_per_sequence_regime_distribution(
    per_sequence: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot absolute regime counts for each sequence."""

    sequences = per_sequence[
        "sequence"
    ].tolist()

    positions = np.arange(
        len(sequences)
    )

    figure, axes = plt.subplots(
        figsize=(14, 7)
    )

    bottom = np.zeros(
        len(sequences),
        dtype=np.float64,
    )

    for regime_name in REGIME_NAMES:
        counts = per_sequence[
            f"{regime_name}_count"
        ].to_numpy(
            dtype=np.float64
        )

        axes.bar(
            positions,
            counts,
            bottom=bottom,
            label=regime_name,
        )

        bottom += counts

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        sequences
    )

    axes.set_xlabel(
        "KITTI training sequence"
    )

    axes.set_ylabel(
        "Number of transitions"
    )

    axes.set_title(
        "Motion-regime counts by KITTI training sequence"
    )

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.legend()

    save_figure(
        figure,
        path,
        dpi,
    )


def plot_per_sequence_regime_fraction(
    per_sequence: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot normalized regime proportions by sequence."""

    sequences = per_sequence[
        "sequence"
    ].tolist()

    positions = np.arange(
        len(sequences)
    )

    figure, axes = plt.subplots(
        figsize=(14, 7)
    )

    bottom = np.zeros(
        len(sequences),
        dtype=np.float64,
    )

    for regime_name in REGIME_NAMES:
        fractions = (
            per_sequence[
                f"{regime_name}_fraction"
            ].to_numpy(
                dtype=np.float64
            )
            * 100.0
        )

        axes.bar(
            positions,
            fractions,
            bottom=bottom,
            label=regime_name,
        )

        bottom += fractions

    axes.set_xticks(
        positions
    )

    axes.set_xticklabels(
        sequences
    )

    axes.set_xlabel(
        "KITTI training sequence"
    )

    axes.set_ylabel(
        "Transitions (%)"
    )

    axes.set_ylim(
        0.0,
        100.0,
    )

    axes.set_title(
        "Motion-regime fraction by KITTI training sequence"
    )

    axes.grid(
        axis="y",
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
    low_threshold: float,
    high_threshold: float,
    histogram_bins: int,
    path: Path,
    dpi: int,
) -> None:
    """Plot normalized tc_z histograms for each sequence."""

    global_min = float(
        dataframe["tc_z"].min()
    )

    global_max = float(
        dataframe["tc_z"].max()
    )

    bin_edges = np.linspace(
        global_min,
        global_max,
        histogram_bins + 1,
    )

    figure, axes = plt.subplots(
        figsize=(14, 8)
    )

    for sequence, group in dataframe.groupby(
        "sequence",
        sort=True,
    ):
        values = group[
            "tc_z"
        ].to_numpy(
            dtype=np.float64
        )

        hist, edges = np.histogram(
            values,
            bins=bin_edges,
            density=True,
        )

        centers = (
            edges[:-1]
            + edges[1:]
        ) / 2.0

        axes.plot(
            centers,
            hist,
            label=f"Sequence {sequence}",
            linewidth=1.2,
        )

    axes.axvline(
        low_threshold,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Low/medium boundary "
            f"({low_threshold:g})"
        ),
    )

    axes.axvline(
        high_threshold,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Medium/high boundary "
            f"({high_threshold:g})"
        ),
    )

    axes.set_xlabel(
        "DeepDCT directional translation tc_z"
    )

    axes.set_ylabel(
        "Probability density"
    )

    axes.set_title(
        "DeepDCT tc_z distribution by KITTI training sequence"
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


def make_json_safe(
    value: object,
) -> object:
    """Convert NumPy and nonfinite values for strict JSON."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): make_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            make_json_safe(item)
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
        numeric = float(value)

        if not math.isfinite(
            numeric
        ):
            return None

        return numeric

    return value


def build_summary(
    dataframe: pd.DataFrame,
    per_sequence: pd.DataFrame,
    global_statistics: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Create JSON summary."""

    tc_z = dataframe[
        "tc_z"
    ].to_numpy(
        dtype=np.float64
    )

    tc_norm = dataframe[
        "tc_norm"
    ].to_numpy(
        dtype=np.float64
    )

    correlation = float(
        np.corrcoef(
            tc_z,
            tc_norm,
        )[0, 1]
    )

    absolute_difference = np.abs(
        tc_norm - tc_z
    )

    return {
        "data_root": str(
            args.data_root.resolve()
        ),
        "sequences": [
            str(sequence).zfill(2)
            for sequence in args.sequences
        ],
        "total_transitions": int(
            len(dataframe)
        ),
        "label_order": [
            "tc_x",
            "tc_y",
            "tc_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
        ],
        "regime_definition": {
            "low_motion": (
                f"tc_z < "
                f"{args.low_threshold}"
            ),
            "medium_motion": (
                f"{args.low_threshold} <= tc_z < "
                f"{args.high_threshold}"
            ),
            "high_motion": (
                f"tc_z >= "
                f"{args.high_threshold}"
            ),
        },
        "global_tc_z": {
            "mean": float(
                np.mean(tc_z)
            ),
            "standard_deviation": safe_std(
                tc_z
            ),
            "minimum": float(
                np.min(tc_z)
            ),
            "percentile_25": float(
                np.percentile(
                    tc_z,
                    25.0,
                )
            ),
            "median": float(
                np.median(tc_z)
            ),
            "percentile_75": float(
                np.percentile(
                    tc_z,
                    75.0,
                )
            ),
            "maximum": float(
                np.max(tc_z)
            ),
        },
        "tc_z_vs_translation_norm": {
            "correlation": correlation,
            "mean_absolute_difference": float(
                np.mean(
                    absolute_difference
                )
            ),
            "maximum_absolute_difference": float(
                np.max(
                    absolute_difference
                )
            ),
        },
        "global_regimes": (
            global_statistics.to_dict(
                orient="records"
            )
        ),
        "per_sequence": (
            per_sequence.to_dict(
                orient="records"
            )
        ),
    }


def print_summary(
    per_sequence: pd.DataFrame,
    global_statistics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Print concise terminal summary."""

    print()
    print("=" * 116)
    print(
        "KITTI 00-08 DeepDCT training motion-regime analysis"
    )
    print("=" * 116)

    print(
        f"{'Sequence':<10}"
        f"{'N':>9}"
        f"{'tc_z mean':>14}"
        f"{'Low %':>12}"
        f"{'Medium %':>12}"
        f"{'High %':>12}"
    )

    print("-" * 116)

    for _, row in per_sequence.iterrows():
        print(
            f"{row['sequence']:<10}"
            f"{int(row['transitions']):>9d}"
            f"{row['tc_z_mean']:>14.6f}"
            f"{100.0 * row['low_motion_fraction']:>12.2f}"
            f"{100.0 * row['medium_motion_fraction']:>12.2f}"
            f"{100.0 * row['high_motion_fraction']:>12.2f}"
        )

    print("-" * 116)

    print("Global motion-regime distribution")

    for _, row in global_statistics.iterrows():
        print(
            f"  {row['regime_name']:<16}"
            f"{int(row['count']):>8d} "
            f"({100.0 * row['fraction']:>6.2f}%)"
        )

    print("-" * 116)

    print(
        f"Output directory: {output_dir.resolve()}"
    )

    print("=" * 116)


def main() -> None:
    """Run training-set motion-regime analysis."""

    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
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

    dataframe = load_training_labels(
        data_root=args.data_root,
        sequences=args.sequences,
        label_pattern=args.label_pattern,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    per_sequence = (
        compute_per_sequence_statistics(
            dataframe
        )
    )

    global_statistics = (
        compute_global_regime_statistics(
            dataframe
        )
    )

    # Save every transition so individual regime assignments can
    # subsequently be inspected or used to construct a sampler.
    dataframe.to_csv(
        output_dir
        / "frame_regime_assignments.csv",
        index=False,
    )

    per_sequence.to_csv(
        output_dir
        / "per_sequence_statistics.csv",
        index=False,
    )

    global_statistics.to_csv(
        output_dir
        / "global_regime_statistics.csv",
        index=False,
    )

    plot_global_regime_distribution(
        global_statistics=global_statistics,
        path=(
            plots_dir
            / "global_regime_distribution.png"
        ),
        dpi=args.dpi,
    )

    plot_per_sequence_regime_distribution(
        per_sequence=per_sequence,
        path=(
            plots_dir
            / "per_sequence_regime_distribution.png"
        ),
        dpi=args.dpi,
    )

    plot_per_sequence_regime_fraction(
        per_sequence=per_sequence,
        path=(
            plots_dir
            / "per_sequence_regime_fraction.png"
        ),
        dpi=args.dpi,
    )

    plot_tc_z_distribution_by_sequence(
        dataframe=dataframe,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        histogram_bins=args.histogram_bins,
        path=(
            plots_dir
            / "tc_z_distribution_by_sequence.png"
        ),
        dpi=args.dpi,
    )

    summary = build_summary(
        dataframe=dataframe,
        per_sequence=per_sequence,
        global_statistics=global_statistics,
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
            make_json_safe(
                summary
            ),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    print_summary(
        per_sequence=per_sequence,
        global_statistics=global_statistics,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()