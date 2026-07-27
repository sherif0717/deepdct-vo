"""Analyze DeepDCT-VO DCT-label distributions across KITTI sequences.

The expected label order is:

    tx ty tz rx ry rz

where:

    tx, ty, tz
        Directional-coordinate translation targets.

    rx, ry, rz
        Relative Euler-rotation targets in radians.

The script:

1. Loads label files for selected sequences.
2. Reports the number of transitions per sequence.
3. Computes per-component descriptive statistics.
4. Compares train, validation, and test distributions.
5. Saves CSV and JSON summaries.
6. Produces per-component histograms and boxplots.
7. Produces sequence-level mean and standard-deviation charts.
8. Reports each sequence's contribution to the training set.

Default split:

    train:      00-08
    validation: 09
    test:       10

Example:

    python3 scripts/analyze_dataset_statistics.py

Explicit equivalent:

    python3 scripts/analyze_dataset_statistics.py \
        --label-root data/out_csv \
        --train-sequences 00 01 02 03 04 05 06 07 08 \
        --validation-sequences 09 \
        --test-sequences 10 \
        --output-dir analysis/dataset_statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


COMPONENT_NAMES: Tuple[str, ...] = (
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
)

TRANSLATION_COMPONENTS: Tuple[str, ...] = COMPONENT_NAMES[:3]
ROTATION_COMPONENTS: Tuple[str, ...] = COMPONENT_NAMES[3:]

COMPONENT_LABELS: Mapping[str, str] = {
    "translation_x": "Directional translation x",
    "translation_y": "Directional translation y",
    "translation_z": "Directional translation z",
    "rotation_x": "Relative rotation x (rad)",
    "rotation_y": "Relative rotation y (rad)",
    "rotation_z": "Relative rotation z (rad)",
}

SPLIT_ORDER: Tuple[str, ...] = (
    "train",
    "validation",
    "test",
)


@dataclass(frozen=True)
class ComponentStatistics:
    """Descriptive statistics for one sequence and one label component."""

    split: str
    sequence: str
    component: str
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    percentile_25: float
    median: float
    percentile_75: float
    maximum: float
    absolute_mean: float
    root_mean_square: float


@dataclass(frozen=True)
class SequenceSummary:
    """Summary information for one KITTI sequence."""

    split: str
    sequence: str
    label_path: str
    transitions: int
    fraction_of_split: float
    fraction_of_all_data: float


@dataclass(frozen=True)
class SplitComponentStatistics:
    """Descriptive statistics for one complete dataset split."""

    split: str
    component: str
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    percentile_25: float
    median: float
    percentile_75: float
    maximum: float
    absolute_mean: float
    root_mean_square: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze DCT translation and relative-rotation label "
            "distributions across KITTI sequences."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("data/out_csv"),
        help="Directory containing <sequence>_dct.txt label files.",
    )

    parser.add_argument(
        "--label-pattern",
        type=str,
        default="{sequence}_dct.txt",
        help="Filename pattern for each sequence label file.",
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=[
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
        ],
        help="Training sequences.",
    )

    parser.add_argument(
        "--validation-sequences",
        nargs="+",
        default=["09"],
        help="Validation sequences.",
    )

    parser.add_argument(
        "--test-sequences",
        nargs="+",
        default=["10"],
        help="Held-out test sequences.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/dataset_statistics"),
        help="Directory for reports and charts.",
    )

    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=60,
        help="Number of bins in distribution histograms.",
    )

    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=None,
        help=(
            "Optional symmetric percentile clipping for chart axes. "
            "For example, 99.5 plots values between the 0.5th and "
            "99.5th percentiles without modifying reported statistics."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved chart resolution.",
    )

    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Generate reports without plotting charts.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive.")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    if args.clip_percentile is not None:
        if not 50.0 < args.clip_percentile <= 100.0:
            raise ValueError(
                "--clip-percentile must lie in the interval (50, 100]."
            )

    split_sequences = {
        "train": list(args.train_sequences),
        "validation": list(args.validation_sequences),
        "test": list(args.test_sequences),
    }

    for split, sequences in split_sequences.items():
        if not sequences:
            raise ValueError(
                f"The {split} split must contain at least one sequence."
            )

        duplicates = sorted(
            {
                sequence
                for sequence in sequences
                if sequences.count(sequence) > 1
            }
        )

        if duplicates:
            raise ValueError(
                f"Duplicate sequences in {split}: {duplicates}."
            )

    all_assignments: Dict[str, List[str]] = {}

    for split, sequences in split_sequences.items():
        for sequence in sequences:
            all_assignments.setdefault(sequence, []).append(split)

    overlaps = {
        sequence: splits
        for sequence, splits in all_assignments.items()
        if len(splits) > 1
    }

    if overlaps:
        raise ValueError(
            "Sequences must not occur in multiple splits: "
            f"{overlaps}."
        )


def normalize_sequence(sequence: object) -> str:
    """Normalize a sequence identifier to two digits."""

    text = str(sequence).strip()

    if not text:
        raise ValueError("Sequence identifiers cannot be empty.")

    if text.isdigit():
        return f"{int(text):02d}"

    return text


def split_mapping(args: argparse.Namespace) -> Dict[str, List[str]]:
    """Return normalized sequence identifiers grouped by split."""

    return {
        "train": [
            normalize_sequence(sequence)
            for sequence in args.train_sequences
        ],
        "validation": [
            normalize_sequence(sequence)
            for sequence in args.validation_sequences
        ],
        "test": [
            normalize_sequence(sequence)
            for sequence in args.test_sequences
        ],
    }


def read_numeric_label_rows(path: Path) -> np.ndarray:
    """Read a headerless whitespace- or comma-separated numeric label file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"DCT label file does not exist: {path}"
        )

    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            tokens = line.replace(",", " ").split()

            try:
                values = [
                    float(token)
                    for token in tokens
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric label value in {path} at line "
                    f"{line_number}: {raw_line.rstrip()}"
                ) from exc

            if len(values) < 6:
                raise ValueError(
                    f"Expected at least six values in {path} at line "
                    f"{line_number}, but found {len(values)}."
                )

            # Supports both:
            #
            # tx ty tz rx ry rz
            #
            # and:
            #
            # frame_index tx ty tz rx ry rz
            rows.append(values[-6:])

    if not rows:
        raise ValueError(
            f"No numeric label rows were found in {path}."
        )

    labels = np.asarray(rows, dtype=np.float64)

    if labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError(
            f"Expected labels with shape [N, 6], received "
            f"{labels.shape} from {path}."
        )

    if not np.isfinite(labels).all():
        invalid_indices = np.argwhere(~np.isfinite(labels))

        raise ValueError(
            f"Non-finite values found in {path}; first invalid "
            f"indices: {invalid_indices[:10].tolist()}."
        )

    return labels


def load_all_labels(
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Path],
]:
    """Load labels grouped by split and sequence."""

    sequences_by_split = split_mapping(args)

    labels_by_split: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    paths_by_sequence: Dict[str, Path] = {}

    for split in SPLIT_ORDER:
        labels_by_split[split] = {}

        for sequence in sequences_by_split[split]:
            label_path = (
                args.label_root
                / args.label_pattern.format(
                    sequence=sequence
                )
            )

            labels = read_numeric_label_rows(label_path)

            labels_by_split[split][sequence] = labels
            paths_by_sequence[sequence] = label_path

    return labels_by_split, paths_by_sequence


def descriptive_values(
    values: np.ndarray,
) -> Dict[str, float]:
    """Compute descriptive statistics for a one-dimensional array."""

    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional array, got {values.shape}."
        )

    if values.size == 0:
        raise ValueError(
            "Cannot compute statistics for an empty array."
        )

    standard_deviation = (
        float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0
    )

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": standard_deviation,
        "minimum": float(np.min(values)),
        "percentile_25": float(np.percentile(values, 25.0)),
        "median": float(np.median(values)),
        "percentile_75": float(np.percentile(values, 75.0)),
        "maximum": float(np.max(values)),
        "absolute_mean": float(np.mean(np.abs(values))),
        "root_mean_square": float(
            math.sqrt(float(np.mean(values ** 2)))
        ),
    }


def compute_sequence_statistics(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
) -> List[ComponentStatistics]:
    """Compute per-sequence, per-component statistics."""

    rows: List[ComponentStatistics] = []

    for split in SPLIT_ORDER:
        for sequence, labels in labels_by_split[split].items():
            for component_index, component in enumerate(
                COMPONENT_NAMES
            ):
                statistics = descriptive_values(
                    labels[:, component_index]
                )

                rows.append(
                    ComponentStatistics(
                        split=split,
                        sequence=sequence,
                        component=component,
                        **statistics,
                    )
                )

    return rows


def concatenate_split_labels(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    split: str,
) -> np.ndarray:
    """Concatenate all sequence labels in one split."""

    sequence_arrays = list(
        labels_by_split[split].values()
    )

    if not sequence_arrays:
        raise ValueError(
            f"No label arrays exist for split {split}."
        )

    return np.concatenate(
        sequence_arrays,
        axis=0,
    )


def compute_split_statistics(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
) -> List[SplitComponentStatistics]:
    """Compute aggregate statistics for train, validation, and test."""

    rows: List[SplitComponentStatistics] = []

    for split in SPLIT_ORDER:
        labels = concatenate_split_labels(
            labels_by_split,
            split,
        )

        for component_index, component in enumerate(
            COMPONENT_NAMES
        ):
            statistics = descriptive_values(
                labels[:, component_index]
            )

            rows.append(
                SplitComponentStatistics(
                    split=split,
                    component=component,
                    **statistics,
                )
            )

    return rows


def compute_sequence_summaries(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    paths_by_sequence: Mapping[str, Path],
) -> List[SequenceSummary]:
    """Compute sequence sizes and contribution fractions."""

    total_count = sum(
        labels.shape[0]
        for split_sequences in labels_by_split.values()
        for labels in split_sequences.values()
    )

    rows: List[SequenceSummary] = []

    for split in SPLIT_ORDER:
        split_count = sum(
            labels.shape[0]
            for labels in labels_by_split[split].values()
        )

        for sequence, labels in labels_by_split[split].items():
            count = int(labels.shape[0])

            rows.append(
                SequenceSummary(
                    split=split,
                    sequence=sequence,
                    label_path=str(
                        paths_by_sequence[sequence]
                    ),
                    transitions=count,
                    fraction_of_split=(
                        count / split_count
                        if split_count > 0
                        else float("nan")
                    ),
                    fraction_of_all_data=(
                        count / total_count
                        if total_count > 0
                        else float("nan")
                    ),
                )
            )

    return rows


def write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
) -> None:
    """Write dataclass rows to CSV."""

    if not rows:
        raise ValueError(
            f"Cannot write an empty CSV file: {path}"
        )

    dictionaries = [
        asdict(row)
        for row in rows
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(dictionaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(dictionaries)


def write_raw_labels_csv(
    path: Path,
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
) -> None:
    """Write all labels with split and sequence annotations."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "split",
        "sequence",
        "transition_index",
        *COMPONENT_NAMES,
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for split in SPLIT_ORDER:
            for sequence, labels in labels_by_split[split].items():
                for transition_index, row in enumerate(labels):
                    output_row: Dict[str, object] = {
                        "split": split,
                        "sequence": sequence,
                        "transition_index": transition_index,
                    }

                    output_row.update(
                        {
                            component: float(
                                row[component_index]
                            )
                            for component_index, component
                            in enumerate(COMPONENT_NAMES)
                        }
                    )

                    writer.writerow(output_row)


def chart_limits(
    values: np.ndarray,
    clip_percentile: Optional[float],
) -> Optional[Tuple[float, float]]:
    """Return optional percentile-based chart limits."""

    if clip_percentile is None or clip_percentile >= 100.0:
        return None

    tail = (100.0 - clip_percentile) / 2.0

    lower = float(np.percentile(values, tail))
    upper = float(
        np.percentile(values, 100.0 - tail)
    )

    if not math.isfinite(lower) or not math.isfinite(upper):
        return None

    if lower >= upper:
        return None

    return lower, upper


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
    plt.close(figure)


def plot_sequence_counts(
    sequence_summaries: Sequence[SequenceSummary],
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot the number of transitions contributed by each sequence."""

    sorted_rows = sorted(
        sequence_summaries,
        key=lambda row: (
            SPLIT_ORDER.index(row.split),
            row.sequence,
        ),
    )

    labels = [
        f"{row.sequence} ({row.split})"
        for row in sorted_rows
    ]

    values = [
        row.transitions
        for row in sorted_rows
    ]

    figure, axes = plt.subplots(figsize=(10, 7))

    positions = np.arange(len(labels))

    axes.barh(
        positions,
        values,
    )

    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    axes.invert_yaxis()
    axes.set_xlabel("Number of transitions")
    axes.set_ylabel("KITTI sequence")
    axes.set_title(
        "Label contribution by sequence"
    )
    axes.grid(
        axis="x",
        alpha=0.3,
    )

    for position, value in zip(positions, values):
        axes.text(
            value,
            position,
            f" {value}",
            va="center",
        )

    save_figure(
        figure,
        output_dir / "sequence_transition_counts.png",
        dpi,
    )


def plot_component_histograms(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    output_dir: Path,
    histogram_bins: int,
    clip_percentile: Optional[float],
    dpi: int,
) -> None:
    """Plot one split-comparison histogram for each component."""

    plot_directory = output_dir / "histograms"

    for component_index, component in enumerate(
        COMPONENT_NAMES
    ):
        split_values = {
            split: concatenate_split_labels(
                labels_by_split,
                split,
            )[:, component_index]
            for split in SPLIT_ORDER
        }

        all_values = np.concatenate(
            list(split_values.values())
        )

        limits = chart_limits(
            all_values,
            clip_percentile,
        )

        figure, axes = plt.subplots(figsize=(10, 6))

        for split in SPLIT_ORDER:
            values = split_values[split]

            if limits is not None:
                values = values[
                    (values >= limits[0])
                    & (values <= limits[1])
                ]

            axes.hist(
                values,
                bins=histogram_bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                label=(
                    f"{split} "
                    f"(n={split_values[split].size})"
                ),
            )

        axes.set_xlabel(
            COMPONENT_LABELS[component]
        )
        axes.set_ylabel("Probability density")
        axes.set_title(
            f"Split distribution: {COMPONENT_LABELS[component]}"
        )
        axes.legend()
        axes.grid(alpha=0.3)

        if limits is not None:
            axes.set_xlim(*limits)

        save_figure(
            figure,
            plot_directory / f"{component}_histogram.png",
            dpi,
        )


def plot_component_boxplots(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    output_dir: Path,
    clip_percentile: Optional[float],
    dpi: int,
) -> None:
    """Plot one per-sequence boxplot for each component."""

    plot_directory = output_dir / "boxplots"

    sequence_entries: List[
        Tuple[str, str, np.ndarray]
    ] = []

    for split in SPLIT_ORDER:
        for sequence, labels in labels_by_split[split].items():
            sequence_entries.append(
                (
                    split,
                    sequence,
                    labels,
                )
            )

    for component_index, component in enumerate(
        COMPONENT_NAMES
    ):
        values = [
            labels[:, component_index]
            for _, _, labels in sequence_entries
        ]

        labels = [
            f"{sequence}\n{split}"
            for split, sequence, _ in sequence_entries
        ]

        all_values = np.concatenate(values)

        limits = chart_limits(
            all_values,
            clip_percentile,
        )

        figure, axes = plt.subplots(figsize=(13, 7))

        axes.boxplot(
            values,
            labels=labels,
            showfliers=clip_percentile is None,
        )

        axes.set_xlabel("KITTI sequence and split")
        axes.set_ylabel(
            COMPONENT_LABELS[component]
        )
        axes.set_title(
            f"Per-sequence distribution: "
            f"{COMPONENT_LABELS[component]}"
        )
        axes.grid(
            axis="y",
            alpha=0.3,
        )

        if limits is not None:
            axes.set_ylim(*limits)

        save_figure(
            figure,
            plot_directory / f"{component}_boxplot.png",
            dpi,
        )


def plot_component_means(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot per-sequence means for every component."""

    plot_directory = output_dir / "sequence_means"

    sequence_entries: List[
        Tuple[str, str, np.ndarray]
    ] = []

    for split in SPLIT_ORDER:
        for sequence, labels in labels_by_split[split].items():
            sequence_entries.append(
                (
                    split,
                    sequence,
                    labels,
                )
            )

    labels = [
        f"{sequence}\n{split}"
        for split, sequence, _ in sequence_entries
    ]

    positions = np.arange(len(sequence_entries))

    for component_index, component in enumerate(
        COMPONENT_NAMES
    ):
        means = [
            float(
                np.mean(
                    labels_array[:, component_index]
                )
            )
            for _, _, labels_array in sequence_entries
        ]

        figure, axes = plt.subplots(figsize=(12, 6))

        axes.bar(
            positions,
            means,
        )

        axes.set_xticks(positions)
        axes.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )
        axes.set_xlabel("KITTI sequence and split")
        axes.set_ylabel("Mean")
        axes.set_title(
            f"Per-sequence mean: "
            f"{COMPONENT_LABELS[component]}"
        )
        axes.axhline(
            0.0,
            linewidth=1.0,
        )
        axes.grid(
            axis="y",
            alpha=0.3,
        )

        save_figure(
            figure,
            plot_directory / f"{component}_mean.png",
            dpi,
        )


def plot_component_standard_deviations(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot per-sequence standard deviations for each component."""

    plot_directory = output_dir / "sequence_standard_deviations"

    sequence_entries: List[
        Tuple[str, str, np.ndarray]
    ] = []

    for split in SPLIT_ORDER:
        for sequence, labels in labels_by_split[split].items():
            sequence_entries.append(
                (
                    split,
                    sequence,
                    labels,
                )
            )

    labels = [
        f"{sequence}\n{split}"
        for split, sequence, _ in sequence_entries
    ]

    positions = np.arange(len(sequence_entries))

    for component_index, component in enumerate(
        COMPONENT_NAMES
    ):
        standard_deviations = [
            float(
                np.std(
                    labels_array[:, component_index],
                    ddof=1,
                )
            )
            for _, _, labels_array in sequence_entries
        ]

        figure, axes = plt.subplots(figsize=(12, 6))

        axes.bar(
            positions,
            standard_deviations,
        )

        axes.set_xticks(positions)
        axes.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )
        axes.set_xlabel("KITTI sequence and split")
        axes.set_ylabel("Standard deviation")
        axes.set_title(
            f"Per-sequence standard deviation: "
            f"{COMPONENT_LABELS[component]}"
        )
        axes.grid(
            axis="y",
            alpha=0.3,
        )

        save_figure(
            figure,
            plot_directory
            / f"{component}_standard_deviation.png",
            dpi,
        )


def plot_translation_norms(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    output_dir: Path,
    histogram_bins: int,
    clip_percentile: Optional[float],
    dpi: int,
) -> None:
    """Plot directional-translation vector norms by split."""

    split_norms: Dict[str, np.ndarray] = {}

    for split in SPLIT_ORDER:
        labels = concatenate_split_labels(
            labels_by_split,
            split,
        )

        split_norms[split] = np.linalg.norm(
            labels[:, :3],
            axis=1,
        )

    all_norms = np.concatenate(
        list(split_norms.values())
    )

    limits = chart_limits(
        all_norms,
        clip_percentile,
    )

    figure, axes = plt.subplots(figsize=(10, 6))

    for split in SPLIT_ORDER:
        norms = split_norms[split]

        if limits is not None:
            norms = norms[
                (norms >= limits[0])
                & (norms <= limits[1])
            ]

        axes.hist(
            norms,
            bins=histogram_bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=(
                f"{split} "
                f"(n={split_norms[split].size})"
            ),
        )

    axes.set_xlabel(
        "Directional-translation target norm"
    )
    axes.set_ylabel("Probability density")
    axes.set_title(
        "Directional-translation target magnitude by split"
    )
    axes.legend()
    axes.grid(alpha=0.3)

    if limits is not None:
        axes.set_xlim(*limits)

    save_figure(
        figure,
        output_dir / "translation_norm_histogram.png",
        dpi,
    )


def plot_rotation_norms(
    labels_by_split: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    output_dir: Path,
    histogram_bins: int,
    clip_percentile: Optional[float],
    dpi: int,
) -> None:
    """Plot relative Euler-vector norms by split."""

    split_norms: Dict[str, np.ndarray] = {}

    for split in SPLIT_ORDER:
        labels = concatenate_split_labels(
            labels_by_split,
            split,
        )

        split_norms[split] = np.linalg.norm(
            labels[:, 3:6],
            axis=1,
        )

    all_norms = np.concatenate(
        list(split_norms.values())
    )

    limits = chart_limits(
        all_norms,
        clip_percentile,
    )

    figure, axes = plt.subplots(figsize=(10, 6))

    for split in SPLIT_ORDER:
        norms = split_norms[split]

        if limits is not None:
            norms = norms[
                (norms >= limits[0])
                & (norms <= limits[1])
            ]

        axes.hist(
            norms,
            bins=histogram_bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=(
                f"{split} "
                f"(n={split_norms[split].size})"
            ),
        )

    axes.set_xlabel(
        "Relative Euler-vector norm (rad)"
    )
    axes.set_ylabel("Probability density")
    axes.set_title(
        "Relative-rotation target magnitude by split"
    )
    axes.legend()
    axes.grid(alpha=0.3)

    if limits is not None:
        axes.set_xlim(*limits)

    save_figure(
        figure,
        output_dir / "rotation_norm_histogram.png",
        dpi,
    )


def print_sequence_table(
    sequence_summaries: Sequence[SequenceSummary],
) -> None:
    """Print sequence counts and split fractions."""

    print()
    print("=" * 88)
    print("Sequence contributions")
    print("=" * 88)
    print(
        f"{'Split':<12}"
        f"{'Sequence':<10}"
        f"{'Transitions':>14}"
        f"{'Split share':>16}"
        f"{'Overall share':>16}"
    )
    print("-" * 88)

    for row in sequence_summaries:
        print(
            f"{row.split:<12}"
            f"{row.sequence:<10}"
            f"{row.transitions:>14d}"
            f"{100.0 * row.fraction_of_split:>15.2f}%"
            f"{100.0 * row.fraction_of_all_data:>15.2f}%"
        )

    print("=" * 88)


def print_split_statistics(
    split_statistics: Sequence[
        SplitComponentStatistics
    ],
) -> None:
    """Print aggregate train, validation, and test statistics."""

    rows_by_split: Dict[
        str,
        List[SplitComponentStatistics],
    ] = {
        split: []
        for split in SPLIT_ORDER
    }

    for row in split_statistics:
        rows_by_split[row.split].append(row)

    for split in SPLIT_ORDER:
        print()
        print("=" * 112)
        print(
            f"{split.upper()} split component statistics"
        )
        print("=" * 112)
        print(
            f"{'Component':<22}"
            f"{'Count':>10}"
            f"{'Mean':>14}"
            f"{'Std':>14}"
            f"{'Min':>14}"
            f"{'Median':>14}"
            f"{'Max':>14}"
        )
        print("-" * 112)

        for row in rows_by_split[split]:
            print(
                f"{row.component:<22}"
                f"{row.count:>10d}"
                f"{row.mean:>14.6f}"
                f"{row.standard_deviation:>14.6f}"
                f"{row.minimum:>14.6f}"
                f"{row.median:>14.6f}"
                f"{row.maximum:>14.6f}"
            )

        print("=" * 112)


def build_json_summary(
    args: argparse.Namespace,
    sequence_summaries: Sequence[SequenceSummary],
    sequence_statistics: Sequence[ComponentStatistics],
    split_statistics: Sequence[
        SplitComponentStatistics
    ],
) -> Dict[str, object]:
    """Build a serializable analysis summary."""

    return {
        "label_order": [
            "translation_x",
            "translation_y",
            "translation_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
        ],
        "label_root": str(args.label_root.resolve()),
        "label_pattern": args.label_pattern,
        "splits": split_mapping(args),
        "sequence_summaries": [
            asdict(row)
            for row in sequence_summaries
        ],
        "sequence_component_statistics": [
            asdict(row)
            for row in sequence_statistics
        ],
        "split_component_statistics": [
            asdict(row)
            for row in split_statistics
        ],
    }


def main() -> None:
    """Run the dataset-distribution analysis."""

    args = parse_args()
    validate_args(args)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_by_split, paths_by_sequence = load_all_labels(args)

    sequence_statistics = compute_sequence_statistics(
        labels_by_split
    )

    split_statistics = compute_split_statistics(
        labels_by_split
    )

    sequence_summaries = compute_sequence_summaries(
        labels_by_split,
        paths_by_sequence,
    )

    write_dataclass_csv(
        args.output_dir / "sequence_summary.csv",
        sequence_summaries,
    )

    write_dataclass_csv(
        args.output_dir / "sequence_component_statistics.csv",
        sequence_statistics,
    )

    write_dataclass_csv(
        args.output_dir / "split_component_statistics.csv",
        split_statistics,
    )

    write_raw_labels_csv(
        args.output_dir / "all_labels.csv",
        labels_by_split,
    )

    summary = build_json_summary(
        args=args,
        sequence_summaries=sequence_summaries,
        sequence_statistics=sequence_statistics,
        split_statistics=split_statistics,
    )

    with (
        args.output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    print_sequence_table(sequence_summaries)
    print_split_statistics(split_statistics)

    if not args.skip_charts:
        plot_sequence_counts(
            sequence_summaries=sequence_summaries,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )

        plot_component_histograms(
            labels_by_split=labels_by_split,
            output_dir=args.output_dir,
            histogram_bins=args.histogram_bins,
            clip_percentile=args.clip_percentile,
            dpi=args.dpi,
        )

        plot_component_boxplots(
            labels_by_split=labels_by_split,
            output_dir=args.output_dir,
            clip_percentile=args.clip_percentile,
            dpi=args.dpi,
        )

        plot_component_means(
            labels_by_split=labels_by_split,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )

        plot_component_standard_deviations(
            labels_by_split=labels_by_split,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )

        plot_translation_norms(
            labels_by_split=labels_by_split,
            output_dir=args.output_dir,
            histogram_bins=args.histogram_bins,
            clip_percentile=args.clip_percentile,
            dpi=args.dpi,
        )

        plot_rotation_norms(
            labels_by_split=labels_by_split,
            output_dir=args.output_dir,
            histogram_bins=args.histogram_bins,
            clip_percentile=args.clip_percentile,
            dpi=args.dpi,
        )

    print()
    print("=" * 88)
    print("Dataset analysis complete")
    print("=" * 88)
    print(
        f"Output directory: "
        f"{args.output_dir.resolve()}"
    )
    print(
        f"Charts generated:  "
        f"{not args.skip_charts}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()