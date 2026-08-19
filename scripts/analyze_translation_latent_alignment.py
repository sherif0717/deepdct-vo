#!/usr/bin/env python3
"""
Analyze cross-sequence alignment of DeepDCT-VO translation representations.

This script is a diagnostic for the compact translation representation h_D,
typically h_128 from:

    Conv(C -> 8)
        -> ReLU
        -> AdaptiveAvgPool2d(4, 4)
        -> Flatten
        -> h_128

It performs two complementary analyses.

1. Latent geometry
------------------

Compare representation geometry for:

    training:   sequences 00-08
    validation: sequence 09
    test:       sequence 10

Diagnostics include:

    - representation mean / centroid norm
    - total variance
    - per-dimension variance
    - covariance eigenspectrum
    - centroid distance
    - covariance Frobenius distance
    - RMS scale ratio
    - principal angles between PCA subspaces
    - common-PCA projections

2. Linear probing
-----------------

Four diagnostic conditions are constructed:

    A. fit 00-08 train portions -> held-out 00-08 portions
    B. fit all 00-08            -> sequence 09
    C. fit all 00-08            -> sequence 10
    D. fit sequence-10 60%      -> validate 20% -> test final 20%

Condition D is particularly important.

If C fails but D succeeds, then sequence 10 retains linearly decodable
translation information, but a mapping learned on 00-08 does not remain
aligned with sequence 10's representation geometry.

Expected per-sequence input directory
-------------------------------------

Each supplied sequence directory must contain:

    translation_representations.npz
    frame_predictions.csv

The CSV must contain:

    translation_gt_x
    translation_gt_y
    translation_gt_z

The NPZ should contain a 2-D representation array such as:

    translation_rep

Example
-------

python scripts/analyze_translation_latent_alignment.py \\
    --sequence-input 00=experiments/latent_alignment/00 \\
    --sequence-input 01=experiments/latent_alignment/01 \\
    --sequence-input 02=experiments/latent_alignment/02 \\
    --sequence-input 03=experiments/latent_alignment/03 \\
    --sequence-input 04=experiments/latent_alignment/04 \\
    --sequence-input 05=experiments/latent_alignment/05 \\
    --sequence-input 06=experiments/latent_alignment/06 \\
    --sequence-input 07=experiments/latent_alignment/07 \\
    --sequence-input 08=experiments/latent_alignment/08 \\
    --sequence-input 09=experiments/latent_alignment/09 \\
    --sequence-input 10=experiments/latent_alignment/10 \\
    --representation-key translation_rep \\
    --output-dir experiments/translation_latent_alignment

Notes
-----

- No model parameters are changed.
- No loss is changed.
- Sequence 09 or 10 is never used to fit the 00-08 cross-sequence probe.
- Ridge alpha for the 00-08 probe is selected using held-out portions of
  sequences 00-08 only.
- Sequence-10-local probe D selects alpha using its own middle 20%, then
  reports performance only on its final held-out 20%.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Constants
# ============================================================================

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

DEFAULT_VALIDATION_SEQUENCE = "09"
DEFAULT_TEST_SEQUENCE = "10"

GT_COLUMNS = (
    "translation_gt_x",
    "translation_gt_y",
    "translation_gt_z",
)

AXIS_NAMES = (
    "x",
    "y",
    "z",
)


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class SequenceData:
    """Representation and translation targets for one KITTI sequence."""

    sequence: str
    directory: Path
    representation: np.ndarray
    translation_gt: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.representation.shape[0])

    @property
    def representation_dim(self) -> int:
        return int(self.representation.shape[1])


@dataclass
class RidgeModel:
    """Standardized multivariate ridge regression model."""

    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    weight: np.ndarray
    alpha: float

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x_standardized = (
            (x - self.x_mean)
            / self.x_scale
        )

        return (
            x_standardized @ self.weight
            + self.y_mean
        )


# ============================================================================
# Argument parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cross-sequence geometry and linear decodability "
            "of DeepDCT-VO translation representations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sequence-input",
        action="append",
        required=True,
        metavar="SEQ=DIR",
        help=(
            "Per-sequence evaluation directory containing "
            "translation_representations.npz and frame_predictions.csv. "
            "Repeat once for each sequence."
        ),
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=list(
            DEFAULT_TRAIN_SEQUENCES
        ),
        help="Sequences forming the training domain.",
    )

    parser.add_argument(
        "--validation-sequence",
        default=DEFAULT_VALIDATION_SEQUENCE,
        help="Validation-domain sequence.",
    )

    parser.add_argument(
        "--test-sequence",
        default=DEFAULT_TEST_SEQUENCE,
        help="Held-out test-domain sequence.",
    )

    parser.add_argument(
        "--representation-key",
        type=str,
        default="translation_rep",
        help=(
            "NPZ key containing the translation representation. "
            "If absent, the script attempts to identify a compatible "
            "2-D array automatically."
        ),
    )

    parser.add_argument(
        "--representation-file",
        type=str,
        default="translation_representations.npz",
        help="Representation NPZ filename within each sequence directory.",
    )

    parser.add_argument(
        "--prediction-file",
        type=str,
        default="frame_predictions.csv",
        help="Frame-level prediction CSV filename.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/translation_latent_alignment"
        ),
        help="Analysis output directory.",
    )

    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "geometry",
            "probe",
        ),
        default="all",
        help="Analysis stage to execute.",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help=(
            "Number of leading PCA directions used for "
            "principal-angle comparison."
        ),
    )

    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(
            1.0e-6,
            1.0e-5,
            1.0e-4,
            1.0e-3,
            1.0e-2,
            1.0e-1,
            1.0,
            10.0,
            100.0,
            1000.0,
        ),
        help="Candidate ridge regularization strengths.",
    )

    parser.add_argument(
        "--train-holdout-fraction",
        type=float,
        default=0.20,
        help=(
            "Final temporal fraction of each training sequence "
            "reserved for Probe A and ridge-alpha selection."
        ),
    )

    parser.add_argument(
        "--sequence10-fit-fraction",
        type=float,
        default=0.60,
        help="Initial fraction of sequence 10 used to fit Probe D.",
    )

    parser.add_argument(
        "--sequence10-validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Middle fraction of sequence 10 used to select Probe D alpha. "
            "Remaining samples form the held-out Probe D test set."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Input parsing
# ============================================================================


def parse_sequence_inputs(
    values: Sequence[str],
) -> Dict[str, Path]:
    result: Dict[str, Path] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                "--sequence-input must use SEQ=DIR syntax; "
                f"received {value!r}."
            )

        sequence, directory = value.split(
            "=",
            maxsplit=1,
        )

        sequence = sequence.strip()
        directory = directory.strip()

        if not sequence:
            raise ValueError(
                f"Missing sequence in {value!r}."
            )

        if not directory:
            raise ValueError(
                f"Missing directory in {value!r}."
            )

        if sequence in result:
            raise ValueError(
                "Duplicate --sequence-input for "
                f"sequence {sequence}."
            )

        result[sequence] = Path(
            directory
        ).expanduser().resolve()

    return result


def validate_requested_sequences(
    sequence_directories: Mapping[str, Path],
    train_sequences: Sequence[str],
    validation_sequence: str,
    test_sequence: str,
) -> None:
    required = set(
        train_sequences
    )

    required.add(
        validation_sequence
    )

    required.add(
        test_sequence
    )

    missing = sorted(
        required.difference(
            sequence_directories.keys()
        )
    )

    if missing:
        raise ValueError(
            "Missing --sequence-input directories for sequences: "
            f"{missing}."
        )


# ============================================================================
# Data loading
# ============================================================================


def load_translation_targets(
    csv_path: Path,
) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {csv_path}"
        )

    values: List[List[float]] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
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
            for column in GT_COLUMNS
            if column not in fieldnames
        ]

        if missing_columns:
            raise KeyError(
                f"{csv_path} is missing ground-truth columns: "
                f"{missing_columns}."
            )

        for row_index, row in enumerate(
            reader
        ):
            try:
                values.append(
                    [
                        float(
                            row[
                                "translation_gt_x"
                            ]
                        ),
                        float(
                            row[
                                "translation_gt_y"
                            ]
                        ),
                        float(
                            row[
                                "translation_gt_z"
                            ]
                        ),
                    ]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Could not parse translation target in "
                    f"{csv_path}, row {row_index + 2}."
                ) from error

    targets = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        targets.ndim != 2
        or targets.shape[1] != 3
    ):
        raise RuntimeError(
            f"Expected translation targets [N, 3], got {targets.shape}."
        )

    if not np.all(
        np.isfinite(targets)
    ):
        raise ValueError(
            f"{csv_path} contains non-finite translation targets."
        )

    return targets


def load_representation(
    npz_path: Path,
    requested_key: str,
    expected_samples: int,
) -> Tuple[np.ndarray, str]:
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Representation NPZ does not exist: {npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as archive:
        keys = list(
            archive.keys()
        )

        selected_key: Optional[str] = None

        if requested_key in archive:
            selected_key = requested_key
        else:
            candidates: List[str] = []

            for key in keys:
                array = np.asarray(
                    archive[key]
                )

                if (
                    array.ndim == 2
                    and array.shape[0]
                    == expected_samples
                ):
                    candidates.append(
                        key
                    )

            if len(candidates) == 1:
                selected_key = candidates[0]

                print(
                    "Requested representation key "
                    f"{requested_key!r} not found in {npz_path.name}; "
                    f"using {selected_key!r}."
                )

            elif len(candidates) > 1:
                preferred = [
                    key
                    for key in candidates
                    if "translation" in key.lower()
                    and "rep" in key.lower()
                ]

                if len(preferred) == 1:
                    selected_key = preferred[0]
                else:
                    raise KeyError(
                        "Could not uniquely identify representation "
                        f"array in {npz_path}. Candidate keys: "
                        f"{candidates}."
                    )

            else:
                raise KeyError(
                    f"Representation key {requested_key!r} not found "
                    f"in {npz_path}. Available keys: {keys}."
                )

        representation = np.asarray(
            archive[selected_key],
            dtype=np.float64,
        )

    if representation.ndim != 2:
        raise ValueError(
            "Translation representation must be 2-D [N, D], "
            f"got {representation.shape}."
        )

    if representation.shape[0] != expected_samples:
        raise ValueError(
            "Representation/CSV sample mismatch: "
            f"{representation.shape[0]} vs {expected_samples}."
        )

    if not np.all(
        np.isfinite(representation)
    ):
        raise ValueError(
            f"{npz_path} contains non-finite representation values."
        )

    return (
        representation,
        selected_key,
    )


def load_sequence(
    sequence: str,
    directory: Path,
    representation_filename: str,
    prediction_filename: str,
    representation_key: str,
) -> SequenceData:
    targets = load_translation_targets(
        directory / prediction_filename
    )

    representation, selected_key = (
        load_representation(
            directory / representation_filename,
            representation_key,
            expected_samples=targets.shape[0],
        )
    )

    print(
        f"Loaded sequence {sequence}: "
        f"N={representation.shape[0]}, "
        f"D={representation.shape[1]}, "
        f"representation_key={selected_key!r}"
    )

    return SequenceData(
        sequence=sequence,
        directory=directory,
        representation=representation,
        translation_gt=targets,
    )


# ============================================================================
# CSV / JSON helpers
# ============================================================================


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
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
    value: object,
) -> object:
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.floating,
    ):
        return float(
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
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_ready(
                item
            )
            for key, item in value.items()
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
# Geometry
# ============================================================================


def covariance_matrix(
    x: np.ndarray,
) -> np.ndarray:
    centered = (
        x - np.mean(
            x,
            axis=0,
            keepdims=True,
        )
    )

    denominator = max(
        x.shape[0] - 1,
        1,
    )

    return (
        centered.T @ centered
        / float(denominator)
    )


def sorted_eigendecomposition(
    covariance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = np.maximum(
        eigenvalues[order],
        0.0,
    )

    eigenvectors = eigenvectors[
        :,
        order,
    ]

    return (
        eigenvalues,
        eigenvectors,
    )


def representation_statistics(
    name: str,
    x: np.ndarray,
) -> Dict[str, object]:
    mean_vector = np.mean(
        x,
        axis=0,
    )

    variance = np.var(
        x,
        axis=0,
        ddof=1,
    )

    covariance = covariance_matrix(
        x
    )

    eigenvalues, _ = (
        sorted_eigendecomposition(
            covariance
        )
    )

    total_variance = float(
        np.sum(
            variance
        )
    )

    rms_scale = float(
        np.sqrt(
            np.mean(
                variance
            )
        )
    )

    return {
        "group": name,
        "samples": int(
            x.shape[0]
        ),
        "dimension": int(
            x.shape[1]
        ),
        "centroid_norm": float(
            np.linalg.norm(
                mean_vector
            )
        ),
        "mean_feature_variance": float(
            np.mean(
                variance
            )
        ),
        "median_feature_variance": float(
            np.median(
                variance
            )
        ),
        "total_variance": total_variance,
        "rms_feature_scale": rms_scale,
        "leading_eigenvalue": float(
            eigenvalues[0]
        ),
        "leading_variance_fraction": float(
            eigenvalues[0]
            / max(
                np.sum(
                    eigenvalues
                ),
                1.0e-12,
            )
        ),
    }


def principal_angles_degrees(
    x_a: np.ndarray,
    x_b: np.ndarray,
    components: int,
) -> np.ndarray:
    covariance_a = covariance_matrix(
        x_a
    )

    covariance_b = covariance_matrix(
        x_b
    )

    _, eigenvectors_a = (
        sorted_eigendecomposition(
            covariance_a
        )
    )

    _, eigenvectors_b = (
        sorted_eigendecomposition(
            covariance_b
        )
    )

    k = min(
        components,
        eigenvectors_a.shape[1],
        eigenvectors_b.shape[1],
    )

    basis_a = eigenvectors_a[
        :,
        :k,
    ]

    basis_b = eigenvectors_b[
        :,
        :k,
    ]

    singular_values = np.linalg.svd(
        basis_a.T @ basis_b,
        compute_uv=False,
    )

    singular_values = np.clip(
        singular_values,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(
            singular_values
        )
    )


def pairwise_geometry(
    name_a: str,
    x_a: np.ndarray,
    name_b: str,
    x_b: np.ndarray,
    pca_components: int,
) -> Tuple[
    Dict[str, object],
    List[Dict[str, object]],
]:
    mean_a = np.mean(
        x_a,
        axis=0,
    )

    mean_b = np.mean(
        x_b,
        axis=0,
    )

    covariance_a = covariance_matrix(
        x_a
    )

    covariance_b = covariance_matrix(
        x_b
    )

    variance_a = np.var(
        x_a,
        axis=0,
        ddof=1,
    )

    variance_b = np.var(
        x_b,
        axis=0,
        ddof=1,
    )

    centroid_distance = float(
        np.linalg.norm(
            mean_a - mean_b
        )
    )

    covariance_distance = float(
        np.linalg.norm(
            covariance_a - covariance_b,
            ord="fro",
        )
    )

    rms_a = float(
        np.sqrt(
            np.mean(
                variance_a
            )
        )
    )

    rms_b = float(
        np.sqrt(
            np.mean(
                variance_b
            )
        )
    )

    scale_ratio = float(
        rms_b
        / max(
            rms_a,
            1.0e-12,
        )
    )

    angles = principal_angles_degrees(
        x_a=x_a,
        x_b=x_b,
        components=pca_components,
    )

    summary = {
        "group_a": name_a,
        "group_b": name_b,
        "centroid_distance": centroid_distance,
        "covariance_frobenius_distance": covariance_distance,
        "rms_scale_a": rms_a,
        "rms_scale_b": rms_b,
        "scale_ratio_b_over_a": scale_ratio,
        "mean_principal_angle_deg": float(
            np.mean(
                angles
            )
        ),
        "max_principal_angle_deg": float(
            np.max(
                angles
            )
        ),
    }

    angle_rows = [
        {
            "group_a": name_a,
            "group_b": name_b,
            "component": index + 1,
            "principal_angle_deg": float(
                angle
            ),
        }
        for index, angle in enumerate(
            angles
        )
    ]

    return (
        summary,
        angle_rows,
    )


def build_covariance_spectrum_rows(
    groups: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[
        Dict[str, object]
    ] = []

    for name, x in groups.items():
        eigenvalues, _ = (
            sorted_eigendecomposition(
                covariance_matrix(
                    x
                )
            )
        )

        total = max(
            float(
                np.sum(
                    eigenvalues
                )
            ),
            1.0e-12,
        )

        cumulative = 0.0

        for index, eigenvalue in enumerate(
            eigenvalues
        ):
            fraction = float(
                eigenvalue / total
            )

            cumulative += fraction

            rows.append(
                {
                    "group": name,
                    "component": index + 1,
                    "eigenvalue": float(
                        eigenvalue
                    ),
                    "variance_fraction": fraction,
                    "cumulative_variance_fraction": cumulative,
                }
            )

    return rows


# ============================================================================
# Common PCA visualization
# ============================================================================


def fit_common_pca(
    groups: Mapping[str, np.ndarray],
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    all_x = np.concatenate(
        list(
            groups.values()
        ),
        axis=0,
    )

    mean = np.mean(
        all_x,
        axis=0,
    )

    covariance = covariance_matrix(
        all_x
    )

    _, eigenvectors = (
        sorted_eigendecomposition(
            covariance
        )
    )

    return (
        mean,
        eigenvectors,
    )


def plot_common_pca(
    groups: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    mean, eigenvectors = (
        fit_common_pca(
            groups
        )
    )

    if eigenvectors.shape[1] < 2:
        return

    basis = eigenvectors[
        :,
        :2,
    ]

    figure = plt.figure(
        figsize=(8.0, 6.0)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for name, x in groups.items():
        projected = (
            x - mean
        ) @ basis

        maximum_points = 3000

        if projected.shape[0] > maximum_points:
            indices = np.linspace(
                0,
                projected.shape[0] - 1,
                maximum_points,
            ).astype(
                np.int64
            )

            projected = projected[
                indices
            ]

        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            s=7,
            alpha=0.25,
            label=name,
        )

    axis.set_xlabel(
        "Common PC1"
    )

    axis.set_ylabel(
        "Common PC2"
    )

    axis.set_title(
        "Translation representation: common PCA"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        figure
    )


def plot_eigenspectra(
    groups: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(8.0, 5.5)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for name, x in groups.items():
        eigenvalues, _ = (
            sorted_eigendecomposition(
                covariance_matrix(
                    x
                )
            )
        )

        total = max(
            float(
                np.sum(
                    eigenvalues
                )
            ),
            1.0e-12,
        )

        fractions = (
            eigenvalues / total
        )

        count = min(
            30,
            fractions.shape[0],
        )

        axis.plot(
            np.arange(
                1,
                count + 1,
            ),
            fractions[
                :count
            ],
            marker="o",
            markersize=3,
            label=name,
        )

    axis.set_xlabel(
        "Principal component"
    )

    axis.set_ylabel(
        "Explained variance fraction"
    )

    axis.set_title(
        "Latent covariance eigenspectrum"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        figure
    )


# ============================================================================
# Linear regression
# ============================================================================


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> RidgeModel:
    if x.ndim != 2:
        raise ValueError(
            "x must have shape [N, D]."
        )

    if (
        y.ndim != 2
        or y.shape[1] != 3
    ):
        raise ValueError(
            "y must have shape [N, 3]."
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must contain the same samples."
        )

    x_mean = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    x_scale = np.std(
        x,
        axis=0,
        ddof=0,
        keepdims=True,
    )

    x_scale = np.where(
        x_scale < 1.0e-8,
        1.0,
        x_scale,
    )

    y_mean = np.mean(
        y,
        axis=0,
        keepdims=True,
    )

    x_standardized = (
        (x - x_mean)
        / x_scale
    )

    y_centered = (
        y - y_mean
    )

    dimension = x.shape[1]

    gram = (
        x_standardized.T
        @ x_standardized
    )

    regularized = (
        gram
        + float(alpha)
        * np.eye(
            dimension,
            dtype=np.float64,
        )
    )

    right_hand_side = (
        x_standardized.T
        @ y_centered
    )

    try:
        weight = np.linalg.solve(
            regularized,
            right_hand_side,
        )
    except np.linalg.LinAlgError:
        weight = np.linalg.pinv(
            regularized
        ) @ right_hand_side

    return RidgeModel(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        weight=weight,
        alpha=float(
            alpha
        ),
    )


def scalar_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    - target
                )
                ** 2
            )
        )
    )


def select_ridge_alpha(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    candidate_alphas: Sequence[float],
) -> Tuple[
    float,
    List[Dict[str, object]],
]:
    rows: List[
        Dict[str, object]
    ] = []

    best_alpha: Optional[
        float
    ] = None

    best_rmse = float(
        "inf"
    )

    for alpha in candidate_alphas:
        model = fit_ridge(
            x=x_fit,
            y=y_fit,
            alpha=float(
                alpha
            ),
        )

        prediction = model.predict(
            x_validation
        )

        rmse = scalar_rmse(
            prediction,
            y_validation,
        )

        rows.append(
            {
                "alpha": float(
                    alpha
                ),
                "validation_translation_rmse": rmse,
            }
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(
                alpha
            )

    if best_alpha is None:
        raise RuntimeError(
            "Could not select a ridge alpha."
        )

    return (
        best_alpha,
        rows,
    )


# ============================================================================
# Probe metrics
# ============================================================================


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a_std = float(
        np.std(
            a
        )
    )

    b_std = float(
        np.std(
            b
        )
    )

    if (
        a_std < 1.0e-12
        or b_std < 1.0e-12
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def compute_probe_metrics(
    probe_name: str,
    fit_domain: str,
    test_domain: str,
    alpha: float,
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, object]:
    error = (
        prediction - target
    )

    row: Dict[
        str,
        object,
    ] = {
        "probe": probe_name,
        "fit_domain": fit_domain,
        "test_domain": test_domain,
        "alpha": float(
            alpha
        ),
        "samples": int(
            target.shape[0]
        ),
        "translation_rmse": scalar_rmse(
            prediction,
            target,
        ),
        "translation_mae": float(
            np.mean(
                np.abs(
                    error
                )
            )
        ),
        "vector_rmse": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        error ** 2,
                        axis=1,
                    )
                )
            )
        ),
    }

    for axis_index, axis_name in enumerate(
        AXIS_NAMES
    ):
        gt = target[
            :,
            axis_index,
        ]

        pred = prediction[
            :,
            axis_index,
        ]

        axis_error = (
            pred - gt
        )

        gt_std = float(
            np.std(
                gt
            )
        )

        pred_std = float(
            np.std(
                pred
            )
        )

        row[
            f"{axis_name}_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    axis_error ** 2
                )
            )
        )

        row[
            f"{axis_name}_mae"
        ] = float(
            np.mean(
                np.abs(
                    axis_error
                )
            )
        )

        row[
            f"{axis_name}_bias"
        ] = float(
            np.mean(
                axis_error
            )
        )

        row[
            f"{axis_name}_corr"
        ] = safe_correlation(
            pred,
            gt,
        )

        row[
            f"{axis_name}_gt_std"
        ] = gt_std

        row[
            f"{axis_name}_pred_std"
        ] = pred_std

        row[
            f"{axis_name}_std_ratio"
        ] = (
            pred_std
            / gt_std
            if gt_std > 1.0e-12
            else float(
                "nan"
            )
        )

    return row


def prediction_rows(
    probe_name: str,
    sequence: str,
    sample_indices: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[
        Dict[str, object]
    ] = []

    for local_index in range(
        target.shape[0]
    ):
        rows.append(
            {
                "probe": probe_name,
                "sequence": sequence,
                "sample_index": int(
                    sample_indices[
                        local_index
                    ]
                ),
                "translation_gt_x": float(
                    target[
                        local_index,
                        0,
                    ]
                ),
                "translation_gt_y": float(
                    target[
                        local_index,
                        1,
                    ]
                ),
                "translation_gt_z": float(
                    target[
                        local_index,
                        2,
                    ]
                ),
                "translation_pred_x": float(
                    prediction[
                        local_index,
                        0,
                    ]
                ),
                "translation_pred_y": float(
                    prediction[
                        local_index,
                        1,
                    ]
                ),
                "translation_pred_z": float(
                    prediction[
                        local_index,
                        2,
                    ]
                ),
            }
        )

    return rows


# ============================================================================
# Temporal split helpers
# ============================================================================


def temporal_fit_holdout(
    data: SequenceData,
    holdout_fraction: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = data.num_samples

    split = int(
        math.floor(
            n * (
                1.0
                - holdout_fraction
            )
        )
    )

    split = min(
        max(
            split,
            1,
        ),
        n - 1,
    )

    fit_indices = np.arange(
        0,
        split,
        dtype=np.int64,
    )

    holdout_indices = np.arange(
        split,
        n,
        dtype=np.int64,
    )

    return (
        data.representation[
            fit_indices
        ],
        data.translation_gt[
            fit_indices
        ],
        fit_indices,
        holdout_indices,
    )


def concatenate_indices(
    arrays: Sequence[np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        arrays,
        axis=0,
    )


# ============================================================================
# Probe experiment
# ============================================================================


def run_probes(
    sequence_data: Mapping[str, SequenceData],
    train_sequences: Sequence[str],
    validation_sequence: str,
    test_sequence: str,
    candidate_alphas: Sequence[float],
    train_holdout_fraction: float,
    sequence10_fit_fraction: float,
    sequence10_validation_fraction: float,
    output_dir: Path,
) -> Dict[str, object]:
    print()
    print("=" * 88)
    print("Linear probe analysis")
    print("=" * 88)

    # ------------------------------------------------------------------
    # A: Build temporal fit/holdout partitions within 00-08.
    # ------------------------------------------------------------------

    train_fit_x: List[
        np.ndarray
    ] = []

    train_fit_y: List[
        np.ndarray
    ] = []

    train_holdout_x: List[
        np.ndarray
    ] = []

    train_holdout_y: List[
        np.ndarray
    ] = []

    for sequence in train_sequences:
        data = sequence_data[
            sequence
        ]

        n = data.num_samples

        split = int(
            math.floor(
                n * (
                    1.0
                    - train_holdout_fraction
                )
            )
        )

        split = min(
            max(
                split,
                1,
            ),
            n - 1,
        )

        train_fit_x.append(
            data.representation[
                :split
            ]
        )

        train_fit_y.append(
            data.translation_gt[
                :split
            ]
        )

        train_holdout_x.append(
            data.representation[
                split:
            ]
        )

        train_holdout_y.append(
            data.translation_gt[
                split:
            ]
        )

    x_train_fit = np.concatenate(
        train_fit_x,
        axis=0,
    )

    y_train_fit = np.concatenate(
        train_fit_y,
        axis=0,
    )

    x_train_holdout = np.concatenate(
        train_holdout_x,
        axis=0,
    )

    y_train_holdout = np.concatenate(
        train_holdout_y,
        axis=0,
    )

    # Select alpha using only 00-08.
    cross_alpha, cross_alpha_rows = (
        select_ridge_alpha(
            x_fit=x_train_fit,
            y_fit=y_train_fit,
            x_validation=x_train_holdout,
            y_validation=y_train_holdout,
            candidate_alphas=candidate_alphas,
        )
    )

    for row in cross_alpha_rows:
        row[
            "probe_family"
        ] = "00-08"

    print(
        f"00-08 selected ridge alpha: {cross_alpha:g}"
    )

    # Probe A: 00-08 fit portions -> held-out portions.
    probe_a_model = fit_ridge(
        x=x_train_fit,
        y=y_train_fit,
        alpha=cross_alpha,
    )

    probe_a_prediction = (
        probe_a_model.predict(
            x_train_holdout
        )
    )

    probe_rows: List[
        Dict[str, object]
    ] = []

    probe_rows.append(
        compute_probe_metrics(
            probe_name="A_train_in_domain_holdout",
            fit_domain="00-08 temporal fit portions",
            test_domain="00-08 temporal holdout portions",
            alpha=cross_alpha,
            prediction=probe_a_prediction,
            target=y_train_holdout,
        )
    )

    # ------------------------------------------------------------------
    # B/C: Fit final 00-08 model using all training-domain samples.
    # Alpha remains selected entirely inside 00-08.
    # ------------------------------------------------------------------

    x_train_all = np.concatenate(
        [
            sequence_data[
                sequence
            ].representation
            for sequence in train_sequences
        ],
        axis=0,
    )

    y_train_all = np.concatenate(
        [
            sequence_data[
                sequence
            ].translation_gt
            for sequence in train_sequences
        ],
        axis=0,
    )

    cross_model = fit_ridge(
        x=x_train_all,
        y=y_train_all,
        alpha=cross_alpha,
    )

    validation_data = sequence_data[
        validation_sequence
    ]

    test_data = sequence_data[
        test_sequence
    ]

    probe_b_prediction = (
        cross_model.predict(
            validation_data.representation
        )
    )

    probe_c_prediction = (
        cross_model.predict(
            test_data.representation
        )
    )

    probe_rows.append(
        compute_probe_metrics(
            probe_name="B_train_to_validation",
            fit_domain="00-08 all",
            test_domain=validation_sequence,
            alpha=cross_alpha,
            prediction=probe_b_prediction,
            target=validation_data.translation_gt,
        )
    )

    probe_rows.append(
        compute_probe_metrics(
            probe_name="C_train_to_test",
            fit_domain="00-08 all",
            test_domain=test_sequence,
            alpha=cross_alpha,
            prediction=probe_c_prediction,
            target=test_data.translation_gt,
        )
    )

    # ------------------------------------------------------------------
    # D: Sequence-10-local temporal 60/20/20 probe.
    # ------------------------------------------------------------------

    sequence10 = test_data

    n10 = sequence10.num_samples

    fit_end = int(
        math.floor(
            n10
            * sequence10_fit_fraction
        )
    )

    validation_end = int(
        math.floor(
            n10
            * (
                sequence10_fit_fraction
                + sequence10_validation_fraction
            )
        )
    )

    fit_end = min(
        max(
            fit_end,
            1,
        ),
        n10 - 2,
    )

    validation_end = min(
        max(
            validation_end,
            fit_end + 1,
        ),
        n10 - 1,
    )

    x10_fit = (
        sequence10.representation[
            :fit_end
        ]
    )

    y10_fit = (
        sequence10.translation_gt[
            :fit_end
        ]
    )

    x10_validation = (
        sequence10.representation[
            fit_end:validation_end
        ]
    )

    y10_validation = (
        sequence10.translation_gt[
            fit_end:validation_end
        ]
    )

    x10_test = (
        sequence10.representation[
            validation_end:
        ]
    )

    y10_test = (
        sequence10.translation_gt[
            validation_end:
        ]
    )

    sequence10_alpha, sequence10_alpha_rows = (
        select_ridge_alpha(
            x_fit=x10_fit,
            y_fit=y10_fit,
            x_validation=x10_validation,
            y_validation=y10_validation,
            candidate_alphas=candidate_alphas,
        )
    )

    for row in sequence10_alpha_rows:
        row[
            "probe_family"
        ] = "sequence_10_local"

    print(
        "Sequence-10-local selected ridge alpha: "
        f"{sequence10_alpha:g}"
    )

    # Once alpha is selected, refit on the first 80% and test
    # strictly on the final 20%.
    x10_fit_final = (
        sequence10.representation[
            :validation_end
        ]
    )

    y10_fit_final = (
        sequence10.translation_gt[
            :validation_end
        ]
    )

    sequence10_model = fit_ridge(
        x=x10_fit_final,
        y=y10_fit_final,
        alpha=sequence10_alpha,
    )

    probe_d_prediction = (
        sequence10_model.predict(
            x10_test
        )
    )

    probe_rows.append(
        compute_probe_metrics(
            probe_name="D_test_local_holdout",
            fit_domain=(
                f"{test_sequence} first "
                f"{validation_end}/{n10}"
            ),
            test_domain=(
                f"{test_sequence} final "
                f"{n10 - validation_end}/{n10}"
            ),
            alpha=sequence10_alpha,
            prediction=probe_d_prediction,
            target=y10_test,
        )
    )

    # ------------------------------------------------------------------
    # Save per-frame predictions for B/C/D.
    # ------------------------------------------------------------------

    prediction_output_rows: List[
        Dict[str, object]
    ] = []

    prediction_output_rows.extend(
        prediction_rows(
            probe_name="B_train_to_validation",
            sequence=validation_sequence,
            sample_indices=np.arange(
                validation_data.num_samples,
                dtype=np.int64,
            ),
            prediction=probe_b_prediction,
            target=validation_data.translation_gt,
        )
    )

    prediction_output_rows.extend(
        prediction_rows(
            probe_name="C_train_to_test",
            sequence=test_sequence,
            sample_indices=np.arange(
                test_data.num_samples,
                dtype=np.int64,
            ),
            prediction=probe_c_prediction,
            target=test_data.translation_gt,
        )
    )

    prediction_output_rows.extend(
        prediction_rows(
            probe_name="D_test_local_holdout",
            sequence=test_sequence,
            sample_indices=np.arange(
                validation_end,
                n10,
                dtype=np.int64,
            ),
            prediction=probe_d_prediction,
            target=y10_test,
        )
    )

    write_csv(
        output_dir
        / "probe_results.csv",
        probe_rows,
    )

    write_csv(
        output_dir
        / "probe_alpha_search.csv",
        (
            cross_alpha_rows
            + sequence10_alpha_rows
        ),
    )

    write_csv(
        output_dir
        / "probe_predictions.csv",
        prediction_output_rows,
    )

    print()
    print("-" * 88)
    print("Probe summary")
    print("-" * 88)

    for row in probe_rows:
        print(
            f"{row['probe']:<30} "
            f"n={row['samples']:>5} "
            f"RMSE={row['translation_rmse']:.6f} "
            f"z_RMSE={row['z_rmse']:.6f} "
            f"z_corr={row['z_corr']:.4f} "
            f"z_std_ratio={row['z_std_ratio']:.4f}"
        )

    print("-" * 88)

    return {
        "selected_alpha_00_08": cross_alpha,
        "selected_alpha_sequence10": sequence10_alpha,
        "sequence10_split": {
            "fit_samples": fit_end,
            "validation_samples": (
                validation_end
                - fit_end
            ),
            "test_samples": (
                n10
                - validation_end
            ),
        },
        "results": probe_rows,
    }


# ============================================================================
# Geometry experiment
# ============================================================================


def run_geometry(
    sequence_data: Mapping[str, SequenceData],
    train_sequences: Sequence[str],
    validation_sequence: str,
    test_sequence: str,
    pca_components: int,
    output_dir: Path,
) -> Dict[str, object]:
    print()
    print("=" * 88)
    print("Translation latent geometry analysis")
    print("=" * 88)

    train_representation = np.concatenate(
        [
            sequence_data[
                sequence
            ].representation
            for sequence in train_sequences
        ],
        axis=0,
    )

    validation_representation = (
        sequence_data[
            validation_sequence
        ].representation
    )

    test_representation = (
        sequence_data[
            test_sequence
        ].representation
    )

    groups = {
        "train_00_08": train_representation,
        f"validation_{validation_sequence}": (
            validation_representation
        ),
        f"test_{test_sequence}": (
            test_representation
        ),
    }

    statistics_rows = [
        representation_statistics(
            name,
            x,
        )
        for name, x in groups.items()
    ]

    pair_rows: List[
        Dict[str, object]
    ] = []

    principal_angle_rows: List[
        Dict[str, object]
    ] = []

    group_names = list(
        groups.keys()
    )

    for first_index in range(
        len(
            group_names
        )
    ):
        for second_index in range(
            first_index + 1,
            len(
                group_names
            ),
        ):
            name_a = group_names[
                first_index
            ]

            name_b = group_names[
                second_index
            ]

            summary, angles = (
                pairwise_geometry(
                    name_a=name_a,
                    x_a=groups[
                        name_a
                    ],
                    name_b=name_b,
                    x_b=groups[
                        name_b
                    ],
                    pca_components=pca_components,
                )
            )

            pair_rows.append(
                summary
            )

            principal_angle_rows.extend(
                angles
            )

    spectrum_rows = (
        build_covariance_spectrum_rows(
            groups
        )
    )

    write_csv(
        output_dir
        / "latent_statistics.csv",
        statistics_rows,
    )

    write_csv(
        output_dir
        / "group_distances.csv",
        pair_rows,
    )

    write_csv(
        output_dir
        / "principal_angles.csv",
        principal_angle_rows,
    )

    write_csv(
        output_dir
        / "covariance_spectrum.csv",
        spectrum_rows,
    )

    plots_directory = (
        output_dir / "plots"
    )

    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_common_pca(
        groups=groups,
        output_path=(
            plots_directory
            / "common_pca.png"
        ),
    )

    plot_eigenspectra(
        groups=groups,
        output_path=(
            plots_directory
            / "covariance_eigenspectrum.png"
        ),
    )

    print()
    print("Representation statistics")
    print("-" * 88)

    for row in statistics_rows:
        print(
            f"{row['group']:<20} "
            f"n={row['samples']:>6} "
            f"centroid_norm={row['centroid_norm']:.6f} "
            f"total_var={row['total_variance']:.6f} "
            f"rms_scale={row['rms_feature_scale']:.6f}"
        )

    print()
    print("Cross-group geometry")
    print("-" * 88)

    for row in pair_rows:
        print(
            f"{row['group_a']:<20} -> "
            f"{row['group_b']:<20} "
            f"centroid={row['centroid_distance']:.6f} "
            f"scale_ratio={row['scale_ratio_b_over_a']:.6f} "
            f"mean_angle={row['mean_principal_angle_deg']:.3f} deg"
        )

    print("-" * 88)

    return {
        "groups": statistics_rows,
        "pairwise": pair_rows,
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    if not (
        0.0
        < args.train_holdout_fraction
        < 1.0
    ):
        raise ValueError(
            "--train-holdout-fraction must lie in (0, 1)."
        )

    if not (
        0.0
        < args.sequence10_fit_fraction
        < 1.0
    ):
        raise ValueError(
            "--sequence10-fit-fraction must lie in (0, 1)."
        )

    if not (
        0.0
        < args.sequence10_validation_fraction
        < 1.0
    ):
        raise ValueError(
            "--sequence10-validation-fraction must lie in (0, 1)."
        )

    if (
        args.sequence10_fit_fraction
        + args.sequence10_validation_fraction
        >= 1.0
    ):
        raise ValueError(
            "Sequence-10 fit + validation fractions must leave "
            "a non-empty final test partition."
        )

    if args.pca_components <= 0:
        raise ValueError(
            "--pca-components must be positive."
        )

    if any(
        alpha < 0.0
        for alpha in args.ridge_alphas
    ):
        raise ValueError(
            "--ridge-alphas cannot contain negative values."
        )

    sequence_directories = (
        parse_sequence_inputs(
            args.sequence_input
        )
    )

    validate_requested_sequences(
        sequence_directories=sequence_directories,
        train_sequences=args.train_sequences,
        validation_sequence=(
            args.validation_sequence
        ),
        test_sequence=args.test_sequence,
    )

    args.output_dir = (
        args.output_dir.expanduser().resolve()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_sequences = list(
        args.train_sequences
    )

    required_sequences.extend(
        [
            args.validation_sequence,
            args.test_sequence,
        ]
    )

    representation_data: Dict[
        str,
        SequenceData,
    ] = {}

    print("=" * 88)
    print("Loading translation representations")
    print("=" * 88)

    for sequence in required_sequences:
        representation_data[
            sequence
        ] = load_sequence(
            sequence=sequence,
            directory=sequence_directories[
                sequence
            ],
            representation_filename=(
                args.representation_file
            ),
            prediction_filename=(
                args.prediction_file
            ),
            representation_key=(
                args.representation_key
            ),
        )

    dimensions = {
        data.representation_dim
        for data in representation_data.values()
    }

    if len(
        dimensions
    ) != 1:
        raise RuntimeError(
            "Representation dimensions differ across sequences: "
            f"{sorted(dimensions)}."
        )

    representation_dimension = next(
        iter(
            dimensions
        )
    )

    print("-" * 88)
    print(
        f"Common representation dimension: {representation_dimension}"
    )
    print("=" * 88)

    summary: Dict[
        str,
        object,
    ] = {
        "stage": args.stage,
        "representation_dimension": (
            representation_dimension
        ),
        "train_sequences": list(
            args.train_sequences
        ),
        "validation_sequence": (
            args.validation_sequence
        ),
        "test_sequence": (
            args.test_sequence
        ),
        "sequence_samples": {
            sequence: data.num_samples
            for sequence, data
            in representation_data.items()
        },
    }

    if args.stage in (
        "all",
        "geometry",
    ):
        summary[
            "geometry"
        ] = run_geometry(
            sequence_data=representation_data,
            train_sequences=(
                args.train_sequences
            ),
            validation_sequence=(
                args.validation_sequence
            ),
            test_sequence=(
                args.test_sequence
            ),
            pca_components=(
                args.pca_components
            ),
            output_dir=(
                args.output_dir
            ),
        )

    if args.stage in (
        "all",
        "probe",
    ):
        summary[
            "probes"
        ] = run_probes(
            sequence_data=representation_data,
            train_sequences=(
                args.train_sequences
            ),
            validation_sequence=(
                args.validation_sequence
            ),
            test_sequence=(
                args.test_sequence
            ),
            candidate_alphas=(
                args.ridge_alphas
            ),
            train_holdout_fraction=(
                args.train_holdout_fraction
            ),
            sequence10_fit_fraction=(
                args.sequence10_fit_fraction
            ),
            sequence10_validation_fraction=(
                args.sequence10_validation_fraction
            ),
            output_dir=(
                args.output_dir
            ),
        )

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

    print()
    print("=" * 88)
    print("Translation latent alignment analysis complete")
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