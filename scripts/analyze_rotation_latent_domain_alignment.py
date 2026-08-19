#!/usr/bin/env python3
"""
Full-resolution DeepDCT-VO rotation latent alignment / domain audit.

This audit operates directly on the exact rotation representation:

    h_R in R^D

where, for the current paper-style RotationHead,

    D = 120 * 120 = 14400.

NO random projection is applied.

The script investigates three complementary views:

1. Full-resolution latent statistics
   ---------------------------------
   For train 00-08, validation 09, and test 10:

       centroid
       centroid norm
       RMS feature scale
       total variance
       mean feature std
       standardized centroid shift
       diagonal covariance/scale shift

2. Exact trained-readout subspace
   ------------------------------
   The learned rotation readout is:

       r_hat = W h + b

       W shape = [3, D]

   We compute an orthonormal basis Q for span(W.T), then decompose:

       h = h_parallel + h_perp

   where:

       h_parallel lies in the exact 3-D rotation-readout subspace
       h_perp     lies in its orthogonal complement

   We compare train/validation/test behavior in this exact subspace.

3. Full-resolution covariance geometry
   -----------------------------------
   IncrementalPCA is fit DIRECTLY to the D-dimensional h vectors for:

       train 00-08
       validation 09
       test 10

   Principal angles between the resulting subspaces quantify whether
   the dominant latent directions rotate across domains.

Important
---------
IncrementalPCA reduces memory pressure, but every input vector remains
the original full-resolution 14400-D representation. There is no
random feature projection.

Expected input structure
------------------------

    <sequence_dir>/00/
        rotation_representations.npz
        frame_predictions.csv

    ...
    <sequence_dir>/10/
        rotation_representations.npz
        frame_predictions.csv

Typical command
---------------

python scripts/analyze_rotation_latent_domain_alignment.py \\
    --input-root experiments/semantic_depth_rotation_rep_inputs \\
    --checkpoint experiments/semantic_depth_identity_output/best_validation.pt \\
    --output-dir experiments/semantic_depth_rotation_latent_domain_audit \\
    --pca-components 16 \\
    --chunk-size 128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sklearn.decomposition import IncrementalPCA
except ImportError as error:
    raise ImportError(
        "This audit requires scikit-learn. Install it with:\n"
        "  pip install scikit-learn"
    ) from error


AXES = (
    "x",
    "y",
    "z",
)

TRAIN_SEQUENCES = tuple(
    f"{index:02d}"
    for index in range(9)
)

VALIDATION_SEQUENCES = (
    "09",
)

TEST_SEQUENCES = (
    "10",
)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-resolution DeepDCT-VO rotation latent "
            "alignment/domain audit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory containing sequence subdirectories 00 ... 10. "
            "Each must contain rotation_representations.npz and "
            "frame_predictions.csv."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Checkpoint supplying rotation_head.dense.weight/bias. "
            "Use the checkpoint whose representation is being audited."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--representation-file",
        type=str,
        default="rotation_representations.npz",
    )

    parser.add_argument(
        "--representation-key",
        type=str,
        default="rotation_rep",
    )

    parser.add_argument(
        "--prediction-file",
        type=str,
        default="frame_predictions.csv",
    )

    parser.add_argument(
        "--weight-key",
        type=str,
        default="rotation_head.dense.weight",
    )

    parser.add_argument(
        "--bias-key",
        type=str,
        default="rotation_head.dense.bias",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=16,
        help=(
            "Number of full-resolution IncrementalPCA components "
            "fit separately to train, validation, and test."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help=(
            "Rows processed at once for statistics and PCA."
        ),
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-8,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: {args.input_root}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.checkpoint}"
        )

    if args.pca_components <= 0:
        raise ValueError(
            "--pca-components must be positive."
        )

    if args.chunk_size < args.pca_components:
        raise ValueError(
            "--chunk-size must be >= --pca-components because "
            "IncrementalPCA requires enough samples per partial_fit()."
        )

    if args.epsilon <= 0.0:
        raise ValueError(
            "--epsilon must be positive."
        )


# ============================================================================
# File loading
# ============================================================================


def sequence_representation_path(
    root: Path,
    sequence: str,
    filename: str,
) -> Path:
    return (
        root
        / sequence
        / filename
    )


def sequence_prediction_path(
    root: Path,
    sequence: str,
    filename: str,
) -> Path:
    return (
        root
        / sequence
        / filename
    )


def load_representation(
    path: Path,
    key: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing representation file: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        if key not in archive:
            raise KeyError(
                f"{path} does not contain key {key!r}. "
                f"Available: {archive.files}"
            )

        representation = np.asarray(
            archive[key],
            dtype=np.float32,
        )

    if representation.ndim > 2:
        representation = representation.reshape(
            representation.shape[0],
            -1,
        )

    if representation.ndim != 2:
        raise ValueError(
            f"Expected [N,D], received {representation.shape}."
        )

    if not np.all(
        np.isfinite(
            representation
        )
    ):
        raise ValueError(
            f"Non-finite representation values: {path}"
        )

    return representation


def load_rotation_targets(
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    rows: List[
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

        required = (
            "rotation_gt_x",
            "rotation_gt_y",
            "rotation_gt_z",
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        missing = [
            field
            for field in required
            if field not in fieldnames
        ]

        if missing:
            raise KeyError(
                f"{path} missing columns: {missing}"
            )

        for row in reader:
            rows.append(
                [
                    float(
                        row[
                            "rotation_gt_x"
                        ]
                    ),
                    float(
                        row[
                            "rotation_gt_y"
                        ]
                    ),
                    float(
                        row[
                            "rotation_gt_z"
                        ]
                    ),
                ]
            )

    result = np.asarray(
        rows,
        dtype=np.float64,
    )

    if (
        result.ndim != 2
        or result.shape[1] != 3
    ):
        raise ValueError(
            f"Unexpected rotation target shape: {result.shape}"
        )

    return result


# ============================================================================
# Checkpoint / exact readout subspace
# ============================================================================


def remove_module_prefix(
    state_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    result: Dict[
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

        result[key] = value

    return result


def load_readout(
    checkpoint_path: Path,
    weight_key: str,
    bias_key: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    checkpoint = torch.load(
        checkpoint_path,
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

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    state_dict = remove_module_prefix(
        state_dict
    )

    if weight_key not in state_dict:
        raise KeyError(
            f"Missing checkpoint key: {weight_key}"
        )

    if bias_key not in state_dict:
        raise KeyError(
            f"Missing checkpoint key: {bias_key}"
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
            "Expected rotation bias shape [3], "
            f"received {bias.shape}."
        )

    return (
        weight,
        bias,
    )


def readout_subspace_basis(
    weight: np.ndarray,
) -> np.ndarray:
    """
    Return orthonormal basis Q for span(W.T).

    W:
        [3, D]

    W.T:
        [D, 3]

    Q:
        [D, rank]
    """

    q, r = np.linalg.qr(
        weight.T
    )

    diagonal = np.abs(
        np.diag(
            r
        )
    )

    tolerance = (
        max(
            weight.shape
        )
        * np.max(
            diagonal
        )
        * np.finfo(
            np.float64
        ).eps
        if diagonal.size
        else 0.0
    )

    rank = int(
        np.count_nonzero(
            diagonal > tolerance
        )
    )

    if rank <= 0:
        raise RuntimeError(
            "Rotation readout has zero numerical rank."
        )

    return q[
        :,
        :rank,
    ]


# ============================================================================
# Chunk iterator
# ============================================================================


def iter_sequence_chunks(
    root: Path,
    sequences: Sequence[str],
    representation_filename: str,
    representation_key: str,
    chunk_size: int,
) -> Iterable[
    Tuple[
        str,
        np.ndarray,
    ]
]:
    for sequence in sequences:
        path = (
            root
            / sequence
            / representation_filename
        )

        representation = (
            load_representation(
                path,
                representation_key,
            )
        )

        for start in range(
            0,
            representation.shape[0],
            chunk_size,
        ):
            stop = min(
                start
                + chunk_size,
                representation.shape[0],
            )

            yield (
                sequence,
                representation[
                    start:stop
                ],
            )

        del representation


# ============================================================================
# Streaming full-resolution statistics
# ============================================================================


class StreamingMoments:
    """
    Numerically stable feature-wise mean/variance accumulator.

    Stores only O(D) values regardless of sample count.
    """

    def __init__(
        self,
        dimension: int,
    ) -> None:
        self.dimension = int(
            dimension
        )

        self.count = 0

        self.mean = np.zeros(
            self.dimension,
            dtype=np.float64,
        )

        self.m2 = np.zeros(
            self.dimension,
            dtype=np.float64,
        )

        self.sum_squared_norm = 0.0

    def update(
        self,
        x: np.ndarray,
    ) -> None:
        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if (
            x.ndim != 2
            or x.shape[1]
            != self.dimension
        ):
            raise ValueError(
                "StreamingMoments dimension mismatch: "
                f"{x.shape}"
            )

        batch_count = int(
            x.shape[0]
        )

        if batch_count == 0:
            return

        batch_mean = np.mean(
            x,
            axis=0,
        )

        centered = (
            x
            - batch_mean
        )

        batch_m2 = np.sum(
            centered ** 2,
            axis=0,
        )

        self.sum_squared_norm += float(
            np.sum(
                x ** 2
            )
        )

        if self.count == 0:
            self.count = batch_count

            self.mean = (
                batch_mean
            )

            self.m2 = (
                batch_m2
            )

            return

        total = (
            self.count
            + batch_count
        )

        delta = (
            batch_mean
            - self.mean
        )

        self.mean = (
            self.mean
            + delta
            * batch_count
            / total
        )

        self.m2 = (
            self.m2
            + batch_m2
            + delta ** 2
            * self.count
            * batch_count
            / total
        )

        self.count = total

    def variance(
        self,
    ) -> np.ndarray:
        if self.count <= 1:
            return np.zeros(
                self.dimension,
                dtype=np.float64,
            )

        return (
            self.m2
            / (
                self.count
                - 1
            )
        )

    def std(
        self,
    ) -> np.ndarray:
        return np.sqrt(
            np.maximum(
                self.variance(),
                0.0,
            )
        )

    def total_variance(
        self,
    ) -> float:
        return float(
            np.sum(
                self.variance()
            )
        )

    def rms_scale(
        self,
    ) -> float:
        if self.count <= 0:
            return float(
                "nan"
            )

        return float(
            np.sqrt(
                self.sum_squared_norm
                / (
                    self.count
                    * self.dimension
                )
            )
        )


def infer_representation_dimension(
    root: Path,
    representation_filename: str,
    representation_key: str,
) -> int:
    sample = load_representation(
        root
        / "00"
        / representation_filename,
        representation_key,
    )

    dimension = int(
        sample.shape[1]
    )

    del sample

    return dimension


def compute_group_moments(
    root: Path,
    sequences: Sequence[str],
    representation_filename: str,
    representation_key: str,
    chunk_size: int,
    dimension: int,
) -> StreamingMoments:
    moments = StreamingMoments(
        dimension
    )

    for sequence, chunk in iter_sequence_chunks(
        root=root,
        sequences=sequences,
        representation_filename=(
            representation_filename
        ),
        representation_key=(
            representation_key
        ),
        chunk_size=chunk_size,
    ):
        moments.update(
            chunk
        )

    return moments


# ============================================================================
# Exact readout-subspace statistics
# ============================================================================


class ReadoutSubspaceAccumulator:
    def __init__(
        self,
        rank: int,
    ) -> None:
        self.count = 0

        self.parallel_sum = np.zeros(
            rank,
            dtype=np.float64,
        )

        self.parallel_sq_sum = np.zeros(
            rank,
            dtype=np.float64,
        )

        self.parallel_energy = 0.0
        self.total_energy = 0.0

    def update(
        self,
        x: np.ndarray,
        basis: np.ndarray,
    ) -> None:
        x64 = np.asarray(
            x,
            dtype=np.float64,
        )

        coordinates = (
            x64
            @ basis
        )

        self.count += (
            x64.shape[0]
        )

        self.parallel_sum += np.sum(
            coordinates,
            axis=0,
        )

        self.parallel_sq_sum += np.sum(
            coordinates ** 2,
            axis=0,
        )

        self.parallel_energy += float(
            np.sum(
                coordinates ** 2
            )
        )

        self.total_energy += float(
            np.sum(
                x64 ** 2
            )
        )

    def summary(
        self,
    ) -> Dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError(
                "Empty readout-subspace accumulator."
            )

        mean = (
            self.parallel_sum
            / self.count
        )

        variance = (
            self.parallel_sq_sum
            / self.count
            - mean ** 2
        )

        variance = np.maximum(
            variance,
            0.0,
        )

        parallel_fraction = (
            self.parallel_energy
            / self.total_energy
            if self.total_energy > 0.0
            else float(
                "nan"
            )
        )

        return {
            "count": int(
                self.count
            ),
            "coordinate_mean": (
                mean.tolist()
            ),
            "coordinate_std": (
                np.sqrt(
                    variance
                ).tolist()
            ),
            "parallel_energy_fraction": float(
                parallel_fraction
            ),
            "orthogonal_energy_fraction": float(
                1.0
                - parallel_fraction
            ),
        }


def compute_readout_subspace_stats(
    root: Path,
    sequences: Sequence[str],
    representation_filename: str,
    representation_key: str,
    chunk_size: int,
    basis: np.ndarray,
) -> Dict[str, Any]:
    accumulator = ReadoutSubspaceAccumulator(
        rank=basis.shape[1]
    )

    for sequence, chunk in iter_sequence_chunks(
        root=root,
        sequences=sequences,
        representation_filename=(
            representation_filename
        ),
        representation_key=(
            representation_key
        ),
        chunk_size=chunk_size,
    ):
        accumulator.update(
            chunk,
            basis,
        )

    return accumulator.summary()


# ============================================================================
# Full-resolution Incremental PCA
# ============================================================================


def fit_incremental_pca(
    root: Path,
    sequences: Sequence[str],
    representation_filename: str,
    representation_key: str,
    chunk_size: int,
    components: int,
) -> IncrementalPCA:
    ipca = IncrementalPCA(
        n_components=components,
        batch_size=chunk_size,
    )

    pending: List[
        np.ndarray
    ] = []

    pending_rows = 0

    for sequence, chunk in iter_sequence_chunks(
        root=root,
        sequences=sequences,
        representation_filename=(
            representation_filename
        ),
        representation_key=(
            representation_key
        ),
        chunk_size=chunk_size,
    ):
        pending.append(
            np.asarray(
                chunk,
                dtype=np.float64,
            )
        )

        pending_rows += int(
            chunk.shape[0]
        )

        if pending_rows >= chunk_size:
            batch = np.concatenate(
                pending,
                axis=0,
            )

            # Keep a sufficiently sized remainder for another fit.
            while (
                batch.shape[0]
                >= chunk_size
            ):
                current = batch[
                    :chunk_size
                ]

                ipca.partial_fit(
                    current
                )

                batch = batch[
                    chunk_size:
                ]

            pending = (
                [batch]
                if batch.shape[0] > 0
                else []
            )

            pending_rows = (
                batch.shape[0]
            )

    if pending:
        remainder = np.concatenate(
            pending,
            axis=0,
        )

        if remainder.shape[0] >= components:
            ipca.partial_fit(
                remainder
            )

    if not hasattr(
        ipca,
        "components_",
    ):
        raise RuntimeError(
            "IncrementalPCA received insufficient data."
        )

    return ipca


def principal_angles_degrees(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> np.ndarray:
    """
    basis_a, basis_b:
        row-wise orthonormal PCA components [K,D].
    """

    overlap = (
        basis_a
        @ basis_b.T
    )

    singular_values = np.linalg.svd(
        overlap,
        compute_uv=False,
    )

    singular_values = np.clip(
        singular_values,
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(
            singular_values
        )
    )


# ============================================================================
# Readout output diagnostics from exact full-resolution h
# ============================================================================


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if (
        np.std(
            a
        )
        <= 1.0e-12
        or np.std(
            b
        )
        <= 1.0e-12
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


def compute_exact_readout_quality(
    root: Path,
    sequences: Sequence[str],
    representation_filename: str,
    representation_key: str,
    prediction_filename: str,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Dict[str, Any]:
    all_gt: List[
        np.ndarray
    ] = []

    all_pred: List[
        np.ndarray
    ] = []

    for sequence in sequences:
        representation = load_representation(
            root
            / sequence
            / representation_filename,
            representation_key,
        )

        gt = load_rotation_targets(
            root
            / sequence
            / prediction_filename
        )

        if representation.shape[0] != gt.shape[0]:
            raise ValueError(
                f"Sequence {sequence}: representation/target "
                f"count mismatch."
            )

        prediction = (
            representation.astype(
                np.float64
            )
            @ weight.T
            + bias[
                None,
                :
            ]
        )

        all_gt.append(
            gt
        )

        all_pred.append(
            prediction
        )

        del representation

    gt = np.concatenate(
        all_gt,
        axis=0,
    )

    prediction = np.concatenate(
        all_pred,
        axis=0,
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for axis_index, axis in enumerate(
        AXES
    ):
        gt_axis = gt[
            :,
            axis_index
        ]

        pred_axis = prediction[
            :,
            axis_index
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
                "bias": float(
                    np.mean(
                        error
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
                    if gt_std
                    > 1.0e-12
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

    return {
        "samples": int(
            gt.shape[0]
        ),
        "axis_statistics": rows,
        "vector_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        prediction
                        - gt
                    )
                    ** 2
                )
            )
        ),
    }


# ============================================================================
# Domain comparison
# ============================================================================


def group_summary(
    name: str,
    moments: StreamingMoments,
) -> Dict[str, Any]:
    std = moments.std()

    return {
        "group": name,
        "samples": int(
            moments.count
        ),
        "dimension": int(
            moments.dimension
        ),
        "centroid_norm": float(
            np.linalg.norm(
                moments.mean
            )
        ),
        "rms_scale": float(
            moments.rms_scale()
        ),
        "total_variance": float(
            moments.total_variance()
        ),
        "mean_feature_std": float(
            np.mean(
                std
            )
        ),
        "median_feature_std": float(
            np.median(
                std
            )
        ),
    }


def compare_domains(
    name_a: str,
    moments_a: StreamingMoments,
    name_b: str,
    moments_b: StreamingMoments,
    epsilon: float,
) -> Dict[str, Any]:
    mean_a = moments_a.mean
    mean_b = moments_b.mean

    std_a = moments_a.std()
    std_b = moments_b.std()

    centroid_delta = (
        mean_b
        - mean_a
    )

    pooled_scale = np.sqrt(
        0.5
        * (
            std_a ** 2
            + std_b ** 2
        )
        + epsilon
    )

    standardized_shift = (
        centroid_delta
        / pooled_scale
    )

    log_std_ratio = np.log(
        (
            std_b
            + epsilon
        )
        / (
            std_a
            + epsilon
        )
    )

    cosine_denominator = (
        np.linalg.norm(
            mean_a
        )
        * np.linalg.norm(
            mean_b
        )
    )

    centroid_cosine = (
        float(
            np.dot(
                mean_a,
                mean_b,
            )
            / cosine_denominator
        )
        if cosine_denominator
        > epsilon
        else float(
            "nan"
        )
    )

    return {
        "group_a": name_a,
        "group_b": name_b,

        "centroid_distance": float(
            np.linalg.norm(
                centroid_delta
            )
        ),

        "centroid_cosine": (
            centroid_cosine
        ),

        "standardized_centroid_shift_rms": float(
            np.sqrt(
                np.mean(
                    standardized_shift ** 2
                )
            )
        ),

        "standardized_centroid_shift_max_abs": float(
            np.max(
                np.abs(
                    standardized_shift
                )
            )
        ),

        "mean_abs_log_std_ratio": float(
            np.mean(
                np.abs(
                    log_std_ratio
                )
            )
        ),

        "rms_log_std_ratio": float(
            np.sqrt(
                np.mean(
                    log_std_ratio ** 2
                )
            )
        ),

        "total_variance_ratio": float(
            moments_b.total_variance()
            / moments_a.total_variance()
            if moments_a.total_variance()
            > epsilon
            else float(
                "nan"
            )
        ),
    }


# ============================================================================
# CSV / JSON output
# ============================================================================


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
        Mapping,
    ):
        return {
            str(
                key
            ): json_safe(
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
            json_safe(
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
# Plots
# ============================================================================


def plot_pca_variance(
    pca_by_group: Mapping[
        str,
        IncrementalPCA,
    ],
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(
            9,
            6,
        )
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for group_name, pca in pca_by_group.items():
        cumulative = np.cumsum(
            pca.explained_variance_ratio_
        )

        axis.plot(
            np.arange(
                1,
                len(
                    cumulative
                )
                + 1,
            ),
            cumulative,
            marker="o",
            label=group_name,
        )

    axis.set_xlabel(
        "Full-resolution PCA component count"
    )

    axis.set_ylabel(
        "Cumulative explained variance ratio"
    )

    axis.set_title(
        "Rotation latent covariance concentration"
    )

    axis.grid(
        True
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


def plot_principal_angles(
    angle_rows: Sequence[
        Mapping[str, Any]
    ],
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

    grouped: Dict[
        str,
        List[
            Tuple[
                int,
                float,
            ]
        ],
    ] = {}

    for row in angle_rows:
        label = (
            f"{row['group_a']}→{row['group_b']}"
        )

        grouped.setdefault(
            label,
            [],
        ).append(
            (
                int(
                    row[
                        "component_index"
                    ]
                ),
                float(
                    row[
                        "angle_deg"
                    ]
                ),
            )
        )

    for label, values in grouped.items():
        values = sorted(
            values
        )

        axis.plot(
            [
                item[0]
                for item in values
            ],
            [
                item[1]
                for item in values
            ],
            marker="o",
            label=label,
        )

    axis.set_xlabel(
        "Principal-angle index"
    )

    axis.set_ylabel(
        "Angle (degrees)"
    )

    axis.set_title(
        "Full-resolution rotation latent subspace angles"
    )

    axis.grid(
        True
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
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    args.input_root = (
        args.input_root
        .expanduser()
        .resolve()
    )

    args.checkpoint = (
        args.checkpoint
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

    dimension = infer_representation_dimension(
        root=args.input_root,
        representation_filename=(
            args.representation_file
        ),
        representation_key=(
            args.representation_key
        ),
    )

    print("=" * 88)
    print(
        "DeepDCT-VO full-resolution rotation latent domain audit"
    )
    print("=" * 88)
    print(
        f"Input root:              {args.input_root}"
    )
    print(
        f"Checkpoint:              {args.checkpoint}"
    )
    print(
        f"Representation dimension:{dimension:>8d}"
    )
    print(
        f"PCA components:          {args.pca_components}"
    )
    print(
        f"Chunk size:              {args.chunk_size}"
    )
    print(
        "Random projection:       NONE"
    )
    print("=" * 88)

    if args.pca_components > dimension:
        raise ValueError(
            "--pca-components cannot exceed representation dimension."
        )

    weight, bias = load_readout(
        checkpoint_path=args.checkpoint,
        weight_key=args.weight_key,
        bias_key=args.bias_key,
    )

    if weight.shape[1] != dimension:
        raise ValueError(
            "Checkpoint readout dimension does not match representation: "
            f"{weight.shape[1]} vs {dimension}."
        )

    readout_basis = (
        readout_subspace_basis(
            weight
        )
    )

    print(
        f"Readout numerical rank:  {readout_basis.shape[1]}"
    )

    group_sequences = {
        "train_00_08": (
            TRAIN_SEQUENCES
        ),
        "validation_09": (
            VALIDATION_SEQUENCES
        ),
        "test_10": (
            TEST_SEQUENCES
        ),
    }

    # ------------------------------------------------------------------
    # Full-resolution streaming statistics
    # ------------------------------------------------------------------

    moments_by_group: Dict[
        str,
        StreamingMoments,
    ] = {}

    group_rows: List[
        Dict[str, Any]
    ] = []

    for group_name, sequences in group_sequences.items():
        print()
        print(
            f"Computing full-resolution moments: {group_name}"
        )

        moments = compute_group_moments(
            root=args.input_root,
            sequences=sequences,
            representation_filename=(
                args.representation_file
            ),
            representation_key=(
                args.representation_key
            ),
            chunk_size=args.chunk_size,
            dimension=dimension,
        )

        moments_by_group[
            group_name
        ] = moments

        group_rows.append(
            group_summary(
                group_name,
                moments,
            )
        )

    write_csv(
        args.output_dir
        / "full_resolution_group_statistics.csv",
        group_rows,
    )

    # ------------------------------------------------------------------
    # Full-dimensional pairwise domain shift
    # ------------------------------------------------------------------

    comparison_pairs = (
        (
            "train_00_08",
            "validation_09",
        ),
        (
            "train_00_08",
            "test_10",
        ),
        (
            "validation_09",
            "test_10",
        ),
    )

    domain_rows: List[
        Dict[str, Any]
    ] = []

    for group_a, group_b in comparison_pairs:
        domain_rows.append(
            compare_domains(
                group_a,
                moments_by_group[
                    group_a
                ],
                group_b,
                moments_by_group[
                    group_b
                ],
                epsilon=args.epsilon,
            )
        )

    write_csv(
        args.output_dir
        / "full_resolution_domain_shift.csv",
        domain_rows,
    )

    # ------------------------------------------------------------------
    # Exact trained-readout subspace
    # ------------------------------------------------------------------

    readout_rows: List[
        Dict[str, Any]
    ] = []

    readout_details: Dict[
        str,
        Any,
    ] = {}

    for group_name, sequences in group_sequences.items():
        print(
            f"Computing readout-subspace statistics: {group_name}"
        )

        stats = (
            compute_readout_subspace_stats(
                root=args.input_root,
                sequences=sequences,
                representation_filename=(
                    args.representation_file
                ),
                representation_key=(
                    args.representation_key
                ),
                chunk_size=args.chunk_size,
                basis=readout_basis,
            )
        )

        readout_details[
            group_name
        ] = stats

        row: Dict[
            str,
            Any,
        ] = {
            "group": group_name,
            "samples": stats[
                "count"
            ],
            "parallel_energy_fraction": (
                stats[
                    "parallel_energy_fraction"
                ]
            ),
            "orthogonal_energy_fraction": (
                stats[
                    "orthogonal_energy_fraction"
                ]
            ),
        }

        for index, value in enumerate(
            stats[
                "coordinate_mean"
            ]
        ):
            row[
                f"readout_coord_{index}_mean"
            ] = value

        for index, value in enumerate(
            stats[
                "coordinate_std"
            ]
        ):
            row[
                f"readout_coord_{index}_std"
            ] = value

        readout_rows.append(
            row
        )

    write_csv(
        args.output_dir
        / "readout_subspace_statistics.csv",
        readout_rows,
    )

    # ------------------------------------------------------------------
    # Exact readout quality for current checkpoint
    # ------------------------------------------------------------------

    readout_quality: Dict[
        str,
        Any,
    ] = {}

    readout_quality_rows: List[
        Dict[str, Any]
    ] = []

    for group_name, sequences in group_sequences.items():
        quality = (
            compute_exact_readout_quality(
                root=args.input_root,
                sequences=sequences,
                representation_filename=(
                    args.representation_file
                ),
                representation_key=(
                    args.representation_key
                ),
                prediction_filename=(
                    args.prediction_file
                ),
                weight=weight,
                bias=bias,
            )
        )

        readout_quality[
            group_name
        ] = quality

        for axis_row in quality[
            "axis_statistics"
        ]:
            row = {
                "group": group_name,
                **axis_row,
            }

            readout_quality_rows.append(
                row
            )

    write_csv(
        args.output_dir
        / "exact_readout_quality.csv",
        readout_quality_rows,
    )

    # ------------------------------------------------------------------
    # Full-resolution Incremental PCA
    # ------------------------------------------------------------------

    pca_by_group: Dict[
        str,
        IncrementalPCA,
    ] = {}

    pca_rows: List[
        Dict[str, Any]
    ] = []

    for group_name, sequences in group_sequences.items():
        print()
        print(
            f"Fitting full-resolution IncrementalPCA: {group_name}"
        )

        pca = fit_incremental_pca(
            root=args.input_root,
            sequences=sequences,
            representation_filename=(
                args.representation_file
            ),
            representation_key=(
                args.representation_key
            ),
            chunk_size=args.chunk_size,
            components=args.pca_components,
        )

        pca_by_group[
            group_name
        ] = pca

        for component_index in range(
            args.pca_components
        ):
            pca_rows.append(
                {
                    "group": group_name,
                    "component_index": (
                        component_index
                        + 1
                    ),
                    "explained_variance": float(
                        pca.explained_variance_[
                            component_index
                        ]
                    ),
                    "explained_variance_ratio": float(
                        pca.explained_variance_ratio_[
                            component_index
                        ]
                    ),
                    "cumulative_explained_variance_ratio": float(
                        np.sum(
                            pca.explained_variance_ratio_[
                                :component_index
                                + 1
                            ]
                        )
                    ),
                }
            )

    write_csv(
        args.output_dir
        / "full_resolution_pca_variance.csv",
        pca_rows,
    )

    # ------------------------------------------------------------------
    # Principal angles between full-resolution covariance subspaces
    # ------------------------------------------------------------------

    angle_rows: List[
        Dict[str, Any]
    ] = []

    angle_summaries: List[
        Dict[str, Any]
    ] = []

    for group_a, group_b in comparison_pairs:
        angles = (
            principal_angles_degrees(
                pca_by_group[
                    group_a
                ].components_,
                pca_by_group[
                    group_b
                ].components_,
            )
        )

        for index, angle in enumerate(
            angles
        ):
            angle_rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "component_index": (
                        index + 1
                    ),
                    "angle_deg": float(
                        angle
                    ),
                }
            )

        angle_summaries.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "mean_angle_deg": float(
                    np.mean(
                        angles
                    )
                ),
                "median_angle_deg": float(
                    np.median(
                        angles
                    )
                ),
                "min_angle_deg": float(
                    np.min(
                        angles
                    )
                ),
                "max_angle_deg": float(
                    np.max(
                        angles
                    )
                ),
            }
        )

    write_csv(
        args.output_dir
        / "full_resolution_principal_angles.csv",
        angle_rows,
    )

    write_csv(
        args.output_dir
        / "full_resolution_principal_angle_summary.csv",
        angle_summaries,
    )

    # ------------------------------------------------------------------
    # Readout-subspace pairwise shift
    # ------------------------------------------------------------------

    readout_shift_rows: List[
        Dict[str, Any]
    ] = []

    for group_a, group_b in comparison_pairs:
        a = readout_details[
            group_a
        ]

        b = readout_details[
            group_b
        ]

        mean_a = np.asarray(
            a[
                "coordinate_mean"
            ],
            dtype=np.float64,
        )

        mean_b = np.asarray(
            b[
                "coordinate_mean"
            ],
            dtype=np.float64,
        )

        std_a = np.asarray(
            a[
                "coordinate_std"
            ],
            dtype=np.float64,
        )

        std_b = np.asarray(
            b[
                "coordinate_std"
            ],
            dtype=np.float64,
        )

        pooled_std = np.sqrt(
            0.5
            * (
                std_a ** 2
                + std_b ** 2
            )
            + args.epsilon
        )

        standardized_mean_shift = (
            (
                mean_b
                - mean_a
            )
            / pooled_std
        )

        row: Dict[
            str,
            Any,
        ] = {
            "group_a": group_a,
            "group_b": group_b,
            "coordinate_centroid_distance": float(
                np.linalg.norm(
                    mean_b
                    - mean_a
                )
            ),
            "standardized_coordinate_shift_rms": float(
                np.sqrt(
                    np.mean(
                        standardized_mean_shift
                        ** 2
                    )
                )
            ),
            "parallel_energy_fraction_a": (
                a[
                    "parallel_energy_fraction"
                ]
            ),
            "parallel_energy_fraction_b": (
                b[
                    "parallel_energy_fraction"
                ]
            ),
        }

        for index in range(
            readout_basis.shape[1]
        ):
            row[
                f"coord_{index}_mean_shift"
            ] = float(
                mean_b[
                    index
                ]
                - mean_a[
                    index
                ]
            )

            row[
                f"coord_{index}_std_ratio"
            ] = float(
                (
                    std_b[
                        index
                    ]
                    + args.epsilon
                )
                / (
                    std_a[
                        index
                    ]
                    + args.epsilon
                )
            )

        readout_shift_rows.append(
            row
        )

    write_csv(
        args.output_dir
        / "readout_subspace_domain_shift.csv",
        readout_shift_rows,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_pca_variance(
        pca_by_group=pca_by_group,
        output_path=(
            plots_dir
            / "full_resolution_pca_variance.png"
        ),
    )

    plot_principal_angles(
        angle_rows=angle_rows,
        output_path=(
            plots_dir
            / "full_resolution_principal_angles.png"
        ),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "input_root": str(
            args.input_root
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "representation_key": (
            args.representation_key
        ),
        "representation_dimension": (
            dimension
        ),
        "random_projection_used": False,
        "pca_method": (
            "IncrementalPCA directly on full-resolution "
            f"{dimension}-D representation"
        ),
        "pca_components": (
            args.pca_components
        ),
        "readout_weight_shape": list(
            weight.shape
        ),
        "readout_subspace_rank": int(
            readout_basis.shape[1]
        ),
        "group_statistics": (
            group_rows
        ),
        "domain_shift": (
            domain_rows
        ),
        "readout_subspace_statistics": (
            readout_details
        ),
        "readout_subspace_domain_shift": (
            readout_shift_rows
        ),
        "exact_readout_quality": (
            readout_quality
        ),
        "principal_angle_summary": (
            angle_summaries
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
    # Console report
    # ------------------------------------------------------------------

    print()
    print("=" * 112)
    print(
        "Full-resolution rotation latent group statistics"
    )
    print("=" * 112)

    print(
        f"{'Group':<20}"
        f"{'N':>9}"
        f"{'Centroid':>14}"
        f"{'RMS scale':>14}"
        f"{'Total var':>14}"
        f"{'Mean std':>14}"
    )

    print("-" * 112)

    for row in group_rows:
        print(
            f"{row['group']:<20}"
            f"{row['samples']:>9d}"
            f"{row['centroid_norm']:>14.6f}"
            f"{row['rms_scale']:>14.6f}"
            f"{row['total_variance']:>14.6f}"
            f"{row['mean_feature_std']:>14.6f}"
        )

    print("=" * 112)

    print()
    print("=" * 112)
    print(
        "Full-resolution pairwise domain shift"
    )
    print("=" * 112)

    print(
        f"{'A':<18}"
        f"{'B':<18}"
        f"{'Centroid dist':>15}"
        f"{'Std-centroid':>15}"
        f"{'|log std|':>15}"
        f"{'Var ratio':>15}"
    )

    print("-" * 112)

    for row in domain_rows:
        print(
            f"{row['group_a']:<18}"
            f"{row['group_b']:<18}"
            f"{row['centroid_distance']:>15.6f}"
            f"{row['standardized_centroid_shift_rms']:>15.6f}"
            f"{row['mean_abs_log_std_ratio']:>15.6f}"
            f"{row['total_variance_ratio']:>15.6f}"
        )

    print("=" * 112)

    print()
    print("=" * 112)
    print(
        "Full-resolution covariance principal angles"
    )
    print("=" * 112)

    print(
        f"{'A':<18}"
        f"{'B':<18}"
        f"{'Mean deg':>14}"
        f"{'Median deg':>14}"
        f"{'Min deg':>14}"
        f"{'Max deg':>14}"
    )

    print("-" * 112)

    for row in angle_summaries:
        print(
            f"{row['group_a']:<18}"
            f"{row['group_b']:<18}"
            f"{row['mean_angle_deg']:>14.4f}"
            f"{row['median_angle_deg']:>14.4f}"
            f"{row['min_angle_deg']:>14.4f}"
            f"{row['max_angle_deg']:>14.4f}"
        )

    print("=" * 112)

    print()
    print("=" * 112)
    print(
        "Exact checkpoint readout quality"
    )
    print("=" * 112)

    for group_name in (
        "train_00_08",
        "validation_09",
        "test_10",
    ):
        print()
        print(
            f"{group_name}:"
        )

        print(
            f"  vector RMSE: "
            f"{readout_quality[group_name]['vector_rmse']:.8f}"
        )

        for row in readout_quality[
            group_name
        ][
            "axis_statistics"
        ]:
            print(
                f"  {row['axis']}: "
                f"corr={row['correlation']:+.4f} "
                f"rmse={row['rmse']:.8f} "
                f"std_ratio={row['std_ratio']:.4f}"
            )

    print()
    print(
        f"Outputs saved to: {args.output_dir}"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()