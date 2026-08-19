#!/usr/bin/env python3
"""
Cross-sequence rotation-representation audit for DeepDCT-VO.

The script analyzes the representation immediately before the rotation
regression layer.

Typical dense RotationHead:

    fusion
      -> Conv C->1
      -> ReLU
      -> Dropout
      -> Flatten(14400)
      -> Linear(14400, 3)

The exact 14400-D vectors are exported by evaluate_deepdct_vo.py.

To keep the offline cross-sequence analysis tractable, the exact vectors
are mapped through ONE fixed deterministic Gaussian random projection.
The same projection is applied to every sequence.

This does NOT alter the DeepDCT-VO model or representation. It is only
an offline analysis transform.

Probes
------
A:
    temporal train/test split inside sequences 00-08.

B:
    train 00-08 -> validation sequence 09.

C:
    train 00-08 -> held-out sequence 10.

D:
    sequence-10 local temporal transfer:
        first 60% train
        middle 20% alpha selection
        first 80% refit
        final 20% test

Outputs
-------
summary.json
group_statistics.csv
group_distances.csv
principal_angles.csv
probe_results.csv
probe_alpha_search.csv
plots/common_pca.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ALPHAS = (
    1.0e-6,
    1.0e-4,
    1.0e-2,
    1.0,
    10.0,
    100.0,
    1000.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DeepDCT-VO rotation representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sequence-input",
        action="append",
        required=True,
        metavar="SEQ=DIR",
        help=(
            "Sequence and evaluator output directory. "
            "Repeat for sequences 00 through 10."
        ),
    )

    parser.add_argument(
        "--representation-key",
        default="rotation_rep",
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        default=256,
        help=(
            "Offline deterministic random-projection dimension. "
            "The original representation is not changed on disk."
        ),
    )

    parser.add_argument(
        "--projection-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def parse_sequence_inputs(
    values: Sequence[str],
) -> Dict[str, Path]:
    result: Dict[str, Path] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                "--sequence-input must be SEQ=DIR, "
                f"received {value!r}."
            )

        sequence, directory = value.split(
            "=",
            1,
        )

        sequence = sequence.strip().zfill(2)

        result[sequence] = (
            Path(directory)
            .expanduser()
            .resolve()
        )

    required = {
        f"{index:02d}"
        for index in range(11)
    }

    missing = sorted(
        required.difference(
            result.keys()
        )
    )

    if missing:
        raise ValueError(
            "Missing sequence inputs: "
            f"{missing}"
        )

    return result


def load_csv_targets(
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction CSV: {path}"
        )

    rows: List[List[float]] = []

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

        fields = reader.fieldnames or []

        missing = [
            key
            for key in required
            if key not in fields
        ]

        if missing:
            raise KeyError(
                f"{path} is missing {missing}."
            )

        for row in reader:
            rows.append(
                [
                    float(row["rotation_gt_x"]),
                    float(row["rotation_gt_y"]),
                    float(row["rotation_gt_z"]),
                ]
            )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def load_representation(
    path: Path,
    key: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing representation: {path}"
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

        x = np.asarray(
            archive[key],
            dtype=np.float32,
        )

    if x.ndim > 2:
        x = x.reshape(
            x.shape[0],
            -1,
        )

    if x.ndim != 2:
        raise ValueError(
            f"Expected [N,D], received {x.shape}."
        )

    if not np.all(
        np.isfinite(x)
    ):
        raise ValueError(
            f"Non-finite representation values in {path}."
        )

    return x


def build_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    if output_dim <= 0:
        raise ValueError(
            "--projection-dim must be positive."
        )

    if output_dim > input_dim:
        raise ValueError(
            "projection dimension cannot exceed "
            f"input dimension {input_dim}."
        )

    rng = np.random.default_rng(
        seed
    )

    projection = rng.standard_normal(
        (input_dim, output_dim),
        dtype=np.float32,
    )

    projection /= np.sqrt(
        float(output_dim)
    )

    return projection


def project_in_chunks(
    x: np.ndarray,
    projection: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    result = np.empty(
        (
            x.shape[0],
            projection.shape[1],
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        x.shape[0],
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            x.shape[0],
        )

        result[start:stop] = (
            x[start:stop]
            @ projection
        )

    return result


def load_sequences(
    sequence_dirs: Mapping[str, Path],
    representation_key: str,
    projection_dim: int,
    projection_seed: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    int,
]:
    raw_first = load_representation(
        sequence_dirs["00"]
        / "rotation_representations.npz",
        representation_key,
    )

    input_dim = int(
        raw_first.shape[1]
    )

    projection = build_projection(
        input_dim=input_dim,
        output_dim=projection_dim,
        seed=projection_seed,
    )

    x_by_sequence: Dict[
        str,
        np.ndarray,
    ] = {}

    y_by_sequence: Dict[
        str,
        np.ndarray,
    ] = {}

    for sequence in sorted(
        sequence_dirs
    ):
        directory = sequence_dirs[
            sequence
        ]

        x = load_representation(
            directory
            / "rotation_representations.npz",
            representation_key,
        )

        if x.shape[1] != input_dim:
            raise ValueError(
                f"Sequence {sequence} has representation "
                f"dimension {x.shape[1]}, expected {input_dim}."
            )

        y = load_csv_targets(
            directory
            / "frame_predictions.csv"
        )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sequence {sequence}: representation/label "
                f"count mismatch {x.shape[0]} vs {y.shape[0]}."
            )

        x_by_sequence[
            sequence
        ] = project_in_chunks(
            x,
            projection,
        )

        y_by_sequence[
            sequence
        ] = y

        print(
            f"Loaded sequence {sequence}: "
            f"N={x.shape[0]} "
            f"raw_D={input_dim} "
            f"projected_D={projection_dim}"
        )

        del x

    return (
        x_by_sequence,
        y_by_sequence,
        input_dim,
    )


def concatenate(
    mapping: Mapping[str, np.ndarray],
    sequences: Sequence[str],
) -> np.ndarray:
    return np.concatenate(
        [
            mapping[sequence]
            for sequence in sequences
        ],
        axis=0,
    )


def standardize_fit(
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(
        x,
        axis=0,
        dtype=np.float64,
    )

    std = np.std(
        x,
        axis=0,
        dtype=np.float64,
    )

    std[
        std < 1.0e-8
    ] = 1.0

    return mean, std


def standardize_apply(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (
        (
            x.astype(
                np.float64
            )
            - mean
        )
        / std
    )


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_mean, x_std = (
        standardize_fit(x)
    )

    xs = standardize_apply(
        x,
        x_mean,
        x_std,
    )

    y_mean = np.mean(
        y,
        axis=0,
    )

    yc = y - y_mean

    gram = (
        xs.T
        @ xs
    )

    regularizer = (
        alpha
        * np.eye(
            gram.shape[0],
            dtype=np.float64,
        )
    )

    weights = np.linalg.solve(
        gram + regularizer,
        xs.T @ yc,
    )

    # Store feature scaling parameters together.
    model = np.vstack(
        [
            x_mean,
            x_std,
        ]
    )

    return (
        model,
        np.vstack(
            [
                y_mean[None, :],
                weights,
            ]
        ),
    )


def ridge_predict(
    x: np.ndarray,
    model: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    x_mean = model[0]
    x_std = model[1]

    y_mean = coefficients[0]
    weights = coefficients[1:]

    xs = standardize_apply(
        x,
        x_mean,
        x_std,
    )

    return (
        xs @ weights
        + y_mean
    )


def axis_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    for index, axis in enumerate(
        ("x", "y", "z")
    ):
        true = y_true[:, index]
        pred = y_pred[:, index]
        error = pred - true

        result[
            f"{axis}_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        result[
            f"{axis}_bias"
        ] = float(
            np.mean(
                error
            )
        )

        if (
            np.std(true) > 1.0e-12
            and np.std(pred) > 1.0e-12
        ):
            correlation = float(
                np.corrcoef(
                    true,
                    pred,
                )[0, 1]
            )
        else:
            correlation = float(
                "nan"
            )

        result[
            f"{axis}_corr"
        ] = correlation

        result[
            f"{axis}_std_ratio"
        ] = (
            float(
                np.std(pred)
                / np.std(true)
            )
            if np.std(true)
            > 1.0e-12
            else float(
                "nan"
            )
        )

    result["vector_rmse"] = float(
        np.sqrt(
            np.mean(
                (
                    y_pred
                    - y_true
                )
                ** 2
            )
        )
    )

    return result


def choose_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    alphas: Sequence[float],
    probe_name: str,
) -> Tuple[
    float,
    List[Dict[str, float]],
]:
    rows: List[
        Dict[str, float]
    ] = []

    best_alpha = None
    best_rmse = float(
        "inf"
    )

    for alpha in alphas:
        model, coefficients = ridge_fit(
            x_train,
            y_train,
            alpha,
        )

        prediction = ridge_predict(
            x_validation,
            model,
            coefficients,
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        prediction
                        - y_validation
                    )
                    ** 2
                )
            )
        )

        rows.append(
            {
                "probe": probe_name,
                "alpha": float(alpha),
                "validation_rmse": rmse,
            }
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(
                alpha
            )

    assert best_alpha is not None

    return (
        best_alpha,
        rows,
    )


def run_probe(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    model, coefficients = ridge_fit(
        x_train,
        y_train,
        alpha,
    )

    prediction = ridge_predict(
        x_test,
        model,
        coefficients,
    )

    result: Dict[
        str,
        float,
    ] = {
        "probe": name,
        "alpha": float(alpha),
        "train_n": int(
            x_train.shape[0]
        ),
        "test_n": int(
            x_test.shape[0]
        ),
    }

    result.update(
        axis_metrics(
            y_test,
            prediction,
        )
    )

    return result


def temporal_split(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    split = int(
        round(
            x.shape[0]
            * fraction
        )
    )

    return (
        x[:split],
        y[:split],
        x[split:],
        y[split:],
    )


def group_geometry(
    x: np.ndarray,
) -> Dict[str, float]:
    centroid = np.mean(
        x,
        axis=0,
    )

    centered = (
        x - centroid
    )

    total_variance = float(
        np.mean(
            np.sum(
                centered ** 2,
                axis=1,
            )
        )
    )

    return {
        "samples": int(
            x.shape[0]
        ),
        "centroid_norm": float(
            np.linalg.norm(
                centroid
            )
        ),
        "total_variance": total_variance,
        "rms_scale": float(
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        ),
    }


def covariance_basis(
    x: np.ndarray,
    components: int,
) -> np.ndarray:
    centered = (
        x
        - np.mean(
            x,
            axis=0,
        )
    )

    covariance = (
        centered.T
        @ centered
    ) / max(
        x.shape[0] - 1,
        1,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    return eigenvectors[
        :,
        order[
            :components
        ],
    ]


def principal_angles(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> np.ndarray:
    singular_values = np.linalg.svd(
        basis_a.T
        @ basis_b,
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


def write_csv(
    path: Path,
    rows: Sequence[Mapping],
) -> None:
    if not rows:
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

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
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir = (
        output_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_dirs = (
        parse_sequence_inputs(
            args.sequence_input
        )
    )

    (
        x_by_sequence,
        y_by_sequence,
        raw_dimension,
    ) = load_sequences(
        sequence_dirs=sequence_dirs,
        representation_key=(
            args.representation_key
        ),
        projection_dim=(
            args.projection_dim
        ),
        projection_seed=(
            args.projection_seed
        ),
    )

    train_sequences = [
        f"{index:02d}"
        for index in range(9)
    ]

    x_train_all = concatenate(
        x_by_sequence,
        train_sequences,
    )

    y_train_all = concatenate(
        y_by_sequence,
        train_sequences,
    )

    x_val = x_by_sequence["09"]
    y_val = y_by_sequence["09"]

    x_test = x_by_sequence["10"]
    y_test = y_by_sequence["10"]

    # ----------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------

    groups = {
        "train_00_08": x_train_all,
        "validation_09": x_val,
        "test_10": x_test,
    }

    geometry_rows = []

    for name, x in groups.items():
        row = {
            "group": name,
        }

        row.update(
            group_geometry(x)
        )

        geometry_rows.append(
            row
        )

    write_csv(
        output_dir
        / "group_statistics.csv",
        geometry_rows,
    )

    distance_rows = []

    group_names = list(
        groups.keys()
    )

    for i in range(
        len(group_names)
    ):
        for j in range(
            i + 1,
            len(group_names),
        ):
            name_a = group_names[i]
            name_b = group_names[j]

            centroid_a = np.mean(
                groups[name_a],
                axis=0,
            )

            centroid_b = np.mean(
                groups[name_b],
                axis=0,
            )

            distance_rows.append(
                {
                    "group_a": name_a,
                    "group_b": name_b,
                    "centroid_distance": float(
                        np.linalg.norm(
                            centroid_a
                            - centroid_b
                        )
                    ),
                }
            )

    write_csv(
        output_dir
        / "group_distances.csv",
        distance_rows,
    )

    component_count = min(
        args.pca_components,
        args.projection_dim,
    )

    bases = {
        name: covariance_basis(
            x,
            component_count,
        )
        for name, x
        in groups.items()
    }

    angle_rows = []

    for name_a, name_b in (
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
    ):
        angles = principal_angles(
            bases[name_a],
            bases[name_b],
        )

        angle_rows.append(
            {
                "group_a": name_a,
                "group_b": name_b,
                "mean_angle_deg": float(
                    np.mean(angles)
                ),
                "max_angle_deg": float(
                    np.max(angles)
                ),
            }
        )

    write_csv(
        output_dir
        / "principal_angles.csv",
        angle_rows,
    )

    # ----------------------------------------------------------
    # Probe A
    #
    # For each 00-08 sequence:
    # 0-60% alpha training
    # 60-80% alpha validation
    # 0-80% final probe training
    # 80-100% test
    # ----------------------------------------------------------

    a_train_x = []
    a_train_y = []

    a_alpha_train_x = []
    a_alpha_train_y = []

    a_alpha_val_x = []
    a_alpha_val_y = []

    a_test_x = []
    a_test_y = []

    for sequence in train_sequences:
        x = x_by_sequence[
            sequence
        ]
        y = y_by_sequence[
            sequence
        ]

        n = x.shape[0]

        n60 = int(
            0.60 * n
        )
        n80 = int(
            0.80 * n
        )

        a_alpha_train_x.append(
            x[:n60]
        )
        a_alpha_train_y.append(
            y[:n60]
        )

        a_alpha_val_x.append(
            x[n60:n80]
        )
        a_alpha_val_y.append(
            y[n60:n80]
        )

        a_train_x.append(
            x[:n80]
        )
        a_train_y.append(
            y[:n80]
        )

        a_test_x.append(
            x[n80:]
        )
        a_test_y.append(
            y[n80:]
        )

    a_alpha_train_x = np.concatenate(
        a_alpha_train_x
    )
    a_alpha_train_y = np.concatenate(
        a_alpha_train_y
    )

    a_alpha_val_x = np.concatenate(
        a_alpha_val_x
    )
    a_alpha_val_y = np.concatenate(
        a_alpha_val_y
    )

    selected_alpha, alpha_rows = (
        choose_alpha(
            a_alpha_train_x,
            a_alpha_train_y,
            a_alpha_val_x,
            a_alpha_val_y,
            args.alphas,
            "A_00_08_internal",
        )
    )

    probe_rows = []

    probe_rows.append(
        run_probe(
            "A_00_08_internal",
            np.concatenate(
                a_train_x
            ),
            np.concatenate(
                a_train_y
            ),
            np.concatenate(
                a_test_x
            ),
            np.concatenate(
                a_test_y
            ),
            selected_alpha,
        )
    )

    # B: train all 00-08 -> 09.
    probe_rows.append(
        run_probe(
            "B_train_to_validation_09",
            x_train_all,
            y_train_all,
            x_val,
            y_val,
            selected_alpha,
        )
    )

    # C: train all 00-08 -> 10.
    probe_rows.append(
        run_probe(
            "C_train_to_test_10",
            x_train_all,
            y_train_all,
            x_test,
            y_test,
            selected_alpha,
        )
    )

    # ----------------------------------------------------------
    # Probe D: Sequence 10 local temporal transfer.
    # ----------------------------------------------------------

    n10 = x_test.shape[0]

    n60 = int(
        0.60 * n10
    )

    n80 = int(
        0.80 * n10
    )

    d_alpha, d_alpha_rows = (
        choose_alpha(
            x_test[:n60],
            y_test[:n60],
            x_test[n60:n80],
            y_test[n60:n80],
            args.alphas,
            "D_test10_local",
        )
    )

    alpha_rows.extend(
        d_alpha_rows
    )

    probe_rows.append(
        run_probe(
            "D_test10_local",
            x_test[:n80],
            y_test[:n80],
            x_test[n80:],
            y_test[n80:],
            d_alpha,
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
        alpha_rows,
    )

    # ----------------------------------------------------------
    # Common 2-D PCA visualization using projected representation.
    # ----------------------------------------------------------

    common = np.concatenate(
        [
            x_train_all,
            x_val,
            x_test,
        ],
        axis=0,
    )

    common_mean = np.mean(
        common,
        axis=0,
    )

    centered = (
        common
        - common_mean
    )

    covariance = (
        centered.T
        @ centered
    ) / max(
        centered.shape[0] - 1,
        1,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    basis_2d = eigenvectors[
        :,
        order[:2]
    ]

    figure = plt.figure(
        figsize=(9, 7)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    for name, x in groups.items():
        projected = (
            (
                x
                - common_mean
            )
            @ basis_2d
        )

        # Keep the plot readable.
        stride = max(
            1,
            projected.shape[0]
            // 2000,
        )

        axis.scatter(
            projected[
                ::stride,
                0,
            ],
            projected[
                ::stride,
                1,
            ],
            s=8,
            alpha=0.35,
            label=name,
        )

    axis.set_xlabel(
        "PC1"
    )

    axis.set_ylabel(
        "PC2"
    )

    axis.set_title(
        "Rotation representation: common PCA"
    )

    axis.grid(
        True
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        plots_dir
        / "common_pca.png",
        dpi=180,
    )

    plt.close(
        figure
    )

    summary = {
        "raw_representation_dimension": (
            raw_dimension
        ),
        "analysis_projection_dimension": (
            args.projection_dim
        ),
        "projection_seed": (
            args.projection_seed
        ),
        "projection_note": (
            "The DeepDCT-VO representation itself is unchanged. "
            "A fixed Gaussian random projection is applied only "
            "for scalable offline geometry and ridge analysis."
        ),
        "groups": geometry_rows,
        "principal_angles": angle_rows,
        "probe_results": probe_rows,
        "selected_train_domain_alpha": (
            selected_alpha
        ),
        "selected_sequence10_local_alpha": (
            d_alpha
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 104)
    print("Rotation representation probes")
    print("=" * 104)

    print(
        f"{'Probe':<30}"
        f"{'alpha':>10}"
        f"{'RMSE':>12}"
        f"{'x corr':>12}"
        f"{'y corr':>12}"
        f"{'z corr':>12}"
        f"{'z std':>12}"
    )

    print("-" * 104)

    for row in probe_rows:
        print(
            f"{row['probe']:<30}"
            f"{row['alpha']:>10.3g}"
            f"{row['vector_rmse']:>12.6f}"
            f"{row['x_corr']:>12.4f}"
            f"{row['y_corr']:>12.4f}"
            f"{row['z_corr']:>12.4f}"
            f"{row['z_std_ratio']:>12.4f}"
        )

    print("=" * 104)

    print()
    print(
        f"Outputs saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()