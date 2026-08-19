#!/usr/bin/env python3
"""
DeepDCT-VO Rotation Geometry Generalization Audit
=================================================

Post-training diagnostic audit for the continuous SO(3)-supervised
rotation-geometry experiment.

The script answers four questions:

1. Latent Distance Correlation
   Does ||z_hat_i - z_hat_j||_2 correlate with the SO(3) geodesic
   distance d_SO3(R_i, R_j) on held-out Sequence 10?

2. Axis Attribution
   Is remaining rotation error concentrated in camera-frame yaw (Y axis)
   during strong turns, or is there persistent low-magnitude X/Z bias?

3. Probe Gap Closing
   Probe C:
       train a linear ridge rotation probe on sequences 00-08,
       evaluate on Sequence 10.

   Probe D:
       train the same probe on the first chronological fraction of
       Sequence 10 and evaluate on the remaining fraction.

   Report:
       gap = Probe-C vector RMSE / Probe-D vector RMSE

   Reference baseline:
       1.82x

4. Sequence Disentanglement
   How accurately can a linear classifier decode KITTI sequence identity
   from the rotation bottleneck?

   Reference baseline:
       38.76 %

Important coordinate convention
-------------------------------
For KITTI camera coordinates, the principal vehicle-yaw rotation is about
the camera Y axis. Therefore this audit defaults to:

    X -> pitch-like component
    Y -> yaw / heading component
    Z -> roll-like component

The script DOES NOT use the Z Euler component as heading.

Rotation convention
-------------------
Euler labels are assumed to use the project's extrinsic xyz convention:

    R = Rz(z) @ Ry(y) @ Rx(x)

and angles are assumed to be radians unless --angles-in-degrees is used.

The script reuses scripts/evaluate_deepdct_vo.py for:

- checkpoint loading;
- checkpoint configuration recovery;
- dataset construction;
- model reconstruction;
- forward evaluation;
- rotation bottleneck extraction.

Python compatibility: Python 3.8+
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


AXES = ("x", "y", "z")

AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}

DEFAULT_AXIS_NAMES = {
    "x": "pitch_like",
    "y": "yaw_heading",
    "z": "roll_like",
}

EPS = 1.0e-12


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit cross-sequence generalization of the DeepDCT-VO "
            "continuous SO(3)-supervised rotation representation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to audit.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--train-sequences",
        nargs="+",
        default=[
            "00", "01", "02", "03", "04",
            "05", "06", "07", "08",
        ],
        help="Sequences used for Probe C training.",
    )

    parser.add_argument(
        "--test-sequence",
        default="10",
        help="Held-out sequence used for the main diagnostics.",
    )

    parser.add_argument(
        "--identity-sequences",
        nargs="+",
        default=[
            "00", "01", "02", "03", "04",
            "05", "06", "07", "08", "09", "10",
        ],
        help=(
            "Sequences used by the sequence-identity decoder. "
            "Use the same sequence set as the baseline when comparing "
            "against 38.76%%."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=None,
        help=(
            "Optional deterministic cap per sequence. "
            "Samples are selected uniformly over the sequence rather "
            "than taking only the beginning."
        ),
    )

    parser.add_argument(
        "--pair-count",
        type=int,
        default=50000,
        help=(
            "Maximum number of random Sequence-10 pairs used for "
            "latent/SO(3) distance correlation."
        ),
    )

    parser.add_argument(
        "--pair-plot-count",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0e-3,
        help="Ridge regularization for rotation probes.",
    )

    parser.add_argument(
        "--identity-ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization for sequence-ID classifier.",
    )

    parser.add_argument(
        "--local-probe-train-fraction",
        type=float,
        default=0.70,
        help=(
            "Chronological fraction of Sequence 10 used to train "
            "Probe D. The remaining suffix is its test set."
        ),
    )

    parser.add_argument(
        "--identity-train-fraction",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--yaw-axis",
        choices=AXES,
        default="y",
        help=(
            "Euler component used for heading/turn-regime attribution. "
            "KITTI camera-frame yaw should normally remain Y."
        ),
    )

    parser.add_argument(
        "--turn-quantiles",
        type=float,
        nargs=2,
        metavar=("LOW_Q", "HIGH_Q"),
        default=(1.0 / 3.0, 2.0 / 3.0),
        help=(
            "Absolute GT yaw quantiles separating low, moderate, "
            "and strong-turn regimes."
        ),
    )

    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help="Interpret GT and predicted Euler rotations as degrees.",
    )

    parser.add_argument(
        "--baseline-probe-gap",
        type=float,
        default=1.82,
    )

    parser.add_argument(
        "--baseline-sequence-decodability",
        type=float,
        default=38.76,
        help="Baseline sequence-identity accuracy in percent.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "Checkpoint does not exist: {}".format(args.checkpoint)
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")

    if args.pair_count <= 0:
        raise ValueError("--pair-count must be positive.")

    if args.pair_plot_count <= 0:
        raise ValueError("--pair-plot-count must be positive.")

    if args.ridge_alpha < 0.0:
        raise ValueError("--ridge-alpha cannot be negative.")

    if args.identity_ridge_alpha < 0.0:
        raise ValueError(
            "--identity-ridge-alpha cannot be negative."
        )

    if not 0.0 < args.local_probe_train_fraction < 1.0:
        raise ValueError(
            "--local-probe-train-fraction must be in (0, 1)."
        )

    if not 0.0 < args.identity_train_fraction < 1.0:
        raise ValueError(
            "--identity-train-fraction must be in (0, 1)."
        )

    low_q, high_q = args.turn_quantiles

    if not 0.0 < low_q < high_q < 1.0:
        raise ValueError(
            "--turn-quantiles must satisfy "
            "0 < LOW_Q < HIGH_Q < 1."
        )

    if (
        args.max_samples_per_sequence is not None
        and args.max_samples_per_sequence <= 0
    ):
        raise ValueError(
            "--max-samples-per-sequence must be positive."
        )


# ============================================================================
# General utilities
# ============================================================================


def normalize_sequence(value: object) -> str:
    text = str(value).strip()

    if text.isdigit():
        return "{:02d}".format(int(text))

    return text


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def write_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent

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
        "Could not identify the deepdct-vo repository root."
    )


def load_evaluator(repo_root: Path) -> Any:
    path = (
        repo_root
        / "scripts"
        / "evaluate_deepdct_vo.py"
    )

    spec = importlib.util.spec_from_file_location(
        "deepdct_rotation_geometry_audit_evaluator",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to load evaluator from {}".format(path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "resolve_device",
        "load_checkpoint",
        "resolve_evaluation_configuration",
        "build_dataset",
        "build_dataloader",
        "build_model",
        "evaluate_model",
    )

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "evaluate_deepdct_vo.py is missing required helpers: "
            "{}".format(missing)
        )

    return module


def evenly_spaced_subset(
    dataset: Any,
    maximum: Optional[int],
) -> Any:
    if maximum is None or len(dataset) <= maximum:
        return dataset

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=maximum,
        dtype=np.int64,
    )

    indices = np.unique(indices)

    return Subset(
        dataset,
        indices.tolist(),
    )


def flatten_representation(
    representation: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        representation,
        dtype=np.float64,
    )

    if array.ndim < 2:
        raise ValueError(
            "Rotation representation must be at least 2-D; "
            "received shape {}.".format(array.shape)
        )

    array = array.reshape(
        array.shape[0],
        -1,
    )

    if not np.isfinite(array).all():
        raise FloatingPointError(
            "Rotation representation contains NaN or infinity."
        )

    return array


def l2_normalize_rows(
    values: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )

    return values / np.maximum(
        norms,
        EPS,
    )


# ============================================================================
# Evaluator reuse / representation extraction
# ============================================================================


def make_helper_args(
    args: argparse.Namespace,
    data_root: Path,
    sequence: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=data_root,
        sequence=sequence,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rotation_loss_weight=None,
        translation_loss_weight=None,
        translation_num_experts=3,
        use_ground_truth_rotation=False,
        log_interval=args.log_interval,
        euler_order="xyz",
        angles_in_degrees=args.angles_in_degrees,
    )


def evaluate_sequence(
    helper: Any,
    model: torch.nn.Module,
    evaluation_configuration: Mapping[str, object],
    device: torch.device,
    args: argparse.Namespace,
    data_root: Path,
    sequence: str,
) -> Dict[str, object]:

    sequence = normalize_sequence(sequence)

    helper_args = make_helper_args(
        args=args,
        data_root=data_root,
        sequence=sequence,
    )

    dataset = helper.build_dataset(
        args=helper_args,
        evaluation_configuration=evaluation_configuration,
    )

    dataset = evenly_spaced_subset(
        dataset,
        args.max_samples_per_sequence,
    )

    dataloader = helper.build_dataloader(
        dataset=dataset,
        args=helper_args,
        device=device,
    )

    print()
    print("-" * 80)
    print(
        "Extracting rotation representation: "
        "sequence={} samples={}".format(
            sequence,
            len(dataset),
        )
    )
    print("-" * 80)

    result = helper.evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        rotation_loss_weight=float(
            evaluation_configuration["rotation_loss_weight"]
        ),
        translation_loss_weight=float(
            evaluation_configuration["translation_loss_weight"]
        ),
        use_ground_truth_rotation=False,
        use_internal_depth=bool(
            evaluation_configuration["use_depth_cues"]
        ),
        log_interval=args.log_interval,
    )

    if len(result) != 8:
        raise RuntimeError(
            "This audit expects evaluate_model() to return 8 values:\n"
            "metrics, frame_predictions, rotation_gt, rotation_pred, "
            "translation_gt, translation_pred, rotation_representation, "
            "translation_representation.\n"
            "Received {} values.".format(len(result))
        )

    (
        metrics,
        frame_predictions,
        rotation_gt,
        rotation_pred,
        translation_gt,
        translation_pred,
        rotation_representation,
        translation_representation,
    ) = result

    del metrics
    del translation_gt
    del translation_pred
    del translation_representation

    representation = flatten_representation(
        rotation_representation
    )

    rotation_gt = np.asarray(
        rotation_gt,
        dtype=np.float64,
    )

    rotation_pred = np.asarray(
        rotation_pred,
        dtype=np.float64,
    )

    if len(representation) != len(rotation_gt):
        raise ValueError(
            "Representation/rotation sample-count mismatch: "
            "{} vs {}.".format(
                len(representation),
                len(rotation_gt),
            )
        )

    frame_prev = []
    frame_curr = []

    for index, row in enumerate(frame_predictions):
        frame_prev.append(
            int(
                getattr(
                    row,
                    "frame_prev",
                    index,
                )
            )
        )

        frame_curr.append(
            int(
                getattr(
                    row,
                    "frame_curr",
                    index + 1,
                )
            )
        )

    return {
        "sequence": sequence,
        "representation": representation,
        "rotation_gt": rotation_gt,
        "rotation_pred": rotation_pred,
        "frame_prev": np.asarray(
            frame_prev,
            dtype=np.int64,
        ),
        "frame_curr": np.asarray(
            frame_curr,
            dtype=np.int64,
        ),
    }


# ============================================================================
# SO(3)
# ============================================================================


def rotation_matrix_x(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(
    angle: float,
) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)

    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def euler_xyz_to_matrix(
    angles: Sequence[float],
    angles_in_degrees: bool,
) -> np.ndarray:
    x = float(angles[0])
    y = float(angles[1])
    z = float(angles[2])

    if angles_in_degrees:
        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

    return (
        rotation_matrix_z(z)
        @ rotation_matrix_y(y)
        @ rotation_matrix_x(x)
    )


def euler_array_to_matrices(
    rotations: np.ndarray,
    angles_in_degrees: bool,
) -> np.ndarray:
    return np.stack(
        [
            euler_xyz_to_matrix(
                row,
                angles_in_degrees,
            )
            for row in rotations
        ],
        axis=0,
    )


def so3_pair_distances(
    matrices: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    # trace(R_i^T R_j) equals the Frobenius inner product.
    traces = np.einsum(
        "nij,nij->n",
        matrices[first],
        matrices[second],
    )

    cosine = np.clip(
        (traces - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return np.arccos(cosine)


def so3_prediction_errors_deg(
    gt: np.ndarray,
    pred: np.ndarray,
    angles_in_degrees: bool,
) -> np.ndarray:
    gt_matrices = euler_array_to_matrices(
        gt,
        angles_in_degrees,
    )

    pred_matrices = euler_array_to_matrices(
        pred,
        angles_in_degrees,
    )

    traces = np.einsum(
        "nij,nij->n",
        gt_matrices,
        pred_matrices,
    )

    cosine = np.clip(
        (traces - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(cosine)
    )


# ============================================================================
# Correlation
# ============================================================================


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    if np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def rankdata(
    values: np.ndarray,
) -> np.ndarray:
    # pandas rank handles ties correctly and is already a project dependency.
    return (
        pd.Series(values)
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    return pearson_correlation(
        rankdata(x),
        rankdata(y),
    )


def sample_distinct_pairs(
    count: int,
    maximum_pairs: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:

    if count < 2:
        raise ValueError(
            "At least two samples are needed for pair analysis."
        )

    max_unique_pairs = (
        count * (count - 1) // 2
    )

    requested = min(
        maximum_pairs,
        max_unique_pairs,
    )

    if max_unique_pairs <= maximum_pairs:
        first, second = np.triu_indices(
            count,
            k=1,
        )

        return first, second

    pairs = set()

    while len(pairs) < requested:
        a = int(rng.randint(0, count))
        b = int(rng.randint(0, count))

        if a == b:
            continue

        if a > b:
            a, b = b, a

        pairs.add((a, b))

    array = np.asarray(
        list(pairs),
        dtype=np.int64,
    )

    return array[:, 0], array[:, 1]


def latent_distance_correlation(
    data: Mapping[str, object],
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, object], pd.DataFrame]:

    representation = np.asarray(
        data["representation"],
        dtype=np.float64,
    )

    rotation_gt = np.asarray(
        data["rotation_gt"],
        dtype=np.float64,
    )

    z_hat = l2_normalize_rows(
        representation
    )

    first, second = sample_distinct_pairs(
        count=len(z_hat),
        maximum_pairs=args.pair_count,
        rng=rng,
    )

    latent_distance = np.linalg.norm(
        z_hat[first] - z_hat[second],
        axis=1,
    )

    matrices = euler_array_to_matrices(
        rotation_gt,
        args.angles_in_degrees,
    )

    so3_distance = so3_pair_distances(
        matrices,
        first,
        second,
    )

    pearson = pearson_correlation(
        latent_distance,
        so3_distance,
    )

    spearman = spearman_correlation(
        latent_distance,
        so3_distance,
    )

    table = pd.DataFrame(
        {
            "index_i": first,
            "index_j": second,
            "latent_distance": latent_distance,
            "so3_distance_rad": so3_distance,
            "so3_distance_deg": np.degrees(
                so3_distance
            ),
        }
    )

    return {
        "pair_count": int(len(table)),
        "representation_normalized": True,
        "latent_distance": "L2",
        "rotation_distance": "SO(3)_geodesic",
        "pearson_r": safe_float(pearson),
        "spearman_rho": safe_float(spearman),
        "positive_pearson": (
            bool(pearson > 0.0)
            if math.isfinite(pearson)
            else None
        ),
    }, table


# ============================================================================
# Axis attribution
# ============================================================================


def rotation_error_in_degrees(
    rotation_gt: np.ndarray,
    rotation_pred: np.ndarray,
    angles_in_degrees: bool,
) -> np.ndarray:
    error = (
        np.asarray(rotation_pred, dtype=np.float64)
        - np.asarray(rotation_gt, dtype=np.float64)
    )

    if angles_in_degrees:
        return error

    return np.degrees(error)


def descriptive_error_stats(
    error_deg: np.ndarray,
) -> Dict[str, float]:
    return {
        "bias_deg": float(
            np.mean(error_deg)
        ),
        "mae_deg": float(
            np.mean(np.abs(error_deg))
        ),
        "rmse_deg": float(
            np.sqrt(
                np.mean(
                    np.square(error_deg)
                )
            )
        ),
        "std_deg": float(
            np.std(error_deg)
        ),
    }


def axis_attribution(
    data: Mapping[str, object],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame]:

    gt = np.asarray(
        data["rotation_gt"],
        dtype=np.float64,
    )

    pred = np.asarray(
        data["rotation_pred"],
        dtype=np.float64,
    )

    errors_deg = rotation_error_in_degrees(
        gt,
        pred,
        args.angles_in_degrees,
    )

    if args.angles_in_degrees:
        gt_deg = gt.copy()
    else:
        gt_deg = np.degrees(gt)

    yaw_index = AXIS_INDEX[
        args.yaw_axis
    ]

    abs_yaw = np.abs(
        gt_deg[:, yaw_index]
    )

    low_q, high_q = args.turn_quantiles

    low_threshold = float(
        np.quantile(abs_yaw, low_q)
    )

    high_threshold = float(
        np.quantile(abs_yaw, high_q)
    )

    p90_threshold = float(
        np.quantile(abs_yaw, 0.90)
    )

    regimes = {
        "low_turn": (
            abs_yaw <= low_threshold
        ),
        "moderate_turn": (
            (abs_yaw > low_threshold)
            & (abs_yaw <= high_threshold)
        ),
        "strong_turn": (
            abs_yaw > high_threshold
        ),
        "sharpest_10pct": (
            abs_yaw >= p90_threshold
        ),
        "all": np.ones(
            len(gt),
            dtype=bool,
        ),
    }

    rows: List[Dict[str, object]] = []

    for regime_name, mask in regimes.items():
        for axis in AXES:
            axis_index = AXIS_INDEX[axis]

            values = errors_deg[
                mask,
                axis_index,
            ]

            stats = descriptive_error_stats(
                values
            )

            rows.append(
                {
                    "regime": regime_name,
                    "axis": axis,
                    "semantic_axis": (
                        DEFAULT_AXIS_NAMES[axis]
                    ),
                    "samples": int(
                        np.sum(mask)
                    ),
                    **stats,
                }
            )

    table = pd.DataFrame(rows)

    all_stats = table[
        table["regime"] == "all"
    ].set_index("axis")

    strong_stats = table[
        table["regime"] == "strong_turn"
    ].set_index("axis")

    yaw_axis = args.yaw_axis

    other_axes = [
        axis
        for axis in AXES
        if axis != yaw_axis
    ]

    yaw_strong_rmse = float(
        strong_stats.loc[
            yaw_axis,
            "rmse_deg",
        ]
    )

    other_strong_rmse = [
        float(
            strong_stats.loc[
                axis,
                "rmse_deg",
            ]
        )
        for axis in other_axes
    ]

    yaw_dominance_ratio = (
        yaw_strong_rmse
        / max(
            max(other_strong_rmse),
            EPS,
        )
    )

    persistent_other_bias = max(
        abs(
            float(
                all_stats.loc[
                    axis,
                    "bias_deg",
                ]
            )
        )
        for axis in other_axes
    )

    geodesic_error = (
        so3_prediction_errors_deg(
            gt=gt,
            pred=pred,
            angles_in_degrees=(
                args.angles_in_degrees
            ),
        )
    )

    summary = {
        "yaw_axis": yaw_axis,
        "yaw_axis_semantics": (
            DEFAULT_AXIS_NAMES[yaw_axis]
        ),
        "turn_thresholds_deg": {
            "low_to_moderate": low_threshold,
            "moderate_to_strong": high_threshold,
            "sharpest_10pct": p90_threshold,
        },
        "overall_so3_error_deg": {
            "mean": float(
                np.mean(geodesic_error)
            ),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            geodesic_error
                        )
                    )
                )
            ),
            "median": float(
                np.median(geodesic_error)
            ),
            "p90": float(
                np.quantile(
                    geodesic_error,
                    0.90,
                )
            ),
        },
        "strong_turn_yaw_rmse_deg": (
            yaw_strong_rmse
        ),
        "strong_turn_other_axis_rmse_deg": {
            axis: float(
                strong_stats.loc[
                    axis,
                    "rmse_deg",
                ]
            )
            for axis in other_axes
        },
        "strong_turn_yaw_dominance_ratio": (
            float(yaw_dominance_ratio)
        ),
        "largest_non_yaw_overall_bias_deg": (
            float(persistent_other_bias)
        ),
    }

    return summary, table


# ============================================================================
# Ridge probes
# ============================================================================


def fit_standardizer(
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(
        x,
        axis=0,
    )

    std = np.std(
        x,
        axis=0,
    )

    std = np.where(
        std < 1.0e-8,
        1.0,
        std,
    )

    return mean, std


def apply_standardizer(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (
        x - mean
    ) / std


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x_augmented = np.concatenate(
        [
            x,
            np.ones(
                (len(x), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    dimension = x_augmented.shape[1]

    regularizer = (
        alpha
        * np.eye(
            dimension,
            dtype=np.float64,
        )
    )

    # Do not regularize the intercept.
    regularizer[-1, -1] = 0.0

    lhs = (
        x_augmented.T
        @ x_augmented
        + regularizer
    )

    rhs = (
        x_augmented.T
        @ y
    )

    try:
        return np.linalg.solve(
            lhs,
            rhs,
        )
    except np.linalg.LinAlgError:
        return (
            np.linalg.pinv(lhs)
            @ rhs
        )


def ridge_predict(
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x_augmented = np.concatenate(
        [
            x,
            np.ones(
                (len(x), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    return (
        x_augmented
        @ weights
    )


def probe_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    angles_in_degrees: bool,
) -> Dict[str, object]:

    if angles_in_degrees:
        target_deg = target
        prediction_deg = prediction
    else:
        target_deg = np.degrees(
            target
        )
        prediction_deg = np.degrees(
            prediction
        )

    error_deg = (
        prediction_deg
        - target_deg
    )

    vector_error_deg = np.linalg.norm(
        error_deg,
        axis=1,
    )

    axis_rmse = np.sqrt(
        np.mean(
            np.square(error_deg),
            axis=0,
        )
    )

    return {
        "samples": int(len(target)),
        "vector_rmse_deg": float(
            np.sqrt(
                np.mean(
                    np.square(
                        vector_error_deg
                    )
                )
            )
        ),
        "vector_mae_deg": float(
            np.mean(
                vector_error_deg
            )
        ),
        "axis_rmse_deg": {
            axis: float(
                axis_rmse[
                    AXIS_INDEX[axis]
                ]
            )
            for axis in AXES
        },
    }


def run_rotation_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    angles_in_degrees: bool,
) -> Tuple[Dict[str, object], np.ndarray]:

    mean, std = fit_standardizer(
        x_train
    )

    x_train_std = apply_standardizer(
        x_train,
        mean,
        std,
    )

    x_test_std = apply_standardizer(
        x_test,
        mean,
        std,
    )

    weights = ridge_fit(
        x_train_std,
        y_train,
        alpha,
    )

    prediction = ridge_predict(
        x_test_std,
        weights,
    )

    metrics = probe_metrics(
        target=y_test,
        prediction=prediction,
        angles_in_degrees=(
            angles_in_degrees
        ),
    )

    return metrics, prediction


def probe_gap_analysis(
    sequence_data: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame]:

    train_rep = []
    train_gt = []

    for sequence in args.train_sequences:
        sequence = normalize_sequence(
            sequence
        )

        data = sequence_data[
            sequence
        ]

        train_rep.append(
            np.asarray(
                data["representation"],
                dtype=np.float64,
            )
        )

        train_gt.append(
            np.asarray(
                data["rotation_gt"],
                dtype=np.float64,
            )
        )

    probe_c_x_train = np.concatenate(
        train_rep,
        axis=0,
    )

    probe_c_y_train = np.concatenate(
        train_gt,
        axis=0,
    )

    test_sequence = normalize_sequence(
        args.test_sequence
    )

    test_data = sequence_data[
        test_sequence
    ]

    test_rep = np.asarray(
        test_data["representation"],
        dtype=np.float64,
    )

    test_gt = np.asarray(
        test_data["rotation_gt"],
        dtype=np.float64,
    )

    # ----------------------------------------------------------
    # Probe C: cross-sequence
    # ----------------------------------------------------------

    probe_c_metrics, probe_c_prediction = (
        run_rotation_probe(
            x_train=probe_c_x_train,
            y_train=probe_c_y_train,
            x_test=test_rep,
            y_test=test_gt,
            alpha=args.ridge_alpha,
            angles_in_degrees=(
                args.angles_in_degrees
            ),
        )
    )

    # ----------------------------------------------------------
    # Probe D: local temporal
    #
    # First chronological portion of sequence 10 -> later suffix.
    # No random train/test leakage across time.
    # ----------------------------------------------------------

    split = int(
        round(
            len(test_rep)
            * args.local_probe_train_fraction
        )
    )

    split = max(
        1,
        min(
            split,
            len(test_rep) - 1,
        ),
    )

    probe_d_metrics, probe_d_prediction = (
        run_rotation_probe(
            x_train=test_rep[:split],
            y_train=test_gt[:split],
            x_test=test_rep[split:],
            y_test=test_gt[split:],
            alpha=args.ridge_alpha,
            angles_in_degrees=(
                args.angles_in_degrees
            ),
        )
    )

    probe_c_rmse = float(
        probe_c_metrics[
            "vector_rmse_deg"
        ]
    )

    probe_d_rmse = float(
        probe_d_metrics[
            "vector_rmse_deg"
        ]
    )

    gap = (
        probe_c_rmse
        / max(
            probe_d_rmse,
            EPS,
        )
    )

    baseline_gap = float(
        args.baseline_probe_gap
    )

    gap_change = (
        gap
        - baseline_gap
    )

    gap_reduction_percent = (
        (
            baseline_gap - gap
        )
        / baseline_gap
        * 100.0
        if baseline_gap > 0.0
        else float("nan")
    )

    summary = {
        "probe_c": {
            "definition": (
                "ridge trained on train sequences and "
                "evaluated on held-out test sequence"
            ),
            "train_sequences": [
                normalize_sequence(x)
                for x in args.train_sequences
            ],
            "test_sequence": test_sequence,
            "train_samples": int(
                len(probe_c_x_train)
            ),
            **probe_c_metrics,
        },
        "probe_d": {
            "definition": (
                "ridge trained on chronological prefix of "
                "test sequence and evaluated on later suffix"
            ),
            "test_sequence": test_sequence,
            "train_fraction": float(
                args.local_probe_train_fraction
            ),
            "train_samples": int(split),
            **probe_d_metrics,
        },
        "probe_c_over_probe_d_gap": float(
            gap
        ),
        "baseline_gap": baseline_gap,
        "gap_change": float(
            gap_change
        ),
        "gap_reduction_percent": safe_float(
            gap_reduction_percent
        ),
        "gap_narrowed": bool(
            gap < baseline_gap
        ),
    }

    rows = []

    for name, metrics in (
        ("Probe_C_cross_sequence", probe_c_metrics),
        ("Probe_D_local_temporal", probe_d_metrics),
    ):
        row = {
            "probe": name,
            "vector_rmse_deg": metrics[
                "vector_rmse_deg"
            ],
            "vector_mae_deg": metrics[
                "vector_mae_deg"
            ],
        }

        for axis in AXES:
            row[
                "{}_rmse_deg".format(axis)
            ] = metrics[
                "axis_rmse_deg"
            ][axis]

        rows.append(row)

    table = pd.DataFrame(rows)

    # Preserve predictions for optional debugging.
    np.savez_compressed(
        args.output_dir
        / "probe_predictions.npz",
        probe_c_prediction=(
            probe_c_prediction
        ),
        probe_c_target=test_gt,
        probe_d_prediction=(
            probe_d_prediction
        ),
        probe_d_target=test_gt[split:],
    )

    return summary, table


# ============================================================================
# Sequence identity decoding
# ============================================================================


def one_hot(
    labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    output = np.zeros(
        (
            len(labels),
            class_count,
        ),
        dtype=np.float64,
    )

    output[
        np.arange(len(labels)),
        labels,
    ] = 1.0

    return output


def sequence_identity_analysis(
    sequence_data: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> Tuple[
    Dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:

    sequences = [
        normalize_sequence(sequence)
        for sequence
        in args.identity_sequences
    ]

    train_x = []
    train_y = []

    test_x = []
    test_y = []

    for class_index, sequence in enumerate(
        sequences
    ):
        representation = np.asarray(
            sequence_data[
                sequence
            ]["representation"],
            dtype=np.float64,
        )

        indices = np.arange(
            len(representation)
        )

        rng.shuffle(indices)

        split = int(
            round(
                len(indices)
                * args.identity_train_fraction
            )
        )

        split = max(
            1,
            min(
                split,
                len(indices) - 1,
            ),
        )

        training = indices[:split]
        testing = indices[split:]

        train_x.append(
            representation[training]
        )

        test_x.append(
            representation[testing]
        )

        train_y.append(
            np.full(
                len(training),
                class_index,
                dtype=np.int64,
            )
        )

        test_y.append(
            np.full(
                len(testing),
                class_index,
                dtype=np.int64,
            )
        )

    x_train = np.concatenate(
        train_x,
        axis=0,
    )

    y_train = np.concatenate(
        train_y,
        axis=0,
    )

    x_test = np.concatenate(
        test_x,
        axis=0,
    )

    y_test = np.concatenate(
        test_y,
        axis=0,
    )

    mean, std = fit_standardizer(
        x_train
    )

    x_train_std = apply_standardizer(
        x_train,
        mean,
        std,
    )

    x_test_std = apply_standardizer(
        x_test,
        mean,
        std,
    )

    targets = one_hot(
        y_train,
        len(sequences),
    )

    weights = ridge_fit(
        x_train_std,
        targets,
        args.identity_ridge_alpha,
    )

    scores = ridge_predict(
        x_test_std,
        weights,
    )

    prediction = np.argmax(
        scores,
        axis=1,
    )

    correct = (
        prediction == y_test
    )

    accuracy = float(
        np.mean(correct)
    )

    accuracy_percent = (
        100.0 * accuracy
    )

    baseline = float(
        args.baseline_sequence_decodability
    )

    chance_percent = (
        100.0
        / len(sequences)
    )

    confusion = np.zeros(
        (
            len(sequences),
            len(sequences),
        ),
        dtype=np.int64,
    )

    for target, predicted in zip(
        y_test,
        prediction,
    ):
        confusion[
            int(target),
            int(predicted),
        ] += 1

    per_sequence_rows = []

    for class_index, sequence in enumerate(
        sequences
    ):
        mask = (
            y_test == class_index
        )

        per_sequence_rows.append(
            {
                "sequence": sequence,
                "test_samples": int(
                    np.sum(mask)
                ),
                "accuracy_percent": float(
                    100.0
                    * np.mean(
                        prediction[mask]
                        == y_test[mask]
                    )
                ),
            }
        )

    confusion_frame = pd.DataFrame(
        confusion,
        index=[
            "gt_{}".format(x)
            for x in sequences
        ],
        columns=[
            "pred_{}".format(x)
            for x in sequences
        ],
    )

    summary = {
        "classifier": (
            "standardized one-vs-all ridge classifier"
        ),
        "sequences": sequences,
        "train_samples": int(
            len(x_train)
        ),
        "test_samples": int(
            len(x_test)
        ),
        "accuracy_percent": (
            accuracy_percent
        ),
        "chance_percent": (
            chance_percent
        ),
        "baseline_accuracy_percent": (
            baseline
        ),
        "change_from_baseline_percentage_points": (
            accuracy_percent
            - baseline
        ),
        "decodability_dropped": bool(
            accuracy_percent
            < baseline
        ),
    }

    return (
        summary,
        pd.DataFrame(
            per_sequence_rows
        ),
        confusion_frame,
    )


# ============================================================================
# Plots
# ============================================================================


def plot_latent_correlation(
    pairs: pd.DataFrame,
    path: Path,
    count: int,
    rng: np.random.RandomState,
    dpi: int,
) -> None:

    if len(pairs) > count:
        indices = rng.choice(
            len(pairs),
            size=count,
            replace=False,
        )

        data = pairs.iloc[
            indices
        ]
    else:
        data = pairs

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    ax.scatter(
        data["so3_distance_deg"],
        data["latent_distance"],
        s=7,
        alpha=0.35,
    )

    ax.set_xlabel(
        "SO(3) geodesic distance (deg)"
    )

    ax.set_ylabel(
        "Normalized latent L2 distance"
    )

    ax.set_title(
        "Sequence 10: latent vs SO(3) distance"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
    )

    plt.close(fig)


def plot_axis_regimes(
    table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:

    regimes = [
        "low_turn",
        "moderate_turn",
        "strong_turn",
        "sharpest_10pct",
    ]

    x = np.arange(
        len(regimes)
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    for offset_index, axis in enumerate(
        AXES
    ):
        values = []

        for regime in regimes:
            row = table[
                (table["regime"] == regime)
                & (table["axis"] == axis)
            ]

            values.append(
                float(
                    row.iloc[0][
                        "rmse_deg"
                    ]
                )
            )

        offset = (
            offset_index - 1
        ) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=axis.upper(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Low",
            "Moderate",
            "Strong",
            "Sharpest 10%",
        ]
    )

    ax.set_ylabel(
        "Euler-component RMSE (deg)"
    )

    ax.set_title(
        "Rotation error by turn regime"
    )

    ax.legend()
    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
    )

    plt.close(fig)


def plot_probe_comparison(
    table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.bar(
        table["probe"],
        table["vector_rmse_deg"],
    )

    ax.set_ylabel(
        "Rotation vector RMSE (deg)"
    )

    ax.set_title(
        "Cross-sequence vs local temporal probe"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
    )

    plt.close(fig)


def plot_confusion_matrix(
    confusion: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:

    matrix = confusion.to_numpy(
        dtype=np.float64,
    )

    row_sum = np.sum(
        matrix,
        axis=1,
        keepdims=True,
    )

    normalized = (
        matrix
        / np.maximum(
            row_sum,
            1.0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 7.0)
    )

    image = ax.imshow(
        normalized,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Row-normalized fraction",
    )

    ax.set_xticks(
        np.arange(
            len(confusion.columns)
        )
    )

    ax.set_yticks(
        np.arange(
            len(confusion.index)
        )
    )

    ax.set_xticklabels(
        [
            x.replace(
                "pred_",
                "",
            )
            for x
            in confusion.columns
        ]
    )

    ax.set_yticklabels(
        [
            x.replace(
                "gt_",
                "",
            )
            for x
            in confusion.index
        ]
    )

    ax.set_xlabel(
        "Predicted sequence"
    )

    ax.set_ylabel(
        "Ground-truth sequence"
    )

    ax.set_title(
        "Sequence identity from rotation latent"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
    )

    plt.close(fig)


# ============================================================================
# Console summary
# ============================================================================


def print_diagnostic_summary(
    latent: Mapping[str, object],
    axis: Mapping[str, object],
    probes: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:

    print()
    print("=" * 88)
    print(
        "DeepDCT-VO rotation geometry "
        "generalization audit"
    )
    print("=" * 88)

    print()
    print(
        "1. LATENT DISTANCE CORRELATION"
    )
    print("-" * 88)

    print(
        "Sequence-10 Pearson r:       "
        "{}".format(
            latent["pearson_r"]
        )
    )

    print(
        "Sequence-10 Spearman rho:     "
        "{}".format(
            latent["spearman_rho"]
        )
    )

    print(
        "Positive Pearson correlation: "
        "{}".format(
            latent["positive_pearson"]
        )
    )

    print()
    print(
        "2. AXIS ATTRIBUTION"
    )
    print("-" * 88)

    so3 = axis[
        "overall_so3_error_deg"
    ]

    print(
        "SO(3) per-frame RMSE:         "
        "{:.6f} deg".format(
            so3["rmse"]
        )
    )

    print(
        "SO(3) per-frame mean:         "
        "{:.6f} deg".format(
            so3["mean"]
        )
    )

    print(
        "Yaw axis:                     "
        "{} ({})".format(
            axis["yaw_axis"].upper(),
            axis["yaw_axis_semantics"],
        )
    )

    print(
        "Strong-turn yaw RMSE:         "
        "{:.6f} deg".format(
            axis[
                "strong_turn_yaw_rmse_deg"
            ]
        )
    )

    print(
        "Strong-turn yaw dominance:    "
        "{:.3f}x".format(
            axis[
                "strong_turn_yaw_dominance_ratio"
            ]
        )
    )

    print(
        "Largest non-yaw bias:         "
        "{:.6f} deg".format(
            axis[
                "largest_non_yaw_overall_bias_deg"
            ]
        )
    )

    print()
    print(
        "3. PROBE GAP"
    )
    print("-" * 88)

    probe_c = probes[
        "probe_c"
    ][
        "vector_rmse_deg"
    ]

    probe_d = probes[
        "probe_d"
    ][
        "vector_rmse_deg"
    ]

    print(
        "Probe C cross-sequence RMSE:  "
        "{:.6f} deg".format(
            probe_c
        )
    )

    print(
        "Probe D local-temporal RMSE:  "
        "{:.6f} deg".format(
            probe_d
        )
    )

    print(
        "Probe C / Probe D:            "
        "{:.3f}x".format(
            probes[
                "probe_c_over_probe_d_gap"
            ]
        )
    )

    print(
        "Baseline gap:                 "
        "{:.3f}x".format(
            probes[
                "baseline_gap"
            ]
        )
    )

    print(
        "Gap narrowed:                 "
        "{}".format(
            probes[
                "gap_narrowed"
            ]
        )
    )

    print()
    print(
        "4. SEQUENCE DISENTANGLEMENT"
    )
    print("-" * 88)

    print(
        "Sequence-ID accuracy:         "
        "{:.2f}%".format(
            identity[
                "accuracy_percent"
            ]
        )
    )

    print(
        "Baseline decodability:        "
        "{:.2f}%".format(
            identity[
                "baseline_accuracy_percent"
            ]
        )
    )

    print(
        "Chance accuracy:              "
        "{:.2f}%".format(
            identity[
                "chance_percent"
            ]
        )
    )

    print(
        "Decodability dropped:         "
        "{}".format(
            identity[
                "decodability_dropped"
            ]
        )
    )

    print()
    print("=" * 88)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    repo_root = find_repo_root()

    if str(repo_root) not in sys.path:
        sys.path.insert(
            0,
            str(repo_root),
        )

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else repo_root
        / args.checkpoint
    ).resolve()

    data_root = (
        args.data_root
        if args.data_root.is_absolute()
        else repo_root
        / args.data_root
    ).resolve()

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root
        / args.output_dir
    ).resolve()

    args.output_dir = output_dir

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

    helper = load_evaluator(
        repo_root
    )

    device = helper.resolve_device(
        args.device
    )

    checkpoint = helper.load_checkpoint(
        checkpoint_path,
        device,
    )

    configuration_args = make_helper_args(
        args=args,
        data_root=data_root,
        sequence=normalize_sequence(
            args.test_sequence
        ),
    )

    evaluation_configuration = (
        helper.resolve_evaluation_configuration(
            args=configuration_args,
            checkpoint=checkpoint,
        )
    )

    model = helper.build_model(
        checkpoint=checkpoint,
        evaluation_configuration=(
            evaluation_configuration
        ),
        device=device,
    )

    print("=" * 88)
    print(
        "DeepDCT-VO rotation geometry "
        "generalization audit"
    )
    print("=" * 88)

    print(
        "Checkpoint:       {}".format(
            checkpoint_path
        )
    )

    print(
        "Checkpoint epoch: {}".format(
            checkpoint.get(
                "epoch"
            )
        )
    )

    print(
        "Device:           {}".format(
            device
        )
    )

    print(
        "Semantic cues:    {}".format(
            evaluation_configuration.get(
                "use_semantic_cues"
            )
        )
    )

    print(
        "Depth cues:       {}".format(
            evaluation_configuration.get(
                "use_depth_cues"
            )
        )
    )

    print(
        "Rotation rep dim: {}".format(
            evaluation_configuration.get(
                "rotation_representation_dimension",
                getattr(
                    getattr(
                        model,
                        "rotation_head",
                        object(),
                    ),
                    "representation_dim",
                    "unknown",
                ),
            )
        )
    )

    print(
        "Yaw axis:         {} "
        "(KITTI camera-frame heading)".format(
            args.yaw_axis.upper()
        )
    )

    print("=" * 88)

    all_required_sequences = sorted(
        set(
            [
                normalize_sequence(x)
                for x in (
                    list(
                        args.train_sequences
                    )
                    + list(
                        args.identity_sequences
                    )
                    + [
                        args.test_sequence
                    ]
                )
            ]
        )
    )

    sequence_data: Dict[
        str,
        Dict[str, object],
    ] = {}

    for sequence in all_required_sequences:
        sequence_data[
            sequence
        ] = evaluate_sequence(
            helper=helper,
            model=model,
            evaluation_configuration=(
                evaluation_configuration
            ),
            device=device,
            args=args,
            data_root=data_root,
            sequence=sequence,
        )

        data = sequence_data[
            sequence
        ]

        np.savez_compressed(
            output_dir
            / "sequence_{}_rotation_representation.npz".format(
                sequence
            ),
            rotation_rep=np.asarray(
                data[
                    "representation"
                ]
            ),
            rotation_gt=np.asarray(
                data[
                    "rotation_gt"
                ]
            ),
            rotation_pred=np.asarray(
                data[
                    "rotation_pred"
                ]
            ),
            frame_prev=np.asarray(
                data[
                    "frame_prev"
                ]
            ),
            frame_curr=np.asarray(
                data[
                    "frame_curr"
                ]
            ),
        )

    rng = np.random.RandomState(
        args.seed
    )

    # ----------------------------------------------------------
    # Diagnostic 1
    # ----------------------------------------------------------

    test_sequence = normalize_sequence(
        args.test_sequence
    )

    (
        latent_summary,
        latent_pairs,
    ) = latent_distance_correlation(
        data=sequence_data[
            test_sequence
        ],
        args=args,
        rng=rng,
    )

    latent_pairs.to_csv(
        output_dir
        / "latent_distance_correlation.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Diagnostic 2
    # ----------------------------------------------------------

    (
        axis_summary,
        axis_table,
    ) = axis_attribution(
        data=sequence_data[
            test_sequence
        ],
        args=args,
    )

    axis_table.to_csv(
        output_dir
        / "axis_regime_errors.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Diagnostic 3
    # ----------------------------------------------------------

    (
        probe_summary,
        probe_table,
    ) = probe_gap_analysis(
        sequence_data=sequence_data,
        args=args,
    )

    probe_table.to_csv(
        output_dir
        / "probe_results.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Diagnostic 4
    # ----------------------------------------------------------

    (
        identity_summary,
        identity_table,
        confusion,
    ) = sequence_identity_analysis(
        sequence_data=sequence_data,
        args=args,
        rng=rng,
    )

    identity_table.to_csv(
        output_dir
        / "sequence_identity_results.csv",
        index=False,
    )

    confusion.to_csv(
        output_dir
        / "sequence_identity_confusion.csv",
    )

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------

    plot_latent_correlation(
        pairs=latent_pairs,
        path=(
            plots_dir
            / "latent_vs_so3_distance.png"
        ),
        count=args.pair_plot_count,
        rng=rng,
        dpi=args.dpi,
    )

    plot_axis_regimes(
        table=axis_table,
        path=(
            plots_dir
            / "axis_error_by_turn_regime.png"
        ),
        dpi=args.dpi,
    )

    plot_probe_comparison(
        table=probe_table,
        path=(
            plots_dir
            / "probe_c_vs_probe_d.png"
        ),
        dpi=args.dpi,
    )

    plot_confusion_matrix(
        confusion=confusion,
        path=(
            plots_dir
            / "sequence_identity_confusion.png"
        ),
        dpi=args.dpi,
    )

    # ----------------------------------------------------------
    # Combined summary
    # ----------------------------------------------------------

    summary = {
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": int(
            checkpoint.get(
                "epoch",
                -1,
            )
        ),
        "test_sequence": (
            test_sequence
        ),
        "configuration": {
            "train_sequences": [
                normalize_sequence(x)
                for x in args.train_sequences
            ],
            "identity_sequences": [
                normalize_sequence(x)
                for x in args.identity_sequences
            ],
            "yaw_axis": args.yaw_axis,
            "euler_order": "xyz",
            "euler_convention": "extrinsic",
            "angles_in_degrees": bool(
                args.angles_in_degrees
            ),
            "ridge_alpha": float(
                args.ridge_alpha
            ),
            "identity_ridge_alpha": float(
                args.identity_ridge_alpha
            ),
            "pair_count_requested": int(
                args.pair_count
            ),
            "max_samples_per_sequence": (
                args.max_samples_per_sequence
            ),
        },
        "diagnostic_1_latent_distance_correlation": (
            latent_summary
        ),
        "diagnostic_2_axis_attribution": (
            axis_summary
        ),
        "diagnostic_3_probe_gap": (
            probe_summary
        ),
        "diagnostic_4_sequence_disentanglement": (
            identity_summary
        ),
    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    print_diagnostic_summary(
        latent=latent_summary,
        axis=axis_summary,
        probes=probe_summary,
        identity=identity_summary,
    )

    print(
        "Outputs saved to: {}".format(
            output_dir
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())