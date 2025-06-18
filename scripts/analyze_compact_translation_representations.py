#!/usr/bin/env python3
"""Analyze compact DeepDCT-VO translation representations across sequences.

The evaluator stores only ``translation_rep`` in translation_representations.npz.
Ground-truth and predicted translations are therefore loaded from the sibling
``frame_predictions.csv`` file unless an explicit CSV path is supplied.

The script compares sequence 09 and sequence 10 using:

* feature variance, dead/near-constant dimensions, and effective rank;
* PCA variance concentration and target correlations;
* prediction errors overall and by shared forward-motion regimes;
* representation mean/covariance shift, MMD, and motion-distribution JS shift;
* cross-sequence ridge probes for linear translation decodability;
* CSV, JSON, text, and PNG outputs.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = ("x", "y", "z")
REGIME_NAMES = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare compact translation representations from KITTI "
            "sequences 09 and 10."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence-09-npz",
        type=Path,
        required=True,
        help="Sequence-09 translation_representations.npz.",
    )
    parser.add_argument(
        "--sequence-10-npz",
        type=Path,
        required=True,
        help="Sequence-10 translation_representations.npz.",
    )
    parser.add_argument(
        "--sequence-09-predictions",
        type=Path,
        default=None,
        help=(
            "Sequence-09 frame_predictions.csv. When omitted, use the "
            "CSV beside --sequence-09-npz."
        ),
    )
    parser.add_argument(
        "--sequence-10-predictions",
        type=Path,
        default=None,
        help=(
            "Sequence-10 frame_predictions.csv. When omitted, use the "
            "CSV beside --sequence-10-npz."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for reports, tables, and plots.",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Sequence-09 forward-translation quantile for low motion.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=2.0 / 3.0,
        help="Sequence-09 forward-translation quantile for high motion.",
    )
    parser.add_argument(
        "--near-constant-std",
        type=float,
        default=1.0e-6,
        help="Feature standard-deviation threshold for near-constant dims.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0e-3,
        help="L2 regularization used by cross-sequence linear probes.",
    )
    parser.add_argument(
        "--mmd-max-samples",
        type=int,
        default=500,
        help="Maximum deterministic samples per sequence for RBF MMD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic diagnostic subsampling.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.sequence_09_npz, args.sequence_10_npz):
        if not path.is_file():
            raise FileNotFoundError("Representation file not found: {}".format(path))

    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError(
            "Quantiles must satisfy 0 < low < high < 1, received {} and {}."
            .format(args.low_quantile, args.high_quantile)
        )
    if args.near_constant_std < 0.0:
        raise ValueError("--near-constant-std cannot be negative.")
    if args.ridge_alpha < 0.0:
        raise ValueError("--ridge-alpha cannot be negative.")
    if args.mmd_max_samples <= 1:
        raise ValueError("--mmd-max-samples must exceed one.")


def resolve_predictions_path(npz_path: Path, csv_path: Optional[Path]) -> Path:
    resolved = csv_path if csv_path is not None else npz_path.parent / "frame_predictions.csv"
    if not resolved.is_file():
        raise FileNotFoundError("Prediction CSV not found: {}".format(resolved))
    return resolved


def load_representation(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as archive:
        if "translation_rep" not in archive.files:
            raise KeyError(
                "{} does not contain 'translation_rep'; keys are {}."
                .format(path, sorted(archive.files))
            )
        representation = np.asarray(archive["translation_rep"], dtype=np.float64)

    if representation.ndim != 2:
        raise ValueError(
            "translation_rep must be [N, D], received {} from {}."
            .format(representation.shape, path)
        )
    if representation.shape[0] == 0 or representation.shape[1] == 0:
        raise ValueError("Representation cannot be empty: {}.".format(path))
    if not np.all(np.isfinite(representation)):
        raise FloatingPointError("Representation contains non-finite values: {}.".format(path))
    return representation


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    gt_columns = tuple("translation_gt_{}".format(axis) for axis in AXES)
    pred_columns = tuple("translation_pred_{}".format(axis) for axis in AXES)
    gt_rows: List[List[float]] = []
    pred_rows: List[List[float]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = set(gt_columns + pred_columns)
        missing = required.difference(fieldnames)
        if missing:
            raise KeyError("{} is missing columns {}.".format(path, sorted(missing)))

        for row in reader:
            gt_rows.append([float(row[column]) for column in gt_columns])
            pred_rows.append([float(row[column]) for column in pred_columns])

    gt = np.asarray(gt_rows, dtype=np.float64)
    pred = np.asarray(pred_rows, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1:] != (3,) or gt.shape[0] == 0:
        raise ValueError("Unexpected translation array shape from {}: {}.".format(path, gt.shape))
    if not np.all(np.isfinite(gt)) or not np.all(np.isfinite(pred)):
        raise FloatingPointError("Prediction CSV contains non-finite values: {}.".format(path))
    return gt, pred


def validate_sequence_arrays(
    name: str,
    representation: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    if representation.shape[0] != target.shape[0]:
        raise ValueError(
            "{} sample mismatch: representation {}, targets {}."
            .format(name, representation.shape[0], target.shape[0])
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "{} prediction shape {} does not match target {}."
            .format(name, prediction.shape, target.shape)
        )


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = math.sqrt(float(np.sum(a_centered ** 2) * np.sum(b_centered ** 2)))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denominator)


def pca_statistics(representation: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    centered = representation - np.mean(representation, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values ** 2
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    positive = ratios[ratios > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) if positive.size else 0.0
    effective_rank = float(np.exp(entropy)) if positive.size else 0.0
    stable_rank = (
        float(total / eigenvalues[0])
        if eigenvalues.size and eigenvalues[0] > 0.0
        else 0.0
    )
    cumulative = np.cumsum(ratios)

    def components_for(fraction: float) -> int:
        if cumulative.size == 0 or total <= 0.0:
            return 0
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    stats = {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "components_90_percent": components_for(0.90),
        "components_95_percent": components_for(0.95),
        "components_99_percent": components_for(0.99),
        "pc1_variance_ratio": float(ratios[0]) if ratios.size else 0.0,
        "pc2_variance_ratio": float(ratios[1]) if ratios.size > 1 else 0.0,
    }
    return ratios, stats


def feature_target_correlations(
    representation: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    correlations = np.zeros((representation.shape[1], 3), dtype=np.float64)
    for feature_index in range(representation.shape[1]):
        for axis_index in range(3):
            correlations[feature_index, axis_index] = safe_correlation(
                representation[:, feature_index], target[:, axis_index]
            )

    summary: Dict[str, Dict[str, float]] = {}
    for axis_index, axis in enumerate(AXES):
        values = np.abs(correlations[:, axis_index])
        order = np.argsort(values)[::-1]
        top_count = min(10, values.size)
        summary[axis] = {
            "max_absolute_correlation": float(values[order[0]]) if values.size else 0.0,
            "mean_top10_absolute_correlation": (
                float(np.mean(values[order[:top_count]])) if top_count else 0.0
            ),
            "best_feature_index": int(order[0]) if values.size else -1,
        }
    return correlations, summary


def prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    error = prediction - target
    axis_mae = np.mean(np.abs(error), axis=0)
    axis_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    denominator = np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2, axis=0)
    numerator = np.sum(error ** 2, axis=0)
    r2 = np.where(denominator > 0.0, 1.0 - numerator / denominator, 0.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "l2_rmse": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "axis_mae": {axis: float(axis_mae[index]) for index, axis in enumerate(AXES)},
        "axis_rmse": {axis: float(axis_rmse[index]) for index, axis in enumerate(AXES)},
        "axis_r2": {axis: float(r2[index]) for index, axis in enumerate(AXES)},
    }


def representation_statistics(
    representation: np.ndarray,
    target: np.ndarray,
    near_constant_std: float,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = np.mean(representation, axis=0)
    feature_std = np.std(representation, axis=0)
    variance_ratios, pca_stats = pca_statistics(representation)
    correlations, correlation_summary = feature_target_correlations(representation, target)
    near_constant = np.flatnonzero(feature_std <= near_constant_std)
    exactly_constant = np.flatnonzero(feature_std == 0.0)
    stats: Dict[str, object] = {
        "samples": int(representation.shape[0]),
        "dimension": int(representation.shape[1]),
        "mean_feature_std": float(np.mean(feature_std)),
        "median_feature_std": float(np.median(feature_std)),
        "minimum_feature_std": float(np.min(feature_std)),
        "maximum_feature_std": float(np.max(feature_std)),
        "near_constant_threshold": float(near_constant_std),
        "near_constant_dimensions": [int(value) for value in near_constant],
        "near_constant_count": int(near_constant.size),
        "exactly_constant_dimensions": [int(value) for value in exactly_constant],
        "exactly_constant_count": int(exactly_constant.size),
        "pca": pca_stats,
        "target_correlations": correlation_summary,
    }
    return stats, feature_mean, feature_std, correlations


def assign_regimes(forward: np.ndarray, low: float, high: float) -> np.ndarray:
    regimes = np.full(forward.shape, 1, dtype=np.int64)
    regimes[forward <= low] = 0
    regimes[forward > high] = 2
    return regimes


def regime_statistics(
    representation: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    regimes: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray]:
    result: Dict[str, object] = {}
    centroids = np.full((3, representation.shape[1]), np.nan, dtype=np.float64)
    for index, name in enumerate(REGIME_NAMES):
        mask = regimes == index
        if not np.any(mask):
            result[name] = {"count": 0}
            continue
        centroids[index] = np.mean(representation[mask], axis=0)
        result[name] = {
            "count": int(np.sum(mask)),
            "forward_gt_mean": float(np.mean(target[mask, 2])),
            "forward_gt_std": float(np.std(target[mask, 2])),
            "prediction_metrics": prediction_metrics(target[mask], prediction[mask]),
        }

    distances: Dict[str, float] = {}
    for first, second in ((0, 1), (1, 2), (0, 2)):
        key = "{}_to_{}".format(REGIME_NAMES[first], REGIME_NAMES[second])
        if np.all(np.isfinite(centroids[[first, second]])):
            distances[key] = float(np.linalg.norm(centroids[first] - centroids[second]))
        else:
            distances[key] = float("nan")
    result["centroid_distances"] = distances
    return result, centroids


def standardize_train_test(
    train_rep: np.ndarray,
    test_rep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_rep, axis=0, keepdims=True)
    std = np.std(train_rep, axis=0, keepdims=True)
    std = np.where(std > 1.0e-12, std, 1.0)
    return (train_rep - mean) / std, (test_rep - mean) / std


def ridge_probe(
    train_rep: np.ndarray,
    train_target: np.ndarray,
    test_rep: np.ndarray,
    test_target: np.ndarray,
    alpha: float,
) -> Dict[str, object]:
    x_train, x_test = standardize_train_test(train_rep, test_rep)
    x_train = np.column_stack([x_train, np.ones(x_train.shape[0])])
    x_test = np.column_stack([x_test, np.ones(x_test.shape[0])])
    identity = np.eye(x_train.shape[1], dtype=np.float64)
    identity[-1, -1] = 0.0
    gram = x_train.T.dot(x_train) + alpha * identity
    weights = np.linalg.solve(gram, x_train.T.dot(train_target))
    prediction = x_test.dot(weights)
    return prediction_metrics(test_target, prediction)


def pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(a ** 2, axis=1, keepdims=True)
        + np.sum(b ** 2, axis=1, keepdims=True).T
        - 2.0 * a.dot(b.T)
    )
    return np.maximum(distances, 0.0)


def rbf_mmd(
    first: np.ndarray,
    second: np.ndarray,
    max_samples: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)

    def sample(array: np.ndarray) -> np.ndarray:
        if array.shape[0] <= max_samples:
            return array
        indices = rng.choice(array.shape[0], size=max_samples, replace=False)
        return array[indices]

    first_sample = sample(first)
    second_sample = sample(second)
    combined = np.vstack([first_sample, second_sample])
    combined_mean = np.mean(combined, axis=0, keepdims=True)
    combined_std = np.std(combined, axis=0, keepdims=True)
    combined_std = np.where(combined_std > 1.0e-12, combined_std, 1.0)
    first_z = (first_sample - combined_mean) / combined_std
    second_z = (second_sample - combined_mean) / combined_std
    combined_z = np.vstack([first_z, second_z])
    distance_probe = pairwise_squared_distances(combined_z, combined_z)
    positive = distance_probe[distance_probe > 0.0]
    bandwidth_sq = float(np.median(positive)) if positive.size else 1.0
    bandwidth_sq = max(bandwidth_sq, 1.0e-12)

    k_xx = np.exp(-pairwise_squared_distances(first_z, first_z) / (2.0 * bandwidth_sq))
    k_yy = np.exp(-pairwise_squared_distances(second_z, second_z) / (2.0 * bandwidth_sq))
    k_xy = np.exp(-pairwise_squared_distances(first_z, second_z) / (2.0 * bandwidth_sq))
    value = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return {
        "rbf_mmd_squared": max(value, 0.0),
        "rbf_bandwidth_squared": bandwidth_sq,
        "samples_sequence_09": int(first_sample.shape[0]),
        "samples_sequence_10": int(second_sample.shape[0]),
    }


def jensen_shannon_histogram(first: np.ndarray, second: np.ndarray, bins: int = 30) -> float:
    lower = float(min(np.min(first), np.min(second)))
    upper = float(max(np.max(first), np.max(second)))
    if not upper > lower:
        return 0.0
    edges = np.linspace(lower, upper, bins + 1)
    first_hist = np.histogram(first, bins=edges)[0].astype(np.float64) + 1.0e-12
    second_hist = np.histogram(second, bins=edges)[0].astype(np.float64) + 1.0e-12
    first_hist /= np.sum(first_hist)
    second_hist /= np.sum(second_hist)
    midpoint = 0.5 * (first_hist + second_hist)
    return float(
        0.5 * np.sum(first_hist * np.log(first_hist / midpoint))
        + 0.5 * np.sum(second_hist * np.log(second_hist / midpoint))
    )


def distribution_shift(
    rep09: np.ndarray,
    rep10: np.ndarray,
    target09: np.ndarray,
    target10: np.ndarray,
    max_samples: int,
    seed: int,
) -> Dict[str, object]:
    mean09 = np.mean(rep09, axis=0)
    mean10 = np.mean(rep10, axis=0)
    pooled_std = np.sqrt(0.5 * (np.var(rep09, axis=0) + np.var(rep10, axis=0)))
    standardized_mean_difference = np.divide(
        mean10 - mean09,
        pooled_std,
        out=np.zeros_like(mean09),
        where=pooled_std > 1.0e-12,
    )
    covariance09 = np.cov(rep09, rowvar=False)
    covariance10 = np.cov(rep10, rowvar=False)
    covariance_denominator = float(
        np.linalg.norm(covariance09, ord="fro")
        + np.linalg.norm(covariance10, ord="fro")
    )
    covariance_shift = (
        float(np.linalg.norm(covariance10 - covariance09, ord="fro"))
        / covariance_denominator
        if covariance_denominator > 0.0
        else 0.0
    )
    result: Dict[str, object] = {
        "mean_l2_distance": float(np.linalg.norm(mean10 - mean09)),
        "standardized_mean_shift_rms": float(
            np.sqrt(np.mean(standardized_mean_difference ** 2))
        ),
        "maximum_absolute_standardized_mean_shift": float(
            np.max(np.abs(standardized_mean_difference))
        ),
        "relative_covariance_frobenius_shift": covariance_shift,
        "forward_motion_js_divergence": jensen_shannon_histogram(
            target09[:, 2], target10[:, 2]
        ),
    }
    result.update(rbf_mmd(rep09, rep10, max_samples=max_samples, seed=seed))
    return result


def combined_pca_projection(rep09: np.ndarray, rep10: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([rep09, rep10])
    mean = np.mean(combined, axis=0, keepdims=True)
    centered = combined - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    projected = centered.dot(components)
    return projected[: rep09.shape[0]], projected[rep09.shape[0] :]


def write_feature_csv(
    path: Path,
    mean09: np.ndarray,
    std09: np.ndarray,
    corr09: np.ndarray,
    mean10: np.ndarray,
    std10: np.ndarray,
    corr10: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature",
                "mean_09", "std_09", "corr_gt_x_09", "corr_gt_y_09", "corr_gt_z_09",
                "mean_10", "std_10", "corr_gt_x_10", "corr_gt_y_10", "corr_gt_z_10",
            ]
        )
        for index in range(mean09.size):
            writer.writerow(
                [index, mean09[index], std09[index]]
                + list(corr09[index])
                + [mean10[index], std10[index]]
                + list(corr10[index])
            )


def plot_outputs(
    output_dir: Path,
    rep09: np.ndarray,
    rep10: np.ndarray,
    target09: np.ndarray,
    target10: np.ndarray,
    std09: np.ndarray,
    std10: np.ndarray,
    pca09: np.ndarray,
    pca10: np.ndarray,
) -> None:
    plt.figure(figsize=(10, 4.8))
    indices = np.arange(std09.size)
    plt.plot(indices, std09, label="sequence 09", linewidth=1.2)
    plt.plot(indices, std10, label="sequence 10", linewidth=1.2)
    plt.xlabel("Feature dimension")
    plt.ylabel("Standard deviation")
    plt.title("Compact translation feature variability")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "feature_standard_deviation.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    count = min(40, pca09.size, pca10.size)
    plt.semilogy(np.arange(1, count + 1), pca09[:count] + 1.0e-15, marker="o", label="sequence 09")
    plt.semilogy(np.arange(1, count + 1), pca10[:count] + 1.0e-15, marker="o", label="sequence 10")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Compact representation spectrum")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "pca_variance_spectrum.png"), dpi=180)
    plt.close()

    projection09, projection10 = combined_pca_projection(rep09, rep10)
    plt.figure(figsize=(8, 6))
    plt.scatter(projection09[:, 0], projection09[:, 1], s=7, alpha=0.45, label="sequence 09")
    plt.scatter(projection10[:, 0], projection10[:, 1], s=7, alpha=0.45, label="sequence 10")
    plt.xlabel("Combined PC1")
    plt.ylabel("Combined PC2")
    plt.title("Sequence separation in compact representation")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "combined_pca_sequence_scatter.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    lower = float(min(np.min(target09[:, 2]), np.min(target10[:, 2])))
    upper = float(max(np.max(target09[:, 2]), np.max(target10[:, 2])))
    bins = np.linspace(lower, upper, 31) if upper > lower else 30
    plt.hist(target09[:, 2], bins=bins, density=True, alpha=0.55, label="sequence 09")
    plt.hist(target10[:, 2], bins=bins, density=True, alpha=0.55, label="sequence 10")
    plt.xlabel("Ground-truth forward translation z")
    plt.ylabel("Density")
    plt.title("Forward-motion distribution")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "forward_motion_distribution.png"), dpi=180)
    plt.close()


def sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_text_report(path: Path, report: Mapping[str, object]) -> None:
    sequence09 = report["sequence_09"]
    sequence10 = report["sequence_10"]
    shift = report["distribution_shift"]
    lines = [
        "Compact translation representation analysis",
        "=" * 72,
        "Sequence 09: N={}, D={}".format(
            sequence09["representation"]["samples"], sequence09["representation"]["dimension"]
        ),
        "Sequence 10: N={}, D={}".format(
            sequence10["representation"]["samples"], sequence10["representation"]["dimension"]
        ),
        "",
        "Prediction metrics",
        "-" * 72,
        "Seq 09 MAE={:.9f}, RMSE={:.9f}".format(
            sequence09["prediction_metrics"]["mae"], sequence09["prediction_metrics"]["rmse"]
        ),
        "Seq 10 MAE={:.9f}, RMSE={:.9f}".format(
            sequence10["prediction_metrics"]["mae"], sequence10["prediction_metrics"]["rmse"]
        ),
        "",
        "Representation health",
        "-" * 72,
        "Seq 09 effective rank={:.3f}, near-constant dims={}".format(
            sequence09["representation"]["pca"]["effective_rank"],
            sequence09["representation"]["near_constant_count"],
        ),
        "Seq 10 effective rank={:.3f}, near-constant dims={}".format(
            sequence10["representation"]["pca"]["effective_rank"],
            sequence10["representation"]["near_constant_count"],
        ),
        "",
        "Cross-sequence shift",
        "-" * 72,
        "Standardized mean shift RMS={:.6f}".format(shift["standardized_mean_shift_rms"]),
        "Relative covariance shift={:.6f}".format(shift["relative_covariance_frobenius_shift"]),
        "RBF MMD^2={:.6f}".format(shift["rbf_mmd_squared"]),
        "Forward-motion JS divergence={:.6f}".format(shift["forward_motion_js_divergence"]),
        "",
        "Cross-sequence linear probes",
        "-" * 72,
        "Train 09 -> test 10: MAE={:.9f}, RMSE={:.9f}".format(
            report["cross_sequence_ridge_probes"]["train_09_test_10"]["mae"],
            report["cross_sequence_ridge_probes"]["train_09_test_10"]["rmse"],
        ),
        "Train 10 -> test 09: MAE={:.9f}, RMSE={:.9f}".format(
            report["cross_sequence_ridge_probes"]["train_10_test_09"]["mae"],
            report["cross_sequence_ridge_probes"]["train_10_test_09"]["rmse"],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    predictions09_path = resolve_predictions_path(
        args.sequence_09_npz, args.sequence_09_predictions
    )
    predictions10_path = resolve_predictions_path(
        args.sequence_10_npz, args.sequence_10_predictions
    )
    rep09 = load_representation(args.sequence_09_npz)
    rep10 = load_representation(args.sequence_10_npz)
    target09, prediction09 = load_predictions(predictions09_path)
    target10, prediction10 = load_predictions(predictions10_path)
    validate_sequence_arrays("sequence 09", rep09, target09, prediction09)
    validate_sequence_arrays("sequence 10", rep10, target10, prediction10)
    if rep09.shape[1] != rep10.shape[1]:
        raise ValueError(
            "Representation dimensions differ: sequence 09 has {}, sequence 10 has {}."
            .format(rep09.shape[1], rep10.shape[1])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats09, mean09, std09, corr09 = representation_statistics(
        rep09, target09, args.near_constant_std
    )
    stats10, mean10, std10, corr10 = representation_statistics(
        rep10, target10, args.near_constant_std
    )
    pca09, _ = pca_statistics(rep09)
    pca10, _ = pca_statistics(rep10)

    low_threshold = float(np.quantile(target09[:, 2], args.low_quantile))
    high_threshold = float(np.quantile(target09[:, 2], args.high_quantile))
    regimes09 = assign_regimes(target09[:, 2], low_threshold, high_threshold)
    regimes10 = assign_regimes(target10[:, 2], low_threshold, high_threshold)
    regime09, _ = regime_statistics(rep09, target09, prediction09, regimes09)
    regime10, _ = regime_statistics(rep10, target10, prediction10, regimes10)

    report: Dict[str, object] = {
        "inputs": {
            "sequence_09_npz": str(args.sequence_09_npz.resolve()),
            "sequence_10_npz": str(args.sequence_10_npz.resolve()),
            "sequence_09_predictions": str(predictions09_path.resolve()),
            "sequence_10_predictions": str(predictions10_path.resolve()),
        },
        "motion_regime_definition": {
            "source": "sequence_09_ground_truth_forward_translation_z",
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "sequence_09": {
            "representation": stats09,
            "prediction_metrics": prediction_metrics(target09, prediction09),
            "motion_regimes": regime09,
        },
        "sequence_10": {
            "representation": stats10,
            "prediction_metrics": prediction_metrics(target10, prediction10),
            "motion_regimes": regime10,
        },
        "distribution_shift": distribution_shift(
            rep09,
            rep10,
            target09,
            target10,
            max_samples=args.mmd_max_samples,
            seed=args.seed,
        ),
        "cross_sequence_ridge_probes": {
            "alpha": float(args.ridge_alpha),
            "train_09_test_10": ridge_probe(
                rep09, target09, rep10, target10, args.ridge_alpha
            ),
            "train_10_test_09": ridge_probe(
                rep10, target10, rep09, target09, args.ridge_alpha
            ),
        },
    }

    clean_report = sanitize_json(report)
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(clean_report, handle, indent=2, sort_keys=True)

    write_text_report(args.output_dir / "analysis_summary.txt", clean_report)
    write_feature_csv(
        args.output_dir / "feature_statistics.csv",
        mean09,
        std09,
        corr09,
        mean10,
        std10,
        corr10,
    )
    plot_outputs(
        args.output_dir,
        rep09,
        rep10,
        target09,
        target10,
        std09,
        std10,
        pca09,
        pca10,
    )

    print((args.output_dir / "analysis_summary.txt").read_text(encoding="utf-8"))
    print("Saved analysis outputs to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
