#!/usr/bin/env python3
"""Motion-regime analysis for compact DeepDCT-VO translation features.

Sequence-09 ground-truth forward translation defines shared low, medium, and
high thresholds. The same thresholds are then applied to sequence 10, avoiding
the misleading comparison produced by independently quantiling each sequence.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = ("x", "y", "z")
REGIMES = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze compact translation features by motion regime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence-09-npz", type=Path, required=True)
    parser.add_argument("--sequence-10-npz", type=Path, required=True)
    parser.add_argument("--sequence-09-predictions", type=Path, default=None)
    parser.add_argument("--sequence-10-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--high-quantile", type=float, default=2.0 / 3.0)
    parser.add_argument("--mmd-max-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.sequence_09_npz, args.sequence_10_npz):
        if not path.is_file():
            raise FileNotFoundError("Missing representation file: {}".format(path))
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError("Require 0 < low quantile < high quantile < 1.")
    if args.mmd_max_samples < 2:
        raise ValueError("--mmd-max-samples must be at least two.")


def predictions_path(npz_path: Path, explicit: Optional[Path]) -> Path:
    path = explicit if explicit is not None else npz_path.parent / "frame_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError("Missing prediction CSV: {}".format(path))
    return path


def load_representation(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as archive:
        if "translation_rep" not in archive.files:
            raise KeyError("{} lacks translation_rep; keys={}".format(path, archive.files))
        representation = np.asarray(archive["translation_rep"], dtype=np.float64)
    if representation.ndim != 2 or min(representation.shape) <= 0:
        raise ValueError("Expected nonempty [N,D] representation, got {}.".format(representation.shape))
    if not np.all(np.isfinite(representation)):
        raise FloatingPointError("Non-finite representation values in {}.".format(path))
    return representation


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    gt_names = tuple("translation_gt_{}".format(axis) for axis in AXES)
    pred_names = tuple("translation_pred_{}".format(axis) for axis in AXES)
    targets: List[List[float]] = []
    predictions: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(gt_names + pred_names).difference(reader.fieldnames or [])
        if missing:
            raise KeyError("{} is missing {}.".format(path, sorted(missing)))
        for row in reader:
            targets.append([float(row[name]) for name in gt_names])
            predictions.append([float(row[name]) for name in pred_names])
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim != 2 or target.shape[1:] != (3,) or target.shape[0] == 0:
        raise ValueError("Unexpected target shape {} from {}.".format(target.shape, path))
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Non-finite translation values in {}.".format(path))
    return target, prediction


def check_arrays(name: str, rep: np.ndarray, target: np.ndarray, pred: np.ndarray) -> None:
    if rep.shape[0] != target.shape[0] or pred.shape != target.shape:
        raise ValueError(
            "{} shapes disagree: rep={}, target={}, pred={}."
            .format(name, rep.shape, target.shape, pred.shape)
        )


def assign_regimes(forward: np.ndarray, low: float, high: float) -> np.ndarray:
    labels = np.full(forward.shape, 1, dtype=np.int64)
    labels[forward <= low] = 0
    labels[forward > high] = 2
    return labels


def effective_rank(rep: np.ndarray) -> float:
    centered = rep - np.mean(rep, axis=0, keepdims=True)
    values = np.linalg.svd(centered, full_matrices=False, compute_uv=False) ** 2
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def error_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    error = pred - target
    axis_mae = np.mean(np.abs(error), axis=0)
    axis_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    bias = np.mean(error, axis=0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "l2_rmse": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "axis_mae": {axis: float(axis_mae[i]) for i, axis in enumerate(AXES)},
        "axis_rmse": {axis: float(axis_rmse[i]) for i, axis in enumerate(AXES)},
        "axis_bias": {axis: float(bias[i]) for i, axis in enumerate(AXES)},
    }


def regime_summary(
    rep: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    result: Dict[str, object] = {}
    centroids = np.full((3, rep.shape[1]), np.nan, dtype=np.float64)
    within_rms = np.full(3, np.nan, dtype=np.float64)
    for index, name in enumerate(REGIMES):
        mask = labels == index
        count = int(np.sum(mask))
        if count == 0:
            result[name] = {"count": 0}
            continue
        selected_rep = rep[mask]
        centroid = np.mean(selected_rep, axis=0)
        centroids[index] = centroid
        within_rms[index] = float(
            np.sqrt(np.mean(np.sum((selected_rep - centroid) ** 2, axis=1)))
        )
        result[name] = {
            "count": count,
            "fraction": float(count / rep.shape[0]),
            "forward_gt_mean": float(np.mean(target[mask, 2])),
            "forward_gt_std": float(np.std(target[mask, 2])),
            "forward_pred_mean": float(np.mean(pred[mask, 2])),
            "representation_effective_rank": effective_rank(selected_rep),
            "representation_within_rms": float(within_rms[index]),
            "errors": error_metrics(target[mask], pred[mask]),
        }
    return result, centroids, within_rms


def separation_summary(
    rep: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    within_rms: np.ndarray,
) -> Dict[str, object]:
    pairs: Dict[str, object] = {}
    for first, second in ((0, 1), (1, 2), (0, 2)):
        key = "{}_to_{}".format(REGIMES[first], REGIMES[second])
        if not np.all(np.isfinite(centroids[[first, second]])):
            pairs[key] = None
            continue
        distance = float(np.linalg.norm(centroids[first] - centroids[second]))
        scale = float(0.5 * (within_rms[first] + within_rms[second]))
        pairs[key] = {
            "centroid_distance": distance,
            "normalized_separation": distance / scale if scale > 0.0 else 0.0,
        }

    valid = np.all(np.isfinite(centroids), axis=1)
    if np.all(valid):
        squared = np.sum((rep[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        nearest_accuracy = float(np.mean(nearest == labels))
        confusion = np.zeros((3, 3), dtype=np.int64)
        for truth, predicted in zip(labels, nearest):
            confusion[int(truth), int(predicted)] += 1
    else:
        nearest_accuracy = float("nan")
        confusion = np.zeros((3, 3), dtype=np.int64)
    return {
        "pairwise": pairs,
        "nearest_centroid_accuracy": nearest_accuracy,
        "nearest_centroid_confusion": confusion.tolist(),
    }


def pairwise_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(a ** 2, axis=1, keepdims=True)
        + np.sum(b ** 2, axis=1, keepdims=True).T
        - 2.0 * a.dot(b.T),
        0.0,
    )


def same_regime_mmd(
    first: np.ndarray,
    second: np.ndarray,
    max_samples: int,
    rng: np.random.RandomState,
) -> float:
    def sample(array: np.ndarray) -> np.ndarray:
        if array.shape[0] <= max_samples:
            return array
        return array[rng.choice(array.shape[0], max_samples, replace=False)]

    first = sample(first)
    second = sample(second)
    combined = np.vstack([first, second])
    mean = np.mean(combined, axis=0, keepdims=True)
    std = np.std(combined, axis=0, keepdims=True)
    std = np.where(std > 1.0e-12, std, 1.0)
    first = (first - mean) / std
    second = (second - mean) / std
    probe = pairwise_squared(np.vstack([first, second]), np.vstack([first, second]))
    positive = probe[probe > 0.0]
    bandwidth = max(float(np.median(positive)) if positive.size else 1.0, 1.0e-12)
    value = (
        np.mean(np.exp(-pairwise_squared(first, first) / (2.0 * bandwidth)))
        + np.mean(np.exp(-pairwise_squared(second, second) / (2.0 * bandwidth)))
        - 2.0 * np.mean(np.exp(-pairwise_squared(first, second) / (2.0 * bandwidth)))
    )
    return max(float(value), 0.0)


def same_regime_shift(
    rep09: np.ndarray,
    rep10: np.ndarray,
    labels09: np.ndarray,
    labels10: np.ndarray,
    max_samples: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.RandomState(seed)
    result: Dict[str, object] = {}
    for index, name in enumerate(REGIMES):
        first = rep09[labels09 == index]
        second = rep10[labels10 == index]
        if first.shape[0] < 2 or second.shape[0] < 2:
            result[name] = {"count_09": int(first.shape[0]), "count_10": int(second.shape[0])}
            continue
        mean09 = np.mean(first, axis=0)
        mean10 = np.mean(second, axis=0)
        pooled_std = np.sqrt(0.5 * (np.var(first, axis=0) + np.var(second, axis=0)))
        standardized = np.divide(
            mean10 - mean09,
            pooled_std,
            out=np.zeros_like(mean09),
            where=pooled_std > 1.0e-12,
        )
        covariance09 = np.cov(first, rowvar=False)
        covariance10 = np.cov(second, rowvar=False)
        denominator = float(
            np.linalg.norm(covariance09, ord="fro")
            + np.linalg.norm(covariance10, ord="fro")
        )
        result[name] = {
            "count_09": int(first.shape[0]),
            "count_10": int(second.shape[0]),
            "centroid_l2_shift": float(np.linalg.norm(mean10 - mean09)),
            "standardized_centroid_shift_rms": float(np.sqrt(np.mean(standardized ** 2))),
            "relative_covariance_shift": (
                float(np.linalg.norm(covariance10 - covariance09, ord="fro")) / denominator
                if denominator > 0.0
                else 0.0
            ),
            "rbf_mmd_squared": same_regime_mmd(first, second, max_samples, rng),
        }
    return result


def sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_regime_csv(path: Path, report: Dict[str, object]) -> None:
    columns = [
        "sequence", "regime", "count", "fraction", "forward_gt_mean",
        "forward_gt_std", "effective_rank", "within_rms", "mae", "rmse",
        "mae_x", "mae_y", "mae_z", "rmse_x", "rmse_y", "rmse_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sequence in ("09", "10"):
            for regime in REGIMES:
                item = report["sequences"][sequence]["regimes"][regime]
                if item.get("count", 0) == 0:
                    writer.writerow({"sequence": sequence, "regime": regime, "count": 0})
                    continue
                errors = item["errors"]
                writer.writerow(
                    {
                        "sequence": sequence,
                        "regime": regime,
                        "count": item["count"],
                        "fraction": item["fraction"],
                        "forward_gt_mean": item["forward_gt_mean"],
                        "forward_gt_std": item["forward_gt_std"],
                        "effective_rank": item["representation_effective_rank"],
                        "within_rms": item["representation_within_rms"],
                        "mae": errors["mae"],
                        "rmse": errors["rmse"],
                        "mae_x": errors["axis_mae"]["x"],
                        "mae_y": errors["axis_mae"]["y"],
                        "mae_z": errors["axis_mae"]["z"],
                        "rmse_x": errors["axis_rmse"]["x"],
                        "rmse_y": errors["axis_rmse"]["y"],
                        "rmse_z": errors["axis_rmse"]["z"],
                    }
                )


def write_shift_csv(path: Path, shift: Dict[str, object]) -> None:
    columns = [
        "regime", "count_09", "count_10", "centroid_l2_shift",
        "standardized_centroid_shift_rms", "relative_covariance_shift", "rbf_mmd_squared",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for regime in REGIMES:
            row = {"regime": regime}
            row.update(shift[regime])
            writer.writerow(row)


def plot_metrics(output_dir: Path, report: Dict[str, object]) -> None:
    x = np.arange(3)
    width = 0.36
    for metric, filename, ylabel in (
        ("mae", "translation_mae_by_regime.png", "Translation MAE"),
        ("rmse", "translation_rmse_by_regime.png", "Translation RMSE"),
    ):
        values09 = [report["sequences"]["09"]["regimes"][name]["errors"][metric] for name in REGIMES]
        values10 = [report["sequences"]["10"]["regimes"][name]["errors"][metric] for name in REGIMES]
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(x - width / 2.0, values09, width, label="sequence 09")
        plt.bar(x + width / 2.0, values10, width, label="sequence 10")
        plt.xticks(x, REGIMES)
        plt.ylabel(ylabel)
        plt.xlabel("Shared forward-motion regime")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(output_dir / filename), dpi=180)
        plt.close()

    counts09 = [report["sequences"]["09"]["regimes"][name]["count"] for name in REGIMES]
    counts10 = [report["sequences"]["10"]["regimes"][name]["count"] for name in REGIMES]
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(x - width / 2.0, counts09, width, label="sequence 09")
    plt.bar(x + width / 2.0, counts10, width, label="sequence 10")
    plt.xticks(x, REGIMES)
    plt.ylabel("Samples")
    plt.xlabel("Shared forward-motion regime")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "regime_sample_counts.png"), dpi=180)
    plt.close()

    shifts = [report["cross_sequence_same_regime_shift"][name]["standardized_centroid_shift_rms"] for name in REGIMES]
    mmd = [report["cross_sequence_same_regime_shift"][name]["rbf_mmd_squared"] for name in REGIMES]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(REGIMES, shifts)
    axes[0].set_title("Standardized centroid shift")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(REGIMES, mmd)
    axes[1].set_title("Same-regime RBF MMD²")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_dir / "same_regime_distribution_shift.png"), dpi=180)
    plt.close(fig)


def write_text(path: Path, report: Dict[str, object]) -> None:
    lines = [
        "Compact translation motion-regime analysis",
        "=" * 88,
        "Shared thresholds from sequence 09: low <= {:.9f}, high > {:.9f}".format(
            report["thresholds"]["low"], report["thresholds"]["high"]
        ),
        "",
        "{:<10} {:>8} {:>8} {:>12} {:>12} {:>12} {:>12}".format(
            "Regime", "N09", "N10", "MAE09", "MAE10", "RMSE09", "RMSE10"
        ),
        "-" * 88,
    ]
    for regime in REGIMES:
        item09 = report["sequences"]["09"]["regimes"][regime]
        item10 = report["sequences"]["10"]["regimes"][regime]
        lines.append(
            "{:<10} {:>8d} {:>8d} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                regime,
                item09["count"],
                item10["count"],
                item09["errors"]["mae"],
                item10["errors"]["mae"],
                item09["errors"]["rmse"],
                item10["errors"]["rmse"],
            )
        )
    lines.extend(["", "Same-regime representation shift", "-" * 88])
    for regime in REGIMES:
        shift = report["cross_sequence_same_regime_shift"][regime]
        lines.append(
            "{:<10} standardized centroid={:.6f}, covariance={:.6f}, MMD²={:.6f}".format(
                regime,
                shift["standardized_centroid_shift_rms"],
                shift["relative_covariance_shift"],
                shift["rbf_mmd_squared"],
            )
        )
    lines.extend(["", "Regime separability", "-" * 88])
    for sequence in ("09", "10"):
        separation = report["sequences"][sequence]["separation"]
        lines.append(
            "Sequence {} nearest-centroid accuracy: {:.6f}".format(
                sequence, separation["nearest_centroid_accuracy"]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    csv09 = predictions_path(args.sequence_09_npz, args.sequence_09_predictions)
    csv10 = predictions_path(args.sequence_10_npz, args.sequence_10_predictions)
    rep09 = load_representation(args.sequence_09_npz)
    rep10 = load_representation(args.sequence_10_npz)
    target09, pred09 = load_predictions(csv09)
    target10, pred10 = load_predictions(csv10)
    check_arrays("sequence 09", rep09, target09, pred09)
    check_arrays("sequence 10", rep10, target10, pred10)
    if rep09.shape[1] != rep10.shape[1]:
        raise ValueError("Representation dimensions differ: {} and {}.".format(rep09.shape[1], rep10.shape[1]))

    low = float(np.quantile(target09[:, 2], args.low_quantile))
    high = float(np.quantile(target09[:, 2], args.high_quantile))
    labels09 = assign_regimes(target09[:, 2], low, high)
    labels10 = assign_regimes(target10[:, 2], low, high)
    regimes09, centroids09, within09 = regime_summary(rep09, target09, pred09, labels09)
    regimes10, centroids10, within10 = regime_summary(rep10, target10, pred10, labels10)

    report: Dict[str, object] = {
        "inputs": {
            "sequence_09_npz": str(args.sequence_09_npz.resolve()),
            "sequence_10_npz": str(args.sequence_10_npz.resolve()),
            "sequence_09_predictions": str(csv09.resolve()),
            "sequence_10_predictions": str(csv10.resolve()),
        },
        "thresholds": {
            "source": "sequence_09_ground_truth_translation_z",
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
            "low": low,
            "high": high,
        },
        "sequences": {
            "09": {
                "regimes": regimes09,
                "separation": separation_summary(rep09, labels09, centroids09, within09),
            },
            "10": {
                "regimes": regimes10,
                "separation": separation_summary(rep10, labels10, centroids10, within10),
            },
        },
        "cross_sequence_same_regime_shift": same_regime_shift(
            rep09,
            rep10,
            labels09,
            labels10,
            max_samples=args.mmd_max_samples,
            seed=args.seed,
        ),
    }
    report = sanitize(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "motion_regime_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_regime_csv(args.output_dir / "motion_regime_metrics.csv", report)
    write_shift_csv(
        args.output_dir / "same_regime_distribution_shift.csv",
        report["cross_sequence_same_regime_shift"],
    )
    write_text(args.output_dir / "motion_regime_summary.txt", report)
    plot_metrics(args.output_dir, report)
    print((args.output_dir / "motion_regime_summary.txt").read_text(encoding="utf-8"))
    print("Saved motion-regime analysis to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
