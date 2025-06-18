#!/usr/bin/env python3
"""Paired dense-vs-compact translation comparison by motion regime.

The script consumes GT/GT ``frame_predictions.csv`` files from A4 dense and
compact aggregation evaluations. It verifies frame/target alignment, defines
shared regimes from sequence-09 ground-truth forward translation, and performs
paired per-frame comparisons with bootstrap confidence intervals.
"""

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


AXES = ("x", "y", "z")
REGIMES = ("low", "medium", "high")
DECODERS = ("dense", "compact")
SEQUENCES = ("09", "10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare A4 dense and compact translation errors by motion regime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dense-sequence-09", type=Path, required=True)
    parser.add_argument("--dense-sequence-10", type=Path, required=True)
    parser.add_argument("--compact-sequence-09", type=Path, required=True)
    parser.add_argument("--compact-sequence-10", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--high-quantile", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Paired frame-bootstrap replicates for MAE-difference confidence intervals.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1.0e-8,
        help="Absolute tolerance when verifying targets across decoder CSVs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.dense_sequence_09,
        args.dense_sequence_10,
        args.compact_sequence_09,
        args.compact_sequence_10,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Missing prediction CSV: {}".format(path))
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError("Require 0 < low quantile < high quantile < 1.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one.")
    if args.target_tolerance < 0.0:
        raise ValueError("--target-tolerance cannot be negative.")


def load_predictions(path: Path) -> Dict[str, object]:
    gt_names = tuple("translation_gt_{}".format(axis) for axis in AXES)
    pred_names = tuple("translation_pred_{}".format(axis) for axis in AXES)
    target_rows: List[List[float]] = []
    prediction_rows: List[List[float]] = []
    frame_keys: List[Tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = set(gt_names + pred_names)
        missing = required.difference(fields)
        if missing:
            raise KeyError("{} is missing columns {}.".format(path, sorted(missing)))
        has_frame_keys = {"sequence", "frame_prev", "frame_curr"}.issubset(fields)
        for row_index, row in enumerate(reader):
            target_rows.append([float(row[name]) for name in gt_names])
            prediction_rows.append([float(row[name]) for name in pred_names])
            if has_frame_keys:
                frame_keys.append((row["sequence"], row["frame_prev"], row["frame_curr"]))
            else:
                frame_keys.append(("", str(row_index), str(row_index + 1)))

    target = np.asarray(target_rows, dtype=np.float64)
    prediction = np.asarray(prediction_rows, dtype=np.float64)
    if target.ndim != 2 or target.shape[1:] != (3,) or target.shape[0] == 0:
        raise ValueError("Unexpected array shape from {}: {}.".format(path, target.shape))
    if prediction.shape != target.shape:
        raise ValueError("Prediction shape {} differs from target {}.".format(prediction.shape, target.shape))
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Non-finite values in {}.".format(path))
    return {"target": target, "prediction": prediction, "frame_keys": frame_keys}


def verify_pair(
    sequence: str,
    dense: Mapping[str, object],
    compact: Mapping[str, object],
    tolerance: float,
) -> None:
    dense_target = dense["target"]
    compact_target = compact["target"]
    if dense_target.shape != compact_target.shape:
        raise ValueError(
            "Sequence {} sample shapes differ: dense {}, compact {}."
            .format(sequence, dense_target.shape, compact_target.shape)
        )
    if dense["frame_keys"] != compact["frame_keys"]:
        raise ValueError("Sequence {} frame ordering differs between decoders.".format(sequence))
    maximum = float(np.max(np.abs(dense_target - compact_target)))
    if maximum > tolerance:
        raise ValueError(
            "Sequence {} targets differ across decoders; max abs difference {:.3e} exceeds {:.3e}."
            .format(sequence, maximum, tolerance)
        )


def assign_regimes(forward: np.ndarray, low: float, high: float) -> np.ndarray:
    labels = np.full(forward.shape, 1, dtype=np.int64)
    labels[forward <= low] = 0
    labels[forward > high] = 2
    return labels


def metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    error = prediction - target
    absolute = np.abs(error)
    squared = error ** 2
    axis_mae = np.mean(absolute, axis=0)
    axis_rmse = np.sqrt(np.mean(squared, axis=0))
    axis_bias = np.mean(error, axis=0)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "l2_rmse": float(np.sqrt(np.mean(np.sum(squared, axis=1)))),
        "axis_mae": {axis: float(axis_mae[index]) for index, axis in enumerate(AXES)},
        "axis_rmse": {axis: float(axis_rmse[index]) for index, axis in enumerate(AXES)},
        "axis_bias": {axis: float(axis_bias[index]) for index, axis in enumerate(AXES)},
    }


def bootstrap_paired_mae_difference(
    dense_frame_mae: np.ndarray,
    compact_frame_mae: np.ndarray,
    replicates: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    difference = compact_frame_mae - dense_frame_mae
    sample_count = difference.size
    bootstrap = np.empty(replicates, dtype=np.float64)
    chunk_size = min(250, replicates)
    start = 0
    while start < replicates:
        count = min(chunk_size, replicates - start)
        indices = rng.randint(0, sample_count, size=(count, sample_count))
        bootstrap[start : start + count] = np.mean(difference[indices], axis=1)
        start += count
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    return {
        "compact_minus_dense_mae": float(np.mean(difference)),
        "confidence": float(confidence),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "compact_better_frame_fraction": float(np.mean(difference < 0.0)),
        "dense_better_frame_fraction": float(np.mean(difference > 0.0)),
        "tie_fraction": float(np.mean(difference == 0.0)),
    }


def comparison_for_subset(
    target: np.ndarray,
    dense_prediction: np.ndarray,
    compact_prediction: np.ndarray,
    bootstrap_samples: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    dense_metrics = metrics(target, dense_prediction)
    compact_metrics = metrics(target, compact_prediction)
    dense_frame_mae = np.mean(np.abs(dense_prediction - target), axis=1)
    compact_frame_mae = np.mean(np.abs(compact_prediction - target), axis=1)
    dense_mae = dense_metrics["mae"]
    compact_mae = compact_metrics["mae"]
    relative = (
        100.0 * (compact_mae - dense_mae) / dense_mae
        if dense_mae > 0.0
        else 0.0
    )
    return {
        "count": int(target.shape[0]),
        "dense": dense_metrics,
        "compact": compact_metrics,
        "compact_relative_mae_change_percent": float(relative),
        "paired": bootstrap_paired_mae_difference(
            dense_frame_mae,
            compact_frame_mae,
            bootstrap_samples,
            confidence,
            rng,
        ),
    }


def analyze_sequence(
    target: np.ndarray,
    dense_prediction: np.ndarray,
    compact_prediction: np.ndarray,
    labels: np.ndarray,
    bootstrap_samples: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "overall": comparison_for_subset(
            target,
            dense_prediction,
            compact_prediction,
            bootstrap_samples,
            confidence,
            rng,
        ),
        "regimes": {},
    }
    total_absolute_compact = float(np.sum(np.abs(compact_prediction - target)))
    total_absolute_dense = float(np.sum(np.abs(dense_prediction - target)))
    for index, name in enumerate(REGIMES):
        mask = labels == index
        if not np.any(mask):
            result["regimes"][name] = {"count": 0}
            continue
        item = comparison_for_subset(
            target[mask],
            dense_prediction[mask],
            compact_prediction[mask],
            bootstrap_samples,
            confidence,
            rng,
        )
        item["fraction"] = float(np.mean(mask))
        item["forward_gt_mean"] = float(np.mean(target[mask, 2]))
        item["dense_absolute_error_contribution"] = (
            float(np.sum(np.abs(dense_prediction[mask] - target[mask]))) / total_absolute_dense
            if total_absolute_dense > 0.0
            else 0.0
        )
        item["compact_absolute_error_contribution"] = (
            float(np.sum(np.abs(compact_prediction[mask] - target[mask]))) / total_absolute_compact
            if total_absolute_compact > 0.0
            else 0.0
        )
        result["regimes"][name] = item
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


def write_metrics_csv(path: Path, report: Mapping[str, object]) -> None:
    columns = [
        "sequence", "regime", "count", "fraction", "decoder", "mae", "rmse",
        "mae_x", "mae_y", "mae_z", "rmse_x", "rmse_y", "rmse_z",
        "absolute_error_contribution",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sequence in SEQUENCES:
            for regime in REGIMES:
                item = report["sequences"][sequence]["regimes"][regime]
                for decoder in DECODERS:
                    decoder_metrics = item[decoder]
                    writer.writerow(
                        {
                            "sequence": sequence,
                            "regime": regime,
                            "count": item["count"],
                            "fraction": item["fraction"],
                            "decoder": decoder,
                            "mae": decoder_metrics["mae"],
                            "rmse": decoder_metrics["rmse"],
                            "mae_x": decoder_metrics["axis_mae"]["x"],
                            "mae_y": decoder_metrics["axis_mae"]["y"],
                            "mae_z": decoder_metrics["axis_mae"]["z"],
                            "rmse_x": decoder_metrics["axis_rmse"]["x"],
                            "rmse_y": decoder_metrics["axis_rmse"]["y"],
                            "rmse_z": decoder_metrics["axis_rmse"]["z"],
                            "absolute_error_contribution": item[
                                "{}_absolute_error_contribution".format(decoder)
                            ],
                        }
                    )


def write_paired_csv(path: Path, report: Mapping[str, object]) -> None:
    columns = [
        "sequence", "regime", "count", "compact_minus_dense_mae", "ci_lower",
        "ci_upper", "compact_relative_mae_change_percent",
        "compact_better_frame_fraction", "dense_better_frame_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sequence in SEQUENCES:
            for regime in REGIMES:
                item = report["sequences"][sequence]["regimes"][regime]
                paired = item["paired"]
                writer.writerow(
                    {
                        "sequence": sequence,
                        "regime": regime,
                        "count": item["count"],
                        "compact_minus_dense_mae": paired["compact_minus_dense_mae"],
                        "ci_lower": paired["ci_lower"],
                        "ci_upper": paired["ci_upper"],
                        "compact_relative_mae_change_percent": item[
                            "compact_relative_mae_change_percent"
                        ],
                        "compact_better_frame_fraction": paired[
                            "compact_better_frame_fraction"
                        ],
                        "dense_better_frame_fraction": paired[
                            "dense_better_frame_fraction"
                        ],
                    }
                )


def write_text(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "Dense versus compact translation comparison by motion regime",
        "=" * 108,
        "Shared sequence-09 thresholds: low <= {:.9f}, high > {:.9f}".format(
            report["thresholds"]["low"], report["thresholds"]["high"]
        ),
        "",
        "{:<4} {:<8} {:>7} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "Seq", "Regime", "N", "Dense MAE", "Compact MAE", "Delta", "Delta %", "95% CI"
        ),
        "-" * 108,
    ]
    for sequence in SEQUENCES:
        for regime in REGIMES:
            item = report["sequences"][sequence]["regimes"][regime]
            paired = item["paired"]
            ci = "[{:.6f},{:.6f}]".format(paired["ci_lower"], paired["ci_upper"])
            lines.append(
                "{:<4} {:<8} {:>7d} {:>12.6f} {:>12.6f} {:>12.6f} {:>11.2f}% {:>18}".format(
                    sequence,
                    regime,
                    item["count"],
                    item["dense"]["mae"],
                    item["compact"]["mae"],
                    paired["compact_minus_dense_mae"],
                    item["compact_relative_mae_change_percent"],
                    ci,
                )
            )
    lines.extend(["", "Positive delta means compact is worse; a CI excluding zero indicates a stable paired difference."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(output_dir: Path, report: Mapping[str, object]) -> None:
    x = np.arange(3)
    width = 0.36
    for sequence in SEQUENCES:
        dense = [report["sequences"][sequence]["regimes"][name]["dense"]["mae"] for name in REGIMES]
        compact = [report["sequences"][sequence]["regimes"][name]["compact"]["mae"] for name in REGIMES]
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(x - width / 2.0, dense, width, label="A4 dense")
        plt.bar(x + width / 2.0, compact, width, label="compact")
        plt.xticks(x, REGIMES)
        plt.xlabel("Shared forward-motion regime")
        plt.ylabel("Translation MAE")
        plt.title("Sequence {} dense versus compact".format(sequence))
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(output_dir / "sequence_{}_mae_by_regime.png".format(sequence)), dpi=180)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, sequence in zip(axes, SEQUENCES):
        deltas = []
        lower = []
        upper = []
        for name in REGIMES:
            paired = report["sequences"][sequence]["regimes"][name]["paired"]
            delta = paired["compact_minus_dense_mae"]
            deltas.append(delta)
            lower.append(delta - paired["ci_lower"])
            upper.append(paired["ci_upper"] - delta)
        axis.errorbar(
            np.arange(3),
            deltas,
            yerr=np.asarray([lower, upper]),
            fmt="o",
            capsize=5,
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks(np.arange(3))
        axis.set_xticklabels(REGIMES)
        axis.set_title("Sequence {}".format(sequence))
        axis.set_xlabel("Motion regime")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Compact MAE − dense MAE")
    fig.tight_layout()
    fig.savefig(str(output_dir / "paired_mae_difference_confidence_intervals.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    data = {
        "dense": {
            "09": load_predictions(args.dense_sequence_09),
            "10": load_predictions(args.dense_sequence_10),
        },
        "compact": {
            "09": load_predictions(args.compact_sequence_09),
            "10": load_predictions(args.compact_sequence_10),
        },
    }
    for sequence in SEQUENCES:
        verify_pair(
            sequence,
            data["dense"][sequence],
            data["compact"][sequence],
            args.target_tolerance,
        )

    target09 = data["dense"]["09"]["target"]
    low = float(np.quantile(target09[:, 2], args.low_quantile))
    high = float(np.quantile(target09[:, 2], args.high_quantile))
    rng = np.random.RandomState(args.seed)
    report: Dict[str, object] = {
        "inputs": {
            "dense_sequence_09": str(args.dense_sequence_09.resolve()),
            "dense_sequence_10": str(args.dense_sequence_10.resolve()),
            "compact_sequence_09": str(args.compact_sequence_09.resolve()),
            "compact_sequence_10": str(args.compact_sequence_10.resolve()),
        },
        "thresholds": {
            "source": "sequence_09_ground_truth_translation_z",
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
            "low": low,
            "high": high,
        },
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "confidence": float(args.confidence),
            "seed": int(args.seed),
        },
        "sequences": {},
    }
    for sequence in SEQUENCES:
        target = data["dense"][sequence]["target"]
        labels = assign_regimes(target[:, 2], low, high)
        report["sequences"][sequence] = analyze_sequence(
            target,
            data["dense"][sequence]["prediction"],
            data["compact"][sequence]["prediction"],
            labels,
            args.bootstrap_samples,
            args.confidence,
            rng,
        )

    report = sanitize(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dense_compact_regime_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_metrics_csv(args.output_dir / "decoder_regime_metrics.csv", report)
    write_paired_csv(args.output_dir / "paired_regime_comparison.csv", report)
    write_text(args.output_dir / "dense_compact_regime_comparison.txt", report)
    plot_comparison(args.output_dir, report)
    print((args.output_dir / "dense_compact_regime_comparison.txt").read_text(encoding="utf-8"))
    print("Saved paired decoder comparison to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
