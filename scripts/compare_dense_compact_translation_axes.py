#!/usr/bin/env python3
"""Paired per-axis dense-versus-compact translation comparison.

Uses the same sequence-09-derived low/medium/high forward-motion thresholds as
the motion-regime comparison. Positive compact-minus-dense error means the
compact decoder is worse on that axis.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = ("x", "y", "z")
REGIMES = ("low", "medium", "high")
SEQUENCES = ("09", "10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense and compact translation errors per axis and motion regime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dense-sequence-09", type=Path, required=True)
    parser.add_argument("--dense-sequence-10", type=Path, required=True)
    parser.add_argument("--compact-sequence-09", type=Path, required=True)
    parser.add_argument("--compact-sequence-10", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--high-quantile", type=float, default=2.0 / 3.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tolerance", type=float, default=1.0e-8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.dense_sequence_09,
        args.dense_sequence_10,
        args.compact_sequence_09,
        args.compact_sequence_10,
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing prediction CSV: {}".format(path))
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError("Require 0 < low quantile < high quantile < 1.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie between zero and one.")
    if args.target_tolerance < 0.0:
        raise ValueError("--target-tolerance cannot be negative.")


def load_csv(path: Path) -> Dict[str, object]:
    gt_columns = tuple("translation_gt_{}".format(axis) for axis in AXES)
    pred_columns = tuple("translation_pred_{}".format(axis) for axis in AXES)
    targets: List[List[float]] = []
    predictions: List[List[float]] = []
    keys: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(gt_columns + pred_columns).difference(fields)
        if missing:
            raise KeyError("{} is missing {}.".format(path, sorted(missing)))
        has_keys = {"sequence", "frame_prev", "frame_curr"}.issubset(fields)
        for row_index, row in enumerate(reader):
            targets.append([float(row[name]) for name in gt_columns])
            predictions.append([float(row[name]) for name in pred_columns])
            if has_keys:
                keys.append((row["sequence"], row["frame_prev"], row["frame_curr"]))
            else:
                keys.append(("", str(row_index), str(row_index + 1)))
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim != 2 or target.shape[1:] != (3,) or target.shape[0] == 0:
        raise ValueError("Unexpected target shape {} in {}.".format(target.shape, path))
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ in {}.".format(path))
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Non-finite values in {}.".format(path))
    return {"target": target, "prediction": prediction, "keys": keys}


def verify_alignment(
    sequence: str,
    dense: Mapping[str, object],
    compact: Mapping[str, object],
    tolerance: float,
) -> None:
    if dense["target"].shape != compact["target"].shape:
        raise ValueError("Sequence {} sample counts differ.".format(sequence))
    if dense["keys"] != compact["keys"]:
        raise ValueError("Sequence {} frame ordering differs.".format(sequence))
    maximum = float(np.max(np.abs(dense["target"] - compact["target"])))
    if maximum > tolerance:
        raise ValueError(
            "Sequence {} targets differ by {:.3e}, exceeding {:.3e}."
            .format(sequence, maximum, tolerance)
        )


def assign_regimes(forward: np.ndarray, low: float, high: float) -> np.ndarray:
    labels = np.full(forward.shape, 1, dtype=np.int64)
    labels[forward <= low] = 0
    labels[forward > high] = 2
    return labels


def bootstrap_difference(
    dense_absolute: np.ndarray,
    compact_absolute: np.ndarray,
    replicates: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    difference = compact_absolute - dense_absolute
    bootstrap = np.empty(replicates, dtype=np.float64)
    chunk_size = min(250, replicates)
    start = 0
    while start < replicates:
        count = min(chunk_size, replicates - start)
        indices = rng.randint(0, difference.size, size=(count, difference.size))
        bootstrap[start : start + count] = np.mean(difference[indices], axis=1)
        start += count
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    dense_mae = float(np.mean(dense_absolute))
    compact_mae = float(np.mean(compact_absolute))
    return {
        "dense_mae": dense_mae,
        "compact_mae": compact_mae,
        "compact_minus_dense_mae": compact_mae - dense_mae,
        "compact_relative_mae_change_percent": (
            100.0 * (compact_mae - dense_mae) / dense_mae if dense_mae > 0.0 else 0.0
        ),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "compact_better_frame_fraction": float(np.mean(difference < 0.0)),
        "dense_better_frame_fraction": float(np.mean(difference > 0.0)),
        "tie_fraction": float(np.mean(difference == 0.0)),
    }


def axis_comparison(
    target: np.ndarray,
    dense_prediction: np.ndarray,
    compact_prediction: np.ndarray,
    axis_index: int,
    replicates: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    dense_error = dense_prediction[:, axis_index] - target[:, axis_index]
    compact_error = compact_prediction[:, axis_index] - target[:, axis_index]
    result: Dict[str, object] = bootstrap_difference(
        np.abs(dense_error),
        np.abs(compact_error),
        replicates,
        confidence,
        rng,
    )
    result.update(
        {
            "dense_rmse": float(np.sqrt(np.mean(dense_error ** 2))),
            "compact_rmse": float(np.sqrt(np.mean(compact_error ** 2))),
            "dense_bias": float(np.mean(dense_error)),
            "compact_bias": float(np.mean(compact_error)),
            "dense_target_correlation": correlation(
                dense_prediction[:, axis_index], target[:, axis_index]
            ),
            "compact_target_correlation": correlation(
                compact_prediction[:, axis_index], target[:, axis_index]
            ),
        }
    )
    return result


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = math.sqrt(float(np.sum(first ** 2) * np.sum(second ** 2)))
    return float(np.sum(first * second) / denominator) if denominator > 0.0 else 0.0


def analyze_sequence(
    target: np.ndarray,
    dense_prediction: np.ndarray,
    compact_prediction: np.ndarray,
    labels: np.ndarray,
    replicates: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    result: Dict[str, object] = {"regimes": {}}
    for regime_index, regime in enumerate(REGIMES):
        mask = labels == regime_index
        item: Dict[str, object] = {
            "count": int(np.sum(mask)),
            "fraction": float(np.mean(mask)),
            "forward_gt_mean": float(np.mean(target[mask, 2])),
            "axes": {},
        }
        for axis_index, axis in enumerate(AXES):
            item["axes"][axis] = axis_comparison(
                target[mask],
                dense_prediction[mask],
                compact_prediction[mask],
                axis_index,
                replicates,
                confidence,
                rng,
            )
        result["regimes"][regime] = item
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


def write_csv(path: Path, report: Mapping[str, object]) -> None:
    columns = [
        "sequence", "regime", "axis", "count", "dense_mae", "compact_mae",
        "compact_minus_dense_mae", "compact_relative_mae_change_percent",
        "ci_lower", "ci_upper", "dense_rmse", "compact_rmse", "dense_bias",
        "compact_bias", "dense_target_correlation", "compact_target_correlation",
        "compact_better_frame_fraction", "dense_better_frame_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sequence in SEQUENCES:
            for regime in REGIMES:
                regime_item = report["sequences"][sequence]["regimes"][regime]
                for axis in AXES:
                    row = {
                        "sequence": sequence,
                        "regime": regime,
                        "axis": axis,
                        "count": regime_item["count"],
                    }
                    row.update(regime_item["axes"][axis])
                    row.pop("confidence", None)
                    row.pop("tie_fraction", None)
                    writer.writerow(row)


def write_text(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "Dense versus compact per-axis translation comparison",
        "=" * 118,
        "Shared thresholds: low <= {:.9f}, high > {:.9f}".format(
            report["thresholds"]["low"], report["thresholds"]["high"]
        ),
        "",
        "{:<4} {:<8} {:<4} {:>7} {:>11} {:>11} {:>11} {:>10} {:>23}".format(
            "Seq", "Regime", "Axis", "N", "Dense MAE", "Compact", "Delta", "Delta %", "95% CI"
        ),
        "-" * 118,
    ]
    for sequence in SEQUENCES:
        for regime in REGIMES:
            item = report["sequences"][sequence]["regimes"][regime]
            for axis in AXES:
                axis_item = item["axes"][axis]
                ci = "[{:.7f},{:.7f}]".format(axis_item["ci_lower"], axis_item["ci_upper"])
                lines.append(
                    "{:<4} {:<8} {:<4} {:>7d} {:>11.7f} {:>11.7f} {:>11.7f} {:>9.2f}% {:>23}".format(
                        sequence,
                        regime,
                        axis,
                        item["count"],
                        axis_item["dense_mae"],
                        axis_item["compact_mae"],
                        axis_item["compact_minus_dense_mae"],
                        axis_item["compact_relative_mae_change_percent"],
                        ci,
                    )
                )
    lines.extend(
        [
            "",
            "Positive delta means compact is worse on that axis.",
            "A confidence interval excluding zero indicates a stable paired difference.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(output_dir: Path, report: Mapping[str, object]) -> None:
    x = np.arange(3)
    width = 0.36
    for sequence in SEQUENCES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
        for axis_plot, axis_name in zip(axes, AXES):
            dense = [
                report["sequences"][sequence]["regimes"][regime]["axes"][axis_name]["dense_mae"]
                for regime in REGIMES
            ]
            compact = [
                report["sequences"][sequence]["regimes"][regime]["axes"][axis_name]["compact_mae"]
                for regime in REGIMES
            ]
            axis_plot.bar(x - width / 2.0, dense, width, label="A4 dense")
            axis_plot.bar(x + width / 2.0, compact, width, label="compact")
            axis_plot.set_xticks(x)
            axis_plot.set_xticklabels(REGIMES)
            axis_plot.set_title("{} axis".format(axis_name))
            axis_plot.set_xlabel("Motion regime")
            axis_plot.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Translation MAE")
        axes[-1].legend()
        fig.suptitle("Sequence {} per-axis dense versus compact".format(sequence))
        fig.tight_layout()
        fig.savefig(str(output_dir / "sequence_{}_per_axis_mae.png".format(sequence)), dpi=180)
        plt.close(fig)

    for sequence in SEQUENCES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
        for axis_plot, axis_name in zip(axes, AXES):
            deltas = []
            lower = []
            upper = []
            for regime in REGIMES:
                item = report["sequences"][sequence]["regimes"][regime]["axes"][axis_name]
                delta = item["compact_minus_dense_mae"]
                deltas.append(delta)
                lower.append(delta - item["ci_lower"])
                upper.append(item["ci_upper"] - delta)
            axis_plot.errorbar(x, deltas, yerr=np.asarray([lower, upper]), fmt="o", capsize=5)
            axis_plot.axhline(0.0, color="black", linewidth=1.0)
            axis_plot.set_xticks(x)
            axis_plot.set_xticklabels(REGIMES)
            axis_plot.set_title("{} axis".format(axis_name))
            axis_plot.set_xlabel("Motion regime")
            axis_plot.grid(alpha=0.25)
        axes[0].set_ylabel("Compact MAE − dense MAE")
        fig.suptitle("Sequence {} paired per-axis MAE differences".format(sequence))
        fig.tight_layout()
        fig.savefig(str(output_dir / "sequence_{}_per_axis_paired_differences.png".format(sequence)), dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    data = {
        "dense": {
            "09": load_csv(args.dense_sequence_09),
            "10": load_csv(args.dense_sequence_10),
        },
        "compact": {
            "09": load_csv(args.compact_sequence_09),
            "10": load_csv(args.compact_sequence_10),
        },
    }
    for sequence in SEQUENCES:
        verify_alignment(
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
    with (args.output_dir / "per_axis_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_csv(args.output_dir / "per_axis_comparison.csv", report)
    write_text(args.output_dir / "per_axis_comparison.txt", report)
    plot_outputs(args.output_dir, report)
    print((args.output_dir / "per_axis_comparison.txt").read_text(encoding="utf-8"))
    print("Saved per-axis comparison to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
