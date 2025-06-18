#!/usr/bin/env python3
"""Paired circular-block bootstrap for dense/compact z correlations.

The low/medium/high GT-z thresholds are estimated from sequence 09 only and
then frozen for sequence 10. Bootstrap blocks are sampled from each complete
chronological trajectory before regime masks are applied. Dense and compact
predictions always use the same sampled transition indices.

The primary statistic is delta-r = r_dense - r_compact. Positive values favor
the dense decoder's within-regime Pearson correlation; negative values favor
the compact decoder. Percentile confidence intervals are primary. Reported
p-values are approximate centered-bootstrap tail probabilities and are Holm
adjusted across the six sequence-by-regime delta-r tests at each block length.
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


SEQUENCES = ("09", "10")
REGIMES = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dense and compact within-regime z correlations using a "
            "paired full-trajectory circular-block bootstrap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dense-sequence-09", required=True, type=Path)
    parser.add_argument("--dense-sequence-10", required=True, type=Path)
    parser.add_argument("--compact-sequence-09", required=True, type=Path)
    parser.add_argument("--compact-sequence-10", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block-lengths", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--low-regime-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--high-regime-quantile", type=float, default=2.0 / 3.0)
    parser.add_argument("--alignment-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
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
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one.")
    if not 0.0 < args.low_regime_quantile < args.high_regime_quantile < 1.0:
        raise ValueError(
            "Require 0 < --low-regime-quantile < --high-regime-quantile < 1."
        )
    if not args.block_lengths or any(length <= 0 for length in args.block_lengths):
        raise ValueError("Every block length must be positive.")
    if len(set(args.block_lengths)) != len(args.block_lengths):
        raise ValueError("--block-lengths cannot contain duplicates.")
    if not 0.0 < args.minimum_valid_fraction <= 1.0:
        raise ValueError("--minimum-valid-fraction must lie in (0, 1].")
    if args.alignment_tolerance < 0.0:
        raise ValueError("--alignment-tolerance cannot be negative.")


def load_predictions(path: Path) -> Dict[str, object]:
    required = {
        "translation_gt_z",
        "translation_pred_z",
    }
    target: List[float] = []
    prediction: List[float] = []
    keys: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required.difference(fields)
        if missing:
            raise KeyError("{} is missing columns {}".format(path, sorted(missing)))
        has_keys = {"sequence", "frame_prev", "frame_curr"}.issubset(fields)
        for index, row in enumerate(reader):
            target.append(float(row["translation_gt_z"]))
            prediction.append(float(row["translation_pred_z"]))
            if has_keys:
                keys.append((row["sequence"], row["frame_prev"], row["frame_curr"]))
            else:
                keys.append(("", str(index), str(index + 1)))
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if target_array.ndim != 1 or target_array.size < 3:
        raise ValueError("Expected at least three transitions in {}".format(path))
    if prediction_array.shape != target_array.shape:
        raise ValueError("Prediction shape differs from GT shape in {}".format(path))
    if not np.all(np.isfinite(target_array)) or not np.all(np.isfinite(prediction_array)):
        raise FloatingPointError("Non-finite z value in {}".format(path))
    return {"target_z": target_array, "prediction_z": prediction_array, "keys": keys}


def verify_alignment(
    sequence: str,
    dense: Mapping[str, object],
    compact: Mapping[str, object],
    tolerance: float,
) -> None:
    if dense["target_z"].shape != compact["target_z"].shape:
        raise ValueError("Sequence {} dense/compact sample counts differ.".format(sequence))
    if dense["keys"] != compact["keys"]:
        raise ValueError("Sequence {} dense/compact frame ordering differs.".format(sequence))
    difference = float(np.max(np.abs(dense["target_z"] - compact["target_z"])))
    if difference > tolerance:
        raise ValueError(
            "Sequence {} GT z differs by {:.3e}, exceeding {:.3e}.".format(
                sequence, difference, tolerance
            )
        )


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3:
        return float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def regime_masks(
    target_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> Dict[str, np.ndarray]:
    return {
        "low": target_z <= low_threshold,
        "medium": (target_z > low_threshold) & (target_z <= high_threshold),
        "high": target_z > high_threshold,
    }


def circular_block_indices(
    sample_count: int,
    block_length: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    block_count = int(math.ceil(float(sample_count) / float(block_length)))
    starts = rng.randint(0, sample_count, size=block_count)
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, None] + offsets[None, :]) % sample_count
    return indices.reshape(-1)[:sample_count]


def bootstrap_sequence(
    target_z: np.ndarray,
    dense_z: np.ndarray,
    compact_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    block_length: int,
    replicates: int,
    rng: np.random.RandomState,
) -> Dict[str, Dict[str, np.ndarray]]:
    output = {
        regime: {
            "dense_r": np.full(replicates, np.nan, dtype=np.float64),
            "compact_r": np.full(replicates, np.nan, dtype=np.float64),
            "delta_r": np.full(replicates, np.nan, dtype=np.float64),
            "count": np.zeros(replicates, dtype=np.int64),
        }
        for regime in REGIMES
    }
    for replicate in range(replicates):
        indices = circular_block_indices(target_z.size, block_length, rng)
        sampled_target = target_z[indices]
        sampled_dense = dense_z[indices]
        sampled_compact = compact_z[indices]
        masks = regime_masks(sampled_target, low_threshold, high_threshold)
        for regime in REGIMES:
            mask = masks[regime]
            count = int(np.sum(mask))
            output[regime]["count"][replicate] = count
            if count < 3:
                continue
            dense_r = pearson(sampled_target[mask], sampled_dense[mask])
            compact_r = pearson(sampled_target[mask], sampled_compact[mask])
            if math.isfinite(dense_r) and math.isfinite(compact_r):
                output[regime]["dense_r"][replicate] = dense_r
                output[regime]["compact_r"][replicate] = compact_r
                output[regime]["delta_r"][replicate] = dense_r - compact_r
    return output


def percentile_interval(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return float(lower), float(upper)


def approximate_centered_p(
    bootstrap_delta: np.ndarray,
    observed_delta: float,
) -> float:
    finite = bootstrap_delta[np.isfinite(bootstrap_delta)]
    if finite.size == 0 or not math.isfinite(observed_delta):
        return float("nan")
    centered = finite - observed_delta
    exceedances = int(np.sum(np.abs(centered) >= abs(observed_delta)))
    return float((exceedances + 1.0) / (finite.size + 1.0))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    adjusted = [float("nan")] * len(p_values)
    finite_indices = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    family_size = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def interval_excludes_zero(lower: float, upper: float) -> bool:
    return bool(math.isfinite(lower) and math.isfinite(upper) and (lower > 0.0 or upper < 0.0))


def analyze_block_length(
    data: Mapping[str, Mapping[str, Mapping[str, object]]],
    low_threshold: float,
    high_threshold: float,
    block_length: int,
    replicates: int,
    confidence: float,
    minimum_valid_fraction: float,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sequence_index, sequence in enumerate(SEQUENCES):
        target_z = data["dense"][sequence]["target_z"]
        dense_z = data["dense"][sequence]["prediction_z"]
        compact_z = data["compact"][sequence]["prediction_z"]
        observed_masks = regime_masks(target_z, low_threshold, high_threshold)
        rng = np.random.RandomState(seed + 1000003 * block_length + 10007 * sequence_index)
        boot = bootstrap_sequence(
            target_z, dense_z, compact_z, low_threshold, high_threshold,
            block_length, replicates, rng,
        )
        for regime in REGIMES:
            mask = observed_masks[regime]
            observed_count = int(np.sum(mask))
            dense_r = pearson(target_z[mask], dense_z[mask])
            compact_r = pearson(target_z[mask], compact_z[mask])
            delta_r = dense_r - compact_r
            valid = np.isfinite(boot[regime]["delta_r"])
            valid_count = int(np.sum(valid))
            valid_fraction = float(valid_count / float(replicates))
            if valid_fraction < minimum_valid_fraction:
                raise RuntimeError(
                    "Only {:.1%} valid replicates for sequence {}, regime {}, block {}; "
                    "minimum is {:.1%}.".format(
                        valid_fraction, sequence, regime, block_length,
                        minimum_valid_fraction,
                    )
                )
            dense_lower, dense_upper = percentile_interval(
                boot[regime]["dense_r"], confidence
            )
            compact_lower, compact_upper = percentile_interval(
                boot[regime]["compact_r"], confidence
            )
            delta_lower, delta_upper = percentile_interval(
                boot[regime]["delta_r"], confidence
            )
            valid_counts = boot[regime]["count"][valid]
            rows.append({
                "block_length": block_length,
                "sequence": sequence,
                "regime": regime,
                "observed_count": observed_count,
                "dense_r": dense_r,
                "dense_ci_lower": dense_lower,
                "dense_ci_upper": dense_upper,
                "compact_r": compact_r,
                "compact_ci_lower": compact_lower,
                "compact_ci_upper": compact_upper,
                "delta_r": delta_r,
                "delta_ci_lower": delta_lower,
                "delta_ci_upper": delta_upper,
                "delta_ci_excludes_zero": interval_excludes_zero(delta_lower, delta_upper),
                "favored_decoder": (
                    "dense" if delta_r > 0.0 else "compact" if delta_r < 0.0 else "tie"
                ),
                "approximate_raw_p": approximate_centered_p(
                    boot[regime]["delta_r"], delta_r
                ),
                "valid_replicates": valid_count,
                "valid_fraction": valid_fraction,
                "bootstrap_regime_count_min": int(np.min(valid_counts)),
                "bootstrap_regime_count_median": float(np.median(valid_counts)),
                "bootstrap_regime_count_max": int(np.max(valid_counts)),
            })
    raw_p = [float(row["approximate_raw_p"]) for row in rows]
    adjusted = holm_adjust(raw_p)
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
    return rows


def transfer_rows(primary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    block_lengths = sorted({int(row["block_length"]) for row in primary_rows})
    for block_length in block_lengths:
        for regime in REGIMES:
            selected = {
                str(row["sequence"]): row
                for row in primary_rows
                if int(row["block_length"]) == block_length and row["regime"] == regime
            }
            row09 = selected["09"]
            row10 = selected["10"]
            delta09 = float(row09["delta_r"])
            delta10 = float(row10["delta_r"])
            same_sign = bool(delta09 * delta10 > 0.0)
            both_exclude = bool(
                row09["delta_ci_excludes_zero"] and row10["delta_ci_excludes_zero"]
            )
            result.append({
                "block_length": block_length,
                "regime": regime,
                "sequence_09_delta_r": delta09,
                "sequence_10_delta_r": delta10,
                "same_sign_across_sequences": same_sign,
                "both_cis_exclude_zero": both_exclude,
                "stable_transfer_evidence": bool(same_sign and both_exclude),
            })
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def plot_delta(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors = {"09": "tab:blue", "10": "tab:orange"}
    offsets = {"09": -0.8, "10": 0.8}
    for axis, regime in zip(axes, REGIMES):
        for sequence in SEQUENCES:
            selected = sorted(
                (
                    row for row in rows
                    if row["regime"] == regime and row["sequence"] == sequence
                ),
                key=lambda row: int(row["block_length"]),
            )
            x = np.asarray(
                [float(row["block_length"]) + offsets[sequence] for row in selected]
            )
            y = np.asarray([float(row["delta_r"]) for row in selected])
            lower = np.asarray([float(row["delta_ci_lower"]) for row in selected])
            upper = np.asarray([float(row["delta_ci_upper"]) for row in selected])
            axis.errorbar(
                x, y, yerr=np.vstack([y - lower, upper - y]), fmt="o-",
                capsize=3, color=colors[sequence], label="sequence {}".format(sequence),
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title("{} GT-z regime".format(regime.capitalize()))
        axis.set_xlabel("Circular block length")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Delta r = dense r - compact r")
    axes[-1].legend(loc="best")
    figure.suptitle("Paired block-bootstrap decoder correlation difference")
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def write_summary(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    transfers: Sequence[Mapping[str, object]],
    low_threshold: float,
    high_threshold: float,
    confidence: float,
) -> None:
    lines = [
        "Dense/compact paired block-bootstrap correlation analysis",
        "=" * 150,
        "Frozen sequence-09 regimes: low <= {:.9f}; medium <= {:.9f}; high > {:.9f}".format(
            low_threshold, high_threshold, high_threshold
        ),
        "Delta r = dense Pearson r - compact Pearson r; positive favors dense.",
        "Intervals are {:.1f}% paired circular-block percentile intervals.".format(
            100.0 * confidence
        ),
        "P-values are approximate centered-bootstrap tails; Holm family = six delta-r tests per block length.",
        "",
        "{:<5} {:<3} {:<7} {:>5} {:>9} {:>9} {:>9} {:>23} {:>10} {:>10}".format(
            "Block", "Seq", "Regime", "N", "Dense r", "Compact", "Delta r",
            "Delta CI", "Raw p", "Holm p"
        ),
        "-" * 150,
    ]
    for row in rows:
        lines.append(
            "{:<5d} {:<3} {:<7} {:>5d} {:>9.4f} {:>9.4f} {:>9.4f} "
            "[{:>9.4f},{:>9.4f}] {:>10.4g} {:>10.4g}".format(
                int(row["block_length"]), row["sequence"], row["regime"],
                int(row["observed_count"]), float(row["dense_r"]),
                float(row["compact_r"]), float(row["delta_r"]),
                float(row["delta_ci_lower"]), float(row["delta_ci_upper"]),
                float(row["approximate_raw_p"]), float(row["holm_adjusted_p"]),
            )
        )
    lines.extend([
        "",
        "Cross-sequence transfer gate",
        "-" * 100,
        "{:<5} {:<7} {:>12} {:>12} {:>11} {:>13} {:>10}".format(
            "Block", "Regime", "Seq09 delta", "Seq10 delta", "Same sign",
            "Both CI excl", "Stable"
        ),
        "-" * 100,
    ])
    for row in transfers:
        lines.append(
            "{:<5d} {:<7} {:>12.5f} {:>12.5f} {:>11} {:>13} {:>10}".format(
                int(row["block_length"]), row["regime"],
                float(row["sequence_09_delta_r"]), float(row["sequence_10_delta_r"]),
                str(row["same_sign_across_sequences"]),
                str(row["both_cis_exclude_zero"]),
                str(row["stable_transfer_evidence"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = {
        "dense": {"09": args.dense_sequence_09, "10": args.dense_sequence_10},
        "compact": {"09": args.compact_sequence_09, "10": args.compact_sequence_10},
    }
    data = {
        decoder: {
            sequence: load_predictions(paths[decoder][sequence])
            for sequence in SEQUENCES
        }
        for decoder in ("dense", "compact")
    }
    for sequence in SEQUENCES:
        verify_alignment(
            sequence, data["dense"][sequence], data["compact"][sequence],
            args.alignment_tolerance,
        )
    calibration_gt_z = data["dense"]["09"]["target_z"]
    low_threshold = float(np.quantile(calibration_gt_z, args.low_regime_quantile))
    high_threshold = float(np.quantile(calibration_gt_z, args.high_regime_quantile))

    rows: List[Dict[str, object]] = []
    for block_length in args.block_lengths:
        rows.extend(analyze_block_length(
            data, low_threshold, high_threshold, block_length,
            args.bootstrap_samples, args.confidence, args.minimum_valid_fraction,
            args.seed,
        ))
    transfers = transfer_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "block_bootstrap_delta_r.csv", rows)
    write_csv(args.output_dir / "transfer_stability.csv", transfers)
    plot_delta(args.output_dir / "delta_r_block_length_sensitivity.png", rows)
    report = finite_json({
        "protocol": {
            "bootstrap": "paired full-trajectory circular moving blocks",
            "bootstrap_samples": args.bootstrap_samples,
            "block_lengths": args.block_lengths,
            "confidence": args.confidence,
            "seed": args.seed,
            "p_value": "approximate centered-bootstrap two-sided tail",
            "multiple_comparisons": "Holm across six delta-r tests within each block length",
        },
        "fixed_regimes": {
            "source_sequence": "09",
            "low_quantile": args.low_regime_quantile,
            "high_quantile": args.high_regime_quantile,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "primary_rows": rows,
        "transfer_rows": transfers,
    })
    with (args.output_dir / "block_bootstrap_correlation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_summary(
        args.output_dir / "block_bootstrap_correlation.txt", rows, transfers,
        low_threshold, high_threshold, args.confidence,
    )
    print((args.output_dir / "block_bootstrap_correlation.txt").read_text(
        encoding="utf-8"
    ))
    print("Saved block-bootstrap analysis to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
