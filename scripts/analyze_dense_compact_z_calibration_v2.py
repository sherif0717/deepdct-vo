#!/usr/bin/env python3
"""Diagnose dense/compact forward-translation shrinkage and recalibration.

Affine calibration is fitted on sequence 09 only and then frozen for sequence
10.  The script never uses sequence 10 to select parameters or regime bins.
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
DECODERS = ("dense", "compact")
SEQUENCES = ("09", "10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit dense/compact z-axis affine calibration on sequence 09 and "
            "evaluate the frozen calibration on sequence 10."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dense-sequence-09", required=True, type=Path)
    parser.add_argument("--dense-sequence-10", required=True, type=Path)
    parser.add_argument("--compact-sequence-09", required=True, type=Path)
    parser.add_argument("--compact-sequence-10", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument(
        "--low-regime-quantile",
        type=float,
        default=1.0 / 3.0,
        help="Sequence-09 GT-z quantile defining the frozen low regime.",
    )
    parser.add_argument(
        "--high-regime-quantile",
        type=float,
        default=2.0 / 3.0,
        help="Sequence-09 GT-z quantile defining the frozen high regime.",
    )
    parser.add_argument("--hexbin-gridsize", type=int, default=45)
    parser.add_argument("--alignment-tolerance", type=float, default=1.0e-8)
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
    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2.")
    if not 0.0 < args.low_regime_quantile < args.high_regime_quantile < 1.0:
        raise ValueError(
            "Require 0 < --low-regime-quantile < --high-regime-quantile < 1."
        )
    if args.hexbin_gridsize < 5:
        raise ValueError("--hexbin-gridsize must be at least 5.")
    if args.alignment_tolerance < 0.0:
        raise ValueError("--alignment-tolerance cannot be negative.")


def load_predictions(path: Path) -> Dict[str, object]:
    gt_columns = ["translation_gt_{}".format(axis) for axis in AXES]
    pred_columns = ["translation_pred_{}".format(axis) for axis in AXES]
    targets: List[List[float]] = []
    predictions: List[List[float]] = []
    keys: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(gt_columns + pred_columns).difference(fields)
        if missing:
            raise KeyError("{} is missing columns {}".format(path, sorted(missing)))
        has_keys = {"sequence", "frame_prev", "frame_curr"}.issubset(fields)
        for index, row in enumerate(reader):
            targets.append([float(row[name]) for name in gt_columns])
            predictions.append([float(row[name]) for name in pred_columns])
            if has_keys:
                keys.append((row["sequence"], row["frame_prev"], row["frame_curr"]))
            else:
                keys.append(("", str(index), str(index + 1)))
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim != 2 or target.shape[1:] != (3,) or target.shape[0] == 0:
        raise ValueError("Unexpected target shape {} in {}".format(target.shape, path))
    if prediction.shape != target.shape:
        raise ValueError("Prediction shape does not match target shape in {}".format(path))
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Non-finite translation value in {}".format(path))
    return {"target": target, "prediction": prediction, "keys": keys}


def verify_alignment(
    sequence: str,
    dense: Mapping[str, object],
    compact: Mapping[str, object],
    tolerance: float,
) -> None:
    if dense["target"].shape != compact["target"].shape:
        raise ValueError("Sequence {} dense/compact shapes differ.".format(sequence))
    if dense["keys"] != compact["keys"]:
        raise ValueError("Sequence {} dense/compact frame ordering differs.".format(sequence))
    difference = float(np.max(np.abs(dense["target"] - compact["target"])))
    if difference > tolerance:
        raise ValueError(
            "Sequence {} targets differ by {:.3e}, exceeding {:.3e}.".format(
                sequence, difference, tolerance
            )
        )


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def fit_affine(predicted: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    design = np.column_stack([predicted, np.ones(predicted.size, dtype=np.float64)])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def vector_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "axis_mae": {
            axis: float(np.mean(np.abs(error[:, index])))
            for index, axis in enumerate(AXES)
        },
        "axis_rmse": {
            axis: float(np.sqrt(np.mean(error[:, index] ** 2)))
            for index, axis in enumerate(AXES)
        },
    }


def z_diagnostics(target_z: np.ndarray, predicted_z: np.ndarray) -> Dict[str, object]:
    error = predicted_z - target_z
    slope, intercept = fit_affine(target_z, predicted_z)
    target_std = float(np.std(target_z, ddof=1))
    prediction_std = float(np.std(predicted_z, ddof=1))
    pearson = correlation(target_z, predicted_z)
    gt_sum_squares = float(np.sum((target_z - np.mean(target_z)) ** 2))
    residual_sum_squares = float(np.sum(error ** 2))
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {
        "count": int(target_z.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "error_variance": float(np.var(error, ddof=1)),
        "pearson": pearson,
        "spearman": correlation(rank_average(target_z), rank_average(predicted_z)),
        "correlation_r_squared": pearson ** 2,
        "predictive_r_squared": (
            1.0 - residual_sum_squares / gt_sum_squares
            if gt_sum_squares > 0.0 else float("nan")
        ),
        "prediction_on_gt_slope": slope,
        "prediction_on_gt_intercept": intercept,
        "gt_mean": float(np.mean(target_z)),
        "prediction_mean": float(np.mean(predicted_z)),
        "gt_std": target_std,
        "prediction_std": prediction_std,
        "prediction_to_gt_std_ratio": (
            prediction_std / target_std if target_std > 0.0 else float("nan")
        ),
        "prediction_to_gt_variance_ratio": (
            (prediction_std ** 2) / (target_std ** 2)
            if target_std > 0.0 else float("nan")
        ),
        "gt_quantiles": {
            "{:.2f}".format(q): float(value)
            for q, value in zip(quantiles, np.quantile(target_z, quantiles))
        },
        "prediction_quantiles": {
            "{:.2f}".format(q): float(value)
            for q, value in zip(quantiles, np.quantile(predicted_z, quantiles))
        },
    }


def quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("GT z has too few distinct values for conditional bins.")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], values, side="right")


def conditional_rows(
    sequence: str,
    decoder: str,
    target_z: np.ndarray,
    raw_z: np.ndarray,
    calibrated_z: np.ndarray,
    edges: np.ndarray,
) -> List[Dict[str, object]]:
    assignments = bin_indices(target_z, edges)
    rows: List[Dict[str, object]] = []
    for index in range(edges.size - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        row: Dict[str, object] = {
            "sequence": sequence,
            "decoder": decoder,
            "bin": index + 1,
            "lower_gt_z": float(edges[index]),
            "upper_gt_z": float(edges[index + 1]),
            "count": int(np.sum(mask)),
            "mean_gt_z": float(np.mean(target_z[mask])),
        }
        for label, values in (("raw", raw_z), ("calibrated", calibrated_z)):
            error = values[mask] - target_z[mask]
            row[label + "_mean_prediction_z"] = float(np.mean(values[mask]))
            row[label + "_bias_z"] = float(np.mean(error))
            row[label + "_error_variance_z"] = float(
                np.var(error, ddof=1) if np.sum(mask) > 1 else 0.0
            )
            row[label + "_mae_z"] = float(np.mean(np.abs(error)))
            row[label + "_rmse_z"] = float(np.sqrt(np.mean(error ** 2)))
        rows.append(row)
    return rows


def fixed_regime_masks(
    target_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> Dict[str, np.ndarray]:
    return {
        "low": target_z <= low_threshold,
        "medium": (target_z > low_threshold) & (target_z <= high_threshold),
        "high": target_z > high_threshold,
    }


def fixed_regime_rows(
    sequence: str,
    decoder: str,
    target_z: np.ndarray,
    raw_z: np.ndarray,
    calibrated_z: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    masks = fixed_regime_masks(target_z, low_threshold, high_threshold)
    for regime in ("low", "medium", "high"):
        mask = masks[regime]
        count = int(np.sum(mask))
        if count < 2:
            raise ValueError(
                "Sequence {} regime {} has only {} sample(s); at least 2 are required."
                .format(sequence, regime, count)
            )
        raw = z_diagnostics(target_z[mask], raw_z[mask])
        calibrated = z_diagnostics(target_z[mask], calibrated_z[mask])
        row: Dict[str, object] = {
            "sequence": sequence,
            "decoder": decoder,
            "regime": regime,
            "count": count,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        }
        for label, diagnostics in (("raw", raw), ("calibrated", calibrated)):
            for field in (
                "gt_mean", "prediction_mean", "gt_std", "prediction_std",
                "prediction_to_gt_std_ratio", "prediction_to_gt_variance_ratio",
                "pearson", "spearman", "correlation_r_squared",
                "predictive_r_squared", "prediction_on_gt_slope",
                "prediction_on_gt_intercept", "bias", "error_variance", "mae", "rmse",
            ):
                row["{}_{}".format(label, field)] = diagnostics[field]
        rows.append(row)
    return rows


def disagreement_diagnostics(
    target: np.ndarray, dense: np.ndarray, compact: np.ndarray
) -> Dict[str, float]:
    disagreement_z = np.abs(dense[:, 2] - compact[:, 2])
    dense_error = np.abs(dense[:, 2] - target[:, 2])
    compact_error = np.abs(compact[:, 2] - target[:, 2])
    advantage = dense_error - compact_error  # Positive means compact is better.
    return {
        "mean_absolute_z_disagreement": float(np.mean(disagreement_z)),
        "correlation_disagreement_with_compact_advantage": correlation(
            disagreement_z, advantage
        ),
        "compact_better_fraction_z": float(np.mean(advantage > 0.0)),
        "mean_dense_minus_compact_absolute_z_error": float(np.mean(advantage)),
    }


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


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    keys: Sequence[Tuple[str, str, str]],
    target: np.ndarray,
    dense_raw: np.ndarray,
    dense_calibrated: np.ndarray,
    compact_raw: np.ndarray,
    compact_calibrated: np.ndarray,
) -> None:
    fields = ["sequence", "frame_prev", "frame_curr"]
    fields += ["translation_gt_{}".format(axis) for axis in AXES]
    for decoder in DECODERS:
        fields += ["{}_raw_{}".format(decoder, axis) for axis in AXES]
        fields += ["{}_calibrated_{}".format(decoder, axis) for axis in AXES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, key in enumerate(keys):
            row: Dict[str, object] = {
                "sequence": key[0], "frame_prev": key[1], "frame_curr": key[2]
            }
            for axis_index, axis in enumerate(AXES):
                row["translation_gt_{}".format(axis)] = target[row_index, axis_index]
                row["dense_raw_{}".format(axis)] = dense_raw[row_index, axis_index]
                row["dense_calibrated_{}".format(axis)] = dense_calibrated[row_index, axis_index]
                row["compact_raw_{}".format(axis)] = compact_raw[row_index, axis_index]
                row["compact_calibrated_{}".format(axis)] = compact_calibrated[row_index, axis_index]
            writer.writerow(row)


def common_limits(data: Mapping[str, Mapping[str, Mapping[str, object]]]) -> Tuple[float, float]:
    values: List[np.ndarray] = []
    for sequence in SEQUENCES:
        values.append(data["dense"][sequence]["target"][:, 2])
        for decoder in DECODERS:
            values.append(data[decoder][sequence]["prediction"][:, 2])
    combined = np.concatenate(values)
    low, high = np.quantile(combined, [0.002, 0.998])
    padding = max(0.05 * float(high - low), 1.0e-3)
    return float(low - padding), float(high + padding)


def plot_hexbin(
    path: Path,
    data: Mapping[str, Mapping[str, Mapping[str, object]]],
    calibrated: Mapping[str, Mapping[str, np.ndarray]],
    coefficients: Mapping[str, Mapping[str, float]],
    gridsize: int,
    use_calibrated: bool,
) -> None:
    low, high = common_limits(data)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for row, sequence in enumerate(SEQUENCES):
        target_z = data["dense"][sequence]["target"][:, 2]
        for column, decoder in enumerate(DECODERS):
            axis = axes[row, column]
            predicted_z = (
                calibrated[decoder][sequence][:, 2]
                if use_calibrated
                else data[decoder][sequence]["prediction"][:, 2]
            )
            image = axis.hexbin(
                target_z, predicted_z, gridsize=gridsize, mincnt=1,
                cmap="viridis", bins="log", extent=(low, high, low, high),
            )
            axis.plot([low, high], [low, high], "r--", linewidth=1.2, label="identity")
            slope, intercept = fit_affine(target_z, predicted_z)
            x_line = np.asarray([low, high])
            axis.plot(x_line, slope * x_line + intercept, color="orange", linewidth=1.3,
                      label="fit slope={:.3f}".format(slope))
            axis.set_title("{} sequence {}{}".format(
                decoder.capitalize(), sequence, " calibrated" if use_calibrated else " raw"
            ))
            axis.grid(alpha=0.15)
            axis.legend(loc="upper left", fontsize=8)
            figure.colorbar(image, ax=axis, label="log count")
    for axis in axes[-1, :]:
        axis.set_xlabel("Ground-truth translation z")
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted translation z")
    figure.suptitle(
        "Frozen sequence-09 affine calibration" if use_calibrated
        else "Raw dense/compact z calibration",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def write_summary(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "Dense/compact z-axis calibration analysis",
        "=" * 104,
        "Calibration fitted on sequence 09 only; coefficients frozen for sequence 10.",
        "",
        "Affine correction: corrected_z = a * predicted_z + b",
        "-" * 104,
    ]
    for decoder in DECODERS:
        calibration = report["calibration"][decoder]
        lines.append("{:<8} a={:.9f}, b={:.9f}".format(
            decoder, calibration["a"], calibration["b"]
        ))
    lines.extend([
        "",
        "{:<5} {:<8} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12}".format(
            "Seq", "Decoder", "Raw z MAE", "Cal z MAE", "Raw all", "Cal all",
            "Raw slope", "Cal slope"
        ),
        "-" * 104,
    ])
    for sequence in SEQUENCES:
        for decoder in DECODERS:
            item = report["sequences"][sequence][decoder]
            lines.append("{:<5} {:<8} {:>11.6f} {:>11.6f} {:>11.6f} {:>11.6f} {:>12.6f} {:>12.6f}".format(
                sequence, decoder,
                item["raw_z"]["mae"], item["calibrated_z"]["mae"],
                item["raw_vector"]["mae"], item["calibrated_vector"]["mae"],
                item["raw_z"]["prediction_on_gt_slope"],
                item["calibrated_z"]["prediction_on_gt_slope"],
            ))
    thresholds = report["fixed_regimes"]
    lines.extend([
        "",
        "Fixed-regime raw z diagnostics",
        "Low: z <= {:.9f}; medium: {:.9f} < z <= {:.9f}; high: z > {:.9f}".format(
            thresholds["low_threshold"], thresholds["low_threshold"],
            thresholds["high_threshold"], thresholds["high_threshold"],
        ),
        "-" * 138,
        "{:<5} {:<8} {:<7} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9} {:>9}".format(
            "Seq", "Decoder", "Regime", "N", "Pearson", "Corr R2", "Pred R2",
            "Std rat", "Var rat", "Slope", "Bias", "MAE",
        ),
        "-" * 138,
    ])
    for row in report["fixed_regimes"]["rows"]:
        lines.append(
            "{:<5} {:<8} {:<7} {:>6d} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} "
            "{:>9.4f} {:>10.4f} {:>9.4f} {:>9.4f}".format(
                row["sequence"], row["decoder"], row["regime"], row["count"],
                row["raw_pearson"], row["raw_correlation_r_squared"],
                row["raw_predictive_r_squared"],
                row["raw_prediction_to_gt_std_ratio"],
                row["raw_prediction_to_gt_variance_ratio"],
                row["raw_prediction_on_gt_slope"], row["raw_bias"], row["raw_mae"],
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
        decoder: {sequence: load_predictions(paths[decoder][sequence]) for sequence in SEQUENCES}
        for decoder in DECODERS
    }
    for sequence in SEQUENCES:
        verify_alignment(sequence, data["dense"][sequence], data["compact"][sequence],
                         args.alignment_tolerance)

    coefficients: Dict[str, Dict[str, float]] = {}
    calibrated: Dict[str, Dict[str, np.ndarray]] = {decoder: {} for decoder in DECODERS}
    for decoder in DECODERS:
        target09_z = data[decoder]["09"]["target"][:, 2]
        predicted09_z = data[decoder]["09"]["prediction"][:, 2]
        a, b = fit_affine(predicted09_z, target09_z)
        coefficients[decoder] = {"a": a, "b": b}
        for sequence in SEQUENCES:
            prediction = data[decoder][sequence]["prediction"].copy()
            prediction[:, 2] = a * prediction[:, 2] + b
            calibrated[decoder][sequence] = prediction

    calibration_gt_z = data["dense"]["09"]["target"][:, 2]
    bin_edges = quantile_edges(calibration_gt_z, args.num_bins)
    low_threshold = float(np.quantile(calibration_gt_z, args.low_regime_quantile))
    high_threshold = float(np.quantile(calibration_gt_z, args.high_regime_quantile))
    conditional: List[Dict[str, object]] = []
    regime_diagnostics: List[Dict[str, object]] = []
    report: Dict[str, object] = {
        "protocol": {
            "calibration_sequence": "09", "frozen_test_sequence": "10",
            "calibrated_axis": "z", "num_conditional_bins": int(bin_edges.size - 1),
            "conditional_bin_source": "sequence 09 GT-z quantiles",
        },
        "calibration": coefficients,
        "fixed_regimes": {
            "source_sequence": "09",
            "low_quantile": float(args.low_regime_quantile),
            "high_quantile": float(args.high_regime_quantile),
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "rows": regime_diagnostics,
        },
        "sequences": {},
    }
    for sequence in SEQUENCES:
        sequence_report: Dict[str, object] = {}
        target = data["dense"][sequence]["target"]
        for decoder in DECODERS:
            raw = data[decoder][sequence]["prediction"]
            corrected = calibrated[decoder][sequence]
            sequence_report[decoder] = {
                "raw_z": z_diagnostics(target[:, 2], raw[:, 2]),
                "calibrated_z": z_diagnostics(target[:, 2], corrected[:, 2]),
                "raw_vector": vector_metrics(target, raw),
                "calibrated_vector": vector_metrics(target, corrected),
            }
            conditional.extend(conditional_rows(
                sequence, decoder, target[:, 2], raw[:, 2], corrected[:, 2], bin_edges
            ))
            regime_diagnostics.extend(fixed_regime_rows(
                sequence, decoder, target[:, 2], raw[:, 2], corrected[:, 2],
                low_threshold, high_threshold,
            ))
        sequence_report["dense_compact_disagreement"] = disagreement_diagnostics(
            target, data["dense"][sequence]["prediction"],
            data["compact"][sequence]["prediction"]
        )
        report["sequences"][sequence] = sequence_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "conditional_z_bin_metrics.csv", conditional)
    write_rows(args.output_dir / "fixed_regime_z_diagnostics.csv", regime_diagnostics)
    for sequence in SEQUENCES:
        write_predictions(
            args.output_dir / "sequence_{}_z_recalibrated_predictions.csv".format(sequence),
            data["dense"][sequence]["keys"], data["dense"][sequence]["target"],
            data["dense"][sequence]["prediction"], calibrated["dense"][sequence],
            data["compact"][sequence]["prediction"], calibrated["compact"][sequence],
        )
    plot_hexbin(args.output_dir / "raw_z_hexbin.png", data, calibrated, coefficients,
                args.hexbin_gridsize, False)
    plot_hexbin(args.output_dir / "recalibrated_z_hexbin.png", data, calibrated, coefficients,
                args.hexbin_gridsize, True)
    clean_report = finite_json(report)
    with (args.output_dir / "z_calibration_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(clean_report, handle, indent=2, sort_keys=True)
    write_summary(args.output_dir / "z_calibration_analysis.txt", clean_report)
    print((args.output_dir / "z_calibration_analysis.txt").read_text(encoding="utf-8"))
    print("Saved z calibration analysis to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
