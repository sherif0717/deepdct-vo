#!/usr/bin/env python3
"""Simulate deployable dense/compact translation routing.

Two routers are calibrated using sequence 09 only and then frozen:

1. whole_vector: choose the complete dense or compact translation vector;
2. z_only: always retain dense x/y and route only the forward z component.

Routing uses the dense decoder's predicted z value, never ground truth. An
oracle GT-z result is reported only as a diagnostic upper bound. When the
project evaluator is importable, the script also computes GT-rotation
trajectory metrics for sequence 09 and sequence 10.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = ("x", "y", "z")
SEQUENCES = ("09", "10")
MODES = ("whole_vector", "z_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate on sequence 09 and test dense/compact routing on sequence 10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dense-sequence-09", type=Path, required=True)
    parser.add_argument("--dense-sequence-10", type=Path, required=True)
    parser.add_argument("--compact-sequence-09", type=Path, required=True)
    parser.add_argument("--compact-sequence-10", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--calibration-objective",
        choices=["mae", "z_mae"],
        default="mae",
        help="Sequence-09 objective minimized when selecting each predicted-z threshold.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--euler-order", choices=["xyz", "zyx"], default="xyz")
    parser.add_argument("--angles-in-degrees", action="store_true")
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip GT-rotation trajectory reconstruction.",
    )
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
    if not 0.0 < args.low_quantile < 1.0:
        raise ValueError("--low-quantile must lie strictly between zero and one.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one.")
    if args.target_tolerance < 0.0:
        raise ValueError("--target-tolerance cannot be negative.")


def load_csv(path: Path) -> Dict[str, object]:
    translation_gt_names = tuple("translation_gt_{}".format(axis) for axis in AXES)
    translation_pred_names = tuple("translation_pred_{}".format(axis) for axis in AXES)
    rotation_gt_names = tuple("rotation_gt_{}".format(axis) for axis in AXES)
    targets: List[List[float]] = []
    predictions: List[List[float]] = []
    rotations: List[List[float]] = []
    frame_keys: List[Tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = set(translation_gt_names + translation_pred_names + rotation_gt_names)
        missing = required.difference(fields)
        if missing:
            raise KeyError("{} is missing columns {}.".format(path, sorted(missing)))
        has_frame_keys = {"sequence", "frame_prev", "frame_curr"}.issubset(fields)
        for row_index, row in enumerate(reader):
            targets.append([float(row[name]) for name in translation_gt_names])
            predictions.append([float(row[name]) for name in translation_pred_names])
            rotations.append([float(row[name]) for name in rotation_gt_names])
            if has_frame_keys:
                frame_keys.append((row["sequence"], row["frame_prev"], row["frame_curr"]))
            else:
                frame_keys.append(("", str(row_index), str(row_index + 1)))

    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    rotation = np.asarray(rotations, dtype=np.float64)
    if target.ndim != 2 or target.shape[1:] != (3,) or target.shape[0] == 0:
        raise ValueError("Unexpected target shape {} from {}.".format(target.shape, path))
    if prediction.shape != target.shape or rotation.shape != target.shape:
        raise ValueError("Prediction/rotation shapes do not match targets in {}.".format(path))
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Non-finite translation values in {}.".format(path))
    if not np.all(np.isfinite(rotation)):
        raise FloatingPointError("Non-finite rotation values in {}.".format(path))
    return {
        "target": target,
        "prediction": prediction,
        "rotation_gt": rotation,
        "frame_keys": frame_keys,
    }


def verify_alignment(
    sequence: str,
    dense: Mapping[str, object],
    compact: Mapping[str, object],
    tolerance: float,
) -> None:
    if dense["target"].shape != compact["target"].shape:
        raise ValueError("Sequence {} sample shapes differ.".format(sequence))
    if dense["frame_keys"] != compact["frame_keys"]:
        raise ValueError("Sequence {} frame ordering differs.".format(sequence))
    target_difference = float(np.max(np.abs(dense["target"] - compact["target"])))
    rotation_difference = float(
        np.max(np.abs(dense["rotation_gt"] - compact["rotation_gt"]))
    )
    if target_difference > tolerance:
        raise ValueError(
            "Sequence {} targets differ by {:.3e}, exceeding {:.3e}."
            .format(sequence, target_difference, tolerance)
        )
    if rotation_difference > tolerance:
        raise ValueError(
            "Sequence {} GT rotations differ by {:.3e}, exceeding {:.3e}."
            .format(sequence, rotation_difference, tolerance)
        )


def combine_predictions(
    dense: np.ndarray,
    compact: np.ndarray,
    use_compact: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "whole_vector":
        return np.where(use_compact[:, None], compact, dense)
    if mode == "z_only":
        result = dense.copy()
        result[use_compact, 2] = compact[use_compact, 2]
        return result
    raise ValueError("Unsupported routing mode: {!r}.".format(mode))


def objective_value(target: np.ndarray, prediction: np.ndarray, objective: str) -> float:
    if objective == "mae":
        return float(np.mean(np.abs(prediction - target)))
    if objective == "z_mae":
        return float(np.mean(np.abs(prediction[:, 2] - target[:, 2])))
    raise ValueError("Unsupported calibration objective: {!r}.".format(objective))


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if unique.size == 1:
        return np.asarray([unique[0] - 1.0, unique[0], unique[0] + 1.0])
    middle = 0.5 * (unique[:-1] + unique[1:])
    span = max(float(unique[-1] - unique[0]), 1.0)
    return np.concatenate(
        [
            np.asarray([unique[0] - span]),
            middle,
            np.asarray([unique[-1] + span]),
        ]
    )


def calibrate_threshold(
    target: np.ndarray,
    dense: np.ndarray,
    compact: np.ndarray,
    mode: str,
    objective: str,
) -> Dict[str, float]:
    candidates = threshold_candidates(dense[:, 2])
    best_threshold = float(candidates[0])
    best_value = float("inf")
    best_compact_fraction = 0.0
    tolerance = 1.0e-15
    for threshold in candidates:
        use_compact = dense[:, 2] > threshold
        prediction = combine_predictions(dense, compact, use_compact, mode)
        value = objective_value(target, prediction, objective)
        compact_fraction = float(np.mean(use_compact))
        if value < best_value - tolerance:
            best_value = value
            best_threshold = float(threshold)
            best_compact_fraction = compact_fraction
        elif abs(value - best_value) <= tolerance:
            # Prefer the less complex route when objectives tie.
            if compact_fraction < best_compact_fraction:
                best_threshold = float(threshold)
                best_compact_fraction = compact_fraction
    return {
        "threshold": best_threshold,
        "calibration_objective_value": best_value,
        "sequence_09_compact_fraction": best_compact_fraction,
        "candidate_count": int(candidates.size),
    }


def metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    error = prediction - target
    axis_mae = np.mean(np.abs(error), axis=0)
    axis_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    axis_bias = np.mean(error, axis=0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "l2_rmse": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "axis_mae": {axis: float(axis_mae[index]) for index, axis in enumerate(AXES)},
        "axis_rmse": {axis: float(axis_rmse[index]) for index, axis in enumerate(AXES)},
        "axis_bias": {axis: float(axis_bias[index]) for index, axis in enumerate(AXES)},
    }


def paired_bootstrap(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    replicates: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    candidate_frame = np.mean(np.abs(candidate - target), axis=1)
    baseline_frame = np.mean(np.abs(baseline - target), axis=1)
    difference = candidate_frame - baseline_frame
    bootstrap = np.empty(replicates, dtype=np.float64)
    chunk = min(250, replicates)
    start = 0
    while start < replicates:
        count = min(chunk, replicates - start)
        indices = rng.randint(0, difference.size, size=(count, difference.size))
        bootstrap[start : start + count] = np.mean(difference[indices], axis=1)
        start += count
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    return {
        "candidate_minus_baseline_mae": float(np.mean(difference)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "candidate_better_frame_fraction": float(np.mean(difference < 0.0)),
    }


def evaluate_router(
    target: np.ndarray,
    dense: np.ndarray,
    compact: np.ndarray,
    threshold: float,
    mode: str,
    gt_low_threshold: float,
    bootstrap_samples: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    predicted_route = dense[:, 2] > threshold
    oracle_route = target[:, 2] > gt_low_threshold
    hybrid = combine_predictions(dense, compact, predicted_route, mode)
    oracle = combine_predictions(dense, compact, oracle_route, mode)
    result: Dict[str, object] = {
        "predicted_route": {
            "threshold": float(threshold),
            "compact_count": int(np.sum(predicted_route)),
            "compact_fraction": float(np.mean(predicted_route)),
            "metrics": metrics(target, hybrid),
            "versus_dense": paired_bootstrap(
                target, hybrid, dense, bootstrap_samples, confidence, rng
            ),
            "versus_compact": paired_bootstrap(
                target, hybrid, compact, bootstrap_samples, confidence, rng
            ),
        },
        "oracle_gt_route": {
            "gt_low_threshold": float(gt_low_threshold),
            "compact_count": int(np.sum(oracle_route)),
            "compact_fraction": float(np.mean(oracle_route)),
            "metrics": metrics(target, oracle),
        },
    }
    return result, hybrid, predicted_route, oracle


def trajectory_metrics(
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    output_dir: Path,
    sequence: str,
    euler_order: str,
    angles_in_degrees: bool,
) -> Dict[str, object]:
    try:
        from evaluate_deepdct_vo import (
            compute_trajectory_metrics,
            integrate_relative_poses,
            save_kitti_trajectory,
        )
    except ImportError as error:
        raise ImportError(
            "Trajectory analysis requires this script to be installed in scripts/ "
            "beside evaluate_deepdct_vo.py. Use --skip-trajectory to omit it."
        ) from error

    gt_trajectory = integrate_relative_poses(
        rotations=rotation_gt,
        translations=translation_gt,
        euler_order=euler_order,
        angles_in_degrees=angles_in_degrees,
    )
    sequence_dir = output_dir / "sequence_{}".format(sequence)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    save_kitti_trajectory(sequence_dir / "ground_truth_trajectory.txt", gt_trajectory)
    results: Dict[str, object] = {}
    for name, prediction in predictions.items():
        trajectory = integrate_relative_poses(
            rotations=rotation_gt,
            translations=prediction,
            euler_order=euler_order,
            angles_in_degrees=angles_in_degrees,
        )
        results[name] = asdict(
            compute_trajectory_metrics(
                ground_truth_trajectory=gt_trajectory,
                predicted_trajectory=trajectory,
            )
        )
        save_kitti_trajectory(
            sequence_dir / "{}_trajectory.txt".format(name), trajectory
        )
    return results


def write_hybrid_csv(
    path: Path,
    frame_keys: List[Tuple[str, str, str]],
    target: np.ndarray,
    dense: np.ndarray,
    compact: np.ndarray,
    hybrid: np.ndarray,
    use_compact: np.ndarray,
    mode: str,
    threshold: float,
) -> None:
    fields = [
        "sequence", "frame_prev", "frame_curr", "mode", "threshold",
        "use_compact", "dense_pred_z", "compact_pred_z",
    ]
    fields += ["translation_gt_{}".format(axis) for axis in AXES]
    fields += ["translation_dense_{}".format(axis) for axis in AXES]
    fields += ["translation_compact_{}".format(axis) for axis in AXES]
    fields += ["translation_hybrid_{}".format(axis) for axis in AXES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, key in enumerate(frame_keys):
            row: Dict[str, object] = {
                "sequence": key[0],
                "frame_prev": key[1],
                "frame_curr": key[2],
                "mode": mode,
                "threshold": threshold,
                "use_compact": int(use_compact[index]),
                "dense_pred_z": dense[index, 2],
                "compact_pred_z": compact[index, 2],
            }
            for axis_index, axis in enumerate(AXES):
                row["translation_gt_{}".format(axis)] = target[index, axis_index]
                row["translation_dense_{}".format(axis)] = dense[index, axis_index]
                row["translation_compact_{}".format(axis)] = compact[index, axis_index]
                row["translation_hybrid_{}".format(axis)] = hybrid[index, axis_index]
            writer.writerow(row)


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


def write_text(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "Dense/compact predicted-z routing diagnostic",
        "=" * 100,
        "Calibration sequence: 09 only",
        "GT low-motion threshold (oracle only): {:.9f}".format(
            report["calibration"]["gt_low_threshold"]
        ),
        "",
        "{:<14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "Mode", "Threshold", "Seq09 MAE", "Seq10 MAE", "Seq10 Dense", "Seq10 Compact"
        ),
        "-" * 100,
    ]
    dense10 = report["sequences"]["10"]["baselines"]["dense"]["mae"]
    compact10 = report["sequences"]["10"]["baselines"]["compact"]["mae"]
    for mode in MODES:
        lines.append(
            "{:<14} {:>14.9f} {:>14.9f} {:>14.9f} {:>14.9f} {:>14.9f}".format(
                mode,
                report["calibration"][mode]["threshold"],
                report["sequences"]["09"]["routers"][mode]["predicted_route"]["metrics"]["mae"],
                report["sequences"]["10"]["routers"][mode]["predicted_route"]["metrics"]["mae"],
                dense10,
                compact10,
            )
        )
    lines.extend(["", "Oracle upper bounds (GT route; not deployable)", "-" * 100])
    for mode in MODES:
        lines.append(
            "{:<14} seq09 MAE={:.9f}, seq10 MAE={:.9f}".format(
                mode,
                report["sequences"]["09"]["routers"][mode]["oracle_gt_route"]["metrics"]["mae"],
                report["sequences"]["10"]["routers"][mode]["oracle_gt_route"]["metrics"]["mae"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_thresholds(
    output_dir: Path,
    dense09_z: np.ndarray,
    dense10_z: np.ndarray,
    calibration: Mapping[str, object],
) -> None:
    lower = float(min(np.min(dense09_z), np.min(dense10_z)))
    upper = float(max(np.max(dense09_z), np.max(dense10_z)))
    bins = np.linspace(lower, upper, 41) if upper > lower else 40
    plt.figure(figsize=(9, 5))
    plt.hist(dense09_z, bins=bins, density=True, alpha=0.5, label="sequence 09 dense pred z")
    plt.hist(dense10_z, bins=bins, density=True, alpha=0.5, label="sequence 10 dense pred z")
    colors = {"whole_vector": "tab:red", "z_only": "tab:purple"}
    for mode in MODES:
        plt.axvline(
            calibration[mode]["threshold"],
            color=colors[mode],
            linestyle="--",
            label="{} threshold".format(mode),
        )
    plt.xlabel("Dense predicted translation z")
    plt.ylabel("Density")
    plt.title("Sequence-09-calibrated routing thresholds")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "predicted_z_routing_thresholds.png"), dpi=180)
    plt.close()


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
    dense09 = data["dense"]["09"]["prediction"]
    compact09 = data["compact"]["09"]["prediction"]
    gt_low_threshold = float(np.quantile(target09[:, 2], args.low_quantile))
    calibration: Dict[str, object] = {
        "sequence": "09",
        "objective": args.calibration_objective,
        "gt_low_threshold": gt_low_threshold,
    }
    for mode in MODES:
        calibration[mode] = calibrate_threshold(
            target09,
            dense09,
            compact09,
            mode,
            args.calibration_objective,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    report: Dict[str, object] = {
        "inputs": {
            "dense_sequence_09": str(args.dense_sequence_09.resolve()),
            "dense_sequence_10": str(args.dense_sequence_10.resolve()),
            "compact_sequence_09": str(args.compact_sequence_09.resolve()),
            "compact_sequence_10": str(args.compact_sequence_10.resolve()),
        },
        "calibration": calibration,
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "confidence": float(args.confidence),
            "seed": int(args.seed),
        },
        "sequences": {},
    }

    for sequence in SEQUENCES:
        target = data["dense"][sequence]["target"]
        dense = data["dense"][sequence]["prediction"]
        compact = data["compact"][sequence]["prediction"]
        sequence_result: Dict[str, object] = {
            "baselines": {
                "dense": metrics(target, dense),
                "compact": metrics(target, compact),
            },
            "routers": {},
        }
        trajectory_inputs: Dict[str, np.ndarray] = {
            "dense": dense,
            "compact": compact,
        }
        for mode in MODES:
            threshold = calibration[mode]["threshold"]
            result, hybrid, route, oracle = evaluate_router(
                target,
                dense,
                compact,
                threshold,
                mode,
                gt_low_threshold,
                args.bootstrap_samples,
                args.confidence,
                rng,
            )
            sequence_result["routers"][mode] = result
            trajectory_inputs["{}_predicted_route".format(mode)] = hybrid
            trajectory_inputs["{}_oracle_route".format(mode)] = oracle
            write_hybrid_csv(
                args.output_dir / "sequence_{}_{}_hybrid_predictions.csv".format(sequence, mode),
                data["dense"][sequence]["frame_keys"],
                target,
                dense,
                compact,
                hybrid,
                route,
                mode,
                threshold,
            )
        if not args.skip_trajectory:
            sequence_result["gt_rotation_trajectory_metrics"] = trajectory_metrics(
                data["dense"][sequence]["rotation_gt"],
                target,
                trajectory_inputs,
                args.output_dir,
                sequence,
                args.euler_order,
                args.angles_in_degrees,
            )
        report["sequences"][sequence] = sequence_result

    report = sanitize(report)
    with (args.output_dir / "routing_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_text(args.output_dir / "routing_diagnostic.txt", report)
    plot_thresholds(
        args.output_dir,
        data["dense"]["09"]["prediction"][:, 2],
        data["dense"]["10"]["prediction"][:, 2],
        calibration,
    )
    print((args.output_dir / "routing_diagnostic.txt").read_text(encoding="utf-8"))
    print("Saved routing diagnostic to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
