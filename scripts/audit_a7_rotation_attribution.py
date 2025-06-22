#!/usr/bin/env python3
"""Attribute A7 trajectory error to predicted Euler-rotation axes.

The script consumes ``frame_predictions.csv`` produced by
``scripts/evaluate_deepdct_vo.py``.  It does not run the network.  Instead, it
reconstructs the trajectory for all eight combinations of predicted (P) and
ground-truth (G) x/y/z relative-rotation components while always retaining the
same predicted directional translations.

This is an oracle diagnostic, not a deployable inference method.  It answers:
"How much trajectory error disappears when a particular predicted rotation
axis is replaced by its ground-truth value?"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Evaluator frame_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for attribution tables, trajectories, and plots.",
    )
    parser.add_argument(
        "--euler-order",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Must match label generation and evaluation (default: xyz).",
    )
    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help="Interpret CSV rotation values as degrees instead of radians.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale predicted translations during reconstruction (default: 1).",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Optional expected sequence ID, such as 09 or 10.",
    )
    return parser.parse_args()


def require_columns(fieldnames: Sequence[str] | None, required: Iterable[str]) -> None:
    present = set(fieldnames or ())
    missing = sorted(set(required) - present)
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))


def read_predictions(path: Path, expected_sequence: str | None) -> Dict[str, object]:
    columns = [
        "sequence", "frame_prev", "frame_curr",
        *[f"rotation_gt_{axis}" for axis in AXES],
        *[f"rotation_pred_{axis}" for axis in AXES],
        *[f"translation_gt_{axis}" for axis in AXES],
        *[f"translation_pred_{axis}" for axis in AXES],
    ]
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, columns)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No prediction rows found in {path}.")

    sequences = {str(row["sequence"]).zfill(2) for row in rows}
    if len(sequences) != 1:
        raise ValueError(f"Expected one sequence, found {sorted(sequences)}.")
    sequence = next(iter(sequences))
    if expected_sequence is not None and sequence != str(expected_sequence).zfill(2):
        raise ValueError(
            f"Sequence mismatch: CSV contains {sequence}, expected "
            f"{str(expected_sequence).zfill(2)}."
        )

    frame_prev = np.asarray([int(row["frame_prev"]) for row in rows], dtype=np.int64)
    frame_curr = np.asarray([int(row["frame_curr"]) for row in rows], dtype=np.int64)
    order = np.lexsort((frame_curr, frame_prev))
    frame_prev, frame_curr = frame_prev[order], frame_curr[order]
    if np.any(frame_curr != frame_prev + 1):
        raise ValueError("Every row must describe a consecutive frame pair.")
    if len(rows) > 1 and np.any(frame_prev[1:] != frame_curr[:-1]):
        raise ValueError("Prediction rows do not form one continuous trajectory.")

    def matrix(prefix: str) -> np.ndarray:
        values = np.asarray(
            [[float(rows[index][f"{prefix}_{axis}"]) for axis in AXES] for index in order],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{prefix} contains non-finite values.")
        return values

    return {
        "sequence": sequence,
        "frame_prev": frame_prev,
        "frame_curr": frame_curr,
        "rotation_gt": matrix("rotation_gt"),
        "rotation_pred": matrix("rotation_pred"),
        "translation_gt": matrix("translation_gt"),
        "translation_pred": matrix("translation_pred"),
    }


def rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_matrix(euler: Sequence[float], order: str, degrees: bool) -> np.ndarray:
    x, y, z = (float(value) for value in euler)
    if degrees:
        x, y, z = map(math.radians, (x, y, z))
    if order == "xyz":
        return rz(z) @ ry(y) @ rx(x)
    return rx(x) @ ry(y) @ rz(z)


def project_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def rotation_sqrt(rotation: np.ndarray) -> np.ndarray:
    rotation = project_so3(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return np.eye(3)
    if abs(math.pi - angle) > 1e-7:
        axis = np.asarray(
            [rotation[2, 1] - rotation[1, 2],
             rotation[0, 2] - rotation[2, 0],
             rotation[1, 0] - rotation[0, 1]],
            dtype=np.float64,
        ) / (2.0 * math.sin(angle))
        axis /= np.linalg.norm(axis)
    else:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
        largest = int(np.argmax(axis))
        if largest == 0:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif largest == 1:
            axis[0] = math.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        else:
            axis[0] = math.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
            axis[1] = math.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        axis /= np.linalg.norm(axis)
    half = 0.5 * angle
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return project_so3(np.eye(3) + math.sin(half) * skew + (1.0 - math.cos(half)) * (skew @ skew))


def integrate(rotations: np.ndarray, translations: np.ndarray, order: str, degrees: bool) -> np.ndarray:
    if rotations.shape != translations.shape or rotations.ndim != 2 or rotations.shape[1] != 3:
        raise ValueError("Rotation and translation arrays must both have shape [N, 3].")
    trajectory = np.repeat(np.eye(4)[None, :, :], rotations.shape[0] + 1, axis=0)
    for index, (euler, directional_translation) in enumerate(zip(rotations, translations)):
        relative_rotation = project_so3(euler_matrix(euler, order, degrees))
        current_rotation = project_so3(trajectory[index, :3, :3])
        next_rotation = project_so3(current_rotation @ relative_rotation)
        delta_world = rotation_sqrt(relative_rotation).T @ next_rotation @ directional_translation
        relative_translation = current_rotation.T @ delta_world
        relative_transform = np.eye(4)
        relative_transform[:3, :3] = relative_rotation
        relative_transform[:3, 3] = relative_translation
        trajectory[index + 1] = trajectory[index] @ relative_transform
        trajectory[index + 1, :3, :3] = project_so3(trajectory[index + 1, :3, :3])
    return trajectory


def trajectory_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt_pos, pred_pos = gt[:, :3, 3], pred[:, :3, 3]
    errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
    gt_path = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum())
    endpoint = float(np.linalg.norm(pred_pos[-1] - gt_pos[-1]))
    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_max": float(np.max(errors)),
        "endpoint_error": endpoint,
        "endpoint_error_percent": 100.0 * endpoint / gt_path if gt_path > 0 else float("nan"),
        "gt_path_length": gt_path,
        "pred_path_length": float(np.linalg.norm(np.diff(pred_pos, axis=0), axis=1).sum()),
    }


def condition_name(gt_axes: frozenset[str]) -> str:
    return "".join("G" if axis in gt_axes else "P" for axis in AXES)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_kitti(path: Path, trajectory: np.ndarray) -> None:
    np.savetxt(path, trajectory[:, :3, :].reshape(-1, 12), fmt="%.12e")


def shapley_attribution(values: Mapping[frozenset[str], float]) -> Dict[str, float]:
    """Allocate total ATE reduction from PPP to GGG among replacement axes."""
    factorial = math.factorial
    count = len(AXES)
    output: Dict[str, float] = {}
    for axis in AXES:
        contribution = 0.0
        others = [candidate for candidate in AXES if candidate != axis]
        for size in range(len(others) + 1):
            for subset_tuple in itertools.combinations(others, size):
                subset = frozenset(subset_tuple)
                weight = factorial(size) * factorial(count - size - 1) / factorial(count)
                contribution += weight * (values[subset] - values[subset | {axis}])
        output[axis] = float(contribution)
    return output


def make_plots(
    output_dir: Path,
    sequence: str,
    gt: np.ndarray,
    trajectories: Mapping[frozenset[str], np.ndarray],
) -> None:
    selected = [frozenset(), frozenset({"x"}), frozenset({"y"}), frozenset({"z"}), frozenset(AXES)]
    colors = {"PPP": "#d62728", "GPP": "#9467bd", "PGP": "#ff7f0e", "PPG": "#2ca02c", "GGG": "#1f77b4"}
    labels = {"PPP": "all predicted", "GPP": "GT x", "PGP": "GT y", "PPG": "GT z", "GGG": "all GT rotation"}

    fig, axis = plt.subplots(figsize=(9, 7))
    gt_pos = gt[:, :3, 3]
    axis.plot(gt_pos[:, 0], gt_pos[:, 2], color="black", linewidth=2.0, label="GT pose")
    for subset in selected:
        name = condition_name(subset)
        pos = trajectories[subset][:, :3, 3]
        axis.plot(pos[:, 0], pos[:, 2], color=colors[name], linewidth=1.25, label=labels[name])
    axis.set_title(f"Sequence {sequence}: rotation-axis oracle attribution")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_xz_axis_attribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for subset in selected:
        name = condition_name(subset)
        pred_pos = trajectories[subset][:, :3, 3]
        error = np.linalg.norm(pred_pos - gt_pos, axis=1)
        axis.plot(error, color=colors[name], linewidth=1.1, label=labels[name])
    axis.set_title(f"Sequence {sequence}: accumulated position error")
    axis.set_xlabel("trajectory frame")
    axis.set_ylabel("position error [m]")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "position_error_by_frame.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    if not math.isfinite(args.translation_scale) or args.translation_scale <= 0.0:
        raise ValueError("--translation-scale must be finite and positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = read_predictions(args.predictions, args.sequence)
    sequence = str(data["sequence"])
    rotation_gt = np.asarray(data["rotation_gt"])
    rotation_pred = np.asarray(data["rotation_pred"])
    translation_gt = np.asarray(data["translation_gt"])
    translation_pred = np.asarray(data["translation_pred"]) * args.translation_scale

    gt_trajectory = integrate(rotation_gt, translation_gt, args.euler_order, args.angles_in_degrees)
    save_kitti(args.output_dir / "ground_truth_trajectory.txt", gt_trajectory)

    trajectories: Dict[frozenset[str], np.ndarray] = {}
    ate_by_subset: Dict[frozenset[str], float] = {}
    metric_rows = []
    for size in range(4):
        for subset_tuple in itertools.combinations(AXES, size):
            subset = frozenset(subset_tuple)
            rotations = rotation_pred.copy()
            for axis_index, axis in enumerate(AXES):
                if axis in subset:
                    rotations[:, axis_index] = rotation_gt[:, axis_index]
            trajectory = integrate(rotations, translation_pred, args.euler_order, args.angles_in_degrees)
            metrics = trajectory_metrics(gt_trajectory, trajectory)
            name = condition_name(subset)
            trajectories[subset] = trajectory
            ate_by_subset[subset] = metrics["ate_rmse"]
            save_kitti(args.output_dir / f"trajectory_{name}.txt", trajectory)
            metric_rows.append({
                "sequence": sequence,
                "condition": name,
                "gt_axes": "".join(axis for axis in AXES if axis in subset) or "none",
                **metrics,
            })
    metric_rows.sort(key=lambda row: str(row["condition"]))
    write_csv(args.output_dir / "rotation_condition_metrics.csv", metric_rows)

    error = rotation_pred - rotation_gt
    unit_scale = 1.0 if args.angles_in_degrees else 180.0 / math.pi
    axis_rows = []
    for index, axis in enumerate(AXES):
        axis_error = error[:, index]
        gt_axis, pred_axis = rotation_gt[:, index], rotation_pred[:, index]
        corr = float(np.corrcoef(gt_axis, pred_axis)[0, 1]) if np.std(gt_axis) > 0 and np.std(pred_axis) > 0 else float("nan")
        axis_rows.append({
            "sequence": sequence,
            "axis": axis,
            "bias": float(np.mean(axis_error)),
            "mae": float(np.mean(np.abs(axis_error))),
            "rmse": float(np.sqrt(np.mean(axis_error ** 2))),
            "bias_degrees": float(np.mean(axis_error) * unit_scale),
            "cumulative_signed_error_degrees": float(np.sum(axis_error) * unit_scale),
            "gt_std": float(np.std(gt_axis)),
            "pred_std": float(np.std(pred_axis)),
            "std_ratio": float(np.std(pred_axis) / np.std(gt_axis)) if np.std(gt_axis) > 0 else float("nan"),
            "correlation": corr,
            "one_axis_replacement_ate": ate_by_subset[frozenset({axis})],
            "one_axis_ate_reduction": ate_by_subset[frozenset()] - ate_by_subset[frozenset({axis})],
        })
    write_csv(args.output_dir / "rotation_axis_statistics.csv", axis_rows)

    shapley = shapley_attribution(ate_by_subset)
    total_reduction = ate_by_subset[frozenset()] - ate_by_subset[frozenset(AXES)]
    shapley_rows = []
    for axis in AXES:
        shapley_rows.append({
            "sequence": sequence,
            "axis": axis,
            "shapley_ate_reduction": shapley[axis],
            "share_of_total_reduction_percent": 100.0 * shapley[axis] / total_reduction if abs(total_reduction) > 1e-12 else float("nan"),
        })
    write_csv(args.output_dir / "rotation_axis_shapley_attribution.csv", shapley_rows)

    summary = {
        "sequence": sequence,
        "samples": int(rotation_gt.shape[0]),
        "source_predictions": str(args.predictions.resolve()),
        "euler_order": args.euler_order,
        "angles_in_degrees": bool(args.angles_in_degrees),
        "translation_scale": float(args.translation_scale),
        "condition_key": "P=predicted rotation axis; G=ground-truth rotation axis; order=x,y,z",
        "ate_all_predicted_rotation": ate_by_subset[frozenset()],
        "ate_all_ground_truth_rotation": ate_by_subset[frozenset(AXES)],
        "total_oracle_ate_reduction": total_reduction,
        "shapley_ate_reduction": shapley,
        "warning": "Oracle attribution diagnostic; GT-axis substitutions are not deployable inference.",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")

    make_plots(args.output_dir, sequence, gt_trajectory, trajectories)

    print("=" * 88)
    print(f"A7 rotation-axis attribution: sequence {sequence}")
    print("Condition order: x y z; P=predicted, G=ground truth")
    print("-" * 88)
    for row in sorted(metric_rows, key=lambda item: float(item["ate_rmse"]), reverse=True):
        print(
            f"{row['condition']}: ATE RMSE={float(row['ate_rmse']):10.6f} m  "
            f"endpoint={float(row['endpoint_error']):10.6f} m"
        )
    print("-" * 88)
    print("Interaction-aware Shapley attribution of PPP -> GGG ATE reduction:")
    for row in shapley_rows:
        print(
            f"axis {row['axis']}: {float(row['shapley_ate_reduction']):10.6f} m  "
            f"({float(row['share_of_total_reduction_percent']):7.2f}%)"
        )
    print(f"Outputs: {args.output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
