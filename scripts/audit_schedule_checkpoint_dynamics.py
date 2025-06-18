#!/usr/bin/env python3
"""Audit A6 schedule-checkpoint prediction dynamics without model inference.

The audit compares epochs 30, 90, 120, and 180 on KITTI sequences 09/10.
It uses existing frame_predictions.csv files, freezes motion regimes from the
sequence-09 GT z distribution, and keeps GT-rotation Model-T diagnostics
separate from predicted-rotation full-VO metrics.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys

import numpy as np


BOUNDARIES = ((30, 32), (90, 16), (120, 8), (180, 2))
AXES = ("x", "y", "z")
REGIMES = ("low", "medium", "high")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-dir",
        default="experiments/track_a/paper_reproduction/a6_paper_schedule_120x120",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--regime-thresholds", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="Fixed GT-z thresholds. Default: sequence-09 terciles.")
    p.add_argument("--autocorrelation-lags", nargs="+", type=int,
                   default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--window-lengths", nargs="+", type=int,
                   default=[10, 20, 50, 100])
    p.add_argument("--segment-lengths", nargs="+", type=float,
                   default=[100, 200, 300, 400, 500, 600, 700, 800])
    p.add_argument("--segment-step", type=int, default=10)
    p.add_argument("--euler-order", choices=("xyz", "zyx"), default="xyz")
    p.add_argument("--angles-in-degrees", action="store_true")
    p.add_argument("--minimum-translation-norm", type=float, default=1e-9)
    return p.parse_args()


def load_metric_library(repo_root):
    scripts = os.path.join(repo_root, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import compare_paper_and_kitti_translation_metrics as library
    except ImportError as exc:
        raise SystemExit(
            "Missing scripts/compare_paper_and_kitti_translation_metrics.py: {}".format(exc)
        )
    return library


def prediction_path(experiment, epoch, batch, sequence):
    return os.path.join(
        experiment, "boundary_evaluations",
        "epoch_{:03d}_batch_{:02d}".format(epoch, batch),
        "sequence_{}".format(sequence), "frame_predictions.csv")


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def predictive_r2(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    keep = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[keep], pred[keep]
    denominator = float(np.sum((gt - np.mean(gt)) ** 2))
    if gt.size < 2 or denominator <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((pred - gt) ** 2)) / denominator


def distribution_distance(a, b):
    """One-dimensional empirical Wasserstein-1 distance without SciPy."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    count = max(len(a), len(b), 2)
    quantiles = np.linspace(0.0, 1.0, count)
    aq = np.interp(quantiles, np.linspace(0.0, 1.0, len(a)), a)
    bq = np.interp(quantiles, np.linspace(0.0, 1.0, len(b)), b)
    return float(np.mean(np.abs(aq - bq)))


def regime_labels(z, thresholds):
    low, high = thresholds
    result = np.empty(len(z), dtype=np.int64)
    result[z <= low] = 0
    result[(z > low) & (z <= high)] = 1
    result[z > high] = 2
    return result


def signal_metrics(gt, pred):
    error = pred - gt
    gt_variance = float(np.var(gt))
    pred_variance = float(np.var(pred))
    pearson = safe_pearson(gt, pred)
    return {
        "count": int(len(gt)),
        "gt_mean": float(np.mean(gt)), "pred_mean": float(np.mean(pred)),
        "gt_std": float(np.std(gt)), "pred_std": float(np.std(pred)),
        "variance_ratio": pred_variance / gt_variance if gt_variance > 1e-15 else float("nan"),
        "bias": float(np.mean(error)), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson": pearson,
        "correlation_r2": pearson ** 2 if math.isfinite(pearson) else float("nan"),
        "predictive_r2": predictive_r2(gt, pred),
    }


def covariance_decomposition(gt, pred, labels):
    gt_mean = float(np.mean(gt))
    pred_mean = float(np.mean(pred))
    total = float(np.mean((gt - gt_mean) * (pred - pred_mean)))
    between = 0.0
    within = 0.0
    for code in range(3):
        keep = labels == code
        if not np.any(keep):
            continue
        weight = float(np.mean(keep))
        group_gt = gt[keep]
        group_pred = pred[keep]
        mean_gt = float(np.mean(group_gt))
        mean_pred = float(np.mean(group_pred))
        between += weight * (mean_gt - gt_mean) * (mean_pred - pred_mean)
        within += weight * float(np.mean((group_gt - mean_gt) * (group_pred - mean_pred)))
    residual_gt = gt.copy()
    residual_pred = pred.copy()
    for code in range(3):
        keep = labels == code
        if np.any(keep):
            residual_gt[keep] -= np.mean(gt[keep])
            residual_pred[keep] -= np.mean(pred[keep])
    return {
        "total_covariance": total, "between_covariance": between,
        "within_covariance": within,
        "between_fraction": between / total if abs(total) > 1e-15 else float("nan"),
        "within_fraction": within / total if abs(total) > 1e-15 else float("nan"),
        "residual_pearson": safe_pearson(residual_gt, residual_pred),
        "decomposition_error": total - between - within,
    }


def autocorrelation(values, lag):
    values = np.asarray(values, dtype=np.float64)
    if lag <= 0 or lag >= len(values):
        return float("nan")
    return safe_pearson(values[:-lag], values[lag:])


def window_bias_rows(error, lengths, common):
    rows = []
    for length in lengths:
        for axis, name in enumerate(AXES):
            windows = []
            for start in range(0, len(error) - length + 1, length):
                windows.append(float(np.mean(error[start:start + length, axis])))
            values = np.asarray(windows, dtype=np.float64)
            rows.append(dict(common, window_length=length, axis=name,
                             windows=int(len(values)),
                             mean_window_bias=safe_mean(values),
                             mean_absolute_window_bias=safe_mean(np.abs(values)),
                             rms_window_bias=float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan"),
                             maximum_absolute_window_bias=float(np.max(np.abs(values))) if values.size else float("nan")))
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    experiment = os.path.abspath(args.experiment_dir)
    repo_root = os.path.abspath(os.path.join(experiment, "../../../.."))
    output = os.path.abspath(args.output_dir or os.path.join(experiment, "checkpoint_dynamics_audit"))
    if args.segment_step < 1:
        raise SystemExit("--segment-step must be positive")
    if any(x <= 0 for x in args.autocorrelation_lags + args.window_lengths):
        raise SystemExit("Lags and window lengths must be positive")
    library = load_metric_library(repo_root)
    data = {}
    for epoch, batch in BOUNDARIES:
        for sequence in ("09", "10"):
            path = prediction_path(experiment, epoch, batch, sequence)
            if not os.path.isfile(path):
                raise SystemExit("Missing prediction CSV: {}".format(path))
            data[(epoch, sequence)] = library.read_predictions(path)
    os.makedirs(output, exist_ok=True)

    reference = data[(30, "09")]
    thresholds = tuple(args.regime_thresholds or np.quantile(
        reference["translation_gt"][:, 2], [1.0 / 3.0, 2.0 / 3.0]).tolist())
    if thresholds[0] >= thresholds[1]:
        raise SystemExit("Regime thresholds must be strictly increasing")

    alignment_rows = []
    for sequence in ("09", "10"):
        base = data[(30, sequence)]
        for epoch, _ in BOUNDARIES:
            item = data[(epoch, sequence)]
            if item["keys"] != base["keys"]:
                raise SystemExit("Frame alignment differs at epoch {}, sequence {}".format(epoch, sequence))
            gt_diff = float(np.max(np.abs(item["translation_gt"] - base["translation_gt"])))
            rot_diff = float(np.max(np.abs(item["rotation_gt"] - base["rotation_gt"])))
            if gt_diff > 1e-10 or rot_diff > 1e-10:
                raise SystemExit("GT targets differ across checkpoints at epoch {}, sequence {}".format(epoch, sequence))
            alignment_rows.append({"epoch": epoch, "sequence": sequence,
                                   "translation_gt_max_difference": gt_diff,
                                   "rotation_gt_max_difference": rot_diff})

    distribution_rows, signal_rows, covariance_rows = [], [], []
    autocorrelation_rows, cumulative_rows, window_rows = [], [], []
    kitti_rows, kitti_length_rows = [], []
    cumulative_series = {}

    class IntegrationArgs(object):
        pass
    integration_args = IntegrationArgs()
    integration_args.euler_order = args.euler_order
    integration_args.angles_in_degrees = args.angles_in_degrees

    for epoch, batch in BOUNDARIES:
        for sequence in ("09", "10"):
            item = data[(epoch, sequence)]
            gt = item["translation_gt"]
            pred = item["translation_pred"]
            error = pred - gt
            labels = regime_labels(gt[:, 2], thresholds)
            common = {"epoch": epoch, "batch_size_stage": batch, "sequence": sequence}

            for axis, name in enumerate(AXES):
                gt_axis, pred_axis = gt[:, axis], pred[:, axis]
                distribution_rows.append(dict(common, axis=name,
                    gt_min=float(np.min(gt_axis)), gt_max=float(np.max(gt_axis)),
                    gt_mean=float(np.mean(gt_axis)), gt_std=float(np.std(gt_axis)),
                    pred_min=float(np.min(pred_axis)), pred_max=float(np.max(pred_axis)),
                    pred_mean=float(np.mean(pred_axis)), pred_std=float(np.std(pred_axis)),
                    pred_inside_gt_range_fraction=float(np.mean((pred_axis >= np.min(gt_axis)) & (pred_axis <= np.max(gt_axis)))),
                    wasserstein_1=distribution_distance(gt_axis, pred_axis)))
                for scope, mask in [("pooled", np.ones(len(gt), dtype=bool))] + [
                        (REGIMES[code], labels == code) for code in range(3)]:
                    metrics = signal_metrics(gt_axis[mask], pred_axis[mask])
                    signal_rows.append(dict(common, scope=scope, axis=name, **metrics))
                for lag in args.autocorrelation_lags:
                    autocorrelation_rows.append(dict(common, axis=name, lag=lag,
                                                     error_autocorrelation=autocorrelation(error[:, axis], lag)))
                cumulative = np.cumsum(error[:, axis])
                cumulative_series[(epoch, sequence, name)] = cumulative
                cumulative_rows.append(dict(common, axis=name,
                    final_cumulative_error=float(cumulative[-1]),
                    maximum_absolute_cumulative_error=float(np.max(np.abs(cumulative))),
                    rms_cumulative_error=float(np.sqrt(np.mean(cumulative ** 2)))))
            error_norm = np.linalg.norm(error, axis=1)
            for lag in args.autocorrelation_lags:
                autocorrelation_rows.append(dict(common, axis="l2_norm", lag=lag,
                                                 error_autocorrelation=autocorrelation(error_norm, lag)))
            window_rows.extend(window_bias_rows(error, args.window_lengths, common))
            covariance_rows.append(dict(common, axis="z", **covariance_decomposition(gt[:, 2], pred[:, 2], labels)))

            gt_poses = library.integrate(item["rotation_gt"], gt, integration_args)
            for rotation_mode in ("ground_truth", "predicted"):
                rotations = item["rotation_gt"] if rotation_mode == "ground_truth" else item["rotation_pred"]
                pred_poses = library.integrate(rotations, pred, integration_args)
                by_length, aggregate = library.kitti_segments(
                    gt_poses, pred_poses, args.segment_lengths, args.segment_step)
                metric_common = dict(common, trajectory_rotation=rotation_mode)
                kitti_rows.append(dict(metric_common, **aggregate))
                for row in by_length:
                    kitti_length_rows.append(dict(metric_common, **row))

    attribution_rows = []
    metric_attribution_rows = []
    for sequence in ("09", "10"):
        early = data[(120, sequence)]
        late = data[(180, sequence)]
        translation_change = late["translation_pred"] - early["translation_pred"]
        rotation_change = late["rotation_pred"] - early["rotation_pred"]
        for axis, name in enumerate(AXES):
            attribution_rows.append({
                "sequence": sequence, "quantity": "translation_prediction_change",
                "axis": name, "mean_change": float(np.mean(translation_change[:, axis])),
                "mae_change": float(np.mean(np.abs(translation_change[:, axis]))),
                "rmse_change": float(np.sqrt(np.mean(translation_change[:, axis] ** 2))),
            })
            multiplier = 1.0 if args.angles_in_degrees else 180.0 / math.pi
            attribution_rows.append({
                "sequence": sequence, "quantity": "rotation_prediction_change_degrees",
                "axis": name, "mean_change": multiplier * float(np.mean(rotation_change[:, axis])),
                "mae_change": multiplier * float(np.mean(np.abs(rotation_change[:, axis]))),
                "rmse_change": multiplier * float(np.sqrt(np.mean(rotation_change[:, axis] ** 2))),
            })
        gt120 = next(r for r in kitti_rows if r["epoch"] == 120 and r["sequence"] == sequence
                     and r["trajectory_rotation"] == "ground_truth")
        gt180 = next(r for r in kitti_rows if r["epoch"] == 180 and r["sequence"] == sequence
                     and r["trajectory_rotation"] == "ground_truth")
        pred120 = next(r for r in kitti_rows if r["epoch"] == 120 and r["sequence"] == sequence
                       and r["trajectory_rotation"] == "predicted")
        pred180 = next(r for r in kitti_rows if r["epoch"] == 180 and r["sequence"] == sequence
                       and r["trajectory_rotation"] == "predicted")
        metric_attribution_rows.append({
            "sequence": sequence,
            "gt_rotation_translation_delta_percent": gt180["translation_percent"] - gt120["translation_percent"],
            "predicted_rotation_translation_delta_percent": pred180["translation_percent"] - pred120["translation_percent"],
            "rotation_drift_delta_deg_per_100m": pred180["rotation_deg_per_100m"] - pred120["rotation_deg_per_100m"],
            "rotation_compounding_gap_epoch_120": pred120["translation_percent"] - gt120["translation_percent"],
            "rotation_compounding_gap_epoch_180": pred180["translation_percent"] - gt180["translation_percent"],
            "rotation_compounding_gap_delta": ((pred180["translation_percent"] - gt180["translation_percent"])
                                                  - (pred120["translation_percent"] - gt120["translation_percent"])),
        })

    write_csv(os.path.join(output, "target_alignment.csv"), alignment_rows,
              ["epoch", "sequence", "translation_gt_max_difference", "rotation_gt_max_difference"])
    write_csv(os.path.join(output, "prediction_target_distributions.csv"), distribution_rows,
              ["epoch", "batch_size_stage", "sequence", "axis", "gt_min", "gt_max", "gt_mean", "gt_std",
               "pred_min", "pred_max", "pred_mean", "pred_std", "pred_inside_gt_range_fraction", "wasserstein_1"])
    write_csv(os.path.join(output, "fixed_regime_signal_metrics.csv"), signal_rows,
              ["epoch", "batch_size_stage", "sequence", "scope", "axis", "count", "gt_mean", "pred_mean",
               "gt_std", "pred_std", "variance_ratio", "bias", "mae", "rmse", "pearson",
               "correlation_r2", "predictive_r2"])
    write_csv(os.path.join(output, "z_covariance_decomposition.csv"), covariance_rows,
              ["epoch", "batch_size_stage", "sequence", "axis", "total_covariance", "between_covariance",
               "within_covariance", "between_fraction", "within_fraction", "residual_pearson", "decomposition_error"])
    write_csv(os.path.join(output, "translation_error_autocorrelation.csv"), autocorrelation_rows,
              ["epoch", "batch_size_stage", "sequence", "axis", "lag", "error_autocorrelation"])
    write_csv(os.path.join(output, "cumulative_signed_translation_error.csv"), cumulative_rows,
              ["epoch", "batch_size_stage", "sequence", "axis", "final_cumulative_error",
               "maximum_absolute_cumulative_error", "rms_cumulative_error"])
    write_csv(os.path.join(output, "contiguous_window_translation_bias.csv"), window_rows,
              ["epoch", "batch_size_stage", "sequence", "window_length", "axis", "windows",
               "mean_window_bias", "mean_absolute_window_bias", "rms_window_bias", "maximum_absolute_window_bias"])
    write_csv(os.path.join(output, "native_kitti_metrics.csv"), kitti_rows,
              ["epoch", "batch_size_stage", "sequence", "trajectory_rotation", "segments",
               "translation_percent", "rotation_deg_per_100m"])
    write_csv(os.path.join(output, "native_kitti_metrics_by_length.csv"), kitti_length_rows,
              ["epoch", "batch_size_stage", "sequence", "trajectory_rotation", "segment_length_m", "segments",
               "translation_percent", "rotation_deg_per_100m"])
    write_csv(os.path.join(output, "epoch_120_to_180_prediction_change.csv"), attribution_rows,
              ["sequence", "quantity", "axis", "mean_change", "mae_change", "rmse_change"])
    write_csv(os.path.join(output, "epoch_120_to_180_metric_attribution.csv"), metric_attribution_rows,
              ["sequence", "gt_rotation_translation_delta_percent",
               "predicted_rotation_translation_delta_percent", "rotation_drift_delta_deg_per_100m",
               "rotation_compounding_gap_epoch_120", "rotation_compounding_gap_epoch_180",
               "rotation_compounding_gap_delta"])

    report = {"regime_thresholds": {"low_max": thresholds[0], "medium_max": thresholds[1]},
              "target_alignment": alignment_rows, "distributions": distribution_rows,
              "signal_metrics": signal_rows, "covariance_decomposition": covariance_rows,
              "autocorrelation": autocorrelation_rows, "cumulative_error": cumulative_rows,
              "window_bias": window_rows, "kitti_metrics": kitti_rows,
              "kitti_by_length": kitti_length_rows, "epoch_120_to_180_change": attribution_rows,
              "epoch_120_to_180_metric_attribution": metric_attribution_rows}
    with open(os.path.join(output, "checkpoint_dynamics_audit.json"), "w") as handle:
        json.dump(json_clean(report), handle, indent=2, sort_keys=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
        for sequence, color in (("09", "tab:blue"), ("10", "tab:orange")):
            selected = [r for r in kitti_rows if r["sequence"] == sequence and r["trajectory_rotation"] == "predicted"]
            axes[0].plot([r["epoch"] for r in selected], [r["translation_percent"] for r in selected],
                         marker="o", color=color, label="Sequence " + sequence)
            for epoch, style in ((30, "--"), (120, "-"), (180, ":")):
                axes[1].plot(cumulative_series[(epoch, sequence, "z")], linestyle=style,
                             color=color, label="Seq {} epoch {}".format(sequence, epoch))
        axes[0].set_title("Full-VO KITTI translation drift")
        axes[0].set_ylabel("Translation drift (%)")
        axes[0].set_xlabel("Epoch")
        axes[1].set_title("Cumulative signed directional-z prediction error")
        axes[1].set_ylabel("Cumulative error")
        axes[1].set_xlabel("Transition")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output, "checkpoint_dynamics.png"), dpi=180)
        plt.close(fig)
    except ImportError:
        print("WARNING: matplotlib unavailable; skipped checkpoint_dynamics.png")

    print("A6 schedule checkpoint-dynamics audit")
    print("=" * 112)
    print("Fixed sequence-09 GT-z regimes: low <= {:.9f}; medium <= {:.9f}; high above.".format(*thresholds))
    print("{:<5} {:<3} {:>12} {:>14} {:>14} {:>13}".format(
        "Epoch", "Seq", "GT-R t (%)", "Pred-R t (%)", "Pred-R r", "z error ACF1"))
    print("-" * 112)
    for epoch, _ in BOUNDARIES:
        for sequence in ("09", "10"):
            gt_row = next(r for r in kitti_rows if r["epoch"] == epoch and r["sequence"] == sequence and r["trajectory_rotation"] == "ground_truth")
            pred_row = next(r for r in kitti_rows if r["epoch"] == epoch and r["sequence"] == sequence and r["trajectory_rotation"] == "predicted")
            acf = next(r["error_autocorrelation"] for r in autocorrelation_rows
                       if r["epoch"] == epoch and r["sequence"] == sequence and r["axis"] == "z" and r["lag"] == 1)
            print("{:<5} {:<3} {:>12.5f} {:>14.5f} {:>10.4f} deg {:>13.4f}".format(
                epoch, sequence, gt_row["translation_percent"], pred_row["translation_percent"],
                pred_row["rotation_deg_per_100m"], acf))
    print("Saved audit outputs to: {}".format(output))


if __name__ == "__main__":
    main()
