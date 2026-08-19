from pathlib import Path

#!/usr/bin/env python3
"""
Fit and evaluate a linear probe from frozen DeepDCT-VO translation representations
to ground-truth relative translation.

Expected inputs
---------------
Each representation .npz must contain:
    translation_rep : [N, D] float array

For labels / original predictions, the script uses a matching frame_predictions.csv
whose rows are in the same transition order as translation_rep. By default, it looks
for frame_predictions.csv in the same directory as the .npz. You can override this
with --csv-suffix or provide directories instead of .npz files.

Required CSV columns:
    translation_gt_x, translation_gt_y, translation_gt_z
    translation_pred_x, translation_pred_y, translation_pred_z

Typical usage
-------------
python scripts/debug/probe_translation_representation.py \
    --train \
      experiments/translation_rep/baseline/00 \
      experiments/translation_rep/baseline/01 \
      experiments/translation_rep/baseline/02 \
      experiments/translation_rep/baseline/03 \
      experiments/translation_rep/baseline/04 \
      experiments/translation_rep/baseline/05 \
      experiments/translation_rep/baseline/06 \
      experiments/translation_rep/baseline/07 \
      experiments/translation_rep/baseline/08 \
      experiments/translation_rep/baseline/09 \
    --test experiments/translation_rep/baseline/10 \
    --output-dir experiments/translation_rep_probe/baseline \
    --ridge 1e-4

The same command can be repeated for semantic_depth_identity_output.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REP_KEYS = ("translation_rep", "translation_representation", "features", "representation")
GT_COLS = ("translation_gt_x", "translation_gt_y", "translation_gt_z")
PRED_COLS = ("translation_pred_x", "translation_pred_y", "translation_pred_z")


def parse_args():
    p = argparse.ArgumentParser(
        description="Linear probe for DeepDCT-VO translation representations."
    )
    p.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Training representation directories or .npz files.",
    )
    p.add_argument(
        "--test",
        nargs="+",
        required=True,
        help="Held-out representation directories or .npz files.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for probe outputs.",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient. Default: 1e-4.",
    )
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization before fitting.",
    )
    p.add_argument(
        "--rep-filename",
        default="translation_representations.npz",
        help="Representation filename when an input is a directory.",
    )
    p.add_argument(
        "--csv-filename",
        default="frame_predictions.csv",
        help="Prediction/GT CSV filename when an input is a directory.",
    )
    return p.parse_args()


def _resolve_pair(path_str: str, rep_filename: str, csv_filename: str) -> Tuple[Path, Path]:
    p = Path(path_str)

    if p.is_dir():
        rep_path = p / rep_filename
        csv_path = p / csv_filename
    elif p.suffix.lower() == ".npz":
        rep_path = p
        csv_path = p.parent / csv_filename
    else:
        raise ValueError(
            f"Input must be a directory or .npz file, got: {p}"
        )

    if not rep_path.exists():
        raise FileNotFoundError(f"Representation file not found: {rep_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    return rep_path, csv_path


def _load_representation(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    key = None
    for candidate in REP_KEYS:
        if candidate in data:
            key = candidate
            break

    if key is None:
        raise KeyError(
            f"{npz_path} does not contain any supported representation key. "
            f"Expected one of: {REP_KEYS}. Found: {list(data.keys())}"
        )

    x = np.asarray(data[key], dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"{key} in {npz_path} must have shape [N, ...], got {x.shape}")

    # Flatten all non-batch dimensions. This permits either [N, D] or a saved
    # convolutional tensor [N, C, H, W].
    x = x.reshape(x.shape[0], -1)

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values found in representation: {npz_path}")

    return x


def _load_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    missing = [c for c in GT_COLS + PRED_COLS if c not in rows[0]]
    if missing:
        raise KeyError(f"{csv_path} is missing columns: {missing}")

    gt = np.asarray(
        [[float(r[c]) for c in GT_COLS] for r in rows],
        dtype=np.float64,
    )
    pred = np.asarray(
        [[float(r[c]) for c in PRED_COLS] for r in rows],
        dtype=np.float64,
    )
    return gt, pred, rows


def load_split(
    inputs: Sequence[str],
    rep_filename: str,
    csv_filename: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    xs, ys, preds = [], [], []
    provenance = []

    expected_dim = None

    for item in inputs:
        rep_path, csv_path = _resolve_pair(item, rep_filename, csv_filename)
        x = _load_representation(rep_path)
        y, pred, _ = _load_csv(csv_path)

        if len(x) != len(y):
            raise ValueError(
                f"Row mismatch for {item}: representation has {len(x)} rows, "
                f"CSV has {len(y)} rows."
            )

        if expected_dim is None:
            expected_dim = x.shape[1]
        elif x.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim}, "
                f"got {x.shape[1]} in {rep_path}"
            )

        xs.append(x)
        ys.append(y)
        preds.append(pred)
        provenance.append(
            {
                "input": str(item),
                "representation_file": str(rep_path),
                "prediction_csv": str(csv_path),
                "samples": int(len(x)),
                "feature_dim": int(x.shape[1]),
            }
        )

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        provenance,
    )


def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def fit_ridge_probe(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """
    Fit Y ~= [X, 1] W using ridge regression.

    Bias is not regularized.
    Returns W with shape [D+1, 3].
    """
    if ridge < 0:
        raise ValueError("--ridge must be >= 0")

    xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)

    reg = np.eye(xb.shape[1], dtype=x.dtype) * ridge
    reg[-1, -1] = 0.0

    lhs = xb.T @ xb + reg
    rhs = xb.T @ y

    try:
        w = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(lhs) @ rhs

    return w


def predict_probe(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return xb @ w


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, object]:
    err = pred - gt
    bias = err.mean(axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)

    sample_l2 = np.linalg.norm(err, axis=1)
    vector_rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    vector_mae = float(np.mean(sample_l2))

    corr = []
    for i in range(3):
        if np.std(pred[:, i]) < 1e-12 or np.std(gt[:, i]) < 1e-12:
            corr.append(float("nan"))
        else:
            corr.append(float(np.corrcoef(pred[:, i], gt[:, i])[0, 1]))

    return {
        "bias": {"x": float(bias[0]), "y": float(bias[1]), "z": float(bias[2])},
        "rmse": {"x": float(rmse[0]), "y": float(rmse[1]), "z": float(rmse[2])},
        "mae": {"x": float(mae[0]), "y": float(mae[1]), "z": float(mae[2])},
        "correlation": {"x": corr[0], "y": corr[1], "z": corr[2]},
        "vector_rmse": vector_rmse,
        "mean_vector_error": vector_mae,
    }


def improvement_summary(original: Dict[str, object], probe: Dict[str, object]) -> Dict[str, object]:
    raw = float(original["vector_rmse"])
    new = float(probe["vector_rmse"])
    delta = raw - new
    pct = (100.0 * delta / raw) if raw > 0 else float("nan")

    per_axis = {}
    for axis in ("x", "y", "z"):
        r = float(original["rmse"][axis])
        n = float(probe["rmse"][axis])
        per_axis[axis] = {
            "original_rmse": r,
            "probe_rmse": n,
            "absolute_improvement": r - n,
            "percent_improvement": (100.0 * (r - n) / r) if r > 0 else float("nan"),
        }

    return {
        "original_vector_rmse": raw,
        "probe_vector_rmse": new,
        "absolute_improvement": delta,
        "percent_improvement": pct,
        "probe_is_better": bool(new < raw),
        "per_axis": per_axis,
    }


def save_predictions_csv(
    path: Path,
    gt: np.ndarray,
    original_pred: np.ndarray,
    probe_pred: np.ndarray,
) -> None:
    fieldnames = [
        "index",
        "translation_gt_x", "translation_gt_y", "translation_gt_z",
        "translation_pred_x", "translation_pred_y", "translation_pred_z",
        "translation_probe_x", "translation_probe_y", "translation_probe_z",
        "original_error_norm", "probe_error_norm",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        original_norm = np.linalg.norm(original_pred - gt, axis=1)
        probe_norm = np.linalg.norm(probe_pred - gt, axis=1)

        for i in range(len(gt)):
            writer.writerow(
                {
                    "index": i,
                    "translation_gt_x": gt[i, 0],
                    "translation_gt_y": gt[i, 1],
                    "translation_gt_z": gt[i, 2],
                    "translation_pred_x": original_pred[i, 0],
                    "translation_pred_y": original_pred[i, 1],
                    "translation_pred_z": original_pred[i, 2],
                    "translation_probe_x": probe_pred[i, 0],
                    "translation_probe_y": probe_pred[i, 1],
                    "translation_probe_z": probe_pred[i, 2],
                    "original_error_norm": original_norm[i],
                    "probe_error_norm": probe_norm[i],
                }
            )


def maybe_save_plots(output_dir: Path, gt: np.ndarray, original: np.ndarray, probe: np.ndarray):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable; plots skipped: {exc}")
        return

    axes = ("x", "y", "z")

    # One plot per axis to keep comparison legible.
    for j, axis in enumerate(axes):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.scatter(gt[:, j], original[:, j], s=8, alpha=0.35, label="existing head")
        ax.scatter(gt[:, j], probe[:, j], s=8, alpha=0.35, label="linear probe")

        lo = min(gt[:, j].min(), original[:, j].min(), probe[:, j].min())
        hi = max(gt[:, j].max(), original[:, j].max(), probe[:, j].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="ideal")

        ax.set_xlabel(f"GT translation {axis}")
        ax.set_ylabel(f"Predicted translation {axis}")
        ax.set_title(f"Translation representation probe: {axis}-axis")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"probe_scatter_{axis}.png", dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Translation Representation Linear Probe")
    print("=" * 80)

    x_train, y_train, pred_train, train_prov = load_split(
        args.train, args.rep_filename, args.csv_filename
    )
    x_test, y_test, pred_test, test_prov = load_split(
        args.test, args.rep_filename, args.csv_filename
    )

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"Train/test feature dimensions differ: "
            f"{x_train.shape[1]} vs {x_test.shape[1]}"
        )

    standardize = not args.no_standardize

    if standardize:
        mean, std = fit_standardizer(x_train)
        x_train_fit = apply_standardizer(x_train, mean, std)
        x_test_fit = apply_standardizer(x_test, mean, std)
    else:
        mean = np.zeros(x_train.shape[1], dtype=np.float64)
        std = np.ones(x_train.shape[1], dtype=np.float64)
        x_train_fit = x_train
        x_test_fit = x_test

    w = fit_ridge_probe(x_train_fit, y_train, args.ridge)

    probe_train = predict_probe(x_train_fit, w)
    probe_test = predict_probe(x_test_fit, w)

    original_train_metrics = metrics(pred_train, y_train)
    probe_train_metrics = metrics(probe_train, y_train)
    original_test_metrics = metrics(pred_test, y_test)
    probe_test_metrics = metrics(probe_test, y_test)

    improvement = improvement_summary(original_test_metrics, probe_test_metrics)

    summary = {
        "configuration": {
            "ridge": args.ridge,
            "standardize": standardize,
            "feature_dim": int(x_train.shape[1]),
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
        },
        "train_inputs": train_prov,
        "test_inputs": test_prov,
        "train": {
            "existing_head": original_train_metrics,
            "linear_probe": probe_train_metrics,
        },
        "test": {
            "existing_head": original_test_metrics,
            "linear_probe": probe_test_metrics,
            "improvement": improvement,
        },
        "interpretation": (
            "HEAD_DECODING_FAILURE_SUPPORTED"
            if improvement["probe_is_better"]
            else "NO_EVIDENCE_PROBE_BEATS_EXISTING_HEAD"
        ),
    }

    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    np.savez_compressed(
        out / "probe_parameters.npz",
        weight=w[:-1],
        bias=w[-1],
        feature_mean=mean,
        feature_std=std,
        ridge=np.asarray(args.ridge),
        standardize=np.asarray(standardize),
    )

    save_predictions_csv(
        out / "test_predictions.csv",
        y_test,
        pred_test,
        probe_test,
    )

    maybe_save_plots(out, y_test, pred_test, probe_test)

    print()
    print("Train shape:", x_train.shape)
    print("Test shape: ", x_test.shape)
    print()
    print("Held-out test results")
    print("-" * 80)
    print(f"Existing head vector RMSE: {original_test_metrics['vector_rmse']:.9f}")
    print(f"Linear probe vector RMSE:  {probe_test_metrics['vector_rmse']:.9f}")
    print(
        f"Improvement:               {improvement['absolute_improvement']:.9f} "
        f"({improvement['percent_improvement']:.2f}%)"
    )
    print()
    for axis in ("x", "y", "z"):
        d = improvement["per_axis"][axis]
        print(
            f"{axis}: existing RMSE={d['original_rmse']:.9f}, "
            f"probe RMSE={d['probe_rmse']:.9f}, "
            f"improvement={d['percent_improvement']:.2f}%"
        )
    print()
    print("Interpretation:", summary["interpretation"])
    print()
    print(f"Saved: {out / 'summary.json'}")
    print(f"Saved: {out / 'probe_parameters.npz'}")
    print(f"Saved: {out / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()

