from pathlib import Path

#!/usr/bin/env python3
"""
Memory-safe translation representation probe for DeepDCT-VO.

This version avoids concatenating all training sequences and avoids forming
a dense D x D normal-equation matrix. It trains a 3-output linear probe with
minibatch Adam over one sequence at a time.

Expected per-sequence directory:
    translation_representations.npz
    frame_predictions.csv

NPZ key:
    translation_rep

CSV columns:
    translation_gt_x, translation_gt_y, translation_gt_z
    translation_pred_x, translation_pred_y, translation_pred_z
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

REP_KEYS = ("translation_rep", "translation_representation", "features", "representation")
GT_COLS = ("translation_gt_x", "translation_gt_y", "translation_gt_z")
PRED_COLS = ("translation_pred_x", "translation_pred_y", "translation_pred_z")


def parse_args():
    p = argparse.ArgumentParser(
        description="Memory-safe linear probe for DeepDCT-VO translation representations."
    )
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rep-filename", default="translation_representations.npz")
    p.add_argument("--csv-filename", default="frame_predictions.csv")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true")
    return p.parse_args()


def resolve_pair(path_str: str, rep_filename: str, csv_filename: str) -> Tuple[Path, Path]:
    p = Path(path_str)
    if p.is_dir():
        rep = p / rep_filename
        csv_path = p / csv_filename
    elif p.suffix.lower() == ".npz":
        rep = p
        csv_path = p.parent / csv_filename
    else:
        raise ValueError(f"Input must be a directory or .npz file, got: {p}")

    if not rep.is_file():
        raise FileNotFoundError(f"Representation file not found: {rep}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")
    return rep, csv_path


def load_rep(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = next((k for k in REP_KEYS if k in data), None)
        if key is None:
            raise KeyError(f"{path}: expected one of {REP_KEYS}, found {list(data.keys())}")
        x = np.asarray(data[key], dtype=np.float32)

    if x.ndim < 2:
        raise ValueError(f"Representation must have shape [N,...], got {x.shape}")
    x = x.reshape(x.shape[0], -1)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite representation values in {path}")
    return x


def load_csv_arrays(path: Path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")

    missing = [c for c in GT_COLS + PRED_COLS if c not in rows[0]]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    gt = np.asarray([[float(r[c]) for c in GT_COLS] for r in rows], dtype=np.float32)
    pred = np.asarray([[float(r[c]) for c in PRED_COLS] for r in rows], dtype=np.float32)
    return gt, pred


def inspect_inputs(inputs, rep_filename, csv_filename):
    info = []
    feature_dim = None
    total = 0

    for item in inputs:
        rep_path, csv_path = resolve_pair(item, rep_filename, csv_filename)
        x = load_rep(rep_path)
        gt, _ = load_csv_arrays(csv_path)

        if len(x) != len(gt):
            raise ValueError(
                f"Row mismatch for {item}: representation={len(x)}, csv={len(gt)}"
            )

        if feature_dim is None:
            feature_dim = x.shape[1]
        elif x.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {feature_dim}, got {x.shape[1]}"
            )

        total += len(x)
        info.append({
            "input": item,
            "samples": int(len(x)),
            "feature_dim": int(x.shape[1]),
        })

        del x, gt

    return feature_dim, total, info


def streaming_feature_stats(inputs, rep_filename, csv_filename, feature_dim):
    count = 0
    sum_x = np.zeros(feature_dim, dtype=np.float64)
    sum_x2 = np.zeros(feature_dim, dtype=np.float64)

    for item in inputs:
        rep_path, _ = resolve_pair(item, rep_filename, csv_filename)
        x = load_rep(rep_path).astype(np.float64, copy=False)
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)
        count += len(x)
        del x

    mean = sum_x / count
    var = np.maximum(sum_x2 / count - mean * mean, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def metrics(pred, gt):
    err = pred - gt
    bias = err.mean(axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    vector_rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

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
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Translation Representation Linear Probe — Memory-Safe")
    print("=" * 80)

    feature_dim, train_samples, train_info = inspect_inputs(
        args.train, args.rep_filename, args.csv_filename
    )
    test_dim, test_samples, test_info = inspect_inputs(
        args.test, args.rep_filename, args.csv_filename
    )

    if feature_dim != test_dim:
        raise ValueError(f"Train/test feature dims differ: {feature_dim} vs {test_dim}")

    print(f"Feature dimension: {feature_dim}")
    print(f"Train samples:     {train_samples}")
    print(f"Test samples:      {test_samples}")
    print(f"Device:            {device}")

    standardize = not args.no_standardize
    if standardize:
        print("Computing streaming feature statistics...")
        mean, std = streaming_feature_stats(
            args.train, args.rep_filename, args.csv_filename, feature_dim
        )
    else:
        mean = np.zeros(feature_dim, dtype=np.float32)
        std = np.ones(feature_dim, dtype=np.float32)

    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    probe = nn.Linear(feature_dim, 3, bias=True).to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=args.learning_rate,
        weight_decay=args.ridge,
    )
    criterion = nn.MSELoss()

    train_items = list(args.train)

    for epoch in range(1, args.epochs + 1):
        probe.train()
        random.shuffle(train_items)
        total_loss = 0.0
        total_count = 0

        for item in train_items:
            rep_path, csv_path = resolve_pair(
                item, args.rep_filename, args.csv_filename
            )
            x = load_rep(rep_path)
            y, _ = load_csv_arrays(csv_path)

            order = np.random.permutation(len(x))

            for start in range(0, len(x), args.batch_size):
                idx = order[start:start + args.batch_size]

                xb = torch.from_numpy(x[idx]).to(device)
                yb = torch.from_numpy(y[idx]).to(device)

                xb = (xb - mean_t) / std_t

                optimizer.zero_grad(set_to_none=True)
                pred = probe(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(idx)
                total_count += len(idx)

            del x, y

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_mse={total_loss / total_count:.9f}"
        )

    probe.eval()
    all_gt = []
    all_original = []
    all_probe = []

    with torch.inference_mode():
        for item in args.test:
            rep_path, csv_path = resolve_pair(
                item, args.rep_filename, args.csv_filename
            )
            x = load_rep(rep_path)
            gt, original = load_csv_arrays(csv_path)

            parts = []
            for start in range(0, len(x), args.batch_size):
                xb = torch.from_numpy(x[start:start + args.batch_size]).to(device)
                xb = (xb - mean_t) / std_t
                parts.append(probe(xb).cpu().numpy())

            probe_pred = np.concatenate(parts, axis=0)
            all_gt.append(gt)
            all_original.append(original)
            all_probe.append(probe_pred)

            del x, gt, original, probe_pred, parts

    gt = np.concatenate(all_gt, axis=0)
    original = np.concatenate(all_original, axis=0)
    probe_pred = np.concatenate(all_probe, axis=0)

    original_metrics = metrics(original, gt)
    probe_metrics = metrics(probe_pred, gt)

    raw = float(original_metrics["vector_rmse"])
    new = float(probe_metrics["vector_rmse"])
    delta = raw - new
    improvement = {
        "original_vector_rmse": raw,
        "probe_vector_rmse": new,
        "absolute_improvement": delta,
        "percent_improvement": 100.0 * delta / raw if raw > 0 else float("nan"),
        "probe_is_better": bool(new < raw),
    }

    summary = {
        "configuration": {
            "method": "minibatch_adam_linear_probe",
            "feature_dim": int(feature_dim),
            "train_samples": int(train_samples),
            "test_samples": int(test_samples),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ridge_weight_decay": args.ridge,
            "standardize": standardize,
            "device": str(device),
        },
        "train_inputs": train_info,
        "test_inputs": test_info,
        "test": {
            "existing_head": original_metrics,
            "linear_probe": probe_metrics,
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
        weight=probe.weight.detach().cpu().numpy(),
        bias=probe.bias.detach().cpu().numpy(),
        feature_mean=mean,
        feature_std=std,
    )

    with (out / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "translation_gt_x", "translation_gt_y", "translation_gt_z",
            "translation_pred_x", "translation_pred_y", "translation_pred_z",
            "translation_probe_x", "translation_probe_y", "translation_probe_z",
            "original_error_norm", "probe_error_norm",
        ])

        original_norm = np.linalg.norm(original - gt, axis=1)
        probe_norm = np.linalg.norm(probe_pred - gt, axis=1)

        for i in range(len(gt)):
            writer.writerow([
                i,
                *gt[i].tolist(),
                *original[i].tolist(),
                *probe_pred[i].tolist(),
                float(original_norm[i]),
                float(probe_norm[i]),
            ])

    print()
    print("Held-out test results")
    print("-" * 80)
    print(f"Existing head vector RMSE: {original_metrics['vector_rmse']:.9f}")
    print(f"Linear probe vector RMSE:  {probe_metrics['vector_rmse']:.9f}")
    print(
        f"Improvement:               {improvement['absolute_improvement']:.9f} "
        f"({improvement['percent_improvement']:.2f}%)"
    )
    print("Interpretation:", summary["interpretation"])
    print(f"Saved: {out / 'summary.json'}")
    print(f"Saved: {out / 'probe_parameters.npz'}")
    print(f"Saved: {out / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()

