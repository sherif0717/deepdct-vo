#!/usr/bin/env python3
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_positions(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 12:
        raise ValueError(f"{path} must contain KITTI 3x4 poses; got {data.shape}")
    return data.reshape(-1, 3, 4)[:, :, 3]

p = argparse.ArgumentParser()
p.add_argument("--ground-truth", type=Path, required=True)
p.add_argument("--prediction", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

gt = load_positions(args.ground_truth)
pred = load_positions(args.prediction)
if len(gt) != len(pred):
    raise ValueError(f"Length mismatch: GT={len(gt)}, pred={len(pred)}")

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(gt[:, 0], gt[:, 2], label="Ground truth", linewidth=2)
ax.plot(pred[:, 0], pred[:, 2], label="Prediction", linewidth=2)
ax.scatter(gt[0, 0], gt[0, 2], label="Start")
ax.scatter(gt[-1, 0], gt[-1, 2], marker="x", label="GT end")
ax.scatter(pred[-1, 0], pred[-1, 2], marker="+", label="Pred end")
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_title("Track-A A4 — Sequence 10 GT vs prediction")
ax.axis("equal")
ax.grid(True)
ax.legend()
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=200)
print(f"Saved: {args.output}")
