#!/usr/bin/env python3
"""Plot a KITTI ground-truth trajectory against a predicted KITTI trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_kitti_poses(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            values = np.fromstring(text, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(
                    f"{path}:{line_no}: expected 12 KITTI pose values, got {values.size}"
                )
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = values.reshape(3, 4)
            rows.append(T)
    if not rows:
        raise ValueError(f"No poses found in {path}")
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-poses", type=Path, required=True)
    parser.add_argument("--pred-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="GT vs predicted trajectory")
    parser.add_argument(
        "--projection", choices=("xz", "xy", "yz"), default="xz",
        help="2-D trajectory projection (KITTI commonly uses x-z).",
    )
    args = parser.parse_args()

    gt = load_kitti_poses(args.gt_poses)
    pred = load_kitti_poses(args.pred_poses)
    n = min(len(gt), len(pred))
    if len(gt) != len(pred):
        print(f"Warning: pose counts differ; plotting first {n} poses")
    gt_t = gt[:n, :3, 3]
    pred_t = pred[:n, :3, 3]

    axes = {"x": 0, "y": 1, "z": 2}
    a, b = args.projection
    ia, ib = axes[a], axes[b]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt_t[:, ia], gt_t[:, ib], label="Ground truth", linewidth=1.8)
    ax.plot(pred_t[:, ia], pred_t[:, ib], label="Prediction", linewidth=1.5)
    ax.scatter(gt_t[0, ia], gt_t[0, ib], marker="o", s=36, label="Start")
    ax.set_xlabel(f"{a} [m]")
    ax.set_ylabel(f"{b} [m]")
    ax.set_title(args.title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
