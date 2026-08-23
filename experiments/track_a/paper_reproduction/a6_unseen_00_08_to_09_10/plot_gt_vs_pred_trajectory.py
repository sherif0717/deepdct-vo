#!/usr/bin/env python3
"""Plot KITTI-format ground-truth and predicted trajectories.

Expected input format:
    one pose per line, 12 floats representing a row-major 3x4 camera pose.

The evaluator already writes:
    ground_truth_trajectory.txt
    predicted_trajectory.txt

Default plot is the KITTI-driving X-Z ground plane.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GT vs predicted KITTI trajectory."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="KITTI-format ground_truth_trajectory.txt.",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="KITTI-format predicted_trajectory.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output image path, e.g. trajectory_gt_vs_pred.png.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Optional sequence ID for the title.",
    )
    parser.add_argument(
        "--plane",
        choices=("xz", "xy", "yz"),
        default="xz",
        help="Trajectory projection plane.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )
    return parser.parse_args()


def load_kitti_poses(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = np.fromstring(stripped, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(
                    f"{path}:{line_number}: expected 12 values, "
                    f"received {values.size}."
                )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = values.reshape(3, 4)
            rows.append(pose)

    if not rows:
        raise ValueError(f"No poses found in {path}")

    return np.stack(rows, axis=0)


def plane_axes(plane: str) -> Tuple[int, int, str, str]:
    mapping = {
        "xz": (0, 2, "X [m]", "Z [m]"),
        "xy": (0, 1, "X [m]", "Y [m]"),
        "yz": (1, 2, "Y [m]", "Z [m]"),
    }
    return mapping[plane]


def main() -> None:
    args = parse_args()
    gt = load_kitti_poses(args.ground_truth)
    pred = load_kitti_poses(args.prediction)

    n = min(len(gt), len(pred))
    if len(gt) != len(pred):
        print(
            "WARNING: trajectory lengths differ; plotting common prefix: "
            f"GT={len(gt)}, pred={len(pred)}, common={n}"
        )

    gt_xyz = gt[:n, :3, 3]
    pred_xyz = pred[:n, :3, 3]

    axis_a, axis_b, label_a, label_b = plane_axes(args.plane)

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.plot(
        gt_xyz[:, axis_a],
        gt_xyz[:, axis_b],
        linewidth=2.0,
        label="Ground truth",
    )
    ax.plot(
        pred_xyz[:, axis_a],
        pred_xyz[:, axis_b],
        linewidth=1.6,
        label="Prediction",
    )

    ax.scatter(
        [gt_xyz[0, axis_a]],
        [gt_xyz[0, axis_b]],
        marker="o",
        s=38,
        label="Start",
    )
    ax.scatter(
        [gt_xyz[-1, axis_a]],
        [gt_xyz[-1, axis_b]],
        marker="x",
        s=48,
        label="GT end",
    )
    ax.scatter(
        [pred_xyz[-1, axis_a]],
        [pred_xyz[-1, axis_b]],
        marker="+",
        s=58,
        label="Pred end",
    )

    title = "DeepDCT-VO: GT vs predicted trajectory"
    if args.sequence:
        title += f" — KITTI {args.sequence}"

    ax.set_title(title)
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved trajectory comparison: {args.output}")


if __name__ == "__main__":
    main()
