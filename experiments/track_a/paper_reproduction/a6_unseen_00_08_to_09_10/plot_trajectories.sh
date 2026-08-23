#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10"
PLOTTER="$WRAPPER_DIR/plot_gt_vs_pred_trajectory.py"

for seq in 09 10; do
    eval_dir="$WRAPPER_DIR/evaluation_sequence_${seq}"

    python "$PLOTTER" \
      --ground-truth "$eval_dir/ground_truth_trajectory.txt" \
      --prediction "$eval_dir/predicted_trajectory.txt" \
      --output "$eval_dir/trajectory_gt_vs_pred_xz.png" \
      --sequence "$seq" \
      --plane xz
done
