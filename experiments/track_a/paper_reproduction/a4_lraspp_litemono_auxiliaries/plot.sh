#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-experiments/track_a/paper_reproduction/a4_lraspp_litemono_auxiliaries/evaluation_sequence_10}"

python experiments/track_a/paper_reproduction/a4_lraspp_litemono_auxiliaries/plot_trajectory.py   --ground-truth "$OUTPUT_DIR/ground_truth_trajectory.txt"   --prediction "$OUTPUT_DIR/predicted_trajectory.txt"   --output "$OUTPUT_DIR/trajectory_gt_vs_pred.png"
