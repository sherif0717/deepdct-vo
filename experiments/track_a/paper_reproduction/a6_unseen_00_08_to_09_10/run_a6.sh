#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10"

"$WRAPPER_DIR/preflight.sh"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    "$WRAPPER_DIR/train.sh"
fi

"$WRAPPER_DIR/verify_protocol.sh"
"$WRAPPER_DIR/evaluate_09.sh"
"$WRAPPER_DIR/evaluate_10.sh"
"$WRAPPER_DIR/plot_trajectories.sh"

echo
echo "========================================================================"
echo "Track-A A6 complete"
echo "========================================================================"
echo "Sequence 09: $WRAPPER_DIR/evaluation_sequence_09"
echo "Sequence 10: $WRAPPER_DIR/evaluation_sequence_10"
echo "GT-vs-pred plots:"
echo "  sequence 09: trajectory_gt_vs_pred_xz.png"
echo "  sequence 10: trajectory_gt_vs_pred_xz.png"
echo "========================================================================"
