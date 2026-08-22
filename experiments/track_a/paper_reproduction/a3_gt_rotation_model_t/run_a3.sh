#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/preflight.sh"
"$SCRIPT_DIR/train.sh"

ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-track_a/paper_reproduction/a3_gt_rotation_model_t}"
CHECKPOINT="${CHECKPOINT:-experiments/${EXPERIMENT_NAME}/best_validation.pt}"

python "$SCRIPT_DIR/verify_checkpoint.py" "$CHECKPOINT"
CHECKPOINT="$CHECKPOINT" "$SCRIPT_DIR/evaluate.sh"
