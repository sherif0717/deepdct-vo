#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10"
CHECKPOINT="${CHECKPOINT:-$WRAPPER_DIR/checkpoints/latest.pt}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR_09:-$WRAPPER_DIR/evaluation_sequence_09}"

python scripts/evaluate_deepdct_vo.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --sequence 09 \
  --translation-scale-factor 0.975 \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE"
