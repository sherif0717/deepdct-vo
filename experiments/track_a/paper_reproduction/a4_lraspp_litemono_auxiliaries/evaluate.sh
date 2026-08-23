#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-experiments/track_a/paper_reproduction/a4_lraspp_litemono_auxiliaries/checkpoints/best_validation.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/track_a/paper_reproduction/a4_lraspp_litemono_auxiliaries/evaluation_sequence_10}"

mkdir -p "$OUTPUT_DIR"

python scripts/evaluate_deepdct_vo.py   --checkpoint "$CHECKPOINT"   --sequence 10   --batch-size "${BATCH_SIZE:-1}"   --num-workers "${NUM_WORKERS:-0}"   --output-dir "$OUTPUT_DIR"
