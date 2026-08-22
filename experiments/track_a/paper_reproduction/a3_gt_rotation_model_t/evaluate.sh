#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-experiments/track_a/paper_reproduction/a3_gt_rotation_model_t/best_validation.pt}"
TEST_SEQUENCE="${TEST_SEQUENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/track_a/paper_reproduction/a3_gt_rotation_model_t/evaluation_sequence_${TEST_SEQUENCE}}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "========================================================================"
echo "Track-A A3 evaluation: GT rotation -> Model T"
echo "========================================================================"
echo "Checkpoint:          $CHECKPOINT"
echo "Test sequence:       $TEST_SEQUENCE"
echo "Output directory:    $OUTPUT_DIR"
echo "Batch size:          $BATCH_SIZE"
echo "Workers:             $NUM_WORKERS"
echo "Model-T rotation:    GROUND TRUTH"
echo "========================================================================"
echo

python scripts/evaluate_deepdct_vo.py \
  --checkpoint "$CHECKPOINT" \
  --sequence "$TEST_SEQUENCE" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --use-ground-truth-rotation