#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10"

DATA_ROOT="${DATA_ROOT:-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WRAPPER_DIR/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-track_a_a6_unseen_00_08_to_09_10}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

DEPTH_CHECKPOINT_DIR="${DEPTH_CHECKPOINT_DIR:-weights/lite-mono-tiny-640x192}"
DEPTH_MODEL_NAME="${DEPTH_MODEL_NAME:-lite-mono-tiny}"
DEPTH_OUTPUT_MODE="${DEPTH_OUTPUT_MODE:-normalized_depth}"

mkdir -p "$CHECKPOINT_DIR"

echo "========================================================================"
echo "Track-A A6 training: source-only 00-08"
echo "========================================================================"
echo "Train sequences:        00 01 02 03 04 05 06 07 08"
echo "Validation:             DISABLED"
echo "Unseen targets:         09 10"
echo "Checkpoint selection:   final epoch / latest.pt"
echo "Pose objective:         MAE"
echo "Rotation norm scale:    0.175"
echo "Model-T rotation:       GROUND TRUTH"
echo "Semantic auxiliary:     LR-ASPP"
echo "Depth auxiliary:        Lite-Mono"
echo "Translation decoder:    dense"
echo "========================================================================"

python scripts/train_deepdct_vo.py \
  --data-root "$DATA_ROOT" \
  --train-sequences 00 01 02 03 04 05 06 07 08 \
  --no-validation \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --learning-rate "$LEARNING_RATE" \
  --pose-loss-type mae \
  --rotation-normalization-scale 0.175 \
  --translation-decoder dense \
  --use-ground-truth-rotation \
  --use-semantic-cues \
  --freeze-semantic \
  --use-depth-cues \
  --freeze-depth \
  --depth-checkpoint-dir "$DEPTH_CHECKPOINT_DIR" \
  --depth-model-name "$DEPTH_MODEL_NAME" \
  --depth-output-mode "$DEPTH_OUTPUT_MODE" \
  --rotation-geometry-weight 0 \
  --experiment-name "$EXPERIMENT_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo
echo "A6 training complete."
echo "Selected checkpoint: $CHECKPOINT_DIR/latest.pt"
