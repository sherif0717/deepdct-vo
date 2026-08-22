#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Track-A A3: GT rotation -> Model T
#
# A3 inherits the A2 paper-style pose objective:
#   - MAE/L1 pose loss
#   - rotation normalization scale = 0.175
#
# A3 changes only the Model-T conditioning source:
#   - Model T receives physical ground-truth rotation
#
# A4 components remain disabled:
#   - semantic cues OFF
#   - depth cues OFF
#
# Experimental SO(3) geometry supervision remains disabled.
# ---------------------------------------------------------------------------

EXPERIMENT_NAME="${EXPERIMENT_NAME:-track_a_a3_gt_rotation_model_t}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiments/track_a/paper_reproduction/a3_gt_rotation_model_t}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

echo "========================================================================"
echo "Track-A A3 training: GT rotation -> Model T"
echo "========================================================================"
echo "Experiment name:        $EXPERIMENT_NAME"
echo "Checkpoint directory:   $CHECKPOINT_DIR"
echo "Train sequences:        00-08"
echo "Validation sequence:    09"
echo "Epochs:                 $EPOCHS"
echo "Batch size:             $BATCH_SIZE"
echo "Workers:                $NUM_WORKERS"
echo "Learning rate:          $LEARNING_RATE"
echo "Pose loss:              MAE"
echo "Rotation norm scale:    0.175"
echo "Translation decoder:    dense"
echo "Model-T rotation:       GROUND TRUTH"
echo "Semantic cues:          disabled"
echo "Depth cues:             disabled"
echo "Rotation geometry:      disabled"
echo "========================================================================"
echo

python scripts/train_deepdct_vo.py \
  --train-sequences 00 01 02 03 04 05 06 07 08 \
  --validation-sequences 09 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --learning-rate "$LEARNING_RATE" \
  --pose-loss-type mae \
  --rotation-normalization-scale 0.175 \
  --translation-decoder dense \
  --use-ground-truth-rotation \
  --no-use-semantic-cues \
  --no-use-depth-cues \
  --rotation-geometry-weight 0 \
  --experiment-name "$EXPERIMENT_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR"
