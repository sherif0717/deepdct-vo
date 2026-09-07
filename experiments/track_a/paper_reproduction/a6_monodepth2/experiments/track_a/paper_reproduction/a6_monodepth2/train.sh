#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_monodepth2"
DATA_ROOT="${DATA_ROOT:-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WRAPPER_DIR/checkpoints}"
MONODEPTH2_CHECKPOINT_DIR="${MONODEPTH2_CHECKPOINT_DIR:-weights/mono_640x192}"

mkdir -p "$CHECKPOINT_DIR"

python scripts/train_deepdct_vo.py \
  --data-root "$DATA_ROOT" \
  --train-sequences 00 01 02 03 04 05 06 07 08 \
  --no-validation \
  --epochs "${EPOCHS:-15}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --pose-loss-type mae \
  --rotation-normalization-scale 0.175 \
  --translation-decoder dense \
  --use-ground-truth-rotation \
  --use-semantic-cues \
  --freeze-semantic \
  --use-depth-cues \
  --freeze-depth \
  --depth-provider monodepth2 \
  --depth-checkpoint-dir "$MONODEPTH2_CHECKPOINT_DIR" \
  --depth-model-name mono_640x192 \
  --depth-output-mode normalized_depth \
  --depth-normalization-meters 80.0 \
  --rotation-geometry-weight 0 \
  --experiment-name track_a_a6_monodepth2 \
  --checkpoint-dir "$CHECKPOINT_DIR"
