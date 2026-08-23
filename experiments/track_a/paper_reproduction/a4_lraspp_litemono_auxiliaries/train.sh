#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-track_a_a4_lraspp_litemono}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiments/track_a/paper_reproduction/a4_lraspp_litemono_auxiliaries/checkpoints}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

mkdir -p "$CHECKPOINT_DIR"

python scripts/train_deepdct_vo.py   --train-sequences 00 01 02 03 04 05 06 07 08   --validation-sequences 09   --epochs "$EPOCHS"   --batch-size "$BATCH_SIZE"   --num-workers "$NUM_WORKERS"   --learning-rate "$LEARNING_RATE"   --pose-loss-type mae   --rotation-normalization-scale 0.175   --translation-decoder dense   --use-ground-truth-rotation   --use-semantic-cues   --use-depth-cues   --freeze-semantic   --freeze-depth   --depth-checkpoint-dir weights/lite-mono-tiny-640x192   --depth-model-name lite-mono-tiny   --depth-output-mode normalized_depth   --rotation-geometry-weight 0   --experiment-name "$EXPERIMENT_NAME"   --checkpoint-dir "$CHECKPOINT_DIR"
