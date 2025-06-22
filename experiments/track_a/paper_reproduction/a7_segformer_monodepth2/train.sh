#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/a7_common.sh"

mkdir -p "${A7_CHECKPOINT_DIR}"
python scripts/train_deepdct_vo.py \
  --data-root "${A7_DATA_ROOT}" \
  --train-sequences 00 01 02 03 04 05 06 07 08 \
  --no-validation \
  --epochs 15 \
  --batch-size 1 \
  --num-workers 0 \
  --pose-loss-type mae \
  --rotation-normalization-scale 0.175 \
  --translation-decoder dense \
  --use-ground-truth-rotation \
  --use-semantic-cues \
  --pretrained-semantic \
  --freeze-semantic \
  --semantic-provider segformer \
  --semantic-model-name "${A7_SEMANTIC_MODEL}" \
  --semantic-feed-size 512 512 \
  --semantic-map-mode foreground_probability \
  --use-depth-cues \
  --freeze-depth \
  --depth-provider monodepth2 \
  --depth-checkpoint-dir "${A7_DEPTH_WEIGHTS}" \
  --depth-model-name mono_640x192 \
  --depth-output-mode normalized_depth \
  --depth-normalization-meters 80.0 \
  --checkpoint-dir "${A7_CHECKPOINT_DIR}" \
  --experiment-name track_a_a7_segformer_monodepth2 \
  --source-experiment track_a_a6_monodepth2

