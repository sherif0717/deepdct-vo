#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

WRAPPER="experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing"
CKPT="${A4_CHECKPOINT:-experiments/track_a/paper_reproduction/a4_paper_compatible_auxiliaries/checkpoints/best_validation.pt}"
COMMON=(
  --checkpoint "$CKPT"
  --sequences 09
  --batch-size 1
  --num-workers 0
  --pose-objective mae
  --rotation-normalization-scale 0.175
  --translation-decoder dense
  --use-ground-truth-rotation-for-translation
  --use-semantic-cues
  --use-depth-cues
  --freeze-semantic
  --freeze-depth
  --semantic-model lraspp
  --semantic-map-mode foreground_probability
  --depth-checkpoint-dir weights/lite-mono-tiny-640x192
  --depth-model-name lite-mono-tiny
  --depth-output-mode normalized_depth
  --depth-normalization-max 80.0
)

mkdir -p "$WRAPPER/sequence_09/unscaled" "$WRAPPER/sequence_09/scaled"
python scripts/evaluate_deepdct_vo.py "${COMMON[@]}" \
  --translation-scale-factor 1.0 \
  --output-dir "$WRAPPER/sequence_09/unscaled" | tee "$WRAPPER/logs/sequence_09_unscaled.log"
python scripts/evaluate_deepdct_vo.py "${COMMON[@]}" \
  --translation-scale-factor 0.975 \
  --output-dir "$WRAPPER/sequence_09/scaled" | tee "$WRAPPER/logs/sequence_09_scaled.log"
