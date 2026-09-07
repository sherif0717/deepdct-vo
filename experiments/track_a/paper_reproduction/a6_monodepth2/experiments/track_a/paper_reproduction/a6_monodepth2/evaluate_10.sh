#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_monodepth2"
python scripts/evaluate_deepdct_vo.py \
  --checkpoint "${CHECKPOINT:-$WRAPPER_DIR/checkpoints/latest.pt}" \
  --data-root "${DATA_ROOT:-data}" --sequence 10 \
  --translation-scale-factor "${TRANSLATION_SCALE_10:-1.007}" \
  --output-dir "${OUTPUT_DIR_10:-$WRAPPER_DIR/evaluation_sequence_10}" \
  --device "${DEVICE:-auto}"
