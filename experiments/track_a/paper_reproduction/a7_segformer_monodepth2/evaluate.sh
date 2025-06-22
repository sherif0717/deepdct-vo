#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/a7_common.sh"
test -f "${A7_CHECKPOINT}"

for sequence in 09 10; do
  if [[ "${sequence}" == "09" ]]; then
    translation_scale=0.975
  else
    translation_scale=1.007
  fi

  python scripts/evaluate_deepdct_vo.py \
    --checkpoint "${A7_CHECKPOINT}" \
    --data-root "${A7_DATA_ROOT}" \
    --sequence "${sequence}" \
    --require-a6-protocol \
    --device "${A7_DEVICE}" \
    --model-t-rotation-source checkpoint \
    --trajectory-rotation-source predicted \
    --translation-scale-factor "${translation_scale}" \
    --output-dir \
      "${A7_OUTPUT_ROOT}/evaluation_sequence_${sequence}"
done
