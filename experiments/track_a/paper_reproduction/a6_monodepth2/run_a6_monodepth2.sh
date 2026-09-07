#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_monodepth2"
"$WRAPPER_DIR/preflight.sh"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  "$WRAPPER_DIR/train.sh"
fi
"$WRAPPER_DIR/evaluate_09.sh"
"$WRAPPER_DIR/evaluate_10.sh"
