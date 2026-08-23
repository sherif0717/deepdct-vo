#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER_DIR="experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10"
CHECKPOINT="${CHECKPOINT:-$WRAPPER_DIR/checkpoints/latest.pt}"

python "$WRAPPER_DIR/verify_protocol.py" --checkpoint "$CHECKPOINT"
