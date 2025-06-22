#!/usr/bin/env bash
set -euo pipefail
A7_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${A7_SCRIPT_DIR}/preflight.sh"
"${A7_SCRIPT_DIR}/train.sh"
"${A7_SCRIPT_DIR}/evaluate.sh"
"${A7_SCRIPT_DIR}/evaluate_gt_gt.sh"

