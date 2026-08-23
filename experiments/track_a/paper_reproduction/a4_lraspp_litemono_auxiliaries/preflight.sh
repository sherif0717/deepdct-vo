#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

echo "Track-A A4 preflight — LR-ASPP + Lite-Mono"

for f in   deepdct/models/deepdct_vo.py   scripts/train_deepdct_vo.py   scripts/evaluate_deepdct_vo.py   tests/models/test_deepdct_forward.py
do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
  echo "FOUND: $f"
done

test -d weights/lite-mono-tiny-640x192 || {
  echo "MISSING: weights/lite-mono-tiny-640x192"
  exit 2
}

python scripts/train_deepdct_vo.py --help > /tmp/a4_train_help.txt
for flag in   --use-semantic-cues   --use-depth-cues   --freeze-semantic   --freeze-depth   --depth-checkpoint-dir   --depth-model-name   --depth-output-mode   --translation-decoder   --use-ground-truth-rotation
do
  grep -q -- "$flag" /tmp/a4_train_help.txt || {
    echo "MISSING TRAIN FLAG: $flag"
    exit 3
  }
done

pytest -q tests/models/test_deepdct_forward.py
echo "A4 preflight PASS"
