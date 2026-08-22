#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

echo "========================================================================"
echo "Track-A A3 preflight: GT rotation -> Model T"
echo "========================================================================"
echo "Repository root: $ROOT_DIR"
echo

required_files=(
  "scripts/train_deepdct_vo.py"
  "scripts/evaluate_deepdct_vo.py"
  "deepdct/models/deepdct_vo.py"
  "deepdct/training/train_one_epoch.py"
  "deepdct/training/validate_one_epoch.py"
  "data/out_csv/00_dct.txt"
  "data/out_csv/01_dct.txt"
  "data/out_csv/02_dct.txt"
  "data/out_csv/03_dct.txt"
  "data/out_csv/04_dct.txt"
  "data/out_csv/05_dct.txt"
  "data/out_csv/06_dct.txt"
  "data/out_csv/07_dct.txt"
  "data/out_csv/08_dct.txt"
  "data/out_csv/09_dct.txt"
  "data/out_csv/10_dct.txt"
)

for f in "${required_files[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "FAIL: missing required path: $f" >&2
    exit 1
  fi
done

echo "PASS: required repository files are present."

if ! python scripts/train_deepdct_vo.py --help 2>&1 | grep -q -- "--use-ground-truth-rotation"; then
  echo "FAIL: scripts/train_deepdct_vo.py does not expose --use-ground-truth-rotation" >&2
  exit 1
fi

if ! python scripts/evaluate_deepdct_vo.py --help 2>&1 | grep -q -- "--use-ground-truth-rotation"; then
  echo "FAIL: scripts/evaluate_deepdct_vo.py does not expose --use-ground-truth-rotation" >&2
  exit 1
fi

echo
echo "A3 intended protocol"
echo "  train sequences:       00-08"
echo "  validation sequence:   09"
echo "  held-out test:         10"
echo "  semantic cues:         disabled"
echo "  depth cues:            disabled"
echo "  translation decoder:   dense"
echo "  translation rotation:  GROUND TRUTH"
echo "  rotation geometry:     disabled"
echo "  warm start/resume:     disabled"
echo
echo "A3 PREFLIGHT PASS"
