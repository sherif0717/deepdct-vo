#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

A4_CHECKPOINT="${A4_CHECKPOINT:-experiments/track_a/paper_reproduction/a4_paper_compatible_auxiliaries/checkpoints/best_validation.pt}"

fail=0
for f in scripts/evaluate_deepdct_vo.py data/poses/09.txt data/poses/10.txt "$A4_CHECKPOINT"; do
  if [[ ! -e "$f" ]]; then
    echo "[MISSING] $f"
    fail=1
  else
    echo "[OK] $f"
  fi
done

if ! python scripts/evaluate_deepdct_vo.py --help 2>/dev/null | grep -q -- '--translation-scale-factor'; then
  echo "[MISSING] evaluator CLI --translation-scale-factor"
  echo "Apply patches/EVALUATOR_PLACEMENT_GUIDE.md first."
  fail=1
else
  echo "[OK] evaluator exposes --translation-scale-factor"
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "A5 preflight PASS"
