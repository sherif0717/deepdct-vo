#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-data}"
DEPTH_CHECKPOINT_DIR="${DEPTH_CHECKPOINT_DIR:-weights/lite-mono-tiny-640x192}"

echo "========================================================================"
echo "Track-A A6 preflight"
echo "========================================================================"
echo "Repository root:        $ROOT_DIR"
echo "Data root:              $DATA_ROOT"
echo "Train sequences:        00 01 02 03 04 05 06 07 08"
echo "Unseen targets:         09 10"
echo "Target validation:      DISALLOWED"
echo "========================================================================"

test -f scripts/train_deepdct_vo.py
test -f scripts/evaluate_deepdct_vo.py
test -d "$DATA_ROOT/sequences"
test -d "$DATA_ROOT/out_csv"
test -d "$DATA_ROOT/poses"
test -d "$DEPTH_CHECKPOINT_DIR"

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    test -d "$DATA_ROOT/sequences/$seq" || {
        echo "ERROR: missing $DATA_ROOT/sequences/$seq" >&2
        exit 1
    }
    test -f "$DATA_ROOT/out_csv/${seq}_dct.txt" || {
        echo "ERROR: missing $DATA_ROOT/out_csv/${seq}_dct.txt" >&2
        exit 1
    }
    test -f "$DATA_ROOT/poses/${seq}.txt" || {
        echo "ERROR: missing $DATA_ROOT/poses/${seq}.txt" >&2
        exit 1
    }
done

if ! python scripts/train_deepdct_vo.py --help 2>&1 | grep -q -- "--no-validation"; then
    cat >&2 <<'EOF'
ERROR: scripts/train_deepdct_vo.py does not expose --no-validation.

A6 requires a source-only training path so that neither sequence 09 nor 10
participates in validation, early stopping, or checkpoint selection.
See README.md in this wrapper for the required trainer update.
EOF
    exit 1
fi

if ! python scripts/evaluate_deepdct_vo.py --help >/dev/null 2>&1; then
    echo "ERROR: evaluate_deepdct_vo.py --help failed." >&2
    exit 1
fi

echo
echo "A6 preflight PASS."
