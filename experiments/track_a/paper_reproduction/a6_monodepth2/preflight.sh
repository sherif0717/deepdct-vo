#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
DATA_ROOT="${DATA_ROOT:-data}"
WEIGHTS="${MONODEPTH2_CHECKPOINT_DIR:-weights/mono_640x192}"

test -f deepdct/models/auxiliary/monodepth2.py
test -f deepdct/models/auxiliary/monodepth2_vendor/layers.py
test -f deepdct/models/auxiliary/monodepth2_vendor/networks/__init__.py
test -f "$WEIGHTS/encoder.pth"
test -f "$WEIGHTS/depth.pth"
for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  test -d "$DATA_ROOT/sequences/$seq"
  test -f "$DATA_ROOT/out_csv/${seq}_dct.txt"
  test -f "$DATA_ROOT/poses/${seq}.txt"
done
python scripts/train_deepdct_vo.py --help | grep -q -- "--depth-provider"
python scripts/evaluate_deepdct_vo.py --help >/dev/null
echo "A6 Monodepth2 preflight PASS"
