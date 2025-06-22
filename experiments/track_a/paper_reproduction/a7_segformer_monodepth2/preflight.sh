#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/a7_common.sh"

test -f scripts/train_deepdct_vo.py
test -f scripts/evaluate_deepdct_vo.py
test -f deepdct/models/auxiliary/segformer.py
test -d "${A7_DEPTH_WEIGHTS}"
for sequence in 00 01 02 03 04 05 06 07 08 09 10; do
  test -d "${A7_DATA_ROOT}/sequences/${sequence}"
  test -f "${A7_DATA_ROOT}/out_csv/${sequence}_dct.txt"
done

python - <<'PY'
import torch
import transformers
from deepdct.models.auxiliary.segformer import SegFormerSemanticBranch

branch = SegFormerSemanticBranch(
    model_name_or_path=__import__("os").environ["A7_SEMANTIC_MODEL"],
    freeze_pretrained=True,
)
sample = torch.rand(1, 3, 120, 120)
with torch.no_grad():
    cue = branch(sample)
assert cue.shape == (1, 1, 120, 120)
assert torch.isfinite(cue).all()
assert cue.min() >= 0 and cue.max() <= 1
assert all(not parameter.requires_grad for parameter in branch.parameters())
print("A7 SegFormer preflight: PASS")
print("transformers:", transformers.__version__)
print("foreground class IDs:", branch.foreground_class_ids)
PY

python -m pytest -q tests/models/test_deepdct_forward.py

