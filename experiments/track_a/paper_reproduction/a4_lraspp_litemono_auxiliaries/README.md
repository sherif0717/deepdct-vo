# Track-A A4 — LR-ASPP + Lite-Mono Auxiliaries

A4 inherits A3 and adds the Jetson-Nano-oriented auxiliary stack:

- semantic cue: LR-ASPP
- depth cue: Lite-Mono
- both auxiliaries frozen
- pose loss: MAE/L1
- rotation normalization scale: 0.175
- Model T conditioning: ground-truth physical rotation
- translation decoder: dense
- SO(3) geometry loss disabled
- split remains train 00-08 / validation 09 / held-out test 10

This is an intentional paper-compatible adaptation rather than an exact SegFormer + Monodepth2 auxiliary reproduction.

Existing files expected to need A4 updates:
1. deepdct/models/deepdct_vo.py
2. scripts/train_deepdct_vo.py
3. scripts/evaluate_deepdct_vo.py
4. tests/models/test_deepdct_forward.py

No A4-specific logic changes should be needed in:
- deepdct/training/train_one_epoch.py
- deepdct/training/validate_one_epoch.py
- deepdct/data/training_dataset.py

Included utility:
- plot_trajectory.py
