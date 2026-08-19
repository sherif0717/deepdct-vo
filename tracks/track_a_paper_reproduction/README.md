# Track A — DeepDCT-VO Paper Reproduction

Objective:

Reimplement the published DeepDCT-VO methodology in PyTorch as faithfully
as practical and reproduce comparable KITTI behavior.

Primary scope:

- Attention Residual U-Net
- Separate Model R and Model T
- Semantic auxiliary input
- Depth auxiliary input
- Directional coordinate transformation
- Paper rotation normalization
- MAE rotation and translation objectives
- Ground-truth rotation supplied to Model T
- Paper-style translation scale correction
- KITTI unseen-sequence protocol
- KITTI 50%-target-sequence training protocol

This track prioritizes reproduction of the published experimental problem
over strict end-to-end inference.
