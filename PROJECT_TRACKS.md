# DeepDCT-VO PyTorch Reimplementation

The repository is organized around two research tracks.

## Track A — Paper reproduction

Goal: reproduce the published DeepDCT-VO methodology in PyTorch.

Location:

    tracks/track_a_paper_reproduction/

Key objectives:

1. Attention Residual U-Net
2. Separate rotation and translation models
3. Semantic and depth auxiliary cues
4. Directional coordinate transformation
5. Published normalization conventions
6. MAE training
7. Ground-truth rotation conditioning for Model T
8. Translation scaling
9. Unseen KITTI sequence protocol
10. 50%-target-sequence protocol


## Track B — End-to-end extension

Goal: remove paper simplifications and investigate robust predicted-pose
generalization.

Location:

    tracks/track_b_end_to_end_extension/

Key objectives:

1. Predicted rotation conditioning
2. Compact translation representations
3. Representation diagnostics
4. Continuous SO(3) supervision
5. Turn-aware and cross-sequence generalization


## Shared implementation

The reusable PyTorch implementation remains at repository root:

    deepdct/
    tests/
    scripts/train_deepdct_vo.py
    scripts/evaluate_deepdct_vo.py

This prevents track organization from duplicating or breaking the shared
model implementation.
