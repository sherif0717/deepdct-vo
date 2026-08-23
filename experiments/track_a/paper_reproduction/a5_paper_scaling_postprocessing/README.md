# Track-A A5 — Paper Scaling / Post-processing

Purpose: apply the paper-compatible KITTI translation scaling at evaluation time, without retraining the A4 model.

## Fixed KITTI scale factors

- Sequence 09: `0.975`
- Sequence 10: `1.007`

The scale is applied to the **predicted relative translation vector only** before trajectory composition. Rotation is not scaled.

## Expected prerequisite

A4 checkpoint, preferably:

`experiments/track_a/paper_reproduction/a4_paper_compatible_auxiliaries/checkpoints/best_validation.pt`

Override with `A4_CHECKPOINT=/path/to/best_validation.pt`.

## Run

```bash
bash experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/preflight.sh
bash experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/run_a5.sh
```

Or run one held-out sequence:

```bash
bash experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/evaluate_seq09.sh
bash experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/evaluate_seq10.sh
```

## Plot GT vs prediction

```bash
python experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/plot_trajectory.py \
  --gt-poses data/poses/10.txt \
  --pred-poses experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/sequence_10/scaled/predicted_trajectory.txt \
  --output experiments/track_a/paper_reproduction/a5_paper_scaling_postprocessing/plots/sequence_10_gt_vs_pred.png \
  --title "Track-A A5 — KITTI 10"
```

The plotting script accepts standard KITTI pose files containing 12 floats per row (3x4 transform).

## Required evaluator update

`scripts/evaluate_deepdct_vo.py` needs one A5 feature: a CLI/config value for translation scaling, applied to predicted relative translation immediately before constructing/composing the predicted SE(3) transform. Keep the default at `1.0` so A1–A4 behavior is unchanged.

See `patches/EVALUATOR_PLACEMENT_GUIDE.md`.
