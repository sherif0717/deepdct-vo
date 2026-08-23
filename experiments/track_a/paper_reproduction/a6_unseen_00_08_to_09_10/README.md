# Track-A A6 — unseen 00–08 → 09/10 protocol

This wrapper implements the experiment orchestration for the A6 protocol:

- source training sequences: `00 01 02 03 04 05 06 07 08`
- target evaluation sequences: `09` and `10`
- **no sequence 09/10 data may be used for training, validation, checkpoint selection, scaling calibration, or early stopping**
- A4 architecture/settings are retained:
  - paper MAE pose objective
  - rotation normalization scale `0.175`
  - GT rotation → Model T
  - LR-ASPP semantic cues
  - Lite-Mono depth cues
  - dense translation decoder
- A5 sequence-specific translation scaling is retained only as fixed post-processing:
  - sequence 09: `0.975`
  - sequence 10: `1.007`

## Required core-script updates before running

### 1. `scripts/train_deepdct_vo.py` — REQUIRED

Add a source-only / no-target-validation training mode, recommended CLI:

```text
--no-validation
```

In this mode the trainer must:

1. allow training on `00..08` without constructing a validation dataset;
2. skip train/validation overlap validation when no validation split exists;
3. skip `validate_one_epoch(...)`;
4. skip validation-driven early stopping;
5. skip `ReduceLROnPlateau.step(validation_loss)` or replace it with a deterministic, predeclared schedule;
6. save ordinary per-epoch checkpoints and `latest.pt`;
7. record metadata such as:
   - `validation_enabled = false`
   - `train_sequences = [00..08]`
   - `checkpoint_selection = "final_epoch"`
8. not create or use `best_validation.pt` as the A6 selected checkpoint.

Recommended A6 checkpoint: `latest.pt` after the predeclared number of epochs.

### 2. `scripts/evaluate_deepdct_vo.py` — SMALL UPDATE

The current evaluator already accepts `--sequence` and produces trajectory files.
Generalize remaining hard-coded wording/assumptions:

- replace descriptions such as “held-out sequence 10” with the selected `args.sequence`;
- do not require or imply that the selected checkpoint is `best_validation.pt`;
- make checkpoint-comparison reporting tolerate a source-only checkpoint with no validation metrics;
- use the selected sequence in plot titles.

No model, loss, dataset, `train_one_epoch.py`, or `validate_one_epoch.py` change is required for the A6 protocol itself.

## New standalone script

`plot_gt_vs_pred_trajectory.py`

It reads the evaluator-generated:

- `ground_truth_trajectory.txt`
- `predicted_trajectory.txt`

and creates a publication-friendly X–Z trajectory comparison.

## Wrapper files

- `preflight.sh` — checks data, weights, core scripts, and A6 CLI support
- `train.sh` — source-only training on 00–08
- `evaluate_09.sh` — unseen sequence 09 evaluation with fixed A5 scale 0.975
- `evaluate_10.sh` — unseen sequence 10 evaluation with fixed A5 scale 1.007
- `plot_gt_vs_pred_trajectory.py` — standalone GT-vs-pred plotter
- `plot_trajectories.sh` — plots both target sequences
- `run_a6.sh` — preflight → train → evaluate 09/10 → plot
- `verify_protocol.py` — checks checkpoint metadata for target leakage
- `verify_protocol.sh` — wrapper for protocol verification

## Usage

From the repository root:

```bash
chmod +x experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/*.sh
experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/run_a6.sh
```

To evaluate an already-trained A6 checkpoint without retraining:

```bash
SKIP_TRAIN=1 \
CHECKPOINT=experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/checkpoints/latest.pt \
experiments/track_a/paper_reproduction/a6_unseen_00_08_to_09_10/run_a6.sh
```

Environment overrides include `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`,
`LEARNING_RATE`, `DEVICE`, `DATA_ROOT`, `CHECKPOINT_DIR`, and `CHECKPOINT`.
