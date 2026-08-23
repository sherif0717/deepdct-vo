# A5 evaluator placement guide

Only `scripts/evaluate_deepdct_vo.py` needs a core-code update for A5.

## 1. Add the CLI argument

Place this beside the evaluator's other evaluation/post-processing scalar arguments:

```python
parser.add_argument(
    "--translation-scale-factor",
    type=float,
    default=1.0,
    help=(
        "Multiplicative scale applied to each predicted relative translation "
        "before SE(3) trajectory composition. Default 1.0 preserves A1-A4 behavior."
    ),
)
```

## 2. Validate it after argument parsing

Place this with the evaluator's other scalar argument validation:

```python
if args.translation_scale_factor <= 0.0:
    raise ValueError("--translation-scale-factor must be > 0")
```

## 3. Apply the scale at the physical-pose boundary

Find the block where a network translation prediction has already been restored to its **physical units** and is about to be used to create the relative predicted transform / compose the trajectory.

Immediately before construction of the relative transform, add:

```python
pred_translation_unscaled = pred_translation.copy()
pred_translation = pred_translation * args.translation_scale_factor
```

If `pred_translation` is still a torch tensor there, use:

```python
pred_translation_unscaled = pred_translation.clone()
pred_translation = pred_translation * float(args.translation_scale_factor)
```

Do **not** multiply the normalized training target, normalized rotation, Euler angles, or rotation matrix.

## 4. Preserve both raw and scaled diagnostics

Where per-frame predictions are written, retain the normal scaled prediction columns and, if convenient, add raw translation columns:

```python
row["pred_tx_unscaled"] = float(pred_translation_unscaled[0])
row["pred_ty_unscaled"] = float(pred_translation_unscaled[1])
row["pred_tz_unscaled"] = float(pred_translation_unscaled[2])
row["translation_scale_factor"] = float(args.translation_scale_factor)
```

This is recommended rather than required because the wrapper separately runs factor `1.0` and the paper factor.

## 5. Print and checkpoint/evaluation metadata

Add the factor to the evaluation header/configuration:

```python
print(f"Translation scale:     {args.translation_scale_factor:.6f}")
```

and, if the evaluator serializes an evaluation configuration dict:

```python
"translation_scale_factor": float(args.translation_scale_factor),
```

## A5 invariants

- no optimizer/training changes;
- no `train_one_epoch.py` changes;
- no `validate_one_epoch.py` changes;
- no `deepdct/models/*` changes;
- translation only is scaled;
- scaling occurs after conversion to physical translation and before trajectory composition;
- factor `1.0` must reproduce the A4 evaluator numerically (up to file/rounding effects).
