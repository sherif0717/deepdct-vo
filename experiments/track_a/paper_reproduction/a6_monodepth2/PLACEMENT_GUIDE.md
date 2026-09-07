# Monodepth2 compatibility changes

This guide targets the supplied versions of `deepdct_vo.py`,
`train_deepdct_vo.py`, and `evaluate_deepdct_vo.py`. The dataset,
`train_one_epoch.py`, and `validate_one_epoch.py` require no functional change:
their existing `use_internal_depth` routing already sends `depth_curr=None`.

## 1. Add the adapter and vendor files

Copy this package's `deepdct/models/auxiliary/monodepth2.py` to the same path in
the repository.

Place the official Monodepth2 sources in:

```text
deepdct/models/auxiliary/monodepth2_vendor/
├── __init__.py
├── layers.py
└── networks/
    ├── __init__.py
    ├── depth_decoder.py
    └── resnet_encoder.py
```

In the copied `networks/depth_decoder.py`, replace its upstream import:

```python
from layers import ConvBlock, Conv3x3, upsample
```

with the package-relative import:

```python
from ..layers import ConvBlock, Conv3x3, upsample
```

Keep the official `networks/__init__.py` exports for `DepthDecoder` and
`ResnetEncoder`. Put `encoder.pth` and `depth.pth` from `mono_640x192` in
`weights/mono_640x192/`.

## 2. Update `deepdct/models/deepdct_vo.py`

### 2.1 Import the adapter

Immediately after the existing Lite-Mono import, add:

```python
from .auxiliary.monodepth2 import Monodepth2DepthBranch
```

### 2.2 Add a provider constructor argument

In `DeepDCTVO.__init__`, immediately before `depth_checkpoint_dir`, add:

```python
depth_provider: str = "lite_mono",
```

Immediately before `self.depth_model_name = str(depth_model_name)`, add:

```python
self.depth_provider = str(depth_provider).lower().replace("-", "_")
if self.depth_provider not in {"lite_mono", "monodepth2"}:
    raise ValueError(
        "depth_provider must be 'lite_mono' or 'monodepth2', "
        f"but received {depth_provider!r}."
    )
```

### 2.3 Replace the depth-model construction

Find the block beginning `if depth_model is not None:` and replace the complete
block through the existing `LiteMonoDepthBranch(...)` construction with:

```python
if depth_model is not None:
    self.depth_model = depth_model
elif self.depth_provider == "lite_mono":
    self.depth_model = LiteMonoDepthBranch(
        checkpoint_dir=depth_checkpoint_dir,
        model_name=self.depth_model_name,
        feed_size=depth_feed_size,
        output_mode=self.depth_output_mode,
        normalization_depth=self.depth_normalization_meters,
        freeze_pretrained=freeze_depth,
    )
else:
    if depth_checkpoint_dir is None:
        raise ValueError(
            "depth_checkpoint_dir is required for Monodepth2."
        )
    self.depth_model = Monodepth2DepthBranch(
        checkpoint_dir=depth_checkpoint_dir,
        num_layers=18,
        feed_size=depth_feed_size,
        output_mode=self.depth_output_mode,
        normalization_depth=self.depth_normalization_meters,
        freeze_pretrained=freeze_depth,
        strict_checkpoint=True,
    )
```

No change is needed in `_depth_cue`, `forward`, or `_validate_depth`.

## 3. Update `scripts/train_deepdct_vo.py`

### 3.1 Add the provider CLI option

Immediately before `--depth-checkpoint-dir`, add:

```python
parser.add_argument(
    "--depth-provider",
    choices=("lite_mono", "monodepth2"),
    default="lite_mono",
    help="Internal monocular-depth implementation.",
)
```

Replace the current `--depth-model-name` argument with:

```python
parser.add_argument(
    "--depth-model-name",
    type=str,
    default="lite-mono-tiny",
    help=(
        "Checkpoint/model identity: e.g. lite-mono-tiny or "
        "mono_640x192. Architecture selection is controlled by "
        "--depth-provider."
    ),
)
```

### 3.2 Generalize Track-A validation

In `validate_args`, replace the unconditional Lite-Mono model-name check:

```python
if args.depth_model_name != "lite-mono-tiny":
    ...
```

with:

```python
if (
    args.depth_provider == "lite_mono"
    and args.depth_model_name != "lite-mono-tiny"
):
    raise ValueError(
        "The Lite-Mono Track-A baseline requires "
        "--depth-model-name lite-mono-tiny."
    )

if (
    args.depth_provider == "monodepth2"
    and args.depth_model_name != "mono_640x192"
):
    raise ValueError(
        "The A6 Monodepth2 comparison requires "
        "--depth-model-name mono_640x192."
    )
```

Keep the existing requirements for frozen depth, `normalized_depth`, and
normalization depth `80.0`; these preserve the controlled depth-map contract.

### 3.3 Pass the provider into the model

In `build_model`, immediately before `depth_checkpoint_dir=...`, add:

```python
depth_provider=args.depth_provider,
```

### 3.4 Save correct metadata

In `save_checkpoint`, make both replacements below.

Inside `cue_config`, replace the hardcoded depth-model provider expression with:

```python
"depth_model": (
    args.depth_provider
    if args.use_depth_cues
    else None
),
```

Inside `configuration`, replace:

```python
"depth_model": "lite_mono",
```

with:

```python
"depth_provider": args.depth_provider,
"depth_model": args.depth_provider,
```

The duplicate `depth_model` field is retained for compatibility with older
evaluation code. New code should prefer `depth_provider`.

### 3.5 Generalize console reporting

Where the trainer prints `Lite-Mono`, replace the fixed label with:

```python
depth_label = (
    "Monodepth2"
    if args.depth_provider == "monodepth2"
    else "Lite-Mono"
)
```

and print `depth_label`. This is reporting-only but prevents mislabeled runs.

## 4. Update `scripts/evaluate_deepdct_vo.py`

### 4.1 Resolve the provider from checkpoint metadata

In `resolve_evaluation_configuration`, immediately after resolving
`use_depth_cues`, add:

```python
depth_provider = str(
    configuration.get(
        "depth_provider",
        configuration.get("depth_model", "lite_mono"),
    )
).lower().replace("-", "_")

if depth_provider not in {"lite_mono", "monodepth2"}:
    raise ValueError(
        f"Unsupported checkpoint depth provider: {depth_provider!r}."
    )
```

In that function's returned dictionary, add:

```python
"depth_provider": depth_provider,
```

Retain the existing `depth_model_name`, output-mode, and normalization entries.
The fallback makes all old A4/A6 Lite-Mono checkpoints continue to work.

### 4.2 Construct the correct provider before strict loading

In `build_model`, immediately before `depth_checkpoint_dir=...`, add:

```python
depth_provider=str(
    evaluation_configuration["depth_provider"]
),
```

Keep `model.load_state_dict(..., strict=True)`. Once the checkpoint-selected
provider is constructed, strict loading is the desired compatibility check.

### 4.3 Correct evaluation labels

Replace fixed `internal Lite-Mono` / `Lite-Mono` output strings with a label
derived from `evaluation_configuration["depth_provider"]`:

```python
depth_provider = str(evaluation_configuration["depth_provider"])
depth_label = (
    "Monodepth2"
    if depth_provider == "monodepth2"
    else "Lite-Mono"
)
```

## 5. Files that do not need functional changes

- `deepdct/data/training_dataset.py`
- `deepdct/training/train_one_epoch.py`
- `deepdct/training/validate_one_epoch.py`
- `deepdct/models/auxiliary/lite_mono.py`
- the Lite-Mono `depth_encoder.py`

Their existing contract is already provider-neutral at the `[B, 1, H, W]`
boundary. Comments mentioning Lite-Mono may be changed to “internal depth
provider,” but that is optional.

## 6. Install and run the wrapper

Copy the supplied wrapper directory to:

```text
experiments/track_a/paper_reproduction/a6_monodepth2/
```

Then run:

```bash
chmod +x experiments/track_a/paper_reproduction/a6_monodepth2/*.sh

experiments/track_a/paper_reproduction/a6_monodepth2/preflight.sh
experiments/track_a/paper_reproduction/a6_monodepth2/run_a6_monodepth2.sh
```

Individual commands:

```bash
experiments/track_a/paper_reproduction/a6_monodepth2/train.sh
experiments/track_a/paper_reproduction/a6_monodepth2/evaluate_09.sh
experiments/track_a/paper_reproduction/a6_monodepth2/evaluate_10.sh
```

For an unscaled provider comparison, override the inherited A5 trajectory
scales for both evaluations:

```bash
TRANSLATION_SCALE_09=1.0 \
  experiments/track_a/paper_reproduction/a6_monodepth2/evaluate_09.sh

TRANSLATION_SCALE_10=1.0 \
  experiments/track_a/paper_reproduction/a6_monodepth2/evaluate_10.sh
```

The default wrapper keeps `0.975` for sequence 09 and `1.007` for sequence 10
to match the supplied Lite-Mono A6 wrappers.

## 7. Required smoke checks before full training

```bash
python -m compileall \
  deepdct/models/auxiliary/monodepth2.py \
  deepdct/models/deepdct_vo.py \
  scripts/train_deepdct_vo.py \
  scripts/evaluate_deepdct_vo.py

pytest -q tests/models/test_deepdct_forward.py
```

Also verify one Monodepth2 forward pass returns a finite float tensor shaped
`[B, 1, 120, 120]`, and confirm that every depth-model parameter has
`requires_grad=False` in this A6 experiment.
