# A7 SegFormer + Monodepth2 drop-in

This bundle is based on the supplied A6 files and keeps the A6 source-only
protocol. A7 changes only the semantic auxiliary from LR-ASPP to SegFormer.

## 1. Install the additional dependency

From the activated project virtual environment:

```bash
python -m pip install "transformers==4.46.3"
```

The first preflight run downloads
`nvidia/segformer-b0-finetuned-ade-512-512` unless `A7_SEMANTIC_MODEL` points to
a local model directory. After it is cached, set
`A7_SEMANTIC_MODEL=/path/to/local/model` for offline use.

## 2. Place the files

Copy the bundle contents over the repository root:

```bash
cp -a a7_segformer_monodepth2_dropin/. /media/sherifdeen/ext_hd250/projects/deepdct-vo/
cd /media/sherifdeen/ext_hd250/projects/deepdct-vo
chmod +x experiments/track_a/paper_reproduction/a7_segformer_monodepth2/*.sh
```

The copied paths are:

- `deepdct/models/auxiliary/segformer.py` — new provider.
- `deepdct/models/deepdct_vo.py` — complete A7 replacement based on the
  supplied `deepdct_vo(7).py`.
- `scripts/train_deepdct_vo.py` — complete A7 replacement based on the
  supplied `train_deepdct_vo(7).py`.
- `scripts/evaluate_deepdct_vo.py` — complete A7 replacement based on the
  supplied `evaluate_deepdct_vo(20260907-123710).py`.
- `tests/models/test_deepdct_forward.py` — complete A7 replacement based on
  the supplied test file.
- `experiments/track_a/paper_reproduction/a7_segformer_monodepth2/` — wrapper.

## 3. Exact modification anchors

### `deepdct/models/deepdct_vo.py`

1. Immediately after the `lraspp_fig2` import, import
   `SegFormerSemanticBranch` and `DEFAULT_FOREGROUND_LABELS`.
2. In `DeepDCTVO.__init__`, immediately after `semantic_map_mode`, add the
   `semantic_provider`, model identity, foreground IDs/labels, feed size,
   local-files-only, and injectable `semantic_model` parameters.
3. Immediately after `self.semantic_map_mode`, normalize and validate the new
   provider configuration.
4. Replace the unconditional `self.semantic_model = LRASPPSemanticBranch(...)`
   block with the three-way injected/LR-ASPP/SegFormer construction block.
5. Freeze the selected semantic module generically after construction. The
   existing `train()` override continues forcing it into evaluation mode.

### `scripts/train_deepdct_vo.py`

1. In `parse_args()`, immediately after the pretrained-semantic defaults, add
   all `--semantic-*` SegFormer options.
2. In `validate_args()`, immediately after depth-normalization validation, add
   semantic feed-size, model-name, map-mode, and frozen-provider checks.
3. In `build_model()`, immediately after `semantic_map_mode=...`, forward the
   complete semantic-provider configuration to `DeepDCTVO`.
4. In both checkpoint configuration dictionaries, replace hard-coded
   `semantic_model: lraspp` and record provider, model ID, resolved class IDs,
   label names, feed size, and local-files-only state.
5. In the training configuration report, replace the hard-coded LR-ASPP label
   with the selected provider and print the SegFormer-specific configuration.

### `scripts/evaluate_deepdct_vo.py`

1. In `resolve_evaluation_configuration()`, immediately after resolving
   `semantic_map_mode`, recover the provider and all SegFormer parameters.
   `semantic_model` remains a backward-compatible fallback for older
   checkpoints.
2. Add those values to the returned evaluation configuration.
3. In `build_model()`, pass them to `DeepDCTVO` before the existing strict
   `load_state_dict(..., strict=True)` call. This ordering is essential.
4. Replace hard-coded LR-ASPP reporting with provider-aware reporting.

### `tests/models/test_deepdct_forward.py`

1. Give `DummySemanticBranch` a trainable scalar so freezing can be asserted.
2. Append tests for injected SegFormer-provider configuration, one-channel
   output shapes, parameter freezing/eval mode, and invalid-provider rejection.
3. The real model download/inference smoke test belongs in `preflight.sh`, not
   the ordinary unit suite.

## 4. Run A7

Run each phase independently:

```bash
experiments/track_a/paper_reproduction/a7_segformer_monodepth2/preflight.sh
experiments/track_a/paper_reproduction/a7_segformer_monodepth2/train.sh
experiments/track_a/paper_reproduction/a7_segformer_monodepth2/evaluate.sh
experiments/track_a/paper_reproduction/a7_segformer_monodepth2/evaluate_gt_gt.sh
```

Or run the full pipeline:

```bash
experiments/track_a/paper_reproduction/a7_segformer_monodepth2/run_a7.sh
```

Optional environment overrides:

```bash
export A7_DATA_ROOT=/media/sherifdeen/ext_hd250/projects/deepdct-vo/data
export A7_DEPTH_WEIGHTS=/media/sherifdeen/ext_hd250/projects/deepdct-vo/weights/mono_640x192
export A7_SEMANTIC_MODEL=nvidia/segformer-b0-finetuned-ade-512-512
export A7_DEVICE=cuda
```

## 5. Controlled-comparison contract

- Train sequences: 00–08 only.
- Validation: disabled.
- Selected checkpoint: fixed epoch 15 `latest.pt`.
- Test sequences: unseen 09 and 10.
- Model-T conditioning during training: ground-truth rotation.
- Semantic cue: frozen SegFormer B0 foreground probability.
- Depth cue: frozen Monodepth2 `mono_640x192`, normalized to 80 m.
- Translation decoder: dense.
- Evaluation scale: 1.0, so no A5 trajectory rescaling contaminates A7.

The default foreground set contains dynamic road participants. The resolved
numeric IDs are stored in the checkpoint and reused by evaluation, preventing
class-map drift during strict reconstruction.

