# A6 Monodepth2 drop-in package

Contents:

- `deepdct/models/auxiliary/monodepth2.py`: provider adapter preserving the
  existing normalized-depth contract.
- `experiments/track_a/paper_reproduction/a6_monodepth2/`: source-only A6
  training and sequence 09/10 evaluation wrappers.
- `PLACEMENT_GUIDE.md`: exact code insertion and replacement instructions.

The package intentionally excludes the upstream Monodepth2 source and weights.
Use the official non-commercially licensed implementation and pretrained
`mono_640x192` checkpoint, then follow `PLACEMENT_GUIDE.md`.
