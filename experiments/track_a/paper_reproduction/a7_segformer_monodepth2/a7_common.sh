#!/usr/bin/env bash
set -euo pipefail

A7_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A7_REPO_ROOT="$(cd "${A7_SCRIPT_DIR}/../../../.." && pwd)"
A7_OUTPUT_ROOT="${A7_REPO_ROOT}/experiments/track_a/paper_reproduction/a7_segformer_monodepth2"
A7_CHECKPOINT_DIR="${A7_OUTPUT_ROOT}/checkpoints"
A7_CHECKPOINT="${A7_CHECKPOINT_DIR}/latest.pt"
A7_DATA_ROOT="${A7_DATA_ROOT:-${A7_REPO_ROOT}/data}"
A7_DEPTH_WEIGHTS="${A7_DEPTH_WEIGHTS:-${A7_REPO_ROOT}/weights/mono_640x192}"
A7_SEMANTIC_MODEL="${A7_SEMANTIC_MODEL:-nvidia/segformer-b0-finetuned-ade-512-512}"
A7_DEVICE="${A7_DEVICE:-auto}"

export A7_SCRIPT_DIR A7_REPO_ROOT A7_OUTPUT_ROOT A7_CHECKPOINT_DIR
export A7_CHECKPOINT A7_DATA_ROOT A7_DEPTH_WEIGHTS A7_SEMANTIC_MODEL A7_DEVICE

cd "${A7_REPO_ROOT}"

