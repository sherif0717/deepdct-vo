#!/usr/bin/env bash

set -euo pipefail


# ============================================================================
# DeepDCT-VO two-track repository restructuring
#
# Track A:
#   Paper-faithful PyTorch reproduction
#
# Track B:
#   Robust predicted-rotation / end-to-end extension
#
# IMPORTANT:
#   Shared implementation code remains in:
#
#       deepdct/
#       tests/
#       scripts/train_deepdct_vo.py
#       scripts/evaluate_deepdct_vo.py
#
#   This avoids breaking imports during restructuring.
# ============================================================================


ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "${ROOT}" ]]; then
    echo "ERROR: Run this script inside the DeepDCT-VO git repository."
    exit 1
fi

cd "${ROOT}"

echo "Repository root: ${ROOT}"


# ----------------------------------------------------------------------------
# Directory structure
# ----------------------------------------------------------------------------

mkdir -p \
    tracks/track_a_paper_reproduction/configs \
    tracks/track_a_paper_reproduction/scripts \
    tracks/track_a_paper_reproduction/docs \
    tracks/track_b_end_to_end_extension/configs \
    tracks/track_b_end_to_end_extension/scripts/diagnostics \
    tracks/track_b_end_to_end_extension/scripts/debug \
    tracks/track_b_end_to_end_extension/scripts/workflows \
    tracks/track_b_end_to_end_extension/docs


# ----------------------------------------------------------------------------
# Helper: move file using git mv if tracked, otherwise ordinary mv.
# ----------------------------------------------------------------------------

move_file()
{
    local src="$1"
    local dst="$2"

    if [[ ! -e "${src}" ]]; then
        echo "SKIP: ${src}"
        return
    fi

    mkdir -p "$(dirname "${dst}")"

    if git ls-files --error-unmatch "${src}" >/dev/null 2>&1; then
        echo "git mv: ${src} -> ${dst}"
        git mv "${src}" "${dst}"
    else
        echo "mv:     ${src} -> ${dst}"
        mv "${src}" "${dst}"
    fi
}


remove_file()
{
    local path="$1"

    if [[ ! -e "${path}" ]]; then
        echo "SKIP: ${path}"
        return
    fi

    if git ls-files --error-unmatch "${path}" >/dev/null 2>&1; then
        echo "git rm: ${path}"
        git rm "${path}"
    else
        echo "rm:     ${path}"
        rm -f "${path}"
    fi
}


# ============================================================================
# TRACK B — diagnostic / representation research
# ============================================================================

TRACK_B_DIAGNOSTICS=(
    scripts/analyze_motion_regimes.py
    scripts/analyze_pooled_refined_prediction_regimes.py
    scripts/analyze_pose_error_attribution.py
    scripts/analyze_refined_motion_regimes.py
    scripts/analyze_refined_prediction_regimes.py
    scripts/analyze_representation_regimes.py
    scripts/analyze_rotation_bottleneck_alignment.py
    scripts/analyze_rotation_conditional_alignment.py
    scripts/analyze_rotation_error.py
    scripts/analyze_rotation_geometry_alignment.py
    scripts/analyze_rotation_latent_domain_alignment.py
    scripts/analyze_rotation_representation.py
    scripts/analyze_training_motion_regimes.py
    scripts/analyze_translation_aggregation_representation.py
    scripts/analyze_translation_latent_alignment.py
    scripts/analyze_translation_motion_regimes.py
    scripts/analyze_translation_predictions.py
    scripts/audit_rotation_geometry_generalization.py
    scripts/audit_rotation_representation_readout.py
    scripts/build_rotation_motion_alignment_pairs.py
    scripts/fine_tune_translation_head.py
    scripts/fit_translation_affine_calibration.py
    scripts/apply_translation_affine_calibration.py
)

for src in "${TRACK_B_DIAGNOSTICS[@]}"; do
    filename="$(basename "${src}")"

    move_file \
        "${src}" \
        "tracks/track_b_end_to_end_extension/scripts/diagnostics/${filename}"
done


# ----------------------------------------------------------------------------
# Track-B debug probes
# ----------------------------------------------------------------------------

TRACK_B_DEBUG=(
    scripts/debug/audit_translation_representation_reconstruction.py
    scripts/debug/probe_translation_representation.py
    scripts/debug/probe_translation_representation_memory_safe.py
)

for src in "${TRACK_B_DEBUG[@]}"; do
    filename="$(basename "${src}")"

    move_file \
        "${src}" \
        "tracks/track_b_end_to_end_extension/scripts/debug/${filename}"
done


# ----------------------------------------------------------------------------
# Experimental workflow
# ----------------------------------------------------------------------------

move_file \
    run_semantic_depth_after_baseline.sh \
    tracks/track_b_end_to_end_extension/scripts/workflows/run_semantic_depth_after_baseline.sh


# ----------------------------------------------------------------------------
# Experimental configs
# ----------------------------------------------------------------------------

move_file \
    configs/integrated_comparison.json \
    tracks/track_b_end_to_end_extension/configs/integrated_comparison.json

move_file \
    configs/integrated_comparison_layers.json \
    tracks/track_b_end_to_end_extension/configs/integrated_comparison_layers.json


# ============================================================================
# Remove obsolete duplicate/archive files
# ============================================================================

remove_file scripts/evaluate_deepdct_vo.py.before_directional_decode
remove_file deepdct/models.zip
remove_file deepdct/training.zip
remove_file integrated_comparison_inputs.tar.gz


# ============================================================================
# Track documentation
# ============================================================================

cat > tracks/track_a_paper_reproduction/README.md <<'EOF'
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
EOF


cat > tracks/track_b_end_to_end_extension/README.md <<'EOF'
# Track B — End-to-End DeepDCT-VO Extension

Objective:

Remove simplifying assumptions from the paper reproduction and investigate
robust end-to-end visual odometry using predicted rotation.

Current research topics:

- Predicted rotation supplied to Model T
- Compact pooled translation representations
- Rotation and translation representation diagnostics
- Cross-sequence latent alignment
- Continuous SO(3) supervision
- Motion-regime analysis
- Turn-aware rotation modeling
- Sequence / appearance invariance

This directory preserves experimental diagnostics developed before the
paper-reproduction restructuring.
EOF


cat > PROJECT_TRACKS.md <<'EOF'
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
EOF


# ============================================================================
# Track-B implementation notes
# ============================================================================

cat > tracks/track_b_end_to_end_extension/docs/IMPLEMENTATION_COMPONENTS.md <<'EOF'
# Track-B implementation components still located in the shared package

The following modules remain under `deepdct/` because the current shared
training/evaluation implementation imports them:

- `deepdct/training/motion_alignment.py`
- `deepdct/training/rotation_geometry.py`

The Track-A branch may later remove their hooks from the reproduction path.

Do not physically relocate these modules until the shared package has been
split or imports have been refactored.
EOF


echo
echo "========================================================================"
echo "Two-track restructuring complete"
echo "========================================================================"
echo
echo "Shared implementation remains at repository root."
echo
echo "Track A:"
echo "  tracks/track_a_paper_reproduction"
echo
echo "Track B:"
echo "  tracks/track_b_end_to_end_extension"
echo
echo "Next:"
echo "  git status"
echo "  git diff --stat"
echo "========================================================================"