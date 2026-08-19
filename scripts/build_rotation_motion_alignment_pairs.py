#!/usr/bin/env python3
"""Build motion-conditioned rotation-representation alignment pairs.

This standalone script prepares pair assignments for analyzing whether the
compact rotation representation z_R is aligned across KITTI sequences when
conditioned on forward-motion regime.

Expected input
--------------
Each input ``rotation_representations.npz`` should contain at least:

    rotation_rep
        Compact task-relevant rotation representation.
        Shape: [N, D]

    rotation_gt_array
        Ground-truth relative rotation.
        Shape: [N, 3]

    translation_gt_array
        Ground-truth relative translation.
        Shape: [N, 3]

    motion_regime_array
        Training-threshold-derived motion regime.
        Shape: [N]

        Coding:
            0 = low
            1 = medium
            2 = high

The preferred experimental setup exports one NPZ per KITTI sequence.

Purpose
-------
The script builds:

1. Positive alignment pairs
       same motion regime
       preferably different sequences

2. Negative comparison pairs
       different motion regimes
       preferably different sequences

The resulting pair table supports questions such as:

    - Are same-regime representations closer across sequences?
    - Does motion-conditioned training reduce cross-sequence latent shift?
    - Are low/medium/high regimes geometrically separable?
    - Does sequence identity dominate the compact representation?

No model parameters are modified by this script.

Example
-------
python3 scripts/build_rotation_motion_alignment_pairs.py \\
    --input experiments/rotation_alignment/00/rotation_representations.npz \\
    --input experiments/rotation_alignment/01/rotation_representations.npz \\
    --input experiments/rotation_alignment/02/rotation_representations.npz \\
    --input experiments/rotation_alignment/09/rotation_representations.npz \\
    --input experiments/rotation_alignment/10/rotation_representations.npz \\
    --output-dir experiments/rotation_motion_alignment_pairs \\
    --pairs-per-anchor 1 \\
    --negative-pairs-per-anchor 1 \\
    --seed 42

Outputs
-------
<output-dir>/
├── alignment_pairs.csv
├── rotation_motion_alignment_pairs.npz
└── pair_summary.txt

``alignment_pairs.csv`` stores indices, sequences, regimes, rotation and
translation metadata, and representation-distance measurements.

``rotation_motion_alignment_pairs.npz`` stores the actual representation
vectors and compact pair metadata for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REGIME_NAMES = {
    0: "low",
    1: "medium",
    2: "high",
}


@dataclass(frozen=True)
class RepresentationSample:
    """Metadata identifying one exported rotation representation."""

    global_index: int
    source_index: int
    sequence: str
    regime: int

    rotation_gt_x: float
    rotation_gt_y: float
    rotation_gt_z: float

    translation_gt_x: float
    translation_gt_y: float
    translation_gt_z: float


@dataclass(frozen=True)
class AlignmentPair:
    """One representation pair."""

    pair_index: int
    pair_type: str

    anchor_global_index: int
    partner_global_index: int

    anchor_source_index: int
    partner_source_index: int

    anchor_sequence: str
    partner_sequence: str

    anchor_regime: int
    partner_regime: int

    same_sequence: bool
    same_regime: bool

    euclidean_distance: float
    cosine_similarity: float

    rotation_gt_distance: float
    translation_gt_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build positive and negative representation pairs for "
            "motion-conditioned compact rotation alignment analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help=(
            "rotation_representations.npz file. Repeat --input for "
            "each sequence/export to include."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated pair files.",
    )

    parser.add_argument(
        "--pairs-per-anchor",
        type=int,
        default=1,
        help=(
            "Number of same-regime positive partners generated "
            "for each anchor."
        ),
    )

    parser.add_argument(
        "--negative-pairs-per-anchor",
        type=int,
        default=1,
        help=(
            "Number of different-regime negative partners generated "
            "for each anchor."
        ),
    )

    parser.add_argument(
        "--cross-sequence-only",
        action="store_true",
        help=(
            "Require both positive and negative partners to come from "
            "a sequence different from the anchor."
        ),
    )

    parser.add_argument(
        "--max-samples-per-regime-per-sequence",
        type=int,
        default=None,
        help=(
            "Optionally subsample each sequence/regime combination "
            "before pair construction."
        ),
    )

    parser.add_argument(
        "--normalize-representations",
        action="store_true",
        help=(
            "L2-normalize representations before computing pair "
            "distances and writing NPZ vectors."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible pair construction.",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.inputs:
        raise ValueError(
            "At least one --input file is required."
        )

    if args.pairs_per_anchor < 0:
        raise ValueError(
            "--pairs-per-anchor cannot be negative."
        )

    if args.negative_pairs_per_anchor < 0:
        raise ValueError(
            "--negative-pairs-per-anchor cannot be negative."
        )

    if (
        args.pairs_per_anchor == 0
        and args.negative_pairs_per_anchor == 0
    ):
        raise ValueError(
            "At least one positive or negative pair per anchor "
            "must be requested."
        )

    if (
        args.max_samples_per_regime_per_sequence
        is not None
        and args.max_samples_per_regime_per_sequence <= 0
    ):
        raise ValueError(
            "--max-samples-per-regime-per-sequence must be positive."
        )

    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(
                f"Input NPZ does not exist: {path}"
            )


def infer_sequence(
    path: Path,
) -> str:
    """Infer KITTI sequence identifier from the input path.

    The nearest parent/path component consisting solely of 1-2 digits
    is preferred. Otherwise the immediate parent directory name is used.
    """

    for component in reversed(path.parts[:-1]):
        if re.fullmatch(r"\d{1,2}", component):
            return component.zfill(2)

    return path.parent.name


def resolve_npz_key(
    archive: np.lib.npyio.NpzFile,
    candidates: Sequence[str],
    description: str,
) -> str:
    for candidate in candidates:
        if candidate in archive.files:
            return candidate

    raise KeyError(
        f"NPZ is missing {description}. "
        f"Accepted keys: {list(candidates)}. "
        f"Available keys: {archive.files}."
    )


def validate_loaded_arrays(
    path: Path,
    representation: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
    motion_regime: np.ndarray,
) -> None:
    if representation.ndim != 2:
        raise ValueError(
            f"{path}: rotation representation must be [N, D], "
            f"received {representation.shape}."
        )

    if representation.shape[0] == 0:
        raise ValueError(
            f"{path}: representation array is empty."
        )

    n_samples = representation.shape[0]

    if rotation_gt.shape != (n_samples, 3):
        raise ValueError(
            f"{path}: rotation GT must have shape "
            f"({n_samples}, 3), received {rotation_gt.shape}."
        )

    if translation_gt.shape != (n_samples, 3):
        raise ValueError(
            f"{path}: translation GT must have shape "
            f"({n_samples}, 3), received {translation_gt.shape}."
        )

    motion_regime = np.asarray(
        motion_regime
    ).reshape(-1)

    if motion_regime.shape != (n_samples,):
        raise ValueError(
            f"{path}: motion regime must have shape "
            f"({n_samples},), received {motion_regime.shape}."
        )

    for name, array in (
        ("rotation_rep", representation),
        ("rotation_gt", rotation_gt),
        ("translation_gt", translation_gt),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{path}: {name} contains NaN or infinity."
            )

    if not np.all(np.isfinite(motion_regime)):
        raise ValueError(
            f"{path}: motion_regime_array contains NaN or infinity."
        )

    rounded = np.rint(
        motion_regime
    ).astype(np.int64)

    if not np.allclose(
        motion_regime,
        rounded,
    ):
        raise ValueError(
            f"{path}: motion regimes must be integer-valued."
        )

    invalid = sorted(
        set(rounded.tolist())
        - set(REGIME_NAMES.keys())
    )

    if invalid:
        raise ValueError(
            f"{path}: unsupported motion regime IDs: {invalid}."
        )


def l2_normalize_rows(
    matrix: np.ndarray,
    eps: float = 1.0e-12,
) -> np.ndarray:
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        eps,
    )

    return matrix / norms


def load_inputs(
    paths: Sequence[Path],
    normalize_representations: bool,
) -> Tuple[
    np.ndarray,
    List[RepresentationSample],
]:
    """Load and concatenate all representation exports."""

    representation_blocks: List[np.ndarray] = []
    samples: List[RepresentationSample] = []

    expected_dimension: Optional[int] = None
    global_offset = 0

    for path in paths:
        sequence = infer_sequence(
            path
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            representation_key = resolve_npz_key(
                archive,
                (
                    "rotation_rep",
                    "rotation_representation",
                ),
                "rotation representation",
            )

            rotation_key = resolve_npz_key(
                archive,
                (
                    "rotation_gt_array",
                    "rotation_gt",
                ),
                "rotation ground truth",
            )

            translation_key = resolve_npz_key(
                archive,
                (
                    "translation_gt_array",
                    "translation_gt",
                ),
                "translation ground truth",
            )

            regime_key = resolve_npz_key(
                archive,
                (
                    "motion_regime_array",
                    "motion_regime",
                ),
                "motion-regime assignment",
            )

            representation = np.asarray(
                archive[representation_key],
                dtype=np.float64,
            )

            rotation_gt = np.asarray(
                archive[rotation_key],
                dtype=np.float64,
            )

            translation_gt = np.asarray(
                archive[translation_key],
                dtype=np.float64,
            )

            motion_regime = np.asarray(
                archive[regime_key]
            ).reshape(-1)

        validate_loaded_arrays(
            path=path,
            representation=representation,
            rotation_gt=rotation_gt,
            translation_gt=translation_gt,
            motion_regime=motion_regime,
        )

        if expected_dimension is None:
            expected_dimension = int(
                representation.shape[1]
            )

        elif (
            representation.shape[1]
            != expected_dimension
        ):
            raise ValueError(
                "All inputs must contain the same rotation "
                "representation dimension. "
                f"Expected {expected_dimension}, but "
                f"{path} contains {representation.shape[1]}."
            )

        if normalize_representations:
            representation = l2_normalize_rows(
                representation
            )

        motion_regime = np.rint(
            motion_regime
        ).astype(np.int64)

        for source_index in range(
            representation.shape[0]
        ):
            samples.append(
                RepresentationSample(
                    global_index=(
                        global_offset
                        + source_index
                    ),
                    source_index=source_index,
                    sequence=sequence,
                    regime=int(
                        motion_regime[
                            source_index
                        ]
                    ),
                    rotation_gt_x=float(
                        rotation_gt[
                            source_index,
                            0,
                        ]
                    ),
                    rotation_gt_y=float(
                        rotation_gt[
                            source_index,
                            1,
                        ]
                    ),
                    rotation_gt_z=float(
                        rotation_gt[
                            source_index,
                            2,
                        ]
                    ),
                    translation_gt_x=float(
                        translation_gt[
                            source_index,
                            0,
                        ]
                    ),
                    translation_gt_y=float(
                        translation_gt[
                            source_index,
                            1,
                        ]
                    ),
                    translation_gt_z=float(
                        translation_gt[
                            source_index,
                            2,
                        ]
                    ),
                )
            )

        representation_blocks.append(
            representation
        )

        global_offset += (
            representation.shape[0]
        )

        print(
            f"Loaded {path}: "
            f"sequence={sequence} "
            f"N={representation.shape[0]} "
            f"D={representation.shape[1]}"
        )

    representations = np.concatenate(
        representation_blocks,
        axis=0,
    )

    return representations, samples


def subsample_samples(
    samples: Sequence[RepresentationSample],
    maximum_per_group: Optional[int],
    rng: random.Random,
) -> List[RepresentationSample]:
    if maximum_per_group is None:
        return list(samples)

    groups: Dict[
        Tuple[str, int],
        List[RepresentationSample],
    ] = {}

    for sample in samples:
        groups.setdefault(
            (
                sample.sequence,
                sample.regime,
            ),
            [],
        ).append(sample)

    selected: List[RepresentationSample] = []

    for key in sorted(groups):
        group = groups[key]

        if len(group) <= maximum_per_group:
            selected.extend(
                group
            )
            continue

        selected.extend(
            rng.sample(
                group,
                maximum_per_group,
            )
        )

    selected.sort(
        key=lambda sample: sample.global_index
    )

    return selected


def vector_from_rotation(
    sample: RepresentationSample,
) -> np.ndarray:
    return np.asarray(
        [
            sample.rotation_gt_x,
            sample.rotation_gt_y,
            sample.rotation_gt_z,
        ],
        dtype=np.float64,
    )


def vector_from_translation(
    sample: RepresentationSample,
) -> np.ndarray:
    return np.asarray(
        [
            sample.translation_gt_x,
            sample.translation_gt_y,
            sample.translation_gt_z,
        ],
        dtype=np.float64,
    )


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
    eps: float = 1.0e-12,
) -> float:
    denominator = float(
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= eps:
        return 0.0

    return float(
        np.dot(first, second)
        / denominator
    )


def make_pair(
    pair_index: int,
    pair_type: str,
    anchor: RepresentationSample,
    partner: RepresentationSample,
    representations: np.ndarray,
) -> AlignmentPair:
    anchor_rep = representations[
        anchor.global_index
    ]

    partner_rep = representations[
        partner.global_index
    ]

    euclidean_distance = float(
        np.linalg.norm(
            anchor_rep
            - partner_rep
        )
    )

    similarity = cosine_similarity(
        anchor_rep,
        partner_rep,
    )

    rotation_distance = float(
        np.linalg.norm(
            vector_from_rotation(anchor)
            - vector_from_rotation(partner)
        )
    )

    translation_distance = float(
        np.linalg.norm(
            vector_from_translation(anchor)
            - vector_from_translation(partner)
        )
    )

    return AlignmentPair(
        pair_index=pair_index,
        pair_type=pair_type,
        anchor_global_index=anchor.global_index,
        partner_global_index=partner.global_index,
        anchor_source_index=anchor.source_index,
        partner_source_index=partner.source_index,
        anchor_sequence=anchor.sequence,
        partner_sequence=partner.sequence,
        anchor_regime=anchor.regime,
        partner_regime=partner.regime,
        same_sequence=(
            anchor.sequence
            == partner.sequence
        ),
        same_regime=(
            anchor.regime
            == partner.regime
        ),
        euclidean_distance=euclidean_distance,
        cosine_similarity=similarity,
        rotation_gt_distance=rotation_distance,
        translation_gt_distance=translation_distance,
    )


def choose_candidates(
    anchor: RepresentationSample,
    samples: Sequence[RepresentationSample],
    same_regime: bool,
    cross_sequence_only: bool,
) -> List[RepresentationSample]:
    candidates: List[RepresentationSample] = []

    for candidate in samples:
        if (
            candidate.global_index
            == anchor.global_index
        ):
            continue

        if (
            candidate.regime == anchor.regime
        ) != same_regime:
            continue

        if (
            cross_sequence_only
            and candidate.sequence
            == anchor.sequence
        ):
            continue

        candidates.append(
            candidate
        )

    return candidates


def prefer_cross_sequence(
    anchor: RepresentationSample,
    candidates: Sequence[RepresentationSample],
) -> List[RepresentationSample]:
    """Prefer different-sequence candidates when available."""

    cross_sequence = [
        candidate
        for candidate in candidates
        if candidate.sequence
        != anchor.sequence
    ]

    if cross_sequence:
        return cross_sequence

    return list(candidates)


def sample_without_replacement_or_repeat(
    candidates: Sequence[RepresentationSample],
    count: int,
    rng: random.Random,
) -> List[RepresentationSample]:
    if count <= 0 or not candidates:
        return []

    if len(candidates) >= count:
        return rng.sample(
            list(candidates),
            count,
        )

    # When a regime contains too few candidates, include all once
    # rather than silently dropping the anchor.
    return rng.sample(
        list(candidates),
        len(candidates),
    )


def build_pairs(
    samples: Sequence[RepresentationSample],
    representations: np.ndarray,
    positive_pairs_per_anchor: int,
    negative_pairs_per_anchor: int,
    cross_sequence_only: bool,
    rng: random.Random,
) -> List[AlignmentPair]:
    pairs: List[AlignmentPair] = []

    pair_index = 0

    for anchor in samples:
        positive_candidates = choose_candidates(
            anchor=anchor,
            samples=samples,
            same_regime=True,
            cross_sequence_only=cross_sequence_only,
        )

        if not cross_sequence_only:
            positive_candidates = (
                prefer_cross_sequence(
                    anchor,
                    positive_candidates,
                )
            )

        selected_positive = (
            sample_without_replacement_or_repeat(
                positive_candidates,
                positive_pairs_per_anchor,
                rng,
            )
        )

        for partner in selected_positive:
            pairs.append(
                make_pair(
                    pair_index=pair_index,
                    pair_type="positive_same_regime",
                    anchor=anchor,
                    partner=partner,
                    representations=representations,
                )
            )

            pair_index += 1

        negative_candidates = choose_candidates(
            anchor=anchor,
            samples=samples,
            same_regime=False,
            cross_sequence_only=cross_sequence_only,
        )

        if not cross_sequence_only:
            negative_candidates = (
                prefer_cross_sequence(
                    anchor,
                    negative_candidates,
                )
            )

        selected_negative = (
            sample_without_replacement_or_repeat(
                negative_candidates,
                negative_pairs_per_anchor,
                rng,
            )
        )

        for partner in selected_negative:
            pairs.append(
                make_pair(
                    pair_index=pair_index,
                    pair_type="negative_different_regime",
                    anchor=anchor,
                    partner=partner,
                    representations=representations,
                )
            )

            pair_index += 1

    return pairs


def write_pairs_csv(
    path: Path,
    pairs: Sequence[AlignmentPair],
) -> None:
    fieldnames = [
        "pair_index",
        "pair_type",
        "anchor_global_index",
        "partner_global_index",
        "anchor_source_index",
        "partner_source_index",
        "anchor_sequence",
        "partner_sequence",
        "anchor_regime",
        "anchor_regime_name",
        "partner_regime",
        "partner_regime_name",
        "same_sequence",
        "same_regime",
        "euclidean_distance",
        "cosine_similarity",
        "rotation_gt_distance",
        "translation_gt_distance",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for pair in pairs:
            writer.writerow(
                {
                    "pair_index": pair.pair_index,
                    "pair_type": pair.pair_type,
                    "anchor_global_index": (
                        pair.anchor_global_index
                    ),
                    "partner_global_index": (
                        pair.partner_global_index
                    ),
                    "anchor_source_index": (
                        pair.anchor_source_index
                    ),
                    "partner_source_index": (
                        pair.partner_source_index
                    ),
                    "anchor_sequence": (
                        pair.anchor_sequence
                    ),
                    "partner_sequence": (
                        pair.partner_sequence
                    ),
                    "anchor_regime": (
                        pair.anchor_regime
                    ),
                    "anchor_regime_name": (
                        REGIME_NAMES[
                            pair.anchor_regime
                        ]
                    ),
                    "partner_regime": (
                        pair.partner_regime
                    ),
                    "partner_regime_name": (
                        REGIME_NAMES[
                            pair.partner_regime
                        ]
                    ),
                    "same_sequence": int(
                        pair.same_sequence
                    ),
                    "same_regime": int(
                        pair.same_regime
                    ),
                    "euclidean_distance": (
                        pair.euclidean_distance
                    ),
                    "cosine_similarity": (
                        pair.cosine_similarity
                    ),
                    "rotation_gt_distance": (
                        pair.rotation_gt_distance
                    ),
                    "translation_gt_distance": (
                        pair.translation_gt_distance
                    ),
                }
            )


def write_pairs_npz(
    path: Path,
    pairs: Sequence[AlignmentPair],
    representations: np.ndarray,
) -> None:
    if not pairs:
        raise RuntimeError(
            "Cannot save NPZ because no pairs were generated."
        )

    anchor_indices = np.asarray(
        [
            pair.anchor_global_index
            for pair in pairs
        ],
        dtype=np.int64,
    )

    partner_indices = np.asarray(
        [
            pair.partner_global_index
            for pair in pairs
        ],
        dtype=np.int64,
    )

    pair_type = np.asarray(
        [
            1
            if pair.pair_type
            == "positive_same_regime"
            else 0
            for pair in pairs
        ],
        dtype=np.int8,
    )

    anchor_regime = np.asarray(
        [
            pair.anchor_regime
            for pair in pairs
        ],
        dtype=np.int64,
    )

    partner_regime = np.asarray(
        [
            pair.partner_regime
            for pair in pairs
        ],
        dtype=np.int64,
    )

    euclidean_distance = np.asarray(
        [
            pair.euclidean_distance
            for pair in pairs
        ],
        dtype=np.float64,
    )

    cosine = np.asarray(
        [
            pair.cosine_similarity
            for pair in pairs
        ],
        dtype=np.float64,
    )

    np.savez_compressed(
        path,
        anchor_rep=representations[
            anchor_indices
        ].astype(
            np.float32,
            copy=False,
        ),
        partner_rep=representations[
            partner_indices
        ].astype(
            np.float32,
            copy=False,
        ),
        pair_label=pair_type,
        anchor_global_index=anchor_indices,
        partner_global_index=partner_indices,
        anchor_regime=anchor_regime,
        partner_regime=partner_regime,
        same_regime=np.asarray(
            [
                pair.same_regime
                for pair in pairs
            ],
            dtype=np.bool_,
        ),
        same_sequence=np.asarray(
            [
                pair.same_sequence
                for pair in pairs
            ],
            dtype=np.bool_,
        ),
        euclidean_distance=euclidean_distance,
        cosine_similarity=cosine,
    )


def mean_or_nan(
    values: Iterable[float],
) -> float:
    values_array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    if values_array.size == 0:
        return float("nan")

    return float(
        np.mean(values_array)
    )


def summarize_pairs(
    samples: Sequence[RepresentationSample],
    pairs: Sequence[AlignmentPair],
    representation_dim: int,
) -> str:
    positive = [
        pair
        for pair in pairs
        if pair.pair_type
        == "positive_same_regime"
    ]

    negative = [
        pair
        for pair in pairs
        if pair.pair_type
        == "negative_different_regime"
    ]

    sequences = sorted(
        {
            sample.sequence
            for sample in samples
        }
    )

    lines: List[str] = []

    lines.append(
        "=" * 88
    )
    lines.append(
        "Rotation Motion-Alignment Pair Summary"
    )
    lines.append(
        "=" * 88
    )

    lines.append(
        f"Samples:                       {len(samples)}"
    )
    lines.append(
        f"Representation dimension:      {representation_dim}"
    )
    lines.append(
        f"Sequences:                     {' '.join(sequences)}"
    )
    lines.append(
        f"Total pairs:                   {len(pairs)}"
    )
    lines.append(
        f"Positive same-regime pairs:    {len(positive)}"
    )
    lines.append(
        f"Negative diff-regime pairs:    {len(negative)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Sample distribution"
    )
    lines.append(
        "-" * 88
    )

    for sequence in sequences:
        for regime in sorted(
            REGIME_NAMES
        ):
            count = sum(
                1
                for sample in samples
                if (
                    sample.sequence
                    == sequence
                    and sample.regime
                    == regime
                )
            )

            lines.append(
                f"seq={sequence} "
                f"regime={REGIME_NAMES[regime]:<6s} "
                f"n={count:6d}"
            )

    lines.append(
        ""
    )

    lines.append(
        "Representation pair geometry"
    )
    lines.append(
        "-" * 88
    )

    lines.append(
        "Positive mean Euclidean distance: "
        f"{mean_or_nan(pair.euclidean_distance for pair in positive):.8f}"
    )

    lines.append(
        "Negative mean Euclidean distance: "
        f"{mean_or_nan(pair.euclidean_distance for pair in negative):.8f}"
    )

    lines.append(
        "Positive mean cosine similarity:  "
        f"{mean_or_nan(pair.cosine_similarity for pair in positive):.8f}"
    )

    lines.append(
        "Negative mean cosine similarity:  "
        f"{mean_or_nan(pair.cosine_similarity for pair in negative):.8f}"
    )

    cross_positive = [
        pair
        for pair in positive
        if not pair.same_sequence
    ]

    cross_negative = [
        pair
        for pair in negative
        if not pair.same_sequence
    ]

    lines.append(
        ""
    )

    lines.append(
        f"Cross-sequence positive pairs: {len(cross_positive)}"
    )

    lines.append(
        f"Cross-sequence negative pairs: {len(cross_negative)}"
    )

    lines.append(
        "=" * 88
    )

    return "\n".join(
        lines
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    rng = random.Random(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    representations, samples = (
        load_inputs(
            paths=args.inputs,
            normalize_representations=(
                args.normalize_representations
            ),
        )
    )

    selected_samples = subsample_samples(
        samples=samples,
        maximum_per_group=(
            args.max_samples_per_regime_per_sequence
        ),
        rng=rng,
    )

    if not selected_samples:
        raise RuntimeError(
            "No samples remain after filtering."
        )

    pairs = build_pairs(
        samples=selected_samples,
        representations=representations,
        positive_pairs_per_anchor=(
            args.pairs_per_anchor
        ),
        negative_pairs_per_anchor=(
            args.negative_pairs_per_anchor
        ),
        cross_sequence_only=(
            args.cross_sequence_only
        ),
        rng=rng,
    )

    if not pairs:
        raise RuntimeError(
            "No alignment pairs could be constructed. "
            "Check sequence coverage and motion-regime assignments."
        )

    csv_path = (
        args.output_dir
        / "alignment_pairs.csv"
    )

    npz_path = (
        args.output_dir
        / "rotation_motion_alignment_pairs.npz"
    )

    summary_path = (
        args.output_dir
        / "pair_summary.txt"
    )

    write_pairs_csv(
        path=csv_path,
        pairs=pairs,
    )

    write_pairs_npz(
        path=npz_path,
        pairs=pairs,
        representations=representations,
    )

    summary = summarize_pairs(
        samples=selected_samples,
        pairs=pairs,
        representation_dim=(
            representations.shape[1]
        ),
    )

    summary_path.write_text(
        summary + "\n",
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(
        f"Saved pair table: {csv_path}"
    )
    print(
        f"Saved pair archive: {npz_path}"
    )
    print(
        f"Saved summary: {summary_path}"
    )


if __name__ == "__main__":
    main()