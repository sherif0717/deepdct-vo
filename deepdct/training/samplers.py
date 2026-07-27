"""Uniform, tempered, and sequence-balanced sampling utilities."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def collect_sequence_ids(dataset) -> List[str]:
    """Return the KITTI sequence identifier for every dataset sample.

    The dataset must implement:

        sequence_for_index(index) -> str

    Returns
    -------
    List[str]
        Sequence name corresponding to every sample.
    """

    if not hasattr(dataset, "sequence_for_index"):
        raise TypeError(
            "Dataset must implement sequence_for_index(index)."
        )

    return [
        str(dataset.sequence_for_index(index))
        for index in range(len(dataset))
    ]


def build_sequence_balanced_sampler(
    dataset,
    *,
    alpha: float = 1.0,
    seed: int = 42,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> Tuple[
    Sampler[int],
    Dict[str, int],
    Dict[str, float],
]:
    """Create an inverse-frequency sequence sampler.

    Each transition belonging to sequence ``s`` receives weight

        weight_s = count_s ** (-alpha)

    Therefore, the aggregate expected probability of sequence ``s`` is

        probability_s ∝ count_s ** (1 - alpha)

    Important cases
    ---------------
    alpha = 0.0
        All transitions receive equal weight. When replacement=True,
        this is uniform random sampling with replacement.

    alpha = 0.5
        Tempered sequence balancing. Small sequences receive more
        influence without making every sequence equally probable.

    alpha = 1.0
        Full sequence balancing. Every sequence receives equal
        aggregate probability.

    Parameters
    ----------
    dataset
        Dataset implementing ``sequence_for_index(index)``.

    alpha
        Sequence-balancing strength. Must lie between 0.0 and 1.0.

    seed
        Random seed used by the sampler.

    num_samples
        Number of samples drawn per epoch. Defaults to ``len(dataset)``.

    replacement
        Whether sampling is performed with replacement.

    Returns
    -------
    sampler
        Configured WeightedRandomSampler.

    sequence_counts
        Original transition count for each sequence.

    sequence_probabilities
        Expected aggregate probability for each sequence.
    """

    if len(dataset) == 0:
        raise ValueError(
            "Cannot build a sampler for an empty dataset."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"alpha must lie between 0.0 and 1.0; received {alpha}."
        )

    sequence_ids = collect_sequence_ids(dataset)
    sequence_counts = Counter(sequence_ids)

    sample_weights = torch.tensor(
        [
            float(sequence_counts[sequence]) ** (-alpha)
            for sequence in sequence_ids
        ],
        dtype=torch.double,
    )

    if not torch.isfinite(sample_weights).all():
        raise ValueError(
            "Sampler produced non-finite sample weights."
        )

    if num_samples is None:
        num_samples = len(dataset)

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )

    sequence_total_weights = {
        sequence: (
            float(count)
            * float(count) ** (-alpha)
        )
        for sequence, count in sequence_counts.items()
    }

    total_weight = sum(
        sequence_total_weights.values()
    )

    sequence_probabilities = {
        sequence: sequence_total_weights[sequence] / total_weight
        for sequence in sorted(sequence_counts)
    }

    return (
        sampler,
        dict(sorted(sequence_counts.items())),
        sequence_probabilities,
    )


def summarize_sequence_distribution(
    sequence_counts: Dict[str, int],
    sequence_probabilities: Dict[str, float],
) -> str:
    """Return a nicely formatted sampler summary."""

    lines = []

    lines.append("-" * 72)
    lines.append("Sequence-balanced sampling")
    lines.append("-" * 72)

    lines.append(
        f"{'Sequence':<10}"
        f"{'Transitions':>15}"
        f"{'Probability':>18}"
    )

    lines.append("-" * 72)

    for sequence in sorted(sequence_counts):

        lines.append(
            f"{sequence:<10}"
            f"{sequence_counts[sequence]:>15d}"
            f"{sequence_probabilities[sequence]:>18.6f}"
        )

    lines.append("-" * 72)

    return "\n".join(lines)