#!/usr/bin/env python3
"""Print experiment provenance stored in a DeepDCT-VO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect DeepDCT-VO checkpoint provenance."
        )
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint to inspect.",
    )

    return parser.parse_args()


def print_mapping(
    title: str,
    mapping: Mapping[str, Any],
) -> None:
    print()
    print(title)
    print("-" * len(title))

    if not mapping:
        print("<empty>")
        return

    for key in sorted(mapping):
        print(
            f"{key}: {mapping[key]}"
        )


def main() -> None:
    args = parse_args()

    checkpoint_path = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Expected the checkpoint to contain "
            "a mapping."
        )

    print("=" * 72)
    print("DeepDCT-VO checkpoint provenance")
    print("=" * 72)
    print(
        f"Checkpoint: {checkpoint_path}"
    )
    print(
        f"Epoch: {checkpoint.get('epoch')}"
    )
    print(
        "Best validation loss: "
        f"{checkpoint.get('best_validation_loss')}"
    )

    experiment = checkpoint.get(
        "experiment",
        {},
    )

    warm_start = checkpoint.get(
        "warm_start",
        {},
    )

    configuration = checkpoint.get(
        "configuration",
        {},
    )

    if isinstance(experiment, Mapping):
        print_mapping(
            "Experiment",
            experiment,
        )

    if isinstance(warm_start, Mapping):
        print_mapping(
            "Warm-start provenance",
            warm_start,
        )

    if isinstance(configuration, Mapping):
        print_mapping(
            "Configuration",
            configuration,
        )


if __name__ == "__main__":
    main()