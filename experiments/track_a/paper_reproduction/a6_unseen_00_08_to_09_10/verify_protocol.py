#!/usr/bin/env python3
"""Verify that an A6 checkpoint does not declare target-sequence validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch


SOURCE = {"00", "01", "02", "03", "04", "05", "06", "07", "08"}
TARGET = {"09", "10"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    return p.parse_args()


def normalize(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(v).zfill(2) for v in values}


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("configuration", {})
    if not isinstance(cfg, Mapping):
        raise TypeError("checkpoint['configuration'] must be a mapping")

    train = normalize(cfg.get("train_sequences"))
    validation = normalize(cfg.get("validation_sequences"))

    problems = []

    if train != SOURCE:
        problems.append(
            f"train_sequences must be exactly {sorted(SOURCE)}, got {sorted(train)}"
        )

    leaked = (train | validation) & TARGET
    if leaked:
        problems.append(
            "target leakage detected in train/validation metadata: "
            + ", ".join(sorted(leaked))
        )

    validation_enabled = cfg.get("validation_enabled")
    if validation_enabled is True:
        problems.append("validation_enabled=True for source-only A6 checkpoint")

    selection = cfg.get("checkpoint_selection")
    if selection is not None and str(selection) not in {
        "final_epoch", "latest", "fixed_epoch"
    }:
        problems.append(
            f"unexpected checkpoint_selection={selection!r}; "
            "A6 must not use target validation selection"
        )

    print("=" * 72)
    print("Track-A A6 protocol verification")
    print("=" * 72)
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Train sequences:     {sorted(train)}")
    print(f"Validation sequences:{sorted(validation)}")
    print(f"Validation enabled:  {validation_enabled}")
    print(f"Selection metadata:  {selection}")

    if problems:
        print("-" * 72)
        print("A6 protocol status: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("-" * 72)
    print("A6 protocol status: PASS")


if __name__ == "__main__":
    main()
