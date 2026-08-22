#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import torch


def _search_nested(obj: Any, candidates: Iterable[str]):
    candidates = tuple(candidates)
    if isinstance(obj, dict):
        for key in candidates:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = _search_nested(value, candidates)
            if found is not None:
                return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"FAIL: checkpoint does not exist: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    gt_rotation = _search_nested(
        checkpoint,
        ("use_ground_truth_rotation", "ground_truth_rotation", "use_gt_rotation"),
    )
    translation_decoder = _search_nested(
        checkpoint,
        ("translation_decoder", "translation_decoder_type"),
    )
    semantic = _search_nested(
        checkpoint,
        ("use_semantic_cues", "semantic_cues"),
    )
    depth = _search_nested(
        checkpoint,
        ("use_depth_cues", "depth_cues"),
    )
    rotation_geometry_weight = _search_nested(
        checkpoint,
        ("rotation_geometry_weight",),
    )

    print("=" * 72)
    print("Track-A A3 checkpoint verification")
    print("=" * 72)
    print(f"Checkpoint:             {args.checkpoint}")
    print(f"GT rotation:            {gt_rotation}")
    print(f"Translation decoder:    {translation_decoder}")
    print(f"Semantic cues:          {semantic}")
    print(f"Depth cues:             {depth}")
    print(f"Rotation geom. weight:  {rotation_geometry_weight}")

    failures = []

    if gt_rotation is not True:
        failures.append("use_ground_truth_rotation=True not recorded")

    if translation_decoder is not None and str(translation_decoder).lower() != "dense":
        failures.append(f"translation decoder is {translation_decoder!r}, expected 'dense'")

    if semantic is True:
        failures.append("semantic cues are enabled")

    if depth is True:
        failures.append("depth cues are enabled")

    if rotation_geometry_weight not in (None, 0, 0.0):
        failures.append(
            f"rotation_geometry_weight={rotation_geometry_weight!r}, expected disabled"
        )

    print("-" * 72)
    if failures:
        print("A3 CHECKPOINT VERIFICATION: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("A3 CHECKPOINT VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
