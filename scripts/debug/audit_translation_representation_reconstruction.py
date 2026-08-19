#!/usr/bin/env python3
"""
Audit reconstruction of DeepDCT-VO translation-head predictions from the
saved representation immediately entering translation_head.dense.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


AXES = ("x", "y", "z")
REP_KEYS = (
    "translation_rep",
    "translation_representation",
    "features",
    "representation",
)
PRED_COLS = (
    "translation_pred_x",
    "translation_pred_y",
    "translation_pred_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct DeepDCT-VO translation predictions from the saved "
            "translation_head.dense input representation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rep-filename",
        default="translation_representations.npz",
    )
    parser.add_argument(
        "--csv-filename",
        default="frame_predictions.csv",
    )
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.01,
        help="LeakyReLU slope used by the pose head.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--worst-count", type=int, default=50)
    args = parser.parse_args()

    if args.negative_slope < 0.0:
        raise ValueError("--negative-slope cannot be negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("--atol and --rtol cannot be negative.")
    if args.worst_count <= 0:
        raise ValueError("--worst-count must be positive.")

    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested, but CUDA is unavailable."
        )
    return torch.device(name)


def load_checkpoint_state(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a mapping, got "
            f"{type(checkpoint).__name__}."
        )

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(
            "Checkpoint does not contain a mapping named model_state_dict."
        )

    return state, checkpoint


def find_unique_state_key(
    state: Mapping[str, torch.Tensor],
    suffix: str,
) -> str:
    matches = [
        key
        for key in state.keys()
        if str(key).endswith(suffix)
    ]

    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one checkpoint key ending in {suffix!r}; "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def load_dense_parameters(
    state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    weight_key = find_unique_state_key(
        state,
        "translation_head.dense.weight",
    )
    bias_key = find_unique_state_key(
        state,
        "translation_head.dense.bias",
    )

    weight = state[weight_key]
    bias = state[bias_key]

    if not torch.is_tensor(weight) or not torch.is_tensor(bias):
        raise TypeError(
            "Translation dense parameters must be torch tensors."
        )

    weight = weight.detach().to(
        device=device,
        dtype=torch.float32,
    )
    bias = bias.detach().to(
        device=device,
        dtype=torch.float32,
    )

    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError(
            "Unexpected dense parameter shapes: "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )

    if weight.shape[0] != 3 or bias.shape[0] != 3:
        raise ValueError(
            "Expected translation head to produce 3 outputs; "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )

    return weight, bias, weight_key, bias_key


def load_representation(path: Path) -> Tuple[np.ndarray, str, Tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Representation file not found: {path}"
        )

    with np.load(path) as data:
        key = next(
            (candidate for candidate in REP_KEYS if candidate in data),
            None,
        )
        if key is None:
            raise KeyError(
                f"{path} does not contain any supported representation key. "
                f"Expected one of {REP_KEYS}; found {list(data.keys())}."
            )
        raw = np.asarray(data[key])

    raw_shape = tuple(raw.shape)

    if raw.ndim < 2:
        raise ValueError(
            "Representation must have shape [N, ...], got "
            f"{raw_shape}."
        )

    representation = np.asarray(
        raw.reshape(raw.shape[0], -1),
        dtype=np.float32,
    )

    if not np.isfinite(representation).all():
        raise FloatingPointError(
            "Representation contains non-finite values."
        )

    return representation, key, raw_shape


def load_frame_predictions(
    path: Path,
) -> Tuple[List[Dict[str, str]], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Frame predictions CSV not found: {path}"
        )

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"Frame predictions CSV is empty: {path}"
        )

    missing = [
        column
        for column in PRED_COLS
        if column not in rows[0]
    ]
    if missing:
        raise KeyError(
            f"{path} is missing required columns: {missing}"
        )

    prediction = np.asarray(
        [
            [float(row[column]) for column in PRED_COLS]
            for row in rows
        ],
        dtype=np.float32,
    )

    if not np.isfinite(prediction).all():
        raise FloatingPointError(
            "CSV contains non-finite translation predictions."
        )

    return rows, prediction


def reconstruct_predictions(
    representation: np.ndarray,
    weight: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_parts: List[np.ndarray] = []
    activated_parts: List[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(representation), batch_size):
            stop = min(
                start + batch_size,
                len(representation),
            )

            x = torch.from_numpy(
                representation[start:stop]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            raw = F.linear(
                x,
                weight,
                bias,
            )
            activated = raw


            raw_parts.append(
                raw.detach().cpu().numpy()
            )
            activated_parts.append(
                activated.detach().cpu().numpy()
            )

    return (
        np.concatenate(raw_parts, axis=0),
        np.concatenate(activated_parts, axis=0),
    )


def error_statistics(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, object]:
    signed = reconstructed - reference
    absolute = np.abs(signed)

    per_axis_max = np.max(absolute, axis=0)
    per_axis_mean = np.mean(absolute, axis=0)
    per_axis_rmse = np.sqrt(
        np.mean(np.square(signed), axis=0)
    )

    row_l2 = np.linalg.norm(
        signed,
        axis=1,
    )

    isclose = np.isclose(
        reconstructed,
        reference,
        atol=atol,
        rtol=rtol,
    )
    row_matches = np.all(isclose, axis=1)
    all_match = bool(np.all(row_matches))

    first_mismatch = (
        None
        if all_match
        else int(np.flatnonzero(~row_matches)[0])
    )

    return {
        "allclose": all_match,
        "atol": float(atol),
        "rtol": float(rtol),
        "matching_elements": int(np.count_nonzero(isclose)),
        "total_elements": int(isclose.size),
        "matching_rows": int(np.count_nonzero(row_matches)),
        "total_rows": int(len(row_matches)),
        "first_mismatch_index": first_mismatch,
        "maximum_absolute_difference": float(
            np.max(absolute)
        ),
        "mean_absolute_difference": float(
            np.mean(absolute)
        ),
        "rmse_all_elements": float(
            np.sqrt(np.mean(np.square(signed)))
        ),
        "maximum_row_l2_difference": float(
            np.max(row_l2)
        ),
        "mean_row_l2_difference": float(
            np.mean(row_l2)
        ),
        "per_axis": {
            axis: {
                "max_abs_difference": float(
                    per_axis_max[index]
                ),
                "mean_abs_difference": float(
                    per_axis_mean[index]
                ),
                "rmse": float(
                    per_axis_rmse[index]
                ),
            }
            for index, axis in enumerate(AXES)
        },
    }


def metadata_value(
    row: Mapping[str, str],
    key: str,
    default: object,
) -> object:
    value = row.get(key)
    if value in (None, ""):
        return default
    return value


def write_reconstruction_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    reference: np.ndarray,
    raw: np.ndarray,
    reconstructed: np.ndarray,
) -> None:
    fieldnames = [
        "index",
        "sequence",
        "frame_prev",
        "frame_curr",
        "translation_pred_x",
        "translation_pred_y",
        "translation_pred_z",
        "translation_reconstructed_raw_x",
        "translation_reconstructed_raw_y",
        "translation_reconstructed_raw_z",
        "translation_reconstructed_x",
        "translation_reconstructed_y",
        "translation_reconstructed_z",
        "difference_x",
        "difference_y",
        "difference_z",
        "max_abs_difference",
        "l2_difference",
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

        for index in range(len(reference)):
            diff = reconstructed[index] - reference[index]

            writer.writerow(
                {
                    "index": index,
                    "sequence": metadata_value(
                        rows[index],
                        "sequence",
                        "",
                    ),
                    "frame_prev": metadata_value(
                        rows[index],
                        "frame_prev",
                        "",
                    ),
                    "frame_curr": metadata_value(
                        rows[index],
                        "frame_curr",
                        "",
                    ),
                    "translation_pred_x": float(reference[index, 0]),
                    "translation_pred_y": float(reference[index, 1]),
                    "translation_pred_z": float(reference[index, 2]),
                    "translation_reconstructed_raw_x": float(raw[index, 0]),
                    "translation_reconstructed_raw_y": float(raw[index, 1]),
                    "translation_reconstructed_raw_z": float(raw[index, 2]),
                    "translation_reconstructed_x": float(reconstructed[index, 0]),
                    "translation_reconstructed_y": float(reconstructed[index, 1]),
                    "translation_reconstructed_z": float(reconstructed[index, 2]),
                    "difference_x": float(diff[0]),
                    "difference_y": float(diff[1]),
                    "difference_z": float(diff[2]),
                    "max_abs_difference": float(
                        np.max(np.abs(diff))
                    ),
                    "l2_difference": float(
                        np.linalg.norm(diff)
                    ),
                }
            )


def write_worst_rows(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    reference: np.ndarray,
    reconstructed: np.ndarray,
    count: int,
) -> None:
    diff = reconstructed - reference
    l2 = np.linalg.norm(diff, axis=1)
    order = np.argsort(l2)[::-1][
        : min(count, len(l2))
    ]

    fieldnames = [
        "rank",
        "index",
        "sequence",
        "frame_prev",
        "frame_curr",
        "l2_difference",
        "max_abs_difference",
        "difference_x",
        "difference_y",
        "difference_z",
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

        for rank, index in enumerate(order, start=1):
            row_diff = diff[index]

            writer.writerow(
                {
                    "rank": rank,
                    "index": int(index),
                    "sequence": metadata_value(
                        rows[index],
                        "sequence",
                        "",
                    ),
                    "frame_prev": metadata_value(
                        rows[index],
                        "frame_prev",
                        "",
                    ),
                    "frame_curr": metadata_value(
                        rows[index],
                        "frame_curr",
                        "",
                    ),
                    "l2_difference": float(l2[index]),
                    "max_abs_difference": float(
                        np.max(np.abs(row_diff))
                    ),
                    "difference_x": float(row_diff[0]),
                    "difference_y": float(row_diff[1]),
                    "difference_z": float(row_diff[2]),
                }
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    representation_path = (
        args.representation_dir
        / args.rep_filename
    )
    csv_path = (
        args.representation_dir
        / args.csv_filename
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Translation Representation Reconstruction Audit")
    print("=" * 80)

    state, checkpoint = load_checkpoint_state(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    (
        dense_weight,
        dense_bias,
        weight_key,
        bias_key,
    ) = load_dense_parameters(
        state=state,
        device=device,
    )

    (
        representation,
        representation_key,
        representation_raw_shape,
    ) = load_representation(
        representation_path
    )

    csv_rows, reference_prediction = (
        load_frame_predictions(csv_path)
    )

    if len(representation) != len(reference_prediction):
        raise ValueError(
            "Sample-count mismatch: representation has "
            f"{len(representation)} rows, frame_predictions.csv has "
            f"{len(reference_prediction)} rows."
        )

    expected_features = int(dense_weight.shape[1])
    actual_features = int(representation.shape[1])

    if actual_features != expected_features:
        raise ValueError(
            "Representation dimension does not match the checkpoint's "
            "translation_head.dense input dimension: "
            f"representation D={actual_features}, dense expects "
            f"D={expected_features}. This audit requires the saved "
            "forward-pre-hook input of translation_head.dense."
        )

    print(f"Checkpoint:             {args.checkpoint}")
    print(f"Checkpoint epoch:       {checkpoint.get('epoch')}")
    print(f"Representation:         {representation_path}")
    print(f"Representation key:     {representation_key}")
    print(f"Raw representation:     {representation_raw_shape}")
    print(f"Flattened shape:        {tuple(representation.shape)}")
    print(f"Prediction CSV:         {csv_path}")
    print(f"Samples:                {len(representation)}")
    print(f"Dense weight key:       {weight_key}")
    print(f"Dense bias key:         {bias_key}")
    print(f"Dense weight shape:     {tuple(dense_weight.shape)}")
    print(f"Negative slope:         {args.negative_slope}")
    print(f"Device:                 {device}")
    print("-" * 80)

    raw_output, reconstructed = reconstruct_predictions(
        representation=representation,
        weight=dense_weight,
        bias=dense_bias,
        negative_slope=args.negative_slope,
        batch_size=args.batch_size,
        device=device,
    )

    stats = error_statistics(
        reconstructed=reconstructed,
        reference=reference_prediction,
        atol=args.atol,
        rtol=args.rtol,
    )

    status = "PASS" if stats["allclose"] else "FAIL"

    summary = {
        "status": status,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": (
            int(checkpoint["epoch"])
            if checkpoint.get("epoch") is not None
            else None
        ),
        "representation_file": str(
            representation_path.resolve()
        ),
        "frame_predictions_file": str(
            csv_path.resolve()
        ),
        "representation_key": representation_key,
        "representation_raw_shape": list(
            representation_raw_shape
        ),
        "representation_flattened_shape": list(
            representation.shape
        ),
        "dense_weight_key": weight_key,
        "dense_bias_key": bias_key,
        "dense_weight_shape": list(
            dense_weight.shape
        ),
        "dense_bias_shape": list(
            dense_bias.shape
        ),
        "negative_slope": float(
            args.negative_slope
        ),
        "comparison": stats,
        "interpretation": (
            "SAVED_REPRESENTATION_RECONSTRUCTS_EXISTING_HEAD_OUTPUT"
            if stats["allclose"]
            else "RECONSTRUCTION_MISMATCH_REQUIRES_INVESTIGATION"
        ),
    }

    with (
        args.output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    write_reconstruction_csv(
        path=(
            args.output_dir
            / "reconstruction.csv"
        ),
        rows=csv_rows,
        reference=reference_prediction,
        raw=raw_output,
        reconstructed=reconstructed,
    )

    write_worst_rows(
        path=(
            args.output_dir
            / "worst_reconstruction_rows.csv"
        ),
        rows=csv_rows,
        reference=reference_prediction,
        reconstructed=reconstructed,
        count=args.worst_count,
    )

    print(f"Audit status:                    {status}")
    print(
        "Maximum absolute difference:     "
        f"{stats['maximum_absolute_difference']:.12e}"
    )
    print(
        "Mean absolute difference:        "
        f"{stats['mean_absolute_difference']:.12e}"
    )
    print(
        "RMSE across all elements:        "
        f"{stats['rmse_all_elements']:.12e}"
    )
    print(
        "Maximum row L2 difference:       "
        f"{stats['maximum_row_l2_difference']:.12e}"
    )
    print(
        "Matching rows:                   "
        f"{stats['matching_rows']}/{stats['total_rows']}"
    )
    print(
        "First mismatch index:            "
        f"{stats['first_mismatch_index']}"
    )

    for axis in AXES:
        axis_stats = stats["per_axis"][axis]
        print(
            f"{axis.upper()} max/mean abs difference:     "
            f"{axis_stats['max_abs_difference']:.12e} / "
            f"{axis_stats['mean_abs_difference']:.12e}"
        )

    print("-" * 80)
    print(
        "Interpretation: "
        f"{summary['interpretation']}"
    )
    print(f"Saved: {args.output_dir / 'summary.json'}")
    print(f"Saved: {args.output_dir / 'reconstruction.csv'}")
    print(
        "Saved: "
        f"{args.output_dir / 'worst_reconstruction_rows.csv'}"
    )
    print("=" * 80)

    if not stats["allclose"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
