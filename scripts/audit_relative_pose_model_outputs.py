#!/usr/bin/env python3
"""
Audit raw DeepDCT-VO relative-pose model outputs before trajectory reconstruction.

This script reuses the dataset, checkpoint configuration, model builder, and
batch utilities from scripts/evaluate_deepdct_vo.py. It inspects:

1. Ground-truth rotation and directional-translation targets.
2. Final outputs returned by DeepDCTVO.
3. Raw Dense(3) outputs immediately before the pose-head LeakyReLU.
4. Outputs immediately after the pose-head activation.
5. Inputs presented to each regression head.
6. Regression-head parameters and per-axis biases.
7. Predicted-rotation versus ground-truth-rotation conditioning for Model T.

The purpose is to distinguish among:

- target-scale or target-distribution problems;
- upstream feature collapse;
- Dense-layer output collapse;
- bias-dominated predictions;
- output-activation distortion;
- translation degradation caused by predicted rotation;
- non-finite or near-constant model outputs.

Expected repository layout
--------------------------
deepdct-vo/
├── deepdct/
├── scripts/
│   ├── evaluate_deepdct_vo.py
│   └── audit_relative_pose_model_outputs.py   <-- this file
└── data/

Example
-------
Baseline, normal deployable cascade:

    python scripts/audit_relative_pose_model_outputs.py \
        --checkpoint experiments/baseline_rgb_only/best_validation.pt \
        --data-root data \
        --sequence 10 \
        --output-dir experiments/relative_pose_model_output_audit/baseline \
        --device cuda

Semantic + depth:

    python scripts/audit_relative_pose_model_outputs.py \
        --checkpoint experiments/semantic_depth/best_validation.pt \
        --data-root data \
        --sequence 10 \
        --output-dir experiments/relative_pose_model_output_audit/semantic_depth \
        --device cuda

Run both translation-conditioning modes:

    python scripts/audit_relative_pose_model_outputs.py \
        --checkpoint experiments/semantic_depth/best_validation.pt \
        --data-root data \
        --sequence 10 \
        --output-dir experiments/relative_pose_model_output_audit/semantic_depth \
        --device cuda \
        --compare-translation-conditioning

Outputs
-------
relative_pose_model_output_audit/
├── summary.json
├── axis_statistics.csv
├── correlations.csv
├── frame_outputs.csv
├── head_parameter_statistics.csv
├── feature_statistics.csv
├── failure_modes.csv
├── worst_rotation_frames.csv
├── worst_translation_frames.csv
├── conditioning_comparison.csv              # when requested
└── plots/
    ├── rotation_target_vs_raw_dense.png
    ├── rotation_target_vs_final.png
    ├── translation_target_vs_raw_dense.png
    ├── translation_target_vs_final.png
    ├── rotation_output_distributions.png
    ├── translation_output_distributions.png
    ├── raw_vs_activated_rotation.png
    ├── raw_vs_activated_translation.png
    ├── output_timeseries_rotation.png
    ├── output_timeseries_translation.png
    ├── feature_dispersion.png
    └── conditioning_translation_errors.png   # when requested

Python compatibility: Python 3.8+
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor


AXES = ("x", "y", "z")
HEADS = ("rotation", "translation")
SIGNALS = ("target", "raw_dense", "activated", "final")
EPSILON = 1.0e-12


@dataclass
class Capture:
    """One batch of pose-head diagnostic tensors."""

    head_input: Optional[Tensor] = None
    raw_dense: Optional[Tensor] = None
    activated: Optional[Tensor] = None

    def clear(self) -> None:
        self.head_input = None
        self.raw_dense = None
        self.activated = None


class HeadHookCollector:
    """Capture regression-head inputs, Dense outputs, and activation outputs."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.captures: Dict[str, Capture] = {
            "rotation": Capture(),
            "translation": Capture(),
        }
        self.handles: List[Any] = []
        self._register_head("rotation", model.rotation_head)
        self._register_head("translation", model.translation_head)

    def _register_head(self, name: str, head: nn.Module) -> None:
        if not hasattr(head, "dense"):
            raise AttributeError(
                "{} head has no 'dense' module; found {}.".format(
                    name, head.__class__.__name__
                )
            )
        if not hasattr(head, "output_activation"):
            raise AttributeError(
                "{} head has no 'output_activation' module.".format(name)
            )

        def head_pre_hook(
            module: nn.Module,
            inputs: Tuple[object, ...],
        ) -> None:
            del module
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise TypeError(
                    "{} head input hook expected one tensor.".format(name)
                )
            self.captures[name].head_input = inputs[0].detach()

        def dense_hook(
            module: nn.Module,
            inputs: Tuple[object, ...],
            output: object,
        ) -> None:
            del module, inputs
            if not torch.is_tensor(output):
                raise TypeError(
                    "{} dense output is not a tensor.".format(name)
                )
            self.captures[name].raw_dense = output.detach()

        def activation_hook(
            module: nn.Module,
            inputs: Tuple[object, ...],
            output: object,
        ) -> None:
            del module, inputs
            if not torch.is_tensor(output):
                raise TypeError(
                    "{} activation output is not a tensor.".format(name)
                )
            self.captures[name].activated = output.detach()

        self.handles.append(
            head.register_forward_pre_hook(head_pre_hook)
        )
        self.handles.append(
            head.dense.register_forward_hook(dense_hook)
        )
        self.handles.append(
            head.output_activation.register_forward_hook(
                activation_hook
            )
        )

    def clear(self) -> None:
        for capture in self.captures.values():
            capture.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw Dense outputs, activated outputs, features, and "
            "targets of the DeepDCT-VO rotation and translation heads."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="DeepDCT-VO checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="10",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for a fast audit.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--worst-frame-count",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--use-ground-truth-rotation",
        action="store_true",
        help=(
            "Audit Model T while conditioning it on ground-truth rotation."
        ),
    )
    parser.add_argument(
        "--compare-translation-conditioning",
        action="store_true",
        help=(
            "Run two passes: predicted-rotation conditioning and "
            "ground-truth-rotation conditioning."
        ),
    )
    parser.add_argument(
        "--collapse-std-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--minimum-absolute-correlation",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--scale-slope-tolerance",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--bias-fraction-of-target-std",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--activation-change-tolerance",
        type=float,
        default=1.0e-8,
        help=(
            "Absolute raw-to-activated difference regarded as unchanged."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.worst_frame_count <= 0:
        raise ValueError("--worst-frame-count must be positive.")
    if args.histogram_bins <= 1:
        raise ValueError("--histogram-bins must be greater than one.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not 0.0 <= args.minimum_absolute_correlation <= 1.0:
        raise ValueError(
            "--minimum-absolute-correlation must be in [0, 1]."
        )
    for name in (
        "collapse_std_ratio",
        "scale_slope_tolerance",
        "bias_fraction_of_target_std",
        "activation_change_tolerance",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError("--{} cannot be negative.".format(
                name.replace("_", "-")
            ))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_evaluation_module(repo_root: Path) -> Any:
    """Import scripts/evaluate_deepdct_vo.py without requiring scripts package."""
    path = repo_root / "scripts" / "evaluate_deepdct_vo.py"
    if not path.is_file():
        raise FileNotFoundError(
            "Required evaluation script does not exist: {}".format(path)
        )

    spec = importlib.util.spec_from_file_location(
        "deepdct_evaluation_helpers",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to create import specification for {}".format(path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_repo_root(script_path: Path) -> Path:
    candidate = script_path.resolve().parent.parent
    if (
        (candidate / "deepdct").is_dir()
        and (candidate / "scripts" / "evaluate_deepdct_vo.py").is_file()
    ):
        return candidate

    current = Path.cwd().resolve()
    if (
        (current / "deepdct").is_dir()
        and (current / "scripts" / "evaluate_deepdct_vo.py").is_file()
    ):
        return current

    raise FileNotFoundError(
        "Could not identify repository root containing deepdct/ and "
        "scripts/evaluate_deepdct_vo.py."
    )


def safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def tensor_to_numpy(value: Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise FloatingPointError(
            "Captured tensor contains NaN or infinity."
        )
    return array


def metadata_value(
    helper_module: Any,
    batch: Mapping[str, object],
    key: str,
    index: int,
    default: object,
) -> object:
    if key not in batch:
        return default
    try:
        return helper_module.metadata_value(batch, key, index)
    except Exception:
        value = batch[key]
        if torch.is_tensor(value):
            return value[index].item()
        if isinstance(value, (list, tuple)):
            return value[index]
        return value


def descriptive_statistics(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q99": None,
            "max": None,
            "mean_abs": None,
            "rms": None,
            "zero_fraction": None,
            "negative_fraction": None,
            "positive_fraction": None,
        }

    return {
        "count": int(values.size),
        "mean": safe_float(np.mean(values)),
        "std": safe_float(np.std(values)),
        "min": safe_float(np.min(values)),
        "q01": safe_float(np.quantile(values, 0.01)),
        "q25": safe_float(np.quantile(values, 0.25)),
        "median": safe_float(np.median(values)),
        "q75": safe_float(np.quantile(values, 0.75)),
        "q99": safe_float(np.quantile(values, 0.99)),
        "max": safe_float(np.max(values)),
        "mean_abs": safe_float(np.mean(np.abs(values))),
        "rms": safe_float(np.sqrt(np.mean(np.square(values)))),
        "zero_fraction": safe_float(np.mean(values == 0.0)),
        "negative_fraction": safe_float(np.mean(values < 0.0)),
        "positive_fraction": safe_float(np.mean(values > 0.0)),
    }


def correlation_and_fit(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, object]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    mask = np.isfinite(target) & np.isfinite(prediction)
    target = target[mask]
    prediction = prediction[mask]

    result: Dict[str, object] = {
        "count": int(target.size),
        "pearson_r": None,
        "r_squared": None,
        "slope": None,
        "intercept": None,
        "mae": None,
        "rmse": None,
        "bias": None,
    }

    if target.size == 0:
        return result

    error = prediction - target
    result["mae"] = safe_float(np.mean(np.abs(error)))
    result["rmse"] = safe_float(np.sqrt(np.mean(np.square(error))))
    result["bias"] = safe_float(np.mean(error))

    if target.size >= 2:
        target_std = float(np.std(target))
        prediction_std = float(np.std(prediction))
        if target_std > EPSILON and prediction_std > EPSILON:
            r = float(np.corrcoef(target, prediction)[0, 1])
            result["pearson_r"] = safe_float(r)
            result["r_squared"] = safe_float(r * r)
        if target_std > EPSILON:
            slope, intercept = np.polyfit(target, prediction, deg=1)
            result["slope"] = safe_float(slope)
            result["intercept"] = safe_float(intercept)

    return result


def feature_batch_statistics(value: Tensor) -> Dict[str, float]:
    array = tensor_to_numpy(value)
    flattened = array.reshape(array.shape[0], -1)
    per_sample_mean = np.mean(flattened, axis=1)
    per_sample_std = np.std(flattened, axis=1)
    per_sample_l2 = np.linalg.norm(flattened, axis=1)
    per_sample_nonzero = np.mean(flattened != 0.0, axis=1)

    return {
        "mean": float(np.mean(per_sample_mean)),
        "std": float(np.mean(per_sample_std)),
        "l2": float(np.mean(per_sample_l2)),
        "nonzero_fraction": float(np.mean(per_sample_nonzero)),
        "batch_between_sample_std": float(
            np.mean(np.std(flattened, axis=0))
        ),
    }


def append_feature_records(
    records: List[Dict[str, object]],
    batch_start: int,
    head: str,
    value: Tensor,
) -> None:
    array = tensor_to_numpy(value)
    flattened = array.reshape(array.shape[0], -1)
    for offset in range(flattened.shape[0]):
        sample = flattened[offset]
        records.append({
            "sample_index": batch_start + offset,
            "head": head,
            "feature_count": int(sample.size),
            "mean": float(np.mean(sample)),
            "std": float(np.std(sample)),
            "min": float(np.min(sample)),
            "max": float(np.max(sample)),
            "l1": float(np.sum(np.abs(sample))),
            "l2": float(np.linalg.norm(sample)),
            "zero_fraction": float(np.mean(sample == 0.0)),
            "negative_fraction": float(np.mean(sample < 0.0)),
            "positive_fraction": float(np.mean(sample > 0.0)),
        })


def inspect_head_parameters(model: nn.Module) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for head_name in HEADS:
        head = getattr(model, "{}_head".format(head_name))
        for parameter_name, parameter in head.named_parameters():
            array = tensor_to_numpy(parameter)
            record: Dict[str, object] = {
                "head": head_name,
                "parameter": parameter_name,
                "shape": "x".join(map(str, array.shape)),
            }
            record.update(descriptive_statistics(array))
            record["l1_norm"] = safe_float(np.sum(np.abs(array)))
            record["l2_norm"] = safe_float(np.linalg.norm(array))
            records.append(record)

        dense = head.dense
        weight = tensor_to_numpy(dense.weight)
        bias = tensor_to_numpy(dense.bias)
        for axis_index, axis in enumerate(AXES):
            records.append({
                "head": head_name,
                "parameter": "dense.axis_{}.weight".format(axis),
                "shape": "x".join(map(str, weight[axis_index].shape)),
                **descriptive_statistics(weight[axis_index]),
                "l1_norm": safe_float(
                    np.sum(np.abs(weight[axis_index]))
                ),
                "l2_norm": safe_float(
                    np.linalg.norm(weight[axis_index])
                ),
            })
            records.append({
                "head": head_name,
                "parameter": "dense.axis_{}.bias".format(axis),
                "shape": "1",
                **descriptive_statistics(
                    np.asarray([bias[axis_index]])
                ),
                "l1_norm": safe_float(abs(bias[axis_index])),
                "l2_norm": safe_float(abs(bias[axis_index])),
            })

    return pd.DataFrame.from_records(records)


def audit_pass(
    helper: Any,
    model: nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    device: torch.device,
    use_internal_depth: bool,
    use_ground_truth_rotation: bool,
    max_samples: Optional[int],
    log_interval: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    collector = HeadHookCollector(model)
    rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    processed = 0

    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader):
                if max_samples is not None and processed >= max_samples:
                    break

                collector.clear()

                image_prev = helper.move_tensor(
                    batch, "image_prev", device
                )
                image_curr = helper.move_tensor(
                    batch, "image_curr", device
                )
                rotation_gt = helper.move_tensor(
                    batch, "rotation_gt", device
                )
                translation_gt = helper.move_tensor(
                    batch, "translation_gt", device
                )

                if use_internal_depth:
                    depth_curr = None
                else:
                    depth_curr = helper.move_tensor(
                        batch, "depth_curr", device
                    )

                remaining = (
                    image_prev.shape[0]
                    if max_samples is None
                    else min(
                        image_prev.shape[0],
                        max_samples - processed,
                    )
                )

                if remaining < image_prev.shape[0]:
                    image_prev = image_prev[:remaining]
                    image_curr = image_curr[:remaining]
                    rotation_gt = rotation_gt[:remaining]
                    translation_gt = translation_gt[:remaining]
                    if depth_curr is not None:
                        depth_curr = depth_curr[:remaining]

                outputs = model(
                    image_prev=image_prev,
                    image_curr=image_curr,
                    depth_curr=depth_curr,
                    rotation_for_translation=(
                        rotation_gt
                        if use_ground_truth_rotation
                        else None
                    ),
                    use_ground_truth_rotation=use_ground_truth_rotation,
                    return_intermediates=True,
                )

                rotation_final = outputs["rotation"]
                translation_final = outputs[
                    "directional_translation"
                ]

                expected_shape = (remaining, 3)
                for name, tensor in (
                    ("rotation final", rotation_final),
                    ("translation final", translation_final),
                ):
                    if tuple(tensor.shape) != expected_shape:
                        raise ValueError(
                            "{} shape is {}, expected {}.".format(
                                name, tuple(tensor.shape), expected_shape
                            )
                        )
                    if not torch.isfinite(tensor).all():
                        raise FloatingPointError(
                            "{} contains non-finite values.".format(name)
                        )

                capture_arrays: Dict[str, Dict[str, np.ndarray]] = {}
                for head in HEADS:
                    capture = collector.captures[head]
                    if capture.head_input is None:
                        raise RuntimeError(
                            "{} head input was not captured.".format(head)
                        )
                    if capture.raw_dense is None:
                        raise RuntimeError(
                            "{} raw Dense output was not captured.".format(
                                head
                            )
                        )
                    if capture.activated is None:
                        raise RuntimeError(
                            "{} activated output was not captured.".format(
                                head
                            )
                        )

                    if tuple(capture.raw_dense.shape) != expected_shape:
                        raise ValueError(
                            "{} raw Dense shape is {}, expected {}.".format(
                                head,
                                tuple(capture.raw_dense.shape),
                                expected_shape,
                            )
                        )
                    if tuple(capture.activated.shape) != expected_shape:
                        raise ValueError(
                            "{} activated shape is {}, expected {}.".format(
                                head,
                                tuple(capture.activated.shape),
                                expected_shape,
                            )
                        )

                    capture_arrays[head] = {
                        "raw_dense": tensor_to_numpy(
                            capture.raw_dense
                        ),
                        "activated": tensor_to_numpy(
                            capture.activated
                        ),
                    }
                    append_feature_records(
                        feature_rows,
                        batch_start=processed,
                        head=head,
                        value=capture.head_input,
                    )

                arrays = {
                    "rotation": {
                        "target": tensor_to_numpy(rotation_gt),
                        "final": tensor_to_numpy(rotation_final),
                        **capture_arrays["rotation"],
                    },
                    "translation": {
                        "target": tensor_to_numpy(translation_gt),
                        "final": tensor_to_numpy(translation_final),
                        **capture_arrays["translation"],
                    },
                }

                # Hooked activated values must match the model outputs.
                for head in HEADS:
                    maximum_difference = float(np.max(np.abs(
                        arrays[head]["activated"]
                        - arrays[head]["final"]
                    )))
                    if maximum_difference > 1.0e-7:
                        raise RuntimeError(
                            "{} activated hook differs from final output "
                            "by {}.".format(head, maximum_difference)
                        )

                rotation_used = outputs.get(
                    "rotation_used_for_translation",
                    rotation_gt
                    if use_ground_truth_rotation
                    else rotation_final,
                )
                rotation_used_np = tensor_to_numpy(rotation_used)

                for offset in range(remaining):
                    global_index = processed + offset
                    row: Dict[str, object] = {
                        "sample_index": global_index,
                        "conditioning": (
                            "ground_truth_rotation"
                            if use_ground_truth_rotation
                            else "predicted_rotation"
                        ),
                        "sequence": str(metadata_value(
                            helper,
                            batch,
                            "sequence",
                            offset,
                            "",
                        )),
                        "frame_prev": int(metadata_value(
                            helper,
                            batch,
                            "frame_prev",
                            offset,
                            global_index,
                        )),
                        "frame_curr": int(metadata_value(
                            helper,
                            batch,
                            "frame_curr",
                            offset,
                            global_index + 1,
                        )),
                        "image_prev_path": str(metadata_value(
                            helper,
                            batch,
                            "image_prev_path",
                            offset,
                            "",
                        )),
                        "image_curr_path": str(metadata_value(
                            helper,
                            batch,
                            "image_curr_path",
                            offset,
                            "",
                        )),
                    }

                    for head in HEADS:
                        for signal in SIGNALS:
                            values = arrays[head][signal][offset]
                            for axis_index, axis in enumerate(AXES):
                                row[
                                    "{}_{}_{}".format(
                                        head, signal, axis
                                    )
                                ] = float(values[axis_index])

                        target = arrays[head]["target"][offset]
                        final = arrays[head]["final"][offset]
                        raw = arrays[head]["raw_dense"][offset]
                        activated = arrays[head]["activated"][offset]

                        row["{}_target_norm".format(head)] = float(
                            np.linalg.norm(target)
                        )
                        row["{}_final_norm".format(head)] = float(
                            np.linalg.norm(final)
                        )
                        row["{}_error_norm".format(head)] = float(
                            np.linalg.norm(final - target)
                        )
                        row[
                            "{}_raw_to_activated_l2".format(head)
                        ] = float(np.linalg.norm(activated - raw))
                        row[
                            "{}_raw_negative_count".format(head)
                        ] = int(np.sum(raw < 0.0))

                    for axis_index, axis in enumerate(AXES):
                        row[
                            "rotation_used_for_translation_{}".format(axis)
                        ] = float(rotation_used_np[offset, axis_index])

                    rows.append(row)

                processed += remaining

                if (
                    (batch_index + 1) % log_interval == 0
                    or (
                        max_samples is not None
                        and processed >= max_samples
                    )
                ):
                    print(
                        "[audit] conditioning={} batches={} samples={}".format(
                            (
                                "GT"
                                if use_ground_truth_rotation
                                else "predicted"
                            ),
                            batch_index + 1,
                            processed,
                        )
                    )
    finally:
        collector.close()

    if not rows:
        raise RuntimeError("No samples were audited.")

    return (
        pd.DataFrame.from_records(rows),
        pd.DataFrame.from_records(feature_rows),
    )


def build_axis_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for head in HEADS:
        for signal in SIGNALS:
            for axis in AXES:
                column = "{}_{}_{}".format(head, signal, axis)
                record: Dict[str, object] = {
                    "head": head,
                    "signal": signal,
                    "axis": axis,
                    "column": column,
                }
                record.update(descriptive_statistics(
                    frame[column].to_numpy(dtype=np.float64)
                ))
                records.append(record)

        for signal in ("target", "final"):
            norm_column = "{}_{}_norm".format(head, signal)
            record = {
                "head": head,
                "signal": "{}_norm".format(signal),
                "axis": "norm",
                "column": norm_column,
            }
            record.update(descriptive_statistics(
                frame[norm_column].to_numpy(dtype=np.float64)
            ))
            records.append(record)

    return pd.DataFrame.from_records(records)


def build_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for head in HEADS:
        for signal in ("raw_dense", "activated", "final"):
            for axis in AXES:
                target = frame[
                    "{}_target_{}".format(head, axis)
                ].to_numpy(dtype=np.float64)
                prediction = frame[
                    "{}_{}_{}".format(head, signal, axis)
                ].to_numpy(dtype=np.float64)

                target_std = float(np.std(target))
                prediction_std = float(np.std(prediction))

                record: Dict[str, object] = {
                    "head": head,
                    "signal": signal,
                    "axis": axis,
                    "target_mean": safe_float(np.mean(target)),
                    "prediction_mean": safe_float(np.mean(prediction)),
                    "target_std": safe_float(target_std),
                    "prediction_std": safe_float(prediction_std),
                    "prediction_to_target_std_ratio": (
                        safe_float(prediction_std / target_std)
                        if target_std > EPSILON
                        else None
                    ),
                }
                record.update(correlation_and_fit(
                    target, prediction
                ))
                records.append(record)

        target_norm = frame[
            "{}_target_norm".format(head)
        ].to_numpy(dtype=np.float64)
        final_norm = frame[
            "{}_final_norm".format(head)
        ].to_numpy(dtype=np.float64)
        target_std = float(np.std(target_norm))
        prediction_std = float(np.std(final_norm))
        record = {
            "head": head,
            "signal": "final_norm",
            "axis": "norm",
            "target_mean": safe_float(np.mean(target_norm)),
            "prediction_mean": safe_float(np.mean(final_norm)),
            "target_std": safe_float(target_std),
            "prediction_std": safe_float(prediction_std),
            "prediction_to_target_std_ratio": (
                safe_float(prediction_std / target_std)
                if target_std > EPSILON
                else None
            ),
        }
        record.update(correlation_and_fit(
            target_norm, final_norm
        ))
        records.append(record)

    return pd.DataFrame.from_records(records)


def detect_failure_modes(
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    findings: List[Dict[str, object]] = []

    def add(
        head: str,
        signal: str,
        axis: str,
        failure_mode: str,
        severity: str,
        evidence: str,
        next_check: str,
    ) -> None:
        findings.append({
            "head": head,
            "signal": signal,
            "axis": axis,
            "failure_mode": failure_mode,
            "severity": severity,
            "evidence": evidence,
            "next_check": next_check,
        })

    final_rows = correlations[
        correlations["signal"].isin(("final", "final_norm"))
    ]

    for _, row in final_rows.iterrows():
        head = str(row["head"])
        signal = str(row["signal"])
        axis = str(row["axis"])
        target_std = row["target_std"]
        prediction_std = row["prediction_std"]
        ratio = row["prediction_to_target_std_ratio"]
        r = row["pearson_r"]
        slope = row["slope"]
        bias = row["bias"]

        if (
            pd.notna(r)
            and abs(float(r))
            < args.minimum_absolute_correlation
        ):
            add(
                head,
                signal,
                axis,
                "poor_target_correlation",
                "high" if abs(float(r)) < 0.25 else "medium",
                "Pearson r={:.6f}.".format(float(r)),
                "Check frame/target alignment and whether upstream features "
                "vary with the target.",
            )

        if (
            pd.notna(r)
            and float(r) < -args.minimum_absolute_correlation
        ):
            add(
                head,
                signal,
                axis,
                "possible_sign_or_frame_inversion",
                "high",
                "Pearson r={:.6f} is strongly negative.".format(float(r)),
                "Verify relative-pose direction and axis sign conventions.",
            )

        if (
            pd.notna(ratio)
            and float(ratio) < args.collapse_std_ratio
        ):
            add(
                head,
                signal,
                axis,
                "output_under_dispersion_or_collapse",
                "high",
                "Prediction/target std ratio={:.6f}.".format(
                    float(ratio)
                ),
                "Compare head-input dispersion, raw Dense dispersion, and "
                "Dense weight/bias statistics.",
            )

        if (
            pd.notna(slope)
            and pd.notna(r)
            and abs(float(r)) >= args.minimum_absolute_correlation
            and abs(float(slope) - 1.0)
            > args.scale_slope_tolerance
        ):
            add(
                head,
                signal,
                axis,
                "scale_error",
                "high" if abs(float(slope) - 1.0) > 0.75 else "medium",
                "Fit slope={:.6f}; ideal is 1.".format(float(slope)),
                "Audit label scaling, inverse normalization, and units.",
            )

        if (
            pd.notna(target_std)
            and float(target_std) > EPSILON
            and pd.notna(bias)
            and abs(float(bias)) / float(target_std)
            > args.bias_fraction_of_target_std
        ):
            normalized_bias = abs(float(bias)) / float(target_std)
            add(
                head,
                signal,
                axis,
                "systematic_bias",
                "high" if normalized_bias > 1.0 else "medium",
                "Bias is {:.6f} target standard deviations.".format(
                    normalized_bias
                ),
                "Inspect Dense bias, target centering, and training-set "
                "target means.",
            )

    for head in HEADS:
        for axis in AXES:
            raw = frame[
                "{}_raw_dense_{}".format(head, axis)
            ].to_numpy(dtype=np.float64)
            activated = frame[
                "{}_activated_{}".format(head, axis)
            ].to_numpy(dtype=np.float64)
            target = frame[
                "{}_target_{}".format(head, axis)
            ].to_numpy(dtype=np.float64)

            changed = np.abs(activated - raw) > (
                args.activation_change_tolerance
            )
            negative = raw < 0.0
            if np.any(negative):
                negative_fraction = float(np.mean(negative))
                mean_attenuation = float(np.mean(
                    np.divide(
                        activated[negative],
                        raw[negative],
                        out=np.full_like(
                            activated[negative], np.nan
                        ),
                        where=np.abs(raw[negative]) > EPSILON,
                    )
                ))
                add(
                    head,
                    "output_activation",
                    axis,
                    "negative_outputs_attenuated_by_leaky_relu",
                    (
                        "high"
                        if negative_fraction > 0.25
                        and np.mean(target < 0.0) > 0.25
                        else "informational"
                    ),
                    (
                        "{:.2%} raw Dense outputs are negative; mean "
                        "activated/raw ratio on negatives is {:.6f}."
                    ).format(
                        negative_fraction,
                        mean_attenuation,
                    ),
                    "Confirm whether signed pose regression should use a "
                    "final LeakyReLU or an identity activation.",
                )

            changed_fraction = float(np.mean(changed))
            if changed_fraction > 0.0:
                add(
                    head,
                    "output_activation",
                    axis,
                    "activation_changes_dense_output",
                    "informational",
                    "{:.2%} of values change after activation.".format(
                        changed_fraction
                    ),
                    "Compare raw-Dense and final correlations to determine "
                    "whether the activation improves or worsens regression.",
                )

        feature_subset = frame[
            [
                "{}_raw_dense_{}".format(head, axis)
                for axis in AXES
            ]
        ].to_numpy(dtype=np.float64)
        if np.max(np.std(feature_subset, axis=0)) < 1.0e-8:
            add(
                head,
                "raw_dense",
                "all",
                "dense_output_nearly_constant",
                "high",
                "All raw Dense axes have near-zero sample variation.",
                "Inspect head-input feature dispersion and Dense weights.",
            )

    return pd.DataFrame.from_records(findings)


def aggregate_feature_statistics(
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for head, group in feature_frame.groupby("head"):
        for metric in (
            "mean",
            "std",
            "l1",
            "l2",
            "zero_fraction",
            "negative_fraction",
            "positive_fraction",
        ):
            values = group[metric].to_numpy(dtype=np.float64)
            record: Dict[str, object] = {
                "head": head,
                "metric": metric,
            }
            record.update(descriptive_statistics(values))
            records.append(record)
    return pd.DataFrame.from_records(records)


def save_figure(
    figure: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        str(path),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_target_scatter(
    frame: pd.DataFrame,
    head: str,
    signal: str,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for index, axis in enumerate(AXES):
        target = frame[
            "{}_target_{}".format(head, axis)
        ].to_numpy(dtype=np.float64)
        prediction = frame[
            "{}_{}_{}".format(head, signal, axis)
        ].to_numpy(dtype=np.float64)

        current = axes[index]
        current.scatter(target, prediction, s=10, alpha=0.45)
        lower = float(min(np.min(target), np.min(prediction)))
        upper = float(max(np.max(target), np.max(prediction)))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        current.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.2,
            label="Ideal y=x",
        )

        fit = correlation_and_fit(target, prediction)
        if fit["slope"] is not None:
            x_values = np.asarray([lower, upper])
            y_values = (
                float(fit["slope"]) * x_values
                + float(fit["intercept"])
            )
            current.plot(
                x_values,
                y_values,
                linewidth=1.2,
                label="Linear fit",
            )

        current.set_title(axis.upper())
        current.set_xlabel("Target")
        current.set_ylabel(signal.replace("_", " ").title())
        current.grid(True, alpha=0.25)
        current.legend(loc="best")
        current.text(
            0.04,
            0.96,
            "r={}\nslope={}".format(
                (
                    "{:.4f}".format(float(fit["pearson_r"]))
                    if fit["pearson_r"] is not None
                    else "n/a"
                ),
                (
                    "{:.4f}".format(float(fit["slope"]))
                    if fit["slope"] is not None
                    else "n/a"
                ),
            ),
            transform=current.transAxes,
            va="top",
        )

    fig.suptitle(
        "{} target versus {}".format(
            head.title(), signal.replace("_", " ")
        ),
        y=1.02,
    )
    save_figure(fig, path, dpi)


def plot_distributions(
    frame: pd.DataFrame,
    head: str,
    path: Path,
    bins: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for index, axis in enumerate(AXES):
        current = axes[index]
        for signal in SIGNALS:
            current.hist(
                frame[
                    "{}_{}_{}".format(head, signal, axis)
                ].to_numpy(dtype=np.float64),
                bins=bins,
                histtype="step",
                linewidth=1.4,
                label=signal,
                density=False,
            )
        current.set_title(axis.upper())
        current.set_xlabel("Value")
        current.set_ylabel("Count")
        current.grid(True, alpha=0.2)
        current.legend(loc="best")

    fig.suptitle(
        "{} target and output distributions".format(head.title()),
        y=1.01,
    )
    save_figure(fig, path, dpi)


def plot_raw_vs_activated(
    frame: pd.DataFrame,
    head: str,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for index, axis in enumerate(AXES):
        raw = frame[
            "{}_raw_dense_{}".format(head, axis)
        ].to_numpy(dtype=np.float64)
        activated = frame[
            "{}_activated_{}".format(head, axis)
        ].to_numpy(dtype=np.float64)

        current = axes[index]
        current.scatter(raw, activated, s=10, alpha=0.45)
        lower = float(min(np.min(raw), np.min(activated)))
        upper = float(max(np.max(raw), np.max(activated)))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        current.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.2,
        )
        current.set_title(axis.upper())
        current.set_xlabel("Raw Dense output")
        current.set_ylabel("After output activation")
        current.grid(True, alpha=0.25)

    fig.suptitle(
        "{} raw Dense versus activated output".format(head.title()),
        y=1.02,
    )
    save_figure(fig, path, dpi)


def plot_timeseries(
    frame: pd.DataFrame,
    head: str,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    x = np.arange(len(frame))

    for index, axis in enumerate(AXES):
        current = axes[index]
        for signal in SIGNALS:
            current.plot(
                x,
                frame[
                    "{}_{}_{}".format(head, signal, axis)
                ],
                linewidth=1.0,
                alpha=0.85,
                label=signal,
            )
        current.set_ylabel(axis.upper())
        current.grid(True, alpha=0.25)
        current.legend(loc="best")

    axes[-1].set_xlabel("Transition index")
    fig.suptitle(
        "{} model-output time series".format(head.title()),
        y=1.01,
    )
    save_figure(fig, path, dpi)


def plot_feature_dispersion(
    feature_frame: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    positions = np.arange(len(feature_frame))

    for head in HEADS:
        subset = feature_frame[
            feature_frame["head"] == head
        ]
        ax.plot(
            subset["sample_index"],
            subset["std"],
            linewidth=1.0,
            label="{} head input std".format(head),
        )

    ax.set_xlabel("Transition index")
    ax.set_ylabel("Within-sample feature standard deviation")
    ax.set_title("Regression-head input feature dispersion")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    del positions
    save_figure(fig, path, dpi)


def conditioning_comparison(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["sequence", "frame_prev", "frame_curr"]
    pred_columns = keys + [
        "translation_error_norm",
        "translation_final_norm",
    ] + [
        "translation_final_{}".format(axis)
        for axis in AXES
    ]
    gt_columns = keys + [
        "translation_error_norm",
        "translation_final_norm",
    ] + [
        "translation_final_{}".format(axis)
        for axis in AXES
    ]

    left = predicted[pred_columns].copy()
    right = ground_truth[gt_columns].copy()
    left = left.rename(columns={
        column: "{}_predicted_conditioning".format(column)
        for column in left.columns if column not in keys
    })
    right = right.rename(columns={
        column: "{}_gt_conditioning".format(column)
        for column in right.columns if column not in keys
    })

    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    merged["translation_error_improvement_with_gt_rotation"] = (
        merged[
            "translation_error_norm_predicted_conditioning"
        ]
        - merged[
            "translation_error_norm_gt_conditioning"
        ]
    )
    return merged


def plot_conditioning_errors(
    comparison: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(comparison))
    ax.plot(
        x,
        comparison[
            "translation_error_norm_predicted_conditioning"
        ],
        linewidth=1.0,
        label="Predicted rotation conditioning",
    )
    ax.plot(
        x,
        comparison[
            "translation_error_norm_gt_conditioning"
        ],
        linewidth=1.0,
        label="Ground-truth rotation conditioning",
    )
    ax.set_xlabel("Matched transition index")
    ax.set_ylabel("Translation L2 error")
    ax.set_title("Effect of rotation conditioning on Model T")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig, path, dpi)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def dataframe_records_json_safe(
    frame: pd.DataFrame,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for raw_record in frame.to_dict(orient="records"):
        record: Dict[str, object] = {}
        for key, value in raw_record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (np.integer,)):
                record[key] = int(value)
            elif isinstance(value, (np.floating,)):
                record[key] = safe_float(value)
            else:
                record[key] = value
        records.append(record)
    return records


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        seed_everything(args.seed)

        repo_root = find_repo_root(Path(__file__))
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        helper = load_evaluation_module(repo_root)
        device = helper.resolve_device(args.device)

        checkpoint_path = (
            args.checkpoint
            if args.checkpoint.is_absolute()
            else repo_root / args.checkpoint
        )
        data_root = (
            args.data_root
            if args.data_root.is_absolute()
            else repo_root / args.data_root
        )
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else repo_root / args.output_dir
        )
        plots_dir = output_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = helper.load_checkpoint(
            checkpoint_path,
            device,
        )

        helper_args = argparse.Namespace(
            data_root=data_root,
            sequence=args.sequence,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rotation_loss_weight=None,
            translation_loss_weight=None,
        )

        evaluation_configuration = (
            helper.resolve_evaluation_configuration(
                args=helper_args,
                checkpoint=checkpoint,
            )
        )

        dataset = helper.build_dataset(
            args=helper_args,
            evaluation_configuration=evaluation_configuration,
        )
        dataloader = helper.build_dataloader(
            dataset=dataset,
            args=helper_args,
            device=device,
        )
        model = helper.build_model(
            checkpoint=checkpoint,
            evaluation_configuration=evaluation_configuration,
            device=device,
        )

        print("=" * 80)
        print("Relative Pose Model Output Audit")
        print("=" * 80)
        print("Checkpoint:       {}".format(checkpoint_path.resolve()))
        print("Checkpoint epoch: {}".format(checkpoint.get("epoch")))
        print("Sequence:         {}".format(args.sequence))
        print("Dataset samples:  {}".format(len(dataset)))
        print("Device:           {}".format(device))
        print("Semantic cues:    {}".format(
            evaluation_configuration.get("use_semantic_cues")
        ))
        print("Depth cues:       {}".format(
            evaluation_configuration.get("use_depth_cues")
        ))
        print("=" * 80)

        parameter_statistics = inspect_head_parameters(model)
        parameter_statistics.to_csv(
            output_dir / "head_parameter_statistics.csv",
            index=False,
        )

        use_internal_depth = bool(
            evaluation_configuration.get("use_depth_cues", False)
        )

        if args.compare_translation_conditioning:
            predicted_frame, predicted_features = audit_pass(
                helper=helper,
                model=model,
                dataloader=dataloader,
                device=device,
                use_internal_depth=use_internal_depth,
                use_ground_truth_rotation=False,
                max_samples=args.max_samples,
                log_interval=args.log_interval,
            )

            # Rebuild loader to avoid assumptions about reusable iterators.
            dataloader_gt = helper.build_dataloader(
                dataset=dataset,
                args=helper_args,
                device=device,
            )
            gt_frame, gt_features = audit_pass(
                helper=helper,
                model=model,
                dataloader=dataloader_gt,
                device=device,
                use_internal_depth=use_internal_depth,
                use_ground_truth_rotation=True,
                max_samples=args.max_samples,
                log_interval=args.log_interval,
            )

            primary_frame = (
                gt_frame
                if args.use_ground_truth_rotation
                else predicted_frame
            )
            primary_features = (
                gt_features
                if args.use_ground_truth_rotation
                else predicted_features
            )

            predicted_frame.to_csv(
                output_dir / "frame_outputs_predicted_conditioning.csv",
                index=False,
            )
            gt_frame.to_csv(
                output_dir / "frame_outputs_gt_conditioning.csv",
                index=False,
            )

            comparison = conditioning_comparison(
                predicted_frame,
                gt_frame,
            )
            comparison.to_csv(
                output_dir / "conditioning_comparison.csv",
                index=False,
            )
            plot_conditioning_errors(
                comparison,
                plots_dir / "conditioning_translation_errors.png",
                args.dpi,
            )
        else:
            primary_frame, primary_features = audit_pass(
                helper=helper,
                model=model,
                dataloader=dataloader,
                device=device,
                use_internal_depth=use_internal_depth,
                use_ground_truth_rotation=args.use_ground_truth_rotation,
                max_samples=args.max_samples,
                log_interval=args.log_interval,
            )
            comparison = None

        primary_frame.to_csv(
            output_dir / "frame_outputs.csv",
            index=False,
        )
        primary_features.to_csv(
            output_dir / "feature_statistics.csv",
            index=False,
        )

        feature_summary = aggregate_feature_statistics(
            primary_features
        )
        feature_summary.to_csv(
            output_dir / "feature_summary.csv",
            index=False,
        )

        axis_statistics = build_axis_statistics(primary_frame)
        correlations = build_correlations(primary_frame)
        failures = detect_failure_modes(
            primary_frame,
            correlations,
            args,
        )

        axis_statistics.to_csv(
            output_dir / "axis_statistics.csv",
            index=False,
        )
        correlations.to_csv(
            output_dir / "correlations.csv",
            index=False,
        )
        failures.to_csv(
            output_dir / "failure_modes.csv",
            index=False,
        )

        worst_rotation = primary_frame.sort_values(
            "rotation_error_norm",
            ascending=False,
        ).head(args.worst_frame_count)
        worst_translation = primary_frame.sort_values(
            "translation_error_norm",
            ascending=False,
        ).head(args.worst_frame_count)

        worst_rotation.to_csv(
            output_dir / "worst_rotation_frames.csv",
            index=False,
        )
        worst_translation.to_csv(
            output_dir / "worst_translation_frames.csv",
            index=False,
        )

        plot_target_scatter(
            primary_frame,
            "rotation",
            "raw_dense",
            plots_dir / "rotation_target_vs_raw_dense.png",
            args.dpi,
        )
        plot_target_scatter(
            primary_frame,
            "rotation",
            "final",
            plots_dir / "rotation_target_vs_final.png",
            args.dpi,
        )
        plot_target_scatter(
            primary_frame,
            "translation",
            "raw_dense",
            plots_dir / "translation_target_vs_raw_dense.png",
            args.dpi,
        )
        plot_target_scatter(
            primary_frame,
            "translation",
            "final",
            plots_dir / "translation_target_vs_final.png",
            args.dpi,
        )
        plot_distributions(
            primary_frame,
            "rotation",
            plots_dir / "rotation_output_distributions.png",
            args.histogram_bins,
            args.dpi,
        )
        plot_distributions(
            primary_frame,
            "translation",
            plots_dir / "translation_output_distributions.png",
            args.histogram_bins,
            args.dpi,
        )
        plot_raw_vs_activated(
            primary_frame,
            "rotation",
            plots_dir / "raw_vs_activated_rotation.png",
            args.dpi,
        )
        plot_raw_vs_activated(
            primary_frame,
            "translation",
            plots_dir / "raw_vs_activated_translation.png",
            args.dpi,
        )
        plot_timeseries(
            primary_frame,
            "rotation",
            plots_dir / "output_timeseries_rotation.png",
            args.dpi,
        )
        plot_timeseries(
            primary_frame,
            "translation",
            plots_dir / "output_timeseries_translation.png",
            args.dpi,
        )
        plot_feature_dispersion(
            primary_features,
            plots_dir / "feature_dispersion.png",
            args.dpi,
        )

        primary_conditioning = (
            "ground_truth_rotation"
            if args.use_ground_truth_rotation
            else "predicted_rotation"
        )

        summary: Dict[str, object] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "sequence": str(args.sequence),
            "device": str(device),
            "samples_available": int(len(dataset)),
            "samples_analyzed": int(len(primary_frame)),
            "primary_translation_conditioning": primary_conditioning,
            "conditioning_comparison_enabled": bool(
                args.compare_translation_conditioning
            ),
            "evaluation_configuration": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in evaluation_configuration.items()
            },
            "principal_metrics": {},
            "failure_mode_count": int(len(failures)),
            "failure_modes": dataframe_records_json_safe(failures),
            "conditioning_comparison": None,
            "outputs": {
                "frame_outputs": "frame_outputs.csv",
                "axis_statistics": "axis_statistics.csv",
                "correlations": "correlations.csv",
                "feature_statistics": "feature_statistics.csv",
                "feature_summary": "feature_summary.csv",
                "head_parameter_statistics": (
                    "head_parameter_statistics.csv"
                ),
                "failure_modes": "failure_modes.csv",
                "worst_rotation_frames": (
                    "worst_rotation_frames.csv"
                ),
                "worst_translation_frames": (
                    "worst_translation_frames.csv"
                ),
                "plots": "plots/",
            },
        }

        for head in HEADS:
            metrics: Dict[str, object] = {}
            for axis in AXES:
                final_row = correlations[
                    (correlations["head"] == head)
                    & (correlations["signal"] == "final")
                    & (correlations["axis"] == axis)
                ].iloc[0]
                raw_row = correlations[
                    (correlations["head"] == head)
                    & (correlations["signal"] == "raw_dense")
                    & (correlations["axis"] == axis)
                ].iloc[0]
                metrics[axis] = {
                    "final_pearson_r": safe_float(
                        final_row["pearson_r"]
                    ),
                    "final_slope": safe_float(
                        final_row["slope"]
                    ),
                    "final_bias": safe_float(
                        final_row["bias"]
                    ),
                    "final_rmse": safe_float(
                        final_row["rmse"]
                    ),
                    "final_std_ratio": safe_float(
                        final_row[
                            "prediction_to_target_std_ratio"
                        ]
                    ),
                    "raw_dense_pearson_r": safe_float(
                        raw_row["pearson_r"]
                    ),
                    "raw_dense_slope": safe_float(
                        raw_row["slope"]
                    ),
                }
            summary["principal_metrics"][head] = metrics

        if comparison is not None and not comparison.empty:
            predicted_error = comparison[
                "translation_error_norm_predicted_conditioning"
            ].to_numpy(dtype=np.float64)
            gt_error = comparison[
                "translation_error_norm_gt_conditioning"
            ].to_numpy(dtype=np.float64)
            improvement = predicted_error - gt_error
            summary["conditioning_comparison"] = {
                "matched_samples": int(len(comparison)),
                "predicted_conditioning_mean_translation_error": (
                    safe_float(np.mean(predicted_error))
                ),
                "gt_conditioning_mean_translation_error": (
                    safe_float(np.mean(gt_error))
                ),
                "mean_error_reduction_with_gt_rotation": (
                    safe_float(np.mean(improvement))
                ),
                "median_error_reduction_with_gt_rotation": (
                    safe_float(np.median(improvement))
                ),
                "fraction_improved_with_gt_rotation": (
                    safe_float(np.mean(improvement > 0.0))
                ),
            }

        write_json(output_dir / "summary.json", summary)

        print(
            "[audit] complete: samples={} findings={} output={}".format(
                len(primary_frame),
                len(failures),
                output_dir.resolve(),
            )
        )
        return 0

    except Exception as error:
        print(
            "[audit] ERROR: {}".format(error),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
