#!/usr/bin/env python3
"""Read-only gradient, activation, and BatchNorm audit for A6 checkpoints.

The script loads epochs 30, 120, and 180, selects the same source-training
samples from fixed GT-z regimes, performs forward/backward diagnostics without
an optimizer step, and writes parameter-, module-, and branch-level results.
Checkpoint files on disk are never modified.
"""
from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


DEFAULT_BOUNDARIES = ((30, 32), (120, 8), (180, 2))
STRUCTURAL_KEYS = (
    "height", "width", "rotation_pool_size", "rotation_normalization_scale",
    "translation_decoder", "translation_mlp_hidden_dims",
    "translation_projection_channels", "translation_pool_size",
    "translation_aggregation_hidden_dim", "translation_num_experts",
    "share_aresunet_between_models", "use_semantic_cues", "pretrained_semantic",
    "freeze_semantic", "semantic_map_mode", "use_depth_cues",
    "depth_model_name", "depth_output_mode", "freeze_depth",
    "depth_normalization_meters", "use_ground_truth_rotation",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    base = "experiments/track_a/paper_reproduction/a6_paper_schedule_120x120"
    p.add_argument("--experiment-dir", default=base)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--train-script", default="scripts/train_deepdct_vo.py")
    p.add_argument("--data-root", default="data")
    p.add_argument("--depth-checkpoint-dir", default=None,
                   help="Override the depth checkpoint path stored in checkpoints.")
    p.add_argument("--epochs", nargs="+", type=int, default=[30, 120, 180])
    p.add_argument("--samples-per-regime", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--near-zero-threshold", type=float, default=1e-8)
    p.add_argument("--extreme-threshold", type=float, default=10.0)
    return p.parse_args()


def load_training_module(path):
    if not os.path.isfile(path):
        raise SystemExit("Missing training script: {}".format(path))
    spec = importlib.util.spec_from_file_location("deepdct_training_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_path(experiment, epoch):
    matches = [item for item in DEFAULT_BOUNDARIES if item[0] == epoch]
    if matches:
        batch = matches[0][1]
        boundary = os.path.join(
            experiment, "checkpoints",
            "schedule_boundary_epoch_{:03d}_batch_{:02d}.pt".format(epoch, batch))
        if os.path.isfile(boundary):
            return boundary
    ordinary = os.path.join(experiment, "checkpoints", "deepdct_vo_epoch_{:03d}.pt".format(epoch))
    if os.path.isfile(ordinary):
        return ordinary
    raise SystemExit("Missing checkpoint for epoch {} under {}".format(epoch, experiment))


def resolve_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_file(path, map_location):
    """Load trusted local experiment checkpoints without the default warning."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases lacking weights_only.
        return torch.load(path, map_location=map_location)


def normalize_config(checkpoint, args):
    config = dict(checkpoint.get("configuration", {}))
    if not config:
        raise SystemExit("Checkpoint has no configuration mapping")
    config["data_root"] = os.path.abspath(args.data_root)
    # The complete semantic state is restored immediately afterward. Avoid a
    # constructor-time network download while preserving identical topology.
    config["pretrained_semantic"] = False
    if args.depth_checkpoint_dir is not None:
        config["depth_checkpoint_dir"] = os.path.abspath(args.depth_checkpoint_dir)
    for key in ("data_root", "depth_checkpoint_dir"):
        if config.get(key) is not None:
            from pathlib import Path
            config[key] = Path(config[key])
    return Namespace(**config)


def select_samples(dataset, count_per_regime, seed):
    if not hasattr(dataset, "records") or not dataset.records:
        raise SystemExit("Training dataset does not expose non-empty records")
    z = np.asarray([float(record.translation_gt[2]) for record in dataset.records])
    low, high = np.quantile(z, [1.0 / 3.0, 2.0 / 3.0])
    masks = (z <= low, (z > low) & (z <= high), z > high)
    rng = random.Random(seed)
    selected = []
    for regime, mask in zip(("low", "medium", "high"), masks):
        candidates = np.flatnonzero(mask).tolist()
        if len(candidates) < count_per_regime:
            raise SystemExit("Not enough {}-regime samples".format(regime))
        picks = sorted(rng.sample(candidates, count_per_regime))
        selected.extend((index, regime, float(z[index])) for index in picks)
    return selected, (float(low), float(high))


def batch_tensor(sample, key, device):
    value = sample.get(key)
    if not torch.is_tensor(value):
        raise TypeError("dataset sample {!r} must be a tensor".format(key))
    return value.unsqueeze(0).to(device=device, non_blocking=False)


def parameter_group(name):
    for prefix in ("semantic_model", "depth_model", "rotation_aresunet",
                   "translation_aresunet", "rotation_head", "translation_head"):
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return "other"


def tensors_from_output(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(tensors_from_output(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(tensors_from_output(item))
        return result
    return []


class ActivationAccumulator(object):
    def __init__(self, near_zero, extreme):
        self.near_zero = near_zero
        self.extreme = extreme
        self.values = defaultdict(lambda: dict(count=0, total=0.0, square=0.0,
                                                minimum=float("inf"), maximum=float("-inf"),
                                                near=0, negative=0, extreme=0, calls=0))

    def hook(self, name):
        def capture(_module, _inputs, output):
            tensors = tensors_from_output(output)
            if not tensors:
                return
            state = self.values[name]
            state["calls"] += 1
            for tensor in tensors:
                detached = tensor.detach().float()
                finite = detached[torch.isfinite(detached)]
                if finite.numel() == 0:
                    continue
                state["count"] += int(finite.numel())
                state["total"] += float(finite.sum().item())
                state["square"] += float((finite * finite).sum().item())
                state["minimum"] = min(state["minimum"], float(finite.min().item()))
                state["maximum"] = max(state["maximum"], float(finite.max().item()))
                state["near"] += int((finite.abs() <= self.near_zero).sum().item())
                state["negative"] += int((finite < 0).sum().item())
                state["extreme"] += int((finite.abs() >= self.extreme).sum().item())
        return capture

    def rows(self, epoch, module_types):
        rows = []
        for name, state in sorted(self.values.items()):
            count = state["count"]
            if count == 0:
                continue
            mean = state["total"] / count
            variance = max(0.0, state["square"] / count - mean * mean)
            rows.append({
                "epoch": epoch, "module": name, "module_type": module_types[name],
                "calls": state["calls"], "elements": count, "mean": mean,
                "std": math.sqrt(variance), "minimum": state["minimum"],
                "maximum": state["maximum"], "near_zero_fraction": state["near"] / float(count),
                "negative_fraction": state["negative"] / float(count),
                "extreme_fraction": state["extreme"] / float(count),
            })
        return rows


def register_activation_hooks(model, accumulator):
    handles = []
    types = {}
    top = {"semantic_model", "depth_model", "rotation_aresunet",
           "translation_aresunet", "rotation_head", "translation_head"}
    detailed = {"rotation_aresunet", "translation_aresunet",
                "rotation_head", "translation_head"}
    tracked_types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU,
                     nn.Sigmoid, nn.Tanh, nn.AdaptiveAvgPool2d)
    for name, module in model.named_modules():
        if not name:
            continue
        branch = parameter_group(name)
        is_branch_leaf = branch in detailed and len(list(module.children())) == 0 and isinstance(module, tracked_types)
        if name in top or is_branch_leaf:
            types[name] = module.__class__.__name__
            handles.append(module.register_forward_hook(accumulator.hook(name)))
    return handles, types


def gradient_rows(model, epoch):
    rows = []
    grouped = defaultdict(lambda: dict(parameter_count=0, elements=0, grad_square=0.0,
                                       param_square=0.0, zero_grad_elements=0, missing=0))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_group(name)
        state = grouped[group]
        state["parameter_count"] += 1
        state["elements"] += int(parameter.numel())
        parameter_norm = float(parameter.detach().float().norm().item())
        state["param_square"] += parameter_norm ** 2
        if parameter.grad is None:
            grad_norm = float("nan")
            mean_abs = float("nan")
            maximum = float("nan")
            zero_fraction = float("nan")
            state["missing"] += 1
        else:
            grad = parameter.grad.detach().float()
            grad_norm = float(grad.norm().item())
            mean_abs = float(grad.abs().mean().item())
            maximum = float(grad.abs().max().item())
            zero_count = int((grad == 0).sum().item())
            zero_fraction = zero_count / float(grad.numel())
            state["grad_square"] += grad_norm ** 2
            state["zero_grad_elements"] += zero_count
        rows.append({"epoch": epoch, "parameter": name, "group": group,
                     "elements": int(parameter.numel()), "parameter_l2": parameter_norm,
                     "gradient_l2": grad_norm, "gradient_mean_abs": mean_abs,
                     "gradient_max_abs": maximum, "gradient_zero_fraction": zero_fraction,
                     "gradient_to_parameter_ratio": grad_norm / parameter_norm if parameter_norm > 0 and math.isfinite(grad_norm) else float("nan")})
    group_rows = []
    for group, state in sorted(grouped.items()):
        grad_l2 = math.sqrt(state["grad_square"])
        param_l2 = math.sqrt(state["param_square"])
        group_rows.append({"epoch": epoch, "group": group,
                           "parameter_tensors": state["parameter_count"], "elements": state["elements"],
                           "missing_gradient_tensors": state["missing"], "parameter_l2": param_l2,
                           "gradient_l2": grad_l2,
                           "gradient_to_parameter_ratio": grad_l2 / param_l2 if param_l2 > 0 else float("nan"),
                           "gradient_zero_fraction": state["zero_grad_elements"] / float(state["elements"]) if state["elements"] else float("nan")})
    return rows, group_rows


def batchnorm_rows(model, epoch):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            mean = module.running_mean.detach().float()
            variance = module.running_var.detach().float()
            tracked = module.num_batches_tracked
            rows.append({"epoch": epoch, "module": name, "features": int(mean.numel()),
                         "running_mean_l2": float(mean.norm().item()),
                         "running_mean_abs_max": float(mean.abs().max().item()),
                         "running_var_mean": float(variance.mean().item()),
                         "running_var_min": float(variance.min().item()),
                         "running_var_max": float(variance.max().item()),
                         "num_batches_tracked": int(tracked.item()) if tracked is not None else -1,
                         "running_mean_vector": mean.cpu().numpy().tolist(),
                         "running_var_vector": variance.cpu().numpy().tolist()})
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    if args.samples_per_regime < 1:
        raise SystemExit("--samples-per-regime must be positive")
    experiment = os.path.abspath(args.experiment_dir)
    output = os.path.abspath(args.output_dir or os.path.join(experiment, "gradient_activation_health_audit"))
    os.makedirs(output, exist_ok=True)
    device = resolve_device(args.device)
    training_module = load_training_module(os.path.abspath(args.train_script))

    checkpoints = {}
    configurations = {}
    for epoch in args.epochs:
        path = checkpoint_path(experiment, epoch)
        checkpoint = load_checkpoint_file(path, map_location="cpu")
        checkpoints[epoch] = (path, checkpoint)
        configurations[epoch] = dict(checkpoint.get("configuration", {}))
    reference_epoch = args.epochs[0]
    reference_config = configurations[reference_epoch]
    for epoch in args.epochs[1:]:
        for key in STRUCTURAL_KEYS:
            if configurations[epoch].get(key) != reference_config.get(key):
                raise SystemExit("Structural configuration differs at epoch {} for {!r}".format(epoch, key))

    model_args = normalize_config(checkpoints[reference_epoch][1], args)
    dataset = training_module.build_dataset(model_args, list(model_args.train_sequences))
    selected, thresholds = select_samples(dataset, args.samples_per_regime, args.seed)
    sample_rows = [{"dataset_index": index, "regime": regime, "translation_gt_z": z}
                   for index, regime, z in selected]

    activation_rows, parameter_rows, group_rows, bn_rows = [], [], [], []
    loss_rows = []
    for epoch in args.epochs:
        path, checkpoint = checkpoints[epoch]
        current_args = normalize_config(checkpoint, args)
        model = training_module.build_model(current_args, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        model.zero_grad(set_to_none=True)
        accumulator = ActivationAccumulator(args.near_zero_threshold, args.extreme_threshold)
        handles, module_types = register_activation_hooks(model, accumulator)
        total_rotation = 0.0
        total_translation = 0.0
        divisor = float(len(selected))
        for index, _regime, _z in selected:
            sample = dataset[index]
            image_prev = batch_tensor(sample, "image_prev", device)
            image_curr = batch_tensor(sample, "image_curr", device)
            rotation_gt = batch_tensor(sample, "rotation_gt", device)
            translation_gt = batch_tensor(sample, "translation_gt", device)
            depth_curr = None if bool(current_args.use_depth_cues) else batch_tensor(sample, "depth_curr", device)
            outputs = model(image_prev=image_prev, image_curr=image_curr, depth_curr=depth_curr,
                            rotation_for_translation=rotation_gt,
                            use_ground_truth_rotation=bool(current_args.use_ground_truth_rotation),
                            return_intermediates=True)
            rotation_loss = torch.mean(torch.abs(
                outputs["rotation_normalized"] - rotation_gt / float(current_args.rotation_normalization_scale)))
            translation_loss = torch.mean(torch.abs(
                outputs["directional_translation"] - translation_gt))
            loss = (float(current_args.rotation_loss_weight) * rotation_loss
                    + float(current_args.translation_loss_weight) * translation_loss) / divisor
            loss.backward()
            total_rotation += float(rotation_loss.detach().item())
            total_translation += float(translation_loss.detach().item())
        for handle in handles:
            handle.remove()
        activation_rows.extend(accumulator.rows(epoch, module_types))
        parameter, grouped = gradient_rows(model, epoch)
        parameter_rows.extend(parameter)
        group_rows.extend(grouped)
        bn_rows.extend(batchnorm_rows(model, epoch))
        loss_rows.append({"epoch": epoch, "checkpoint": path,
                          "diagnostic_rotation_mae": total_rotation / divisor,
                          "diagnostic_translation_mae": total_translation / divisor,
                          "samples": len(selected), "model_mode": "eval_with_gradients",
                          "optimizer_step_performed": False})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bn_lookup = {(row["epoch"], row["module"]): row for row in bn_rows}
    bn_drift = []
    baseline_modules = sorted(row["module"] for row in bn_rows if row["epoch"] == reference_epoch)
    for epoch in args.epochs[1:]:
        for name in baseline_modules:
            before = bn_lookup[(reference_epoch, name)]
            after = bn_lookup.get((epoch, name))
            if after is None:
                raise SystemExit("BatchNorm module missing at epoch {}: {}".format(epoch, name))
            mean_before = np.asarray(before["running_mean_vector"])
            mean_after = np.asarray(after["running_mean_vector"])
            var_before = np.asarray(before["running_var_vector"])
            var_after = np.asarray(after["running_var_vector"])
            bn_drift.append({"reference_epoch": reference_epoch, "epoch": epoch, "module": name,
                             "running_mean_delta_l2": float(np.linalg.norm(mean_after - mean_before)),
                             "running_mean_relative_delta": float(np.linalg.norm(mean_after - mean_before) / max(np.linalg.norm(mean_before), 1e-12)),
                             "running_var_delta_l2": float(np.linalg.norm(var_after - var_before)),
                             "running_var_relative_delta": float(np.linalg.norm(var_after - var_before) / max(np.linalg.norm(var_before), 1e-12)),
                             "tracked_batch_delta": after["num_batches_tracked"] - before["num_batches_tracked"]})

    activation_fields = ["epoch", "module", "module_type", "calls", "elements", "mean", "std",
                         "minimum", "maximum", "near_zero_fraction", "negative_fraction", "extreme_fraction"]
    parameter_fields = ["epoch", "parameter", "group", "elements", "parameter_l2", "gradient_l2",
                        "gradient_mean_abs", "gradient_max_abs", "gradient_zero_fraction",
                        "gradient_to_parameter_ratio"]
    group_fields = ["epoch", "group", "parameter_tensors", "elements", "missing_gradient_tensors",
                    "parameter_l2", "gradient_l2", "gradient_to_parameter_ratio", "gradient_zero_fraction"]
    bn_fields = ["epoch", "module", "features", "running_mean_l2", "running_mean_abs_max",
                 "running_var_mean", "running_var_min", "running_var_max", "num_batches_tracked"]
    drift_fields = ["reference_epoch", "epoch", "module", "running_mean_delta_l2",
                    "running_mean_relative_delta", "running_var_delta_l2", "running_var_relative_delta",
                    "tracked_batch_delta"]
    write_csv(os.path.join(output, "diagnostic_samples.csv"), sample_rows,
              ["dataset_index", "regime", "translation_gt_z"])
    write_csv(os.path.join(output, "diagnostic_losses.csv"), loss_rows,
              ["epoch", "checkpoint", "diagnostic_rotation_mae", "diagnostic_translation_mae",
               "samples", "model_mode", "optimizer_step_performed"])
    write_csv(os.path.join(output, "activation_statistics.csv"), activation_rows, activation_fields)
    write_csv(os.path.join(output, "parameter_gradient_statistics.csv"), parameter_rows, parameter_fields)
    write_csv(os.path.join(output, "gradient_statistics_by_branch.csv"), group_rows, group_fields)
    write_csv(os.path.join(output, "batchnorm_running_statistics.csv"),
              [{key: row[key] for key in bn_fields} for row in bn_rows], bn_fields)
    write_csv(os.path.join(output, "batchnorm_drift_from_epoch_{:03d}.csv".format(reference_epoch)),
              bn_drift, drift_fields)
    report = {"device": str(device), "epochs": args.epochs,
              "regime_thresholds": {"low_max": thresholds[0], "medium_max": thresholds[1]},
              "samples": sample_rows, "losses": loss_rows, "activations": activation_rows,
              "parameter_gradients": parameter_rows, "branch_gradients": group_rows,
              "batchnorm_statistics": bn_rows, "batchnorm_drift": bn_drift}
    with open(os.path.join(output, "gradient_activation_health_audit.json"), "w") as handle:
        json.dump(clean_json(report), handle, indent=2, sort_keys=True)

    print("A6 checkpoint gradient/activation health audit")
    print("=" * 118)
    print("Device: {}; model mode: eval with gradients; optimizer steps: NONE".format(device))
    print("Fixed training GT-z regimes: low <= {:.9f}; medium <= {:.9f}; high above".format(*thresholds))
    print("{:<6} {:<24} {:>14} {:>14} {:>12} {:>10}".format(
        "Epoch", "Branch", "Gradient L2", "Grad/param", "Zero frac", "Missing"))
    print("-" * 118)
    for row in group_rows:
        print("{:<6} {:<24} {:>14.6e} {:>14.6e} {:>12.5f} {:>10}".format(
            row["epoch"], row["group"], row["gradient_l2"],
            row["gradient_to_parameter_ratio"], row["gradient_zero_fraction"],
            row["missing_gradient_tensors"]))
    worst = sorted(bn_drift, key=lambda row: row["running_var_relative_delta"], reverse=True)[:10]
    print("\nLargest BatchNorm running-variance changes from epoch {}:".format(reference_epoch))
    for row in worst:
        print("  epoch {:3d} {:60s} relative delta={:.6f}".format(
            row["epoch"], row["module"], row["running_var_relative_delta"]))
    print("\nSaved audit outputs to: {}".format(output))


if __name__ == "__main__":
    main()
