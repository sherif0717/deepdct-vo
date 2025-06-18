#!/usr/bin/env python3
"""Recalibrate A6 pose-branch BatchNorm buffers without changing learned weights.

For each intact checkpoint, this script performs one deterministic, no-gradient
pass over training sequences 00--08. Only BatchNorm modules belonging to the
rotation and translation AResUNets are placed in training mode. Their running
statistics are recomputed; every parameter and every other state tensor remains
unchanged. Rotation-only, translation-only, and both-branch evaluation-only
checkpoints are then derived from that single recalibration pass.

The single-pass derivation is valid for Track-A A6 because Model T is conditioned
on ground-truth rotation and the rotation/translation AResUNets are separate.
Frozen semantic and depth auxiliary BatchNorm statistics are never modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


POSE_BRANCH_PREFIXES = {
    "rotation": "rotation_aresunet.",
    "translation": "translation_aresunet.",
}
BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute pose-branch BatchNorm statistics and evaluate the resulting checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-120", type=Path, default=None)
    parser.add_argument("--checkpoint-180", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/track_a/paper_reproduction/a6_paper_schedule_120x120/"
            "checkpoint_batchnorm_recalibration"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--training-sequences",
        nargs="+",
        default=["00", "01", "02", "03", "04", "05", "06", "07", "08"],
    )
    parser.add_argument("--recalibration-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum recalibration batches per checkpoint; zero consumes all training data.",
    )
    parser.add_argument(
        "--reset-policy",
        choices=["reset", "preserve"],
        default="reset",
        help="Reset running statistics before accumulation, or continue from checkpoint values.",
    )
    parser.add_argument(
        "--momentum-policy",
        choices=["cumulative", "module"],
        default="cumulative",
        help="Use cumulative averaging (momentum=None) or each module's configured momentum.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluator", type=Path, default=Path("scripts/evaluate_deepdct_vo.py")
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluation-sequences", nargs="+", default=["09", "10"])
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument(
        "--trajectory-gt-option",
        default="auto",
        help=(
            "GT-trajectory evaluator option: 'auto', a flag, or an option/value "
            "pair such as '--trajectory-rotation-source ground_truth'."
        ),
    )
    parser.add_argument("--extra-evaluator-args", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Rebuild metrics CSV/JSON from existing evaluation summary.json files only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.collect_only and args.evaluate:
        raise ValueError("--collect-only and --evaluate are mutually exclusive")
    if args.collect_only:
        return
    if args.checkpoint_120 is None or args.checkpoint_180 is None:
        raise ValueError(
            "--checkpoint-120 and --checkpoint-180 are required unless --collect-only is used"
        )
    for path in (args.checkpoint_120, args.checkpoint_180, args.evaluator):
        if path is None or not path.is_file():
            raise FileNotFoundError("Missing required file: {}".format(path))
    if not args.data_root.is_dir():
        raise FileNotFoundError("Missing data root: {}".format(args.data_root))
    if args.recalibration_batch_size <= 0 or args.evaluation_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers < 0 or args.max_batches < 0:
        raise ValueError("num-workers and max-batches must be non-negative")
    if len(set(args.training_sequences)) != len(args.training_sequences):
        raise ValueError("Duplicate training sequence")
    if set(args.training_sequences) != {
        "00", "01", "02", "03", "04", "05", "06", "07", "08"
    }:
        raise ValueError("This A6 diagnostic requires training sequences exactly 00--08")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_evaluator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_deepdct_recalibration_evaluator", str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to import evaluator from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in (
        "load_checkpoint", "resolve_evaluation_configuration", "build_model",
        "DeepDCTTrainingDataset",
    ):
        if not hasattr(module, name):
            raise AttributeError("Evaluator lacks required symbol: {}".format(name))
    return module


def load_checkpoint(evaluator: ModuleType, path: Path) -> MutableMapping[str, Any]:
    checkpoint = evaluator.load_checkpoint(path, torch.device("cpu"))
    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("Checkpoint is not mutable mapping: {}".format(path))
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint lacks model_state_dict: {}".format(path))
    return checkpoint


def resolve_configuration(
    evaluator: ModuleType, checkpoint: Mapping[str, Any]
) -> Mapping[str, Any]:
    minimal_args = SimpleNamespace(rotation_loss_weight=None, translation_loss_weight=None)
    configuration = evaluator.resolve_evaluation_configuration(minimal_args, checkpoint)
    if not bool(configuration.get("use_ground_truth_rotation", False)):
        raise RuntimeError(
            "This optimized single-pass A6 diagnostic requires checkpoint "
            "use_ground_truth_rotation=True."
        )
    return configuration


def build_training_loader(
    evaluator: ModuleType,
    configuration: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    dataset = evaluator.DeepDCTTrainingDataset(
        data_root=args.data_root,
        sequences=tuple(args.training_sequences),
        camera=str(configuration["camera"]),
        image_size=(int(configuration["height"]), int(configuration["width"])),
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=False,
    )
    return DataLoader(
        dataset,
        batch_size=args.recalibration_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def normalized_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def state_branch(key: str) -> Optional[str]:
    name = normalized_key(key)
    for branch, prefix in POSE_BRANCH_PREFIXES.items():
        if name.startswith(prefix):
            return branch
    return None


def is_bn_buffer(key: str) -> bool:
    return normalized_key(key).endswith(tuple("." + x for x in BN_BUFFER_SUFFIXES))


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_digest(state: Mapping[str, torch.Tensor], keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_digest(state[key]).encode("ascii"))
    return digest.hexdigest()


def selected_bn_modules(model: nn.Module) -> Dict[str, nn.modules.batchnorm._BatchNorm]:
    selected: Dict[str, nn.modules.batchnorm._BatchNorm] = {}
    for name, module in model.named_modules():
        branch = state_branch(name + ".dummy")
        if branch is not None and isinstance(module, nn.modules.batchnorm._BatchNorm):
            selected[name] = module
    if not selected:
        raise RuntimeError("No pose-branch BatchNorm modules found")
    branches = {state_branch(name + ".dummy") for name in selected}
    if branches != {"rotation", "translation"}:
        raise RuntimeError("Expected BatchNorm modules in both pose branches; found {}".format(branches))
    return selected


def configure_bn_recalibration(
    model: nn.Module, args: argparse.Namespace
) -> Dict[str, nn.modules.batchnorm._BatchNorm]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = selected_bn_modules(model)
    for module in selected.values():
        if args.reset_policy == "reset":
            module.reset_running_stats()
        if args.momentum_policy == "cumulative":
            module.momentum = None
        module.train(True)
    return selected


def move_required(batch: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise TypeError("batch[{!r}] must be a tensor".format(key))
    return value.to(device=device, non_blocking=device.type == "cuda")


def recalibrate_model(
    model: nn.Module,
    loader: DataLoader,
    configuration: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[int, int]:
    batches = 0
    samples = 0
    use_internal_depth = bool(configuration.get("use_depth_cues", False))
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            image_prev = move_required(batch, "image_prev", device)
            image_curr = move_required(batch, "image_curr", device)
            rotation_gt = move_required(batch, "rotation_gt", device)
            depth_curr = None if use_internal_depth else move_required(batch, "depth_curr", device)
            model(
                image_prev=image_prev,
                image_curr=image_curr,
                depth_curr=depth_curr,
                rotation_for_translation=rotation_gt,
                use_ground_truth_rotation=True,
                return_intermediates=False,
            )
            batches += 1
            samples += int(image_prev.shape[0])
            if batches == 1 or batches % 100 == 0:
                print("Recalibration batch {} samples {}".format(batches, samples), flush=True)
    if batches == 0:
        raise RuntimeError("Recalibration DataLoader produced no batches")
    return batches, samples


def verify_recalibrated_state(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> Dict[str, Any]:
    if set(before) != set(after):
        raise RuntimeError("Model state keys changed during recalibration")
    selected = [key for key in before if is_bn_buffer(key) and state_branch(key) is not None]
    forbidden_changes = []
    changed_selected = []
    for key in before:
        equal = torch.equal(before[key].detach().cpu(), after[key].detach().cpu())
        if key in selected:
            if not equal:
                changed_selected.append(key)
        elif not equal:
            forbidden_changes.append(key)
    if forbidden_changes:
        raise RuntimeError(
            "Recalibration changed forbidden state tensors: {}".format(forbidden_changes[:20])
        )
    if not changed_selected:
        raise RuntimeError("No selected BatchNorm buffers changed")
    parameter_keys = [key for key in before if not is_bn_buffer(key)]
    return {
        "selected_bn_buffers": len(selected),
        "changed_bn_buffers": len(changed_selected),
        "unchanged_non_bn_digest_sha256": state_digest(after, parameter_keys),
        "recalibrated_bn_digest_sha256": state_digest(after, selected),
    }


def create_scope_checkpoint(
    checkpoint: Mapping[str, Any],
    recalibrated_state: Mapping[str, torch.Tensor],
    scope: str,
    source_path: Path,
    args: argparse.Namespace,
    batches: int,
    samples: int,
) -> Tuple[MutableMapping[str, Any], Dict[str, Any]]:
    result = copy.deepcopy(checkpoint)
    output_state = result["model_state_dict"]
    keys = [
        key for key in output_state
        if is_bn_buffer(key) and (scope == "both" or state_branch(key) == scope)
    ]
    if not keys:
        raise RuntimeError("No BatchNorm buffers selected for scope {}".format(scope))
    changed = 0
    for key in keys:
        replacement = recalibrated_state[key].detach().cpu().clone()
        if not torch.equal(output_state[key].detach().cpu(), replacement):
            changed += 1
        output_state[key] = replacement
    record = {
        "evaluation_only": True,
        "source_checkpoint": str(source_path.resolve()),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "scope": scope,
        "reset_policy": args.reset_policy,
        "momentum_policy": args.momentum_policy,
        "training_sequences": list(args.training_sequences),
        "recalibration_batch_size": args.recalibration_batch_size,
        "recalibration_batches": batches,
        "recalibration_samples": samples,
        "selected_buffers": len(keys),
        "changed_buffers": changed,
        "warning": "Evaluation only; do not resume optimizer state from this checkpoint.",
    }
    result["batchnorm_recalibration"] = record
    return result, record


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite {}; pass --overwrite".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, str(temporary))
    temporary.replace(path)


def run_command(command: Sequence[str], log_path: Path) -> None:
    print("RUN {}".format(" ".join(shlex.quote(x) for x in command)), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, list(command))


def discover_gt_trajectory_args(args: argparse.Namespace) -> List[str]:
    if args.trajectory_gt_option != "auto":
        parsed = shlex.split(args.trajectory_gt_option)
        if not parsed or not parsed[0].startswith("--"):
            raise ValueError("Invalid --trajectory-gt-option")
        return parsed
    result = subprocess.run(
        [args.python, str(args.evaluator), "--help"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
    )
    help_text = result.stdout
    for flag in (
        "--use-ground-truth-trajectory-rotation",
        "--trajectory-use-ground-truth-rotation",
        "--use-gt-trajectory-rotation",
    ):
        if flag in help_text:
            return [flag]
    for option in ("--trajectory-rotation-source", "--trajectory-rotation"):
        if option in help_text:
            return [option, "ground_truth"]
    raise RuntimeError("Cannot discover GT-trajectory evaluator option; pass --trajectory-gt-option")


def nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def first_value(mapping: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        value = nested_get(mapping, path)
        if value is not None:
            return value
    return None


def collect_summary(
    variant: str, sequence: str, trajectory_mode: str, output_dir: Path
) -> Dict[str, Any]:
    path = output_dir / "summary.json"
    row: Dict[str, Any] = {
        "variant": variant, "sequence": sequence,
        "trajectory_rotation": trajectory_mode,
        "output_dir": str(output_dir.resolve()), "summary_found": path.is_file(),
    }
    if not path.is_file():
        return row
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    trajectory = summary.get("trajectory_metrics_unscaled") or summary.get("trajectory_metrics") or {}
    row.update({
        "checkpoint_epoch": summary.get("checkpoint_epoch", summary.get("epoch")),
        "translation_mae": first_value(summary, (
            ("frame_metrics", "translation_mae"),
            ("aggregate_metrics", "translation_mae"), ("translation_mae",),
        )),
        "rotation_mae": first_value(summary, (
            ("frame_metrics", "rotation_mae"),
            ("aggregate_metrics", "rotation_mae"), ("rotation_mae",),
        )),
        "ate_rmse": trajectory.get("ate_rmse"),
        "rpe_translation_rmse": trajectory.get("rpe_translation_rmse"),
        "rpe_rotation_rmse_degrees": trajectory.get("rpe_rotation_rmse_degrees"),
        "approx_translation_drift_percent": trajectory.get("translational_drift_percent"),
        "approx_rotation_drift_degrees_per_100m": trajectory.get("rotational_drift_degrees_per_100m"),
    })
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_outputs(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(output_dir / "batchnorm_recalibration_metrics.csv", rows)
    with (output_dir / "batchnorm_recalibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"evaluations": list(rows)}, handle, indent=2, sort_keys=True)


def discover_existing_evaluations(output_dir: Path) -> List[Tuple[str, str, str, Path]]:
    root = output_dir / "evaluations"
    if not root.is_dir():
        raise FileNotFoundError("Missing evaluations directory: {}".format(root))
    pattern = re.compile(r"^sequence_(.+)_(gt_gt|gt_pred)$")
    found: List[Tuple[str, str, str, Path]] = []
    for path in sorted(root.glob("*/sequence_*/summary.json")):
        match = pattern.match(path.parent.name)
        if match is None:
            continue
        sequence, token = match.groups()
        found.append((
            path.parent.parent.name, sequence,
            "ground_truth" if token == "gt_gt" else "predicted", path.parent,
        ))
    if not found:
        raise RuntimeError("No existing summary.json files found under {}".format(root))
    return found


def collect_existing(output_dir: Path) -> List[Dict[str, Any]]:
    rows = [collect_summary(*item) for item in discover_existing_evaluations(output_dir)]
    missing = [row for row in rows if row.get("translation_mae") is None or row.get("rotation_mae") is None]
    if missing:
        raise RuntimeError("{} collected rows lack frame-level MAE".format(len(missing)))
    write_metric_outputs(output_dir, rows)
    return rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.collect_only:
        rows = collect_existing(args.output_dir)
        print("Collected {} existing evaluations.".format(len(rows)))
        return

    seed_everything(args.seed)
    device = resolve_device(args.device)
    evaluator = import_evaluator(args.evaluator)
    checkpoint_dir = args.output_dir / "checkpoints"
    variants: Dict[str, Path] = {}
    manifest_rows: List[Dict[str, Any]] = []

    assert args.checkpoint_120 is not None and args.checkpoint_180 is not None
    for checkpoint_path in (args.checkpoint_120, args.checkpoint_180):
        print("\nRecalibrating {}".format(checkpoint_path), flush=True)
        checkpoint = load_checkpoint(evaluator, checkpoint_path)
        configuration = resolve_configuration(evaluator, checkpoint)
        epoch = int(checkpoint.get("epoch", -1))
        # Re-evaluate the intact checkpoint in the same matrix so every
        # recalibrated result has a directly comparable native control.
        variants["native_epoch_{:03d}".format(epoch)] = checkpoint_path
        original_state = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint["model_state_dict"].items()
        }
        model = evaluator.build_model(checkpoint, configuration, device)
        modules = configure_bn_recalibration(model, args)
        loader = build_training_loader(evaluator, configuration, args, device)
        batches, samples = recalibrate_model(model, loader, configuration, args, device)
        model.eval()
        recalibrated_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        verification = verify_recalibrated_state(original_state, recalibrated_state)
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for scope in ("rotation", "translation", "both"):
            variant = "epoch_{:03d}_bn_recal_{}".format(epoch, scope)
            output_path = checkpoint_dir / (variant + ".pt")
            output_checkpoint, record = create_scope_checkpoint(
                checkpoint, recalibrated_state, scope, checkpoint_path,
                args, batches, samples,
            )
            save_checkpoint(output_checkpoint, output_path, args.overwrite)
            variants[variant] = output_path
            row = dict({
                "variant": variant, "checkpoint": str(output_path.resolve()),
                "recalibrated_modules_total": len(modules),
            }, **record, **verification)
            manifest_rows.append(row)
            print("Created {} ({} changed buffers)".format(output_path, record["changed_buffers"]))

    write_csv(args.output_dir / "batchnorm_recalibration_manifest.csv", manifest_rows)
    with (args.output_dir / "batchnorm_recalibration_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"variants": manifest_rows}, handle, indent=2, sort_keys=True)

    if not args.evaluate:
        print("\nRecalibrated checkpoints created. Add --evaluate to run the evaluation matrix.")
        return

    gt_args = discover_gt_trajectory_args(args)
    extra_args = shlex.split(args.extra_evaluator_args)
    metric_rows: List[Dict[str, Any]] = []
    for variant, checkpoint_path in variants.items():
        for sequence in args.evaluation_sequences:
            for trajectory_mode, trajectory_args in (("ground_truth", gt_args), ("predicted", [])):
                token = "gt_gt" if trajectory_mode == "ground_truth" else "gt_pred"
                output_dir = args.output_dir / "evaluations" / variant / "sequence_{}_{}".format(sequence, token)
                if output_dir.exists() and not args.overwrite:
                    raise FileExistsError("Evaluation output exists: {}".format(output_dir))
                command = [
                    args.python, str(args.evaluator),
                    "--checkpoint", str(checkpoint_path),
                    "--data-root", str(args.data_root),
                    "--sequence", sequence,
                    "--output-dir", str(output_dir),
                    "--batch-size", str(args.evaluation_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--device", args.device,
                    "--use-ground-truth-rotation",
                ] + list(trajectory_args) + extra_args
                run_command(command, output_dir / "evaluation.log")
                metric_rows.append(collect_summary(variant, sequence, trajectory_mode, output_dir))
                write_metric_outputs(args.output_dir, metric_rows)

    missing = [row for row in metric_rows if row.get("translation_mae") is None or row.get("rotation_mae") is None]
    if missing:
        raise RuntimeError("{} evaluation rows lack frame-level MAE".format(len(missing)))
    print("\nSaved BatchNorm recalibration audit to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
