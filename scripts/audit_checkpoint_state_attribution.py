#!/usr/bin/env python3
"""Causally attribute checkpoint changes to BatchNorm buffers or learned weights.

The script creates reciprocal hybrids from two architecture-identical checkpoints:

* epoch-180 weights + epoch-120 BatchNorm buffers
* epoch-120 weights + epoch-180 BatchNorm buffers

Each direction is generated for all trainable pose BatchNorm layers, the rotation
branch only, and the translation branch only.  Native checkpoints are never
modified.  Hybrid checkpoints are evaluation-only and must not be resumed for
training because their optimizer state still belongs to the target checkpoint.

Optionally, the existing scripts/evaluate_deepdct_vo.py is run for sequences 09
and 10 with GT rotation supplied to Model T.  Both GT-rotation and predicted-
rotation trajectory reconstructions are produced when the evaluator exposes a
supported trajectory-rotation option.
"""

import argparse
import copy
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


BN_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
BRANCH_PREFIXES = {
    "rotation": "rotation_aresunet.",
    "translation": "translation_aresunet.",
}
STATE_DICT_KEYS = ("model_state_dict", "state_dict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and optionally evaluate reciprocal BatchNorm-state checkpoint hybrids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-120", type=Path, default=None)
    parser.add_argument("--checkpoint-180", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/track_a/paper_reproduction/a6_paper_schedule_120x120/"
            "checkpoint_state_attribution"
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluate_deepdct_vo.py after creating the hybrid checkpoints.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Rebuild the metrics CSV/JSON from existing evaluation summary.json "
            "files. No checkpoint loading, creation, or inference is performed."
        ),
    )
    parser.add_argument(
        "--evaluator", type=Path, default=Path("scripts/evaluate_deepdct_vo.py")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sequences", nargs="+", default=["09", "10"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable used for evaluation."
    )
    parser.add_argument(
        "--trajectory-gt-option",
        default="auto",
        help=(
            "GT-trajectory evaluator option. Use 'auto', a flag such as "
            "'--use-ground-truth-trajectory-rotation', or an option/value pair "
            "such as '--trajectory-rotation-source ground_truth'."
        ),
    )
    parser.add_argument(
        "--extra-evaluator-args",
        default="",
        help="Additional evaluator arguments as one shell-style quoted string.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace previously generated hybrid/evaluation outputs."
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.collect_only and args.evaluate:
        raise ValueError("--collect-only and --evaluate are mutually exclusive")
    if args.collect_only:
        return
    if args.checkpoint_120 is None or args.checkpoint_180 is None:
        raise ValueError(
            "--checkpoint-120 and --checkpoint-180 are required unless "
            "--collect-only is used"
        )
    for path in (args.checkpoint_120, args.checkpoint_180):
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError("Missing checkpoint: {}".format(path))
    if args.evaluate and not args.evaluator.is_file():
        raise FileNotFoundError("Missing evaluator: {}".format(args.evaluator))
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if len(set(args.sequences)) != len(args.sequences):
        raise ValueError("Duplicate sequence in --sequences")


def load_checkpoint(path: Path) -> MutableMapping[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, MutableMapping):
        raise TypeError("Checkpoint is not a mapping: {}".format(path))
    return obj


def state_dict_location(checkpoint: Mapping[str, Any]) -> str:
    for key in STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return key
    raise KeyError(
        "Checkpoint has neither {}. Available keys: {}".format(
            STATE_DICT_KEYS, sorted(str(k) for k in checkpoint.keys())
        )
    )


def normalized_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def is_bn_buffer(key: str) -> bool:
    return normalized_key(key).endswith(tuple("." + suffix for suffix in BN_SUFFIXES))


def buffer_branch(key: str) -> Optional[str]:
    name = normalized_key(key)
    for branch, prefix in BRANCH_PREFIXES.items():
        if name.startswith(prefix):
            return branch
    return None


def validate_state_dicts(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> None:
    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        missing_left = sorted(right_keys - left_keys)[:20]
        missing_right = sorted(left_keys - right_keys)[:20]
        raise RuntimeError(
            "Checkpoint architectures differ. Missing from first: {}; missing from second: {}"
            .format(missing_left, missing_right)
        )
    mismatches = []
    for key in sorted(left_keys):
        if tuple(left[key].shape) != tuple(right[key].shape):
            mismatches.append((key, tuple(left[key].shape), tuple(right[key].shape)))
    if mismatches:
        raise RuntimeError("State tensor shape mismatches: {}".format(mismatches[:20]))


def checkpoint_epoch(checkpoint: Mapping[str, Any], fallback: int) -> int:
    try:
        return int(checkpoint.get("epoch", fallback))
    except (TypeError, ValueError):
        return fallback


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def selected_buffers(state: Mapping[str, torch.Tensor], scope: str) -> List[str]:
    keys = []
    for key in state:
        branch = buffer_branch(key)
        if not is_bn_buffer(key) or branch is None:
            continue
        if scope == "all" or branch == scope:
            keys.append(key)
    return sorted(keys)


def create_hybrid(
    target_checkpoint: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    target_state_key: str,
    source_state_key: str,
    scope: str,
    target_path: Path,
    source_path: Path,
) -> Tuple[MutableMapping[str, Any], Dict[str, Any]]:
    hybrid = copy.deepcopy(target_checkpoint)
    target_state = hybrid[target_state_key]
    source_state = source_checkpoint[source_state_key]
    keys = selected_buffers(target_state, scope)
    if not keys:
        raise RuntimeError("No BatchNorm buffers selected for scope {!r}".format(scope))

    counts = {"rotation": 0, "translation": 0}
    changed = 0
    digest = hashlib.sha256()
    for key in keys:
        if key not in source_state:
            raise KeyError("Source checkpoint lacks selected buffer: {}".format(key))
        old = target_state[key]
        new = source_state[key]
        if tuple(old.shape) != tuple(new.shape) or old.dtype != new.dtype:
            raise RuntimeError(
                "Incompatible buffer {}: target {} {}, source {} {}".format(
                    key, tuple(old.shape), old.dtype, tuple(new.shape), new.dtype
                )
            )
        if not torch.equal(old, new):
            changed += 1
        target_state[key] = new.detach().clone()
        counts[buffer_branch(key)] += 1
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(new).encode("ascii"))

    target_epoch = checkpoint_epoch(target_checkpoint, -1)
    source_epoch = checkpoint_epoch(source_checkpoint, -1)
    record = {
        "evaluation_only": True,
        "target_checkpoint": str(target_path.resolve()),
        "source_checkpoint": str(source_path.resolve()),
        "weights_epoch": target_epoch,
        "batchnorm_source_epoch": source_epoch,
        "batchnorm_scope": scope,
        "swapped_buffers": len(keys),
        "changed_buffers": changed,
        "rotation_buffers": counts["rotation"],
        "translation_buffers": counts["translation"],
        "swapped_buffer_digest_sha256": digest.hexdigest(),
        "warning": "Do not resume training: optimizer state belongs to the weights checkpoint.",
    }
    hybrid["checkpoint_state_attribution"] = record
    return hybrid, record


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite {}; pass --overwrite".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, str(temporary))
    temporary.replace(path)


def evaluator_help(args: argparse.Namespace) -> str:
    result = subprocess.run(
        [args.python, str(args.evaluator), "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return result.stdout


def discover_gt_trajectory_args(args: argparse.Namespace) -> List[str]:
    if args.trajectory_gt_option != "auto":
        parsed = shlex.split(args.trajectory_gt_option)
        if not parsed or not parsed[0].startswith("--"):
            raise ValueError("--trajectory-gt-option must begin with '--'")
        return parsed

    help_text = evaluator_help(args)
    flag_candidates = (
        "--use-ground-truth-trajectory-rotation",
        "--trajectory-use-ground-truth-rotation",
        "--use-gt-trajectory-rotation",
    )
    for flag in flag_candidates:
        if flag in help_text:
            return [flag]
    valued_candidates = (
        "--trajectory-rotation-source",
        "--trajectory-rotation",
    )
    for option in valued_candidates:
        if option in help_text:
            return [option, "ground_truth"]
    raise RuntimeError(
        "Could not discover the evaluator's GT-trajectory option. Pass it explicitly, e.g. "
        "--trajectory-gt-option='--trajectory-rotation-source ground_truth'."
    )


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
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


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
        "variant": variant,
        "sequence": sequence,
        "trajectory_rotation": trajectory_mode,
        "output_dir": str(output_dir.resolve()),
        "summary_found": path.is_file(),
    }
    if not path.is_file():
        return row
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    trajectory = summary.get("trajectory_metrics_unscaled") or summary.get("trajectory_metrics") or {}
    row.update({
        "checkpoint_epoch": summary.get("checkpoint_epoch", summary.get("epoch")),
        # Current evaluator schema stores AggregateMetrics under frame_metrics.
        # Older fallbacks are retained so existing historical outputs remain
        # collectable by the same script.
        "translation_mae": first_value(
            summary,
            (
                ("frame_metrics", "translation_mae"),
                ("aggregate_metrics", "translation_mae"),
                ("metrics", "translation_mae"),
                ("translation_mae",),
            ),
        ),
        "rotation_mae": first_value(
            summary,
            (
                ("frame_metrics", "rotation_mae"),
                ("aggregate_metrics", "rotation_mae"),
                ("metrics", "rotation_mae"),
                ("rotation_mae",),
            ),
        ),
        "ate_rmse": trajectory.get("ate_rmse"),
        "rpe_translation_rmse": trajectory.get("rpe_translation_rmse"),
        "rpe_rotation_rmse_degrees": trajectory.get("rpe_rotation_rmse_degrees"),
        "approx_translation_drift_percent": trajectory.get("translational_drift_percent"),
        "approx_rotation_drift_degrees_per_100m": trajectory.get("rotational_drift_degrees_per_100m"),
    })
    return row


def discover_existing_evaluations(output_dir: Path) -> List[Tuple[str, str, str, Path]]:
    """Return existing evaluations in deterministic experiment-matrix order."""
    evaluations_root = output_dir / "evaluations"
    if not evaluations_root.is_dir():
        raise FileNotFoundError(
            "Missing evaluations directory: {}".format(evaluations_root)
        )

    discovered: List[Tuple[str, str, str, Path]] = []
    pattern = re.compile(r"^sequence_(.+)_(gt_gt|gt_pred)$")
    for summary_path in sorted(evaluations_root.glob("*/sequence_*/summary.json")):
        evaluation_dir = summary_path.parent
        match = pattern.match(evaluation_dir.name)
        if match is None:
            continue
        sequence, mode_token = match.groups()
        trajectory_mode = (
            "ground_truth" if mode_token == "gt_gt" else "predicted"
        )
        variant = evaluation_dir.parent.name
        discovered.append((variant, sequence, trajectory_mode, evaluation_dir))

    if not discovered:
        raise RuntimeError(
            "No summary.json files found under {}".format(evaluations_root)
        )
    return discovered


def write_collected_outputs(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    csv_path = output_dir / "checkpoint_state_attribution_metrics.csv"
    json_path = output_dir / "checkpoint_state_attribution_summary.json"
    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"evaluations": list(rows)}, handle, indent=2, sort_keys=True)


def collect_existing_outputs(output_dir: Path) -> List[Dict[str, Any]]:
    rows = [
        collect_summary(variant, sequence, trajectory_mode, evaluation_dir)
        for variant, sequence, trajectory_mode, evaluation_dir
        in discover_existing_evaluations(output_dir)
    ]
    write_collected_outputs(output_dir, rows)
    missing_frame_metrics = [
        row for row in rows
        if row.get("translation_mae") is None or row.get("rotation_mae") is None
    ]
    if missing_frame_metrics:
        raise RuntimeError(
            "Collected {} evaluations, but {} still lack frame-level MAE. "
            "Inspect one affected summary.json schema before trusting the CSV."
            .format(len(rows), len(missing_frame_metrics))
        )
    return rows


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


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        rows = collect_existing_outputs(args.output_dir)
        print(
            "Recollected {} existing evaluations with frame-level MAE."
            .format(len(rows))
        )
        print("Saved corrected outputs to: {}".format(args.output_dir.resolve()))
        return

    checkpoint_dir = args.output_dir / "checkpoints"

    checkpoint_120 = load_checkpoint(args.checkpoint_120)
    checkpoint_180 = load_checkpoint(args.checkpoint_180)
    state_key_120 = state_dict_location(checkpoint_120)
    state_key_180 = state_dict_location(checkpoint_180)
    validate_state_dicts(checkpoint_120[state_key_120], checkpoint_180[state_key_180])

    epoch_120 = checkpoint_epoch(checkpoint_120, 120)
    epoch_180 = checkpoint_epoch(checkpoint_180, 180)
    variants: Dict[str, Path] = {
        "native_epoch_{:03d}".format(epoch_120): args.checkpoint_120,
        "native_epoch_{:03d}".format(epoch_180): args.checkpoint_180,
    }
    manifest_rows: List[Dict[str, Any]] = []

    directions = (
        (checkpoint_180, checkpoint_120, state_key_180, state_key_120, args.checkpoint_180, args.checkpoint_120),
        (checkpoint_120, checkpoint_180, state_key_120, state_key_180, args.checkpoint_120, args.checkpoint_180),
    )
    for target, source, target_key, source_key, target_path, source_path in directions:
        target_epoch = checkpoint_epoch(target, -1)
        source_epoch = checkpoint_epoch(source, -1)
        for scope in ("all", "rotation", "translation"):
            variant = "weights_{:03d}_bn_{:03d}_{}".format(target_epoch, source_epoch, scope)
            path = checkpoint_dir / (variant + ".pt")
            hybrid, record = create_hybrid(
                target, source, target_key, source_key, scope, target_path, source_path
            )
            save_checkpoint(hybrid, path, args.overwrite)
            variants[variant] = path
            manifest_rows.append(dict({"variant": variant, "checkpoint": str(path.resolve())}, **record))
            print("Created {} ({} buffers; {} changed)".format(path, record["swapped_buffers"], record["changed_buffers"]))

    write_csv(args.output_dir / "hybrid_checkpoint_manifest.csv", manifest_rows)
    with (args.output_dir / "hybrid_checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"variants": manifest_rows}, handle, indent=2, sort_keys=True)

    if not args.evaluate:
        print("\nHybrid checkpoints created. Re-run with --evaluate to execute the attribution matrix.")
        return

    gt_trajectory_args = discover_gt_trajectory_args(args)
    extra_args = shlex.split(args.extra_evaluator_args)
    summary_rows: List[Dict[str, Any]] = []
    for variant, checkpoint_path in variants.items():
        for sequence in args.sequences:
            for trajectory_mode, trajectory_args in (
                ("ground_truth", gt_trajectory_args),
                ("predicted", []),
            ):
                output_dir = args.output_dir / "evaluations" / variant / (
                    "sequence_{}_{}".format(sequence, "gt_gt" if trajectory_mode == "ground_truth" else "gt_pred")
                )
                if output_dir.exists() and not args.overwrite:
                    raise FileExistsError(
                        "Evaluation directory exists: {}. Pass --overwrite or choose a new output directory."
                        .format(output_dir)
                    )
                command = [
                    args.python,
                    str(args.evaluator),
                    "--checkpoint", str(checkpoint_path),
                    "--data-root", str(args.data_root),
                    "--sequence", sequence,
                    "--output-dir", str(output_dir),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--device", args.device,
                    "--use-ground-truth-rotation",
                ] + list(trajectory_args) + extra_args
                run_command(command, output_dir / "evaluation.log")
                summary_rows.append(collect_summary(variant, sequence, trajectory_mode, output_dir))
                write_csv(args.output_dir / "checkpoint_state_attribution_metrics.csv", summary_rows)

    write_collected_outputs(args.output_dir, summary_rows)
    print("\nSaved attribution outputs to: {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
