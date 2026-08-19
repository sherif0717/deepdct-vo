#!/usr/bin/env python3
"""
Orchestrate a paired DeepDCT-VO baseline-versus-semantic+depth comparison.

This coordinator deliberately reuses the repository's existing scripts:

  scripts/evaluate_deepdct_vo.py
  scripts/interpretability/analyze_attention_statistics.py
  scripts/interpretability/inspect_deepdct_vo.py

It creates:

integrated_comparison_inputs/
├── baseline/
│   ├── evaluation_summary.json
│   ├── frame_predictions.csv
│   ├── attention_statistics.csv
│   ├── attention_maps/
│   └── gradcam_maps/
├── semantic_depth/
│   ├── evaluation_summary.json
│   ├── frame_predictions.csv
│   ├── attention_statistics.csv
│   ├── attention_maps/
│   ├── gradcam_maps/
│   ├── semantic_maps/
│   └── depth_maps/
├── matched_frames/
│   └── selected_frames.json
└── experiment_manifest.json

The commands are declared in JSON so small CLI differences can be corrected
without editing this coordinator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


MODEL_NAMES = ("baseline", "semantic_depth")
PAIR_KEYS = ("sequence", "frame_prev", "frame_curr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate integrated baseline versus semantic+depth comparison inputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/integrated_comparison.json"),
    )
    parser.add_argument(
        "--layers",
        type=Path,
        default=Path("configs/integrated_comparison_layers.json"),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "evaluate", "attention", "select", "visualize", "manifest"),
        default="all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow regeneration of existing stage outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print external commands without executing them.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return data


def normalize_sequence(value: Any) -> str:
    text = str(value).strip()
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def format_command(
    template: Sequence[str],
    variables: Mapping[str, Any],
) -> List[str]:
    command: List[str] = []
    for token in template:
        try:
            rendered = str(token).format_map(variables)
        except KeyError as exc:
            raise KeyError(
                f"Command template references unknown placeholder {exc!s}: {token!r}"
            ) from exc
        if rendered != "":
            command.append(rendered)
    return command


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(subprocess.list2cmdline([part]) for part in command)
    print(f"\n$ {printable}")

    if dry_run:
        return

    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(completed.stdout or "")

    if completed.returncode != 0:
        tail = (completed.stdout or "")[-5000:]
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}.\n"
            f"Log: {log_path}\n"
            f"Last output:\n{tail}"
        )


def ensure_empty_or_allowed(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        return
    if any(path.iterdir()) if path.is_dir() else True:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite or choose another output root."
        )


def newest_matching(root: Path, names: Sequence[str]) -> Path:
    candidates: List[Path] = []
    for name in names:
        candidates.extend(root.rglob(name))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any of {list(names)} below {root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: Mapping[str, str], candidates: Sequence[str]) -> Optional[float]:
    for key in candidates:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            return number
    return None


def int_value(row: Mapping[str, str], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise KeyError(f"CSV row is missing {key!r}.")
    return int(float(value))


def pair_id(row: Mapping[str, str]) -> Tuple[str, int, int]:
    return (
        normalize_sequence(row.get("sequence", "")),
        int_value(row, "frame_prev"),
        int_value(row, "frame_curr"),
    )


ROTATION_ERROR_COLUMNS = (
    "rotation_error",
    "rotation_l2_error",
    "rotation_error_l2",
    "rotation_euclidean_error",
)
TRANSLATION_ERROR_COLUMNS = (
    "translation_error",
    "translation_l2_error",
    "translation_error_l2",
    "translation_euclidean_error",
)
TOTAL_ERROR_COLUMNS = (
    "total_error",
    "weighted_total_error",
    "pose_error",
)


def error_value(row: Mapping[str, str], metric: str) -> float:
    if metric == "translation":
        value = numeric(row, TRANSLATION_ERROR_COLUMNS)
    elif metric == "rotation":
        value = numeric(row, ROTATION_ERROR_COLUMNS)
    elif metric == "total":
        value = numeric(row, TOTAL_ERROR_COLUMNS)
        if value is None:
            rotation = numeric(row, ROTATION_ERROR_COLUMNS)
            translation = numeric(row, TRANSLATION_ERROR_COLUMNS)
            if rotation is not None and translation is not None:
                value = rotation + translation
    else:
        raise ValueError(f"Unsupported selection metric: {metric!r}")

    if value is None:
        raise KeyError(
            f"Could not resolve {metric} error. Available CSV columns: {sorted(row.keys())}"
        )
    return value


def rotation_motion(row: Mapping[str, str]) -> float:
    direct = numeric(row, ("rotation_gt_norm", "rotation_target_norm"))
    if direct is not None:
        return abs(direct)

    components: List[float] = []
    for axis in ("x", "y", "z"):
        value = numeric(
            row,
            (
                f"rotation_gt_{axis}",
                f"rotation_target_{axis}",
                f"gt_rotation_{axis}",
            ),
        )
        if value is None:
            raise KeyError(
                "Could not resolve ground-truth rotation components for "
                "the high-rotation frame selection."
            )
        components.append(value)
    return math.sqrt(sum(value * value for value in components))


def nearest_median(rows: Sequence[Mapping[str, str]], metric: str) -> Mapping[str, str]:
    ordered = sorted(rows, key=lambda row: error_value(row, metric))
    return ordered[len(ordered) // 2]


def select_frames(
    baseline_rows: Sequence[Dict[str, str]],
    cue_rows: Sequence[Dict[str, str]],
    *,
    metric: str,
) -> Dict[str, Any]:
    baseline_by_id = {pair_id(row): row for row in baseline_rows}
    cue_by_id = {pair_id(row): row for row in cue_rows}
    common_ids = sorted(set(baseline_by_id).intersection(cue_by_id))

    if not common_ids:
        raise RuntimeError("Baseline and semantic+depth CSV files have no matched frame pairs.")

    pairs: List[Dict[str, Any]] = []
    for key in common_ids:
        baseline = baseline_by_id[key]
        cue = cue_by_id[key]
        baseline_error = error_value(baseline, metric)
        cue_error = error_value(cue, metric)
        pairs.append(
            {
                "key": key,
                "baseline": baseline,
                "semantic_depth": cue,
                "baseline_error": baseline_error,
                "semantic_depth_error": cue_error,
                "improvement": baseline_error - cue_error,
                "rotation_motion": rotation_motion(baseline),
            }
        )

    baseline_sorted = sorted(pairs, key=lambda item: item["baseline_error"])
    categories = {
        "lowest_baseline_error": baseline_sorted[0],
        "median_baseline_error": baseline_sorted[len(baseline_sorted) // 2],
        "highest_baseline_error": baseline_sorted[-1],
        "largest_semantic_depth_improvement": max(
            pairs, key=lambda item: item["improvement"]
        ),
        "largest_semantic_depth_degradation": min(
            pairs, key=lambda item: item["improvement"]
        ),
        "highest_ground_truth_rotation": max(
            pairs, key=lambda item: item["rotation_motion"]
        ),
    }

    selected: List[Dict[str, Any]] = []
    seen: set = set()
    for category, item in categories.items():
        sequence, frame_prev, frame_curr = item["key"]
        identity = (sequence, frame_prev, frame_curr)
        record = {
            "category": category,
            "sequence": sequence,
            "frame_prev": frame_prev,
            "frame_curr": frame_curr,
            "sample_index": frame_prev,
            "selection_metric": metric,
            "baseline_error": item["baseline_error"],
            "semantic_depth_error": item["semantic_depth_error"],
            "improvement": item["improvement"],
            "ground_truth_rotation_norm": item["rotation_motion"],
            "duplicate_of_previous_category": identity in seen,
        }
        selected.append(record)
        seen.add(identity)

    return {
        "schema_version": 1,
        "selection_metric": metric,
        "matched_frame_pair_count": len(common_ids),
        "sample_index_assumption": (
            "For consecutive KITTI transitions sorted by frame number, sample_index "
            "equals frame_prev. The visualizer output should be checked once to confirm."
        ),
        "selected_frames": selected,
    }


def stage_evaluate(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    command_template = config["commands"]["evaluate"]
    common = config["common"]

    for model_name in MODEL_NAMES:
        model_config = config["models"][model_name]
        destination = output_root / model_name
        work_dir = output_root / "_work" / model_name / "evaluation"
        work_dir.mkdir(parents=True, exist_ok=True)

        variables = {
            "python": sys.executable,
            "repo_root": str(repo_root),
            "checkpoint": str(resolve_path(repo_root, model_config["checkpoint"])),
            "data_root": str(resolve_path(repo_root, common["data_root"])),
            "sequence": normalize_sequence(common["sequence"]),
            "output_dir": str(work_dir),
            "batch_size": common["batch_size"],
            "num_workers": common["num_workers"],
            "device": common["device"],
            "seed": common["seed"],
        }
        command = format_command(command_template, variables)
        run_command(
            command,
            cwd=repo_root,
            log_path=output_root / "_logs" / f"{model_name}_evaluation.log",
            dry_run=dry_run,
        )

        if dry_run:
            continue

        summary = newest_matching(work_dir, ("summary.json", "evaluation_summary.json"))
        predictions = newest_matching(work_dir, ("frame_predictions.csv",))
        copy_file(summary, destination / "evaluation_summary.json")
        copy_file(predictions, destination / "frame_predictions.csv")


def stage_attention(
    config: Mapping[str, Any],
    layers: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path,
    dry_run: bool,
) -> None:
    command_template = config["commands"]["attention_statistics"]
    common = config["common"]

    for model_name in MODEL_NAMES:
        model_config = config["models"][model_name]
        destination = output_root / model_name
        work_dir = output_root / "_work" / model_name / "attention_statistics"
        work_dir.mkdir(parents=True, exist_ok=True)

        variables = {
            "python": sys.executable,
            "repo_root": str(repo_root),
            "checkpoint": str(resolve_path(repo_root, model_config["checkpoint"])),
            "data_root": str(resolve_path(repo_root, common["data_root"])),
            "sequence": normalize_sequence(common["sequence"]),
            "output_dir": str(work_dir),
            "batch_size": common["batch_size"],
            "num_workers": common["num_workers"],
            "device": common["device"],
            "seed": common["seed"],
            "max_samples": common.get("attention_max_samples", ""),
        }
        command = format_command(command_template, variables)
        run_command(
            command,
            cwd=repo_root,
            log_path=output_root / "_logs" / f"{model_name}_attention.log",
            dry_run=dry_run,
        )

        if dry_run:
            continue

        statistics = newest_matching(
            work_dir,
            ("frame_attention_statistics.csv", "attention_statistics.csv"),
        )
        copy_file(statistics, destination / "attention_statistics.csv")

        for optional_name in (
            "layer_attention_summary.csv",
            "layer_attention_summary.json",
        ):
            matches = list(work_dir.rglob(optional_name))
            if matches:
                copy_file(matches[-1], destination / optional_name)


def stage_select(
    config: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    baseline_csv = output_root / "baseline" / "frame_predictions.csv"
    cue_csv = output_root / "semantic_depth" / "frame_predictions.csv"

    selection = select_frames(
        read_csv(baseline_csv),
        read_csv(cue_csv),
        metric=config["selection"]["metric"],
    )
    path = output_root / "matched_frames" / "selected_frames.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


def copy_visual_artifacts(
    source_root: Path,
    destination_root: Path,
    *,
    model_name: str,
) -> Dict[str, int]:
    destinations = {
        "attention_maps": destination_root / model_name / "attention_maps",
        "gradcam_maps": destination_root / model_name / "gradcam_maps",
        "semantic_maps": destination_root / model_name / "semantic_maps",
        "depth_maps": destination_root / model_name / "depth_maps",
    }
    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)

    counts = {key: 0 for key in destinations}
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue

        lower = str(source.relative_to(source_root)).lower()
        suffix = source.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".npy", ".npz", ".pt"}:
            continue

        category: Optional[str] = None
        if "gradcam" in lower or "grad_cam" in lower:
            category = "gradcam_maps"
        elif "attention" in lower:
            category = "attention_maps"
        elif model_name == "semantic_depth" and "semantic" in lower:
            category = "semantic_maps"
        elif model_name == "semantic_depth" and "depth" in lower:
            category = "depth_maps"

        if category is None:
            continue

        relative_name = "__".join(source.relative_to(source_root).parts)
        destination = destinations[category] / relative_name
        copy_file(source, destination)
        counts[category] += 1

    return counts


def stage_visualize(
    config: Mapping[str, Any],
    layers: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path,
    dry_run: bool,
) -> None:
    command_template = config["commands"]["visualize"]
    common = config["common"]

    selected_path = (
        output_root
        / "matched_frames"
        / "selected_frames.json"
    )

    if dry_run and not selected_path.is_file():
        sequence = normalize_sequence(common["sequence"])
        selected = [
            {
                "category": "dry_run_example",
                "sequence": sequence,
                "frame_prev": 0,
                "frame_curr": 1,
                "sample_index": 0,
            }
        ]
        print(
            "[dry-run] selected_frames.json does not exist yet; "
            "using frame pair 0->1 only to preview visualization commands."
        )
    else:
        selected = load_json(
            selected_path
        )["selected_frames"]

    target_specs = layers["gradcam_targets"]

    for model_name in MODEL_NAMES:
        model_config = config["models"][model_name]

        for frame in selected:
            frame_stem = (
                f"{frame['category']}__"
                f"seq_{frame['sequence']}__"
                f"{int(frame['frame_prev']):06d}_"
                f"{int(frame['frame_curr']):06d}"
            )

            for spec in target_specs:
                evaluation_output_dir = (
                    output_root
                    / "_work"
                    / model_name
                    / "visualization_evaluations"
                    / frame_stem
                    / spec["name"]
                )

                inspection_output_dir = (
                    output_root
                    / "_work"
                    / model_name
                    / "visualizations"
                    / frame_stem
                    / spec["name"]
                )

                evaluation_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                inspection_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                variables = {
                    "python": sys.executable,
                    "repo_root": str(repo_root),
                    "checkpoint": str(
                        resolve_path(
                            repo_root,
                            model_config["checkpoint"],
                        )
                    ),
                    "data_root": str(
                        resolve_path(
                            repo_root,
                            common["data_root"],
                        )
                    ),
                    "sequence": frame["sequence"],
                    "sample_index": frame["sample_index"],
                    "frame_prev": frame["frame_prev"],
                    "frame_curr": frame["frame_curr"],
                    "evaluation_output_dir": str(
                        evaluation_output_dir
                    ),
                    "inspection_output_dir": str(
                        inspection_output_dir
                    ),
                    "device": common["device"],
                    "seed": common["seed"],
                    "batch_size": common["batch_size"],
                    "num_workers": common["num_workers"],
                    "gradcam_layer": spec["layer"],
                    "gradcam_target": spec["target"],
                }

                command = format_command(
                    command_template,
                    variables,
                )

                run_command(
                    command,
                    cwd=repo_root,
                    log_path=(
                        output_root
                        / "_logs"
                        / (
                            f"{model_name}__"
                            f"{frame_stem}__"
                            f"{spec['name']}.log"
                        )
                    ),
                    dry_run=dry_run,
                )

        if not dry_run:
            counts = copy_visual_artifacts(
                (
                    output_root
                    / "_work"
                    / model_name
                    / "visualizations"
                ),
                output_root,
                model_name=model_name,
            )
            print(
                f"{model_name} copied visual artifacts: "
                f"{counts}"
            )


def build_manifest(
    config: Mapping[str, Any],
    layers: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    common = config["common"]
    model_entries: Dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        model_config = config["models"][model_name]
        checkpoint = resolve_path(repo_root, model_config["checkpoint"])
        model_entries[model_name] = {
            **model_config,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "evaluation_summary": str(
                output_root / model_name / "evaluation_summary.json"
            ),
            "frame_predictions": str(
                output_root / model_name / "frame_predictions.csv"
            ),
            "attention_statistics": str(
                output_root / model_name / "attention_statistics.csv"
            ),
        }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "git_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch": git_output(repo_root, "branch", "--show-current"),
        "git_status_porcelain": git_output(repo_root, "status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "comparison": {
            "test_sequence": normalize_sequence(common["sequence"]),
            "data_root": str(resolve_path(repo_root, common["data_root"])),
            "device": common["device"],
            "batch_size": common["batch_size"],
            "num_workers": common["num_workers"],
            "seed": common["seed"],
            "translation_rotation_conditioning": common[
                "translation_rotation_conditioning"
            ],
            "euler_order": common["euler_order"],
            "angles_in_degrees": common["angles_in_degrees"],
            "selection": config["selection"],
        },
        "models": model_entries,
        "attention_layers": layers["attention_layers"],
        "gradcam_targets": layers["gradcam_targets"],
        "artifact_notes": {
            "attention_coefficients": (
                "AttentionDownBlock maps may be channel-spatial [B,C,H,W]; "
                "AttentionUpBlock maps are spatial [B,1,H,W]. Both are retained "
                "before downstream multiplication."
            ),
            "semantic_map_mode": config["models"]["semantic_depth"].get(
                "semantic_map_mode", "foreground_probability"
            ),
            "depth_output_mode": config["models"]["semantic_depth"].get(
                "depth_output_mode", "normalized_depth"
            ),
            "visualizer_cli": (
                "The visualize command is JSON-configured because the exact "
                "sample-selection flag should be confirmed with "
                "`python scripts/interpretability/inspect_deepdct_vo.py --help`."
            ),
        },
    }


def create_directory_structure(output_root: Path) -> None:
    paths = [
        output_root / "baseline" / "attention_maps",
        output_root / "baseline" / "gradcam_maps",
        output_root / "semantic_depth" / "attention_maps",
        output_root / "semantic_depth" / "gradcam_maps",
        output_root / "semantic_depth" / "semantic_maps",
        output_root / "semantic_depth" / "depth_maps",
        output_root / "matched_frames",
        output_root / "_logs",
        output_root / "_work",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    layers = load_json(args.layers)

    repo_root = resolve_path(Path.cwd(), config.get("repository_root", "."))
    output_root = resolve_path(repo_root, config["output_root"])
    create_directory_structure(output_root)

    stages = (
        ("evaluate", lambda: stage_evaluate(
            config,
            repo_root=repo_root,
            output_root=output_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )),
        ("attention", lambda: stage_attention(
            config,
            layers,
            repo_root=repo_root,
            output_root=output_root,
            dry_run=args.dry_run,
        )),
        ("select", lambda: stage_select(config, output_root=output_root)),
        ("visualize", lambda: stage_visualize(
            config,
            layers,
            repo_root=repo_root,
            output_root=output_root,
            dry_run=args.dry_run,
        )),
        ("manifest", lambda: (
            output_root / "experiment_manifest.json"
        ).write_text(
            json.dumps(
                build_manifest(
                    config,
                    layers,
                    repo_root=repo_root,
                    output_root=output_root,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )),
    )

    for name, function in stages:
        if args.stage not in ("all", name):
            continue
        if args.dry_run and name in ("select", "manifest"):
            print(f"[dry-run] skipping local stage: {name}")
            continue
        print(f"\n{'=' * 88}\nStage: {name}\n{'=' * 88}")
        function()

    print(f"\nIntegrated comparison root: {output_root}")


if __name__ == "__main__":
    main()