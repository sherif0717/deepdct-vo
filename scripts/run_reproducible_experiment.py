"""Run and record a reproducible DeepDCT-VO experiment.

The launcher:

1. Reads an experiment JSON configuration.
2. Requires a clean Git working tree by default.
3. Records the Git commit and diff status.
4. Records Python, PyTorch, CUDA, package, and GPU information.
5. Creates a lightweight dataset fingerprint.
6. Runs scripts/train_deepdct_vo.py.
7. Inspects best_validation.pt.
8. Evaluates the best checkpoint on each configured test sequence.
9. Appends one row per test sequence to experiments/results.csv.

Python compatibility: Python 3.8+
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results.csv"

RESULT_FIELDS = [
    "run_id",
    "experiment_name",
    "timestamp",
    "git_commit",
    "git_branch",
    "git_dirty",
    "train_sequences",
    "validation_sequences",
    "test_sequence",
    "selected_epoch",
    "epochs_requested",
    "batch_size",
    "num_workers",
    "device",
    "sampler",
    "target_normalization",
    "depth_source",
    "semantic_source",
    "best_validation_total_loss",
    "best_validation_rotation_loss",
    "best_validation_translation_loss",
    "test_total_mse",
    "test_rotation_mse",
    "test_translation_mse",
    "test_rotation_rmse",
    "test_translation_rmse",
    "test_rotation_mae",
    "test_translation_mae",
    "checkpoint_path",
    "evaluation_summary_path",
    "config_path",
    "run_directory",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible DeepDCT-VO experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment JSON configuration.",
    )

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow execution with uncommitted Git changes. The diff and "
            "status will still be saved in the run directory."
        ),
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip training and evaluate an existing best checkpoint in the "
            "run checkpoint directory."
        ),
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Run training but skip held-out evaluation.",
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Central experiment-results CSV.",
    )

    return parser.parse_args()


def run_capture(
    command: Sequence[str],
    cwd: Path = PROJECT_ROOT,
    check: bool = True,
) -> str:
    """Run a command and return stdout plus stderr."""

    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if check and completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}:\n"
            f"{format_command(command)}\n\n"
            f"{completed.stdout}"
        )

    return completed.stdout


def run_streaming(
    command: Sequence[str],
    log_path: Path,
    cwd: Path = PROJECT_ROOT,
) -> None:
    """Run a command while streaming and saving its output."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 88)
    print(format_command(command))
    print("=" * 88)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: "
            f"{format_command(command)}"
        )


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def load_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {
        "experiment_name",
        "train_sequences",
        "validation_sequences",
        "test_sequences",
        "epochs",
        "batch_size",
        "num_workers",
    }

    missing = required.difference(config)

    if missing:
        raise KeyError(
            f"Configuration is missing required keys: {sorted(missing)}"
        )

    for key in (
        "train_sequences",
        "validation_sequences",
        "test_sequences",
    ):
        value = config[key]

        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} must be a non-empty list.")

        config[key] = [
            normalize_sequence(sequence)
            for sequence in value
        ]

    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive.")

    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive.")

    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative.")

    return config


def normalize_sequence(value: object) -> str:
    text = str(value).strip()

    if not text:
        raise ValueError("Sequence identifiers cannot be empty.")

    return f"{int(text):02d}" if text.isdigit() else text


def git_information() -> Dict[str, object]:
    commit = run_capture(
        ["git", "rev-parse", "HEAD"]
    ).strip()

    branch = run_capture(
        ["git", "branch", "--show-current"]
    ).strip()

    status = run_capture(
        ["git", "status", "--porcelain"]
    )

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status": status,
    }


def save_git_snapshot(
    run_directory: Path,
    information: Mapping[str, object],
) -> None:
    (run_directory / "git_status.txt").write_text(
        str(information["status"]),
        encoding="utf-8",
    )

    diff = run_capture(
        ["git", "diff", "--binary"],
        check=False,
    )

    (run_directory / "git_diff.patch").write_text(
        diff,
        encoding="utf-8",
    )

    staged_diff = run_capture(
        ["git", "diff", "--cached", "--binary"],
        check=False,
    )

    (run_directory / "git_staged_diff.patch").write_text(
        staged_diff,
        encoding="utf-8",
    )


def save_environment_snapshot(run_directory: Path) -> None:
    environment: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "working_directory": str(PROJECT_ROOT),
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }

    try:
        import torch

        environment["pytorch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }

        if torch.cuda.is_available():
            environment["pytorch"]["gpu_name"] = (
                torch.cuda.get_device_name(0)
            )
            environment["pytorch"]["gpu_capability"] = list(
                torch.cuda.get_device_capability(0)
            )

    except Exception as exc:
        environment["pytorch_error"] = str(exc)

    with (
        run_directory / "environment.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=2, sort_keys=True)

    pip_freeze = run_capture(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
    )

    (run_directory / "requirements_frozen.txt").write_text(
        pip_freeze,
        encoding="utf-8",
    )

    nvidia_smi = run_capture(
        ["nvidia-smi", "-q"],
        check=False,
    )

    (run_directory / "nvidia_smi.txt").write_text(
        nvidia_smi,
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def directory_manifest_hash(
    directory: Path,
    suffixes: Sequence[str],
) -> Tuple[str, int]:
    """Hash relative paths and file sizes without reading all image bytes."""

    digest = hashlib.sha256()
    count = 0

    normalized_suffixes = {
        suffix.lower()
        for suffix in suffixes
    }

    if not directory.is_dir():
        return "", 0

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in normalized_suffixes:
            continue

        relative = path.relative_to(directory)
        entry = f"{relative.as_posix()}\t{path.stat().st_size}\n"

        digest.update(entry.encode("utf-8"))
        count += 1

    return digest.hexdigest(), count


def build_dataset_manifest(
    config: Mapping[str, object],
) -> Dict[str, object]:
    data_root = PROJECT_ROOT / str(
        config.get("data_root", "data")
    )

    label_root = data_root / "out_csv"
    pose_root = data_root / "poses"
    sequence_root = data_root / "sequences"

    camera_directory = str(
        config.get("camera_directory", "image_2")
    )

    sequences = sorted(
        set(
            list(config["train_sequences"])
            + list(config["validation_sequences"])
            + list(config["test_sequences"])
        )
    )

    manifest: Dict[str, object] = {
        "data_root": str(data_root.resolve()),
        "camera_directory": camera_directory,
        "sequences": {},
    }

    sequence_entries: Dict[str, object] = {}

    for sequence in sequences:
        label_path = label_root / f"{sequence}_dct.txt"
        pose_path = pose_root / f"{sequence}.txt"
        image_directory = (
            sequence_root
            / sequence
            / camera_directory
        )

        image_hash, image_count = directory_manifest_hash(
            image_directory,
            suffixes=(".png", ".jpg", ".jpeg"),
        )

        sequence_entries[sequence] = {
            "label_path": str(label_path),
            "label_sha256": (
                sha256_file(label_path)
                if label_path.is_file()
                else None
            ),
            "pose_path": str(pose_path),
            "pose_sha256": (
                sha256_file(pose_path)
                if pose_path.is_file()
                else None
            ),
            "image_directory": str(image_directory),
            "image_manifest_sha256": image_hash or None,
            "image_count": image_count,
        }

    manifest["sequences"] = sequence_entries
    return manifest


def build_train_command(
    config: Mapping[str, object],
    checkpoint_directory: Path,
) -> List[str]:
    command = [
        sys.executable,
        "scripts/train_deepdct_vo.py",
        "--train-sequences",
        *[str(value) for value in config["train_sequences"]],
        "--validation-sequences",
        *[
            str(value)
            for value in config["validation_sequences"]
        ],
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--num-workers",
        str(config["num_workers"]),
        "--checkpoint-dir",
        str(checkpoint_directory),
    ]

    command.extend(
        str(value)
        for value in config.get("train_extra_args", [])
    )

    return command


def build_evaluation_command(
    config: Mapping[str, object],
    checkpoint_path: Path,
    sequence: str,
    output_directory: Path,
) -> List[str]:
    command = [
        sys.executable,
        "scripts/evaluate_deepdct_vo.py",
        "--checkpoint",
        str(checkpoint_path),
        "--sequence",
        sequence,
        "--device",
        str(config.get("evaluation_device", "cuda")),
        "--output-dir",
        str(output_directory),
    ]

    command.extend(
        str(value)
        for value in config.get(
            "evaluation_extra_args",
            ["--skip-trajectory"],
        )
    )

    return command


def load_torch_checkpoint(path: Path) -> Dict[str, object]:
    import torch

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def inspect_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> Dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best-validation checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_torch_checkpoint(checkpoint_path)

    summary = {
        "epoch": checkpoint.get("epoch"),
        "best_validation_loss": checkpoint.get(
            "best_validation_loss"
        ),
        "training_metrics": checkpoint.get(
            "training_metrics"
        ),
        "validation_metrics": checkpoint.get(
            "validation_metrics"
        ),
        "configuration": checkpoint.get("configuration"),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return summary


def append_results_row(
    results_path: Path,
    row: Mapping[str, object],
) -> None:
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = results_path.is_file()

    with results_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                field: row.get(field, "")
                for field in RESULT_FIELDS
            }
        )


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    git = git_information()

    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "The Git working tree contains uncommitted changes.\n\n"
            "Commit, stash, or discard them before running the "
            "experiment, or explicitly pass --allow-dirty.\n\n"
            f"{git['status']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = str(config["experiment_name"])
    run_id = f"{timestamp}_{experiment_name}"

    experiment_root = PROJECT_ROOT / str(
        config.get("experiment_root", "experiments/runs")
    )

    run_directory = experiment_root / run_id
    checkpoint_directory = run_directory / "checkpoints"
    evaluation_root = run_directory / "evaluation"
    log_directory = run_directory / "logs"

    run_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)

    resolved_config = dict(config)
    resolved_config.update(
        {
            "run_id": run_id,
            "run_directory": str(run_directory.resolve()),
            "checkpoint_directory": str(
                checkpoint_directory.resolve()
            ),
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "git_dirty": git["dirty"],
            "launcher_command": format_command(sys.argv),
        }
    )

    with (
        run_directory / "config_resolved.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            resolved_config,
            file,
            indent=2,
            sort_keys=True,
        )

    save_git_snapshot(run_directory, git)
    save_environment_snapshot(run_directory)

    dataset_manifest = build_dataset_manifest(config)

    with (
        run_directory / "dataset_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            dataset_manifest,
            file,
            indent=2,
            sort_keys=True,
        )

    train_command = build_train_command(
        config=config,
        checkpoint_directory=checkpoint_directory,
    )

    (run_directory / "train_command.txt").write_text(
        format_command(train_command) + "\n",
        encoding="utf-8",
    )

    if not args.skip_training:
        run_streaming(
            command=train_command,
            log_path=log_directory / "training.log",
        )

    best_checkpoint = (
        checkpoint_directory
        / "best_validation.pt"
    )

    checkpoint_summary = inspect_checkpoint(
        checkpoint_path=best_checkpoint,
        output_path=run_directory / "best_checkpoint.json",
    )

    if args.skip_evaluation:
        print(
            f"Training complete. Run directory: "
            f"{run_directory.resolve()}"
        )
        return

    validation_metrics = (
        checkpoint_summary.get("validation_metrics") or {}
    )

    if not isinstance(validation_metrics, Mapping):
        validation_metrics = {}

    for test_sequence in config["test_sequences"]:
        evaluation_directory = (
            evaluation_root
            / f"sequence_{test_sequence}"
        )

        evaluation_command = build_evaluation_command(
            config=config,
            checkpoint_path=best_checkpoint,
            sequence=str(test_sequence),
            output_directory=evaluation_directory,
        )

        (
            run_directory
            / f"evaluation_command_sequence_{test_sequence}.txt"
        ).write_text(
            format_command(evaluation_command) + "\n",
            encoding="utf-8",
        )

        run_streaming(
            command=evaluation_command,
            log_path=(
                log_directory
                / f"evaluation_sequence_{test_sequence}.log"
            ),
        )

        summary_path = (
            evaluation_directory
            / "summary.json"
        )

        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Evaluation summary was not generated: {summary_path}"
            )

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            evaluation_summary = json.load(file)

        frame_metrics = evaluation_summary["frame_metrics"]

        result_row = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "git_commit": git["commit"],
            "git_branch": git["branch"],
            "git_dirty": git["dirty"],
            "train_sequences": " ".join(
                config["train_sequences"]
            ),
            "validation_sequences": " ".join(
                config["validation_sequences"]
            ),
            "test_sequence": test_sequence,
            "selected_epoch": checkpoint_summary.get("epoch"),
            "epochs_requested": config["epochs"],
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "device": config.get(
                "evaluation_device",
                "cuda",
            ),
            "sampler": config.get(
                "sampler",
                "transition_uniform",
            ),
            "target_normalization": config.get(
                "target_normalization",
                "none",
            ),
            "depth_source": config.get(
                "depth_source",
                "zero_placeholder",
            ),
            "semantic_source": config.get(
                "semantic_source",
                "pretrained_frozen_lraspp",
            ),
            "best_validation_total_loss": (
                validation_metrics.get("total_loss")
            ),
            "best_validation_rotation_loss": (
                validation_metrics.get("rotation_loss")
            ),
            "best_validation_translation_loss": (
                validation_metrics.get("translation_loss")
            ),
            "test_total_mse": frame_metrics.get("total_mse"),
            "test_rotation_mse": frame_metrics.get(
                "rotation_mse"
            ),
            "test_translation_mse": frame_metrics.get(
                "translation_mse"
            ),
            "test_rotation_rmse": frame_metrics.get(
                "rotation_rmse"
            ),
            "test_translation_rmse": frame_metrics.get(
                "translation_rmse"
            ),
            "test_rotation_mae": frame_metrics.get(
                "rotation_mae"
            ),
            "test_translation_mae": frame_metrics.get(
                "translation_mae"
            ),
            "checkpoint_path": str(
                best_checkpoint.resolve()
            ),
            "evaluation_summary_path": str(
                summary_path.resolve()
            ),
            "config_path": str(config_path),
            "run_directory": str(run_directory.resolve()),
            "notes": config.get("notes", ""),
        }

        append_results_row(
            results_path=args.results_path,
            row=result_row,
        )

    print()
    print("=" * 88)
    print("Reproducible experiment complete")
    print("=" * 88)
    print(f"Run ID:       {run_id}")
    print(f"Run directory:{run_directory.resolve()}")
    print(f"Results table:{args.results_path.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()