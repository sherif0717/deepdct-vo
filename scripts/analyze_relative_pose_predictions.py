#!/usr/bin/env python3
"""
Relative Pose Prediction Audit for DeepDCT-VO.

Version 1 (default core workflow)
---------------------------------
- Load frame_predictions.csv
- Compute per-axis statistics
- Generate GT-vs-prediction scatter plots
- Generate GT-vs-prediction time-series plots
- Produce summary.json

Version 2 additions
-------------------
- Correlation and linear-fit analysis
- Automatic failure-mode detection
- Ranking of worst rotation and translation frames
- Error histograms, translation-norm analysis, and correlation matrix
- Optional comparison of multiple experiments

Python compatibility: Python 3.8+
Required packages: numpy, pandas, matplotlib

Expected pose columns
---------------------
The script automatically recognizes common names such as:

    rotation_gt_x, rotation_gt_y, rotation_gt_z
    rotation_pred_x, rotation_pred_y, rotation_pred_z
    translation_gt_x, translation_gt_y, translation_gt_z
    translation_pred_x, translation_pred_y, translation_pred_z

It also accepts common variants such as rot_gt_x, gt_rotation_x,
translation_prediction_x, pred_tx, and similar names.

Examples
--------
Single experiment:

    python scripts/analyze_relative_pose_predictions.py \
        --input experiments/baseline/frame_predictions.csv \
        --output-dir experiments/relative_pose_prediction_audit/baseline

Multiple experiments:

    python scripts/analyze_relative_pose_predictions.py \
        --experiment baseline=experiments/baseline/frame_predictions.csv \
        --experiment semantic_depth=experiments/semantic_depth/frame_predictions.csv \
        --output-dir experiments/relative_pose_prediction_audit/comparison

Run only the Version 1 core outputs:

    python scripts/analyze_relative_pose_predictions.py \
        --input experiments/baseline/frame_predictions.csv \
        --output-dir experiments/relative_pose_prediction_audit/baseline \
        --version 1
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AXES = ("x", "y", "z")
COMPONENTS = ("rotation", "translation")
ROLES = ("gt", "pred")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def sanitize_label(value: str) -> str:
    """Return a filesystem-safe experiment label."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "experiment"


def finite_array(values: Iterable[float]) -> np.ndarray:
    """Convert values to a one-dimensional finite float array."""
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def safe_float(value: object) -> Optional[float]:
    """Convert a scalar to a JSON-safe float, returning None when non-finite."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write indented JSON, allowing NumPy values through prior normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Save and close a Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def series_statistics(values: np.ndarray) -> Dict[str, Optional[float]]:
    """Return descriptive statistics for finite values."""
    values = finite_array(values)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "rmse": None,
            "mae": None,
        }

    return {
        "count": int(values.size),
        "mean": safe_float(np.mean(values)),
        "std": safe_float(np.std(values, ddof=0)),
        "min": safe_float(np.min(values)),
        "q25": safe_float(np.quantile(values, 0.25)),
        "median": safe_float(np.median(values)),
        "q75": safe_float(np.quantile(values, 0.75)),
        "max": safe_float(np.max(values)),
        "rmse": safe_float(np.sqrt(np.mean(np.square(values)))),
        "mae": safe_float(np.mean(np.abs(values))),
    }


def correlation_and_fit(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Compute Pearson correlation and pred ~= slope * gt + intercept.

    The fit is intentionally prediction-on-ground-truth because a pure output
    scale error appears directly as slope != 1.
    """
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(gt) & np.isfinite(pred)
    gt = gt[mask]
    pred = pred[mask]

    result: Dict[str, Optional[float]] = {
        "count": int(gt.size),
        "pearson_r": None,
        "r_squared": None,
        "slope": None,
        "intercept": None,
    }

    if gt.size < 2:
        return result

    gt_std = float(np.std(gt))
    pred_std = float(np.std(pred))

    if gt_std > 0.0 and pred_std > 0.0:
        r = float(np.corrcoef(gt, pred)[0, 1])
        result["pearson_r"] = safe_float(r)
        result["r_squared"] = safe_float(r * r)

    if gt_std > 0.0:
        slope, intercept = np.polyfit(gt, pred, deg=1)
        result["slope"] = safe_float(slope)
        result["intercept"] = safe_float(intercept)

    return result


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def normalize_column_name(name: str) -> str:
    """Normalize a column name for robust matching."""
    value = str(name).strip().lower()
    value = value.replace("ground_truth", "gt")
    value = value.replace("groundtruth", "gt")
    value = value.replace("prediction", "pred")
    value = value.replace("predicted", "pred")
    value = value.replace("estimate", "pred")
    value = value.replace("estimated", "pred")
    value = value.replace("rotation", "rot")
    value = value.replace("translation", "trans")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def explicit_aliases(component: str, role: str, axis: str) -> List[str]:
    """Generate common aliases for a pose component."""
    short_component = "rot" if component == "rotation" else "trans"
    long_component = component

    aliases = [
        f"{long_component}_{role}_{axis}",
        f"{role}_{long_component}_{axis}",
        f"{short_component}_{role}_{axis}",
        f"{role}_{short_component}_{axis}",
        f"{long_component}_{axis}_{role}",
        f"{short_component}_{axis}_{role}",
    ]

    if component == "rotation":
        aliases.extend(
            [
                f"r_{role}_{axis}",
                f"{role}_r_{axis}",
                f"{role}_r{axis}",
                f"r{axis}_{role}",
                f"{role}_rot{axis}",
                f"rot{axis}_{role}",
            ]
        )
    else:
        aliases.extend(
            [
                f"t_{role}_{axis}",
                f"{role}_t_{axis}",
                f"{role}_t{axis}",
                f"t{axis}_{role}",
                f"{role}_trans{axis}",
                f"trans{axis}_{role}",
            ]
        )

    return [normalize_column_name(alias) for alias in aliases]


def token_match_score(
    normalized: str,
    component: str,
    role: str,
    axis: str,
) -> int:
    """Score a normalized column name as a possible pose-column match."""
    compact = normalized.replace("_", "")
    tokens = normalized.split("_")

    component_tokens = (
        ("rot", "rotation", "r") if component == "rotation"
        else ("trans", "translation", "t")
    )
    role_tokens = ("gt", "truth", "target") if role == "gt" else (
        "pred",
        "prediction",
        "output",
        "estimate",
        "est",
    )

    score = 0

    if normalized in explicit_aliases(component, role, axis):
        score += 100

    if any(token in tokens for token in component_tokens):
        score += 25
    elif any(token in compact for token in component_tokens[:2]):
        score += 15

    if any(token in tokens for token in role_tokens):
        score += 25
    elif any(token in compact for token in role_tokens[:2]):
        score += 15

    if axis in tokens:
        score += 25
    elif compact.endswith(axis):
        score += 15

    # Strong compact-name patterns such as pred_tx or gt_ry.
    prefix = "r" if component == "rotation" else "t"
    if f"{role}{prefix}{axis}" in compact or f"{prefix}{axis}{role}" in compact:
        score += 40

    return score


def discover_pose_columns(
    frame: pd.DataFrame,
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Discover all 12 required columns.

    Returned shape:
        mapping[component][role][axis] = original_dataframe_column
    """
    overrides = dict(overrides or {})
    columns = list(frame.columns)
    normalized_to_original: Dict[str, List[str]] = {}

    for column in columns:
        normalized_to_original.setdefault(
            normalize_column_name(column), []
        ).append(column)

    mapping: Dict[str, Dict[str, Dict[str, str]]] = {
        component: {role: {} for role in ROLES}
        for component in COMPONENTS
    }

    used_columns = set()

    for component in COMPONENTS:
        for role in ROLES:
            for axis in AXES:
                key = f"{component}.{role}.{axis}"

                if key in overrides:
                    chosen = overrides[key]
                    if chosen not in frame.columns:
                        raise ValueError(
                            "Column override {!r} for {} does not exist. "
                            "Available columns: {}".format(
                                chosen, key, ", ".join(map(str, frame.columns))
                            )
                        )
                    mapping[component][role][axis] = chosen
                    used_columns.add(chosen)
                    continue

                candidates: List[Tuple[int, str]] = []

                for alias in explicit_aliases(component, role, axis):
                    for original in normalized_to_original.get(alias, []):
                        candidates.append((1000, original))

                for original in columns:
                    if original in used_columns:
                        continue
                    normalized = normalize_column_name(original)
                    score = token_match_score(
                        normalized, component, role, axis
                    )
                    if score >= 60:
                        candidates.append((score, original))

                if not candidates:
                    raise ValueError(
                        "Could not identify the {} {}-axis {} column.\n"
                        "Available columns:\n  {}".format(
                            role,
                            axis,
                            component,
                            "\n  ".join(map(str, frame.columns)),
                        )
                    )

                candidates.sort(key=lambda item: (-item[0], str(item[1])))
                best_score = candidates[0][0]
                best_columns = sorted(
                    {
                        column
                        for score, column in candidates
                        if score == best_score
                    }
                )

                if len(best_columns) > 1:
                    raise ValueError(
                        "Ambiguous columns for {}: {}. "
                        "Use --column {}=<column-name> to disambiguate.".format(
                            key, ", ".join(best_columns), key
                        )
                    )

                chosen = best_columns[0]
                mapping[component][role][axis] = chosen
                used_columns.add(chosen)

    return mapping


def discover_frame_column(frame: pd.DataFrame) -> Optional[str]:
    """Locate a frame/sample identifier when one exists."""
    aliases = [
        "frame",
        "frame_id",
        "frame_index",
        "sample",
        "sample_id",
        "sample_index",
        "index",
        "transition",
        "transition_index",
        "image_index",
    ]
    normalized = {
        normalize_column_name(column): column for column in frame.columns
    }
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def parse_column_overrides(items: Sequence[str]) -> Dict[str, str]:
    """Parse --column semantic.key=csv_column entries."""
    overrides: Dict[str, str] = {}
    valid_keys = {
        f"{component}.{role}.{axis}"
        for component in COMPONENTS
        for role in ROLES
        for axis in AXES
    }

    for item in items:
        if "=" not in item:
            raise ValueError(
                "Invalid --column value {!r}; expected key=column.".format(item)
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key not in valid_keys:
            raise ValueError(
                "Unknown --column key {!r}. Valid keys: {}".format(
                    key, ", ".join(sorted(valid_keys))
                )
            )
        if not value:
            raise ValueError(
                "Empty CSV column name in --column {!r}.".format(item)
            )
        overrides[key] = value

    return overrides


# ---------------------------------------------------------------------------
# Data preparation and metrics
# ---------------------------------------------------------------------------

def prepare_audit_frame(
    raw: pd.DataFrame,
    mapping: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> Tuple[pd.DataFrame, Optional[str], int]:
    """Create a normalized audit dataframe and drop unusable rows."""
    frame_column = discover_frame_column(raw)

    audit = pd.DataFrame(index=raw.index)
    audit["source_row"] = np.arange(len(raw), dtype=np.int64)

    if frame_column is not None:
        audit["frame"] = raw[frame_column]
    else:
        audit["frame"] = np.arange(len(raw), dtype=np.int64)

    required_columns: List[str] = []
    for component in COMPONENTS:
        for role in ROLES:
            for axis in AXES:
                source = mapping[component][role][axis]
                target = f"{component}_{role}_{axis}"
                audit[target] = pd.to_numeric(raw[source], errors="coerce")
                required_columns.append(target)

    valid_mask = np.ones(len(audit), dtype=bool)
    for column in required_columns:
        valid_mask &= np.isfinite(audit[column].to_numpy(dtype=np.float64))

    dropped = int((~valid_mask).sum())
    audit = audit.loc[valid_mask].reset_index(drop=True)

    if audit.empty:
        raise ValueError(
            "No rows remain after removing rows with missing or non-finite "
            "pose values."
        )

    for component in COMPONENTS:
        gt_cols = [f"{component}_gt_{axis}" for axis in AXES]
        pred_cols = [f"{component}_pred_{axis}" for axis in AXES]

        for axis in AXES:
            gt_col = f"{component}_gt_{axis}"
            pred_col = f"{component}_pred_{axis}"
            error_col = f"{component}_error_{axis}"
            abs_error_col = f"{component}_abs_error_{axis}"
            squared_error_col = f"{component}_squared_error_{axis}"

            audit[error_col] = audit[pred_col] - audit[gt_col]
            audit[abs_error_col] = np.abs(audit[error_col])
            audit[squared_error_col] = np.square(audit[error_col])

        audit[f"{component}_gt_norm"] = np.linalg.norm(
            audit[gt_cols].to_numpy(dtype=np.float64), axis=1
        )
        audit[f"{component}_pred_norm"] = np.linalg.norm(
            audit[pred_cols].to_numpy(dtype=np.float64), axis=1
        )
        audit[f"{component}_error_norm"] = np.linalg.norm(
            audit[pred_cols].to_numpy(dtype=np.float64)
            - audit[gt_cols].to_numpy(dtype=np.float64),
            axis=1,
        )
        audit[f"{component}_norm_error"] = (
            audit[f"{component}_pred_norm"]
            - audit[f"{component}_gt_norm"]
        )

    return audit, frame_column, dropped


def build_statistics(audit: pd.DataFrame) -> pd.DataFrame:
    """Build long-form descriptive statistics for GT, prediction, and error."""
    records: List[Dict[str, object]] = []

    for component in COMPONENTS:
        for axis in AXES:
            for quantity in ("gt", "pred", "error", "abs_error"):
                column = f"{component}_{quantity}_{axis}"
                stats = series_statistics(
                    audit[column].to_numpy(dtype=np.float64)
                )
                record: Dict[str, object] = {
                    "component": component,
                    "axis": axis,
                    "quantity": quantity,
                    "column": column,
                }
                record.update(stats)
                records.append(record)

        for quantity in ("gt_norm", "pred_norm", "error_norm", "norm_error"):
            column = f"{component}_{quantity}"
            stats = series_statistics(
                audit[column].to_numpy(dtype=np.float64)
            )
            record = {
                "component": component,
                "axis": "norm",
                "quantity": quantity,
                "column": column,
            }
            record.update(stats)
            records.append(record)

    return pd.DataFrame.from_records(records)


def build_correlation_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Build per-axis and norm correlation/linear-fit metrics."""
    records: List[Dict[str, object]] = []

    for component in COMPONENTS:
        for axis in AXES:
            gt = audit[f"{component}_gt_{axis}"].to_numpy(dtype=np.float64)
            pred = audit[f"{component}_pred_{axis}"].to_numpy(dtype=np.float64)
            fit = correlation_and_fit(gt, pred)

            error = pred - gt
            gt_std = float(np.std(gt))
            pred_std = float(np.std(pred))
            error_mean = float(np.mean(error))
            error_rmse = float(np.sqrt(np.mean(np.square(error))))

            record: Dict[str, object] = {
                "component": component,
                "axis": axis,
                "gt_mean": safe_float(np.mean(gt)),
                "pred_mean": safe_float(np.mean(pred)),
                "bias": safe_float(error_mean),
                "gt_std": safe_float(gt_std),
                "pred_std": safe_float(pred_std),
                "std_ratio_pred_to_gt": (
                    safe_float(pred_std / gt_std) if gt_std > 0.0 else None
                ),
                "mae": safe_float(np.mean(np.abs(error))),
                "rmse": safe_float(error_rmse),
            }
            record.update(fit)
            records.append(record)

        gt_norm = audit[f"{component}_gt_norm"].to_numpy(dtype=np.float64)
        pred_norm = audit[f"{component}_pred_norm"].to_numpy(dtype=np.float64)
        fit = correlation_and_fit(gt_norm, pred_norm)
        norm_error = pred_norm - gt_norm
        gt_std = float(np.std(gt_norm))
        pred_std = float(np.std(pred_norm))

        record = {
            "component": component,
            "axis": "norm",
            "gt_mean": safe_float(np.mean(gt_norm)),
            "pred_mean": safe_float(np.mean(pred_norm)),
            "bias": safe_float(np.mean(norm_error)),
            "gt_std": safe_float(gt_std),
            "pred_std": safe_float(pred_std),
            "std_ratio_pred_to_gt": (
                safe_float(pred_std / gt_std) if gt_std > 0.0 else None
            ),
            "mae": safe_float(np.mean(np.abs(norm_error))),
            "rmse": safe_float(np.sqrt(np.mean(np.square(norm_error)))),
        }
        record.update(fit)
        records.append(record)

    return pd.DataFrame.from_records(records)


def aggregate_component_metrics(
    audit: pd.DataFrame,
    component: str,
) -> Dict[str, object]:
    """Return overall and per-axis metrics for one pose component."""
    axis_metrics: Dict[str, object] = {}

    all_errors: List[np.ndarray] = []
    for axis in AXES:
        gt = audit[f"{component}_gt_{axis}"].to_numpy(dtype=np.float64)
        pred = audit[f"{component}_pred_{axis}"].to_numpy(dtype=np.float64)
        error = pred - gt
        all_errors.append(error)
        fit = correlation_and_fit(gt, pred)

        axis_metrics[axis] = {
            "gt": series_statistics(gt),
            "prediction": series_statistics(pred),
            "error": series_statistics(error),
            "correlation_and_fit": fit,
        }

    stacked_errors = np.column_stack(all_errors)
    vector_errors = audit[f"{component}_error_norm"].to_numpy(dtype=np.float64)

    return {
        "per_axis": axis_metrics,
        "overall": {
            "axiswise_mae": safe_float(np.mean(np.abs(stacked_errors))),
            "axiswise_rmse": safe_float(
                np.sqrt(np.mean(np.square(stacked_errors)))
            ),
            "mean_vector_error": safe_float(np.mean(vector_errors)),
            "median_vector_error": safe_float(np.median(vector_errors)),
            "vector_error_rmse": safe_float(
                np.sqrt(np.mean(np.square(vector_errors)))
            ),
            "maximum_vector_error": safe_float(np.max(vector_errors)),
            "gt_norm": series_statistics(
                audit[f"{component}_gt_norm"].to_numpy(dtype=np.float64)
            ),
            "prediction_norm": series_statistics(
                audit[f"{component}_pred_norm"].to_numpy(dtype=np.float64)
            ),
            "norm_error": series_statistics(
                audit[f"{component}_norm_error"].to_numpy(dtype=np.float64)
            ),
            "norm_correlation_and_fit": correlation_and_fit(
                audit[f"{component}_gt_norm"].to_numpy(dtype=np.float64),
                audit[f"{component}_pred_norm"].to_numpy(dtype=np.float64),
            ),
        },
    }


# ---------------------------------------------------------------------------
# Automatic failure-mode detection
# ---------------------------------------------------------------------------

def detect_failure_modes(
    audit: pd.DataFrame,
    correlations: pd.DataFrame,
    minimum_correlation: float,
    scale_tolerance: float,
    bias_fraction: float,
    collapse_std_ratio: float,
    outlier_z_threshold: float,
) -> List[Dict[str, object]]:
    """Create interpretable heuristic findings from the audit metrics."""
    findings: List[Dict[str, object]] = []

    def add(
        component: str,
        axis: str,
        failure_mode: str,
        severity: str,
        evidence: str,
        recommendation: str,
    ) -> None:
        findings.append(
            {
                "component": component,
                "axis": axis,
                "failure_mode": failure_mode,
                "severity": severity,
                "evidence": evidence,
                "recommendation": recommendation,
            }
        )

    for _, row in correlations.iterrows():
        component = str(row["component"])
        axis = str(row["axis"])

        gt_std = row.get("gt_std")
        pred_std = row.get("pred_std")
        slope = row.get("slope")
        intercept = row.get("intercept")
        pearson_r = row.get("pearson_r")
        bias = row.get("bias")
        std_ratio = row.get("std_ratio_pred_to_gt")

        gt_std_f = float(gt_std) if pd.notna(gt_std) else 0.0
        pred_std_f = float(pred_std) if pd.notna(pred_std) else 0.0

        if gt_std_f > 0.0 and pred_std_f <= gt_std_f * collapse_std_ratio:
            add(
                component,
                axis,
                "prediction_collapse_or_under-dispersion",
                "high",
                "Prediction standard deviation {:.6g} is only {:.3f} of "
                "the GT standard deviation {:.6g}.".format(
                    pred_std_f,
                    pred_std_f / gt_std_f,
                    gt_std_f,
                ),
                "Inspect output activation, target normalization, loss "
                "weighting, and whether the head is converging toward a "
                "near-constant mean prediction.",
            )

        if pd.notna(pearson_r) and abs(float(pearson_r)) < minimum_correlation:
            severity = "high" if abs(float(pearson_r)) < 0.25 else "medium"
            add(
                component,
                axis,
                "poor_correlation",
                severity,
                "Pearson correlation is {:.4f}, below the configured "
                "threshold {:.4f}.".format(
                    float(pearson_r), minimum_correlation
                ),
                "Check target ordering, frame alignment, representation "
                "decoding, and whether this output axis contains a learnable "
                "signal.",
            )

        if pd.notna(pearson_r) and float(pearson_r) < -minimum_correlation:
            add(
                component,
                axis,
                "possible_sign_or_frame_inversion",
                "high",
                "Pearson correlation is strongly negative ({:.4f}).".format(
                    float(pearson_r)
                ),
                "Verify relative-pose direction, source/target frame order, "
                "axis sign conventions, and inverse-transform handling.",
            )

        if (
            pd.notna(slope)
            and pd.notna(pearson_r)
            and abs(float(pearson_r)) >= minimum_correlation
            and abs(float(slope) - 1.0) > scale_tolerance
        ):
            add(
                component,
                axis,
                "scale_error",
                "high" if abs(float(slope) - 1.0) > 0.75 else "medium",
                "Linear fit pred = slope*GT + intercept gives slope "
                "{:.4f}; ideal slope is 1.0.".format(float(slope)),
                "Audit target scaling, inverse normalization, units, and "
                "directional-coordinate decoding.",
            )

        if gt_std_f > 0.0 and pd.notna(bias):
            normalized_bias = abs(float(bias)) / gt_std_f
            if normalized_bias > bias_fraction:
                add(
                    component,
                    axis,
                    "systematic_bias",
                    "high" if normalized_bias > 1.0 else "medium",
                    "Mean prediction error is {:.6g}, equal to {:.3f} GT "
                    "standard deviations.".format(
                        float(bias), normalized_bias
                    ),
                    "Check target centering, output bias initialization, "
                    "dataset imbalance, and inverse preprocessing.",
                )

        if pd.notna(intercept) and gt_std_f > 0.0:
            normalized_intercept = abs(float(intercept)) / gt_std_f
            if normalized_intercept > bias_fraction:
                add(
                    component,
                    axis,
                    "linear_fit_offset",
                    "medium",
                    "Regression intercept is {:.6g}, equal to {:.3f} GT "
                    "standard deviations.".format(
                        float(intercept), normalized_intercept
                    ),
                    "Check whether predictions or labels were centered, "
                    "standardized, or decoded with a missing offset.",
                )

        if (
            pd.notna(std_ratio)
            and float(std_ratio) > 2.0
            and axis != "norm"
        ):
            add(
                component,
                axis,
                "over-dispersion",
                "medium",
                "Prediction standard deviation is {:.3f} times the GT "
                "standard deviation.".format(float(std_ratio)),
                "Inspect output scale, unstable samples, and whether a small "
                "number of extreme predictions dominate the distribution.",
            )

    for component in COMPONENTS:
        values = audit[f"{component}_error_norm"].to_numpy(dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))

        if mad > 0.0:
            robust_z = 0.67448975 * (values - median) / mad
            count = int(np.sum(robust_z > outlier_z_threshold))
            fraction = count / float(len(values))
            if count > 0:
                add(
                    component,
                    "vector",
                    "large_error_outliers",
                    "medium" if fraction < 0.05 else "high",
                    "{} of {} frames ({:.2%}) exceed robust z-score "
                    "{:.2f} for vector error.".format(
                        count, len(values), fraction, outlier_z_threshold
                    ),
                    "Review the ranked worst-frame CSV and inspect image "
                    "content, motion magnitude, cue quality, and frame "
                    "alignment around those transitions.",
                )

    # Remove exact duplicate findings while preserving order.
    deduplicated: List[Dict[str, object]] = []
    seen = set()
    for item in findings:
        key = (
            item["component"],
            item["axis"],
            item["failure_mode"],
            item["evidence"],
        )
        if key not in seen:
            deduplicated.append(item)
            seen.add(key)

    return deduplicated


# ---------------------------------------------------------------------------
# Tables and plots
# ---------------------------------------------------------------------------

def build_frame_errors_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Select and order frame-level values and errors for CSV export."""
    columns = ["frame", "source_row"]

    for component in COMPONENTS:
        for role in ROLES:
            columns.extend(
                [f"{component}_{role}_{axis}" for axis in AXES]
            )
        columns.extend(
            [f"{component}_error_{axis}" for axis in AXES]
        )
        columns.extend(
            [f"{component}_abs_error_{axis}" for axis in AXES]
        )
        columns.extend(
            [
                f"{component}_gt_norm",
                f"{component}_pred_norm",
                f"{component}_norm_error",
                f"{component}_error_norm",
            ]
        )

    return audit[columns].copy()


def build_worst_frames(
    audit: pd.DataFrame,
    component: str,
    count: int,
) -> pd.DataFrame:
    """Rank frames by vector error for one component."""
    columns = ["frame", "source_row"]

    for role in ROLES:
        columns.extend([f"{component}_{role}_{axis}" for axis in AXES])

    columns.extend([f"{component}_error_{axis}" for axis in AXES])
    columns.extend([f"{component}_abs_error_{axis}" for axis in AXES])
    columns.extend(
        [
            f"{component}_gt_norm",
            f"{component}_pred_norm",
            f"{component}_norm_error",
            f"{component}_error_norm",
        ]
    )

    ranked = audit.sort_values(
        f"{component}_error_norm", ascending=False
    ).head(count)
    ranked = ranked[columns].copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def plot_timeseries(
    audit: pd.DataFrame,
    component: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot GT and prediction over frame index for x/y/z."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(audit))

    for index, axis in enumerate(AXES):
        current = axes[index]
        current.plot(
            x,
            audit[f"{component}_gt_{axis}"],
            label="GT",
            linewidth=1.2,
        )
        current.plot(
            x,
            audit[f"{component}_pred_{axis}"],
            label="Prediction",
            linewidth=1.0,
            alpha=0.85,
        )
        current.set_ylabel(axis.upper())
        current.grid(True, alpha=0.25)
        current.legend(loc="best")

    axes[-1].set_xlabel("Transition index")
    fig.suptitle(
        "{} GT vs prediction time series".format(component.title()),
        y=1.01,
    )
    save_figure(fig, path, dpi)


def plot_scatter(
    audit: pd.DataFrame,
    component: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot per-axis prediction versus GT with ideal and fitted lines."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for index, axis in enumerate(AXES):
        current = axes[index]
        gt = audit[f"{component}_gt_{axis}"].to_numpy(dtype=np.float64)
        pred = audit[f"{component}_pred_{axis}"].to_numpy(dtype=np.float64)

        current.scatter(gt, pred, s=10, alpha=0.45)

        lower = float(min(np.min(gt), np.min(pred)))
        upper = float(max(np.max(gt), np.max(pred)))
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

        fit = correlation_and_fit(gt, pred)
        slope = fit["slope"]
        intercept = fit["intercept"]
        if slope is not None and intercept is not None:
            fit_x = np.array([lower, upper], dtype=np.float64)
            fit_y = float(slope) * fit_x + float(intercept)
            current.plot(
                fit_x,
                fit_y,
                linewidth=1.2,
                label="Linear fit",
            )

        r = fit["pearson_r"]
        annotation = (
            "r={:.4f}\nslope={:.4f}\nintercept={:.4g}".format(
                float(r) if r is not None else float("nan"),
                float(slope) if slope is not None else float("nan"),
                float(intercept) if intercept is not None else float("nan"),
            )
        )
        current.text(
            0.04,
            0.96,
            annotation,
            transform=current.transAxes,
            va="top",
            bbox={"boxstyle": "round", "alpha": 0.8},
        )
        current.set_xlabel("GT {}".format(axis.upper()))
        current.set_ylabel("Prediction {}".format(axis.upper()))
        current.set_title(axis.upper())
        current.grid(True, alpha=0.25)
        current.legend(loc="lower right")

    fig.suptitle(
        "{} GT vs prediction scatter".format(component.title()),
        y=1.02,
    )
    save_figure(fig, path, dpi)


def plot_error_histograms(
    audit: pd.DataFrame,
    component: str,
    path: Path,
    dpi: int,
    bins: int,
) -> None:
    """Plot signed per-axis error histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for index, axis in enumerate(AXES):
        errors = audit[f"{component}_error_{axis}"].to_numpy(
            dtype=np.float64
        )
        current = axes[index]
        current.hist(errors, bins=bins, alpha=0.8)
        current.axvline(0.0, linestyle="--", linewidth=1.2)
        current.axvline(
            float(np.mean(errors)),
            linestyle="-",
            linewidth=1.2,
            label="Mean error",
        )
        current.set_xlabel("Prediction - GT")
        current.set_ylabel("Frame count")
        current.set_title(axis.upper())
        current.grid(True, alpha=0.2)
        current.legend(loc="best")

    fig.suptitle(
        "{} per-axis error distributions".format(component.title()),
        y=1.02,
    )
    save_figure(fig, path, dpi)


def plot_translation_norms(
    audit: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot translation magnitudes and prediction-vs-GT magnitude scatter."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(len(audit))

    gt_norm = audit["translation_gt_norm"].to_numpy(dtype=np.float64)
    pred_norm = audit["translation_pred_norm"].to_numpy(dtype=np.float64)

    axes[0].plot(x, gt_norm, label="GT translation norm", linewidth=1.2)
    axes[0].plot(
        x,
        pred_norm,
        label="Predicted translation norm",
        linewidth=1.0,
        alpha=0.85,
    )
    axes[0].set_xlabel("Transition index")
    axes[0].set_ylabel("Translation norm")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(gt_norm, pred_norm, s=10, alpha=0.45)
    lower = float(min(np.min(gt_norm), np.min(pred_norm)))
    upper = float(max(np.max(gt_norm), np.max(pred_norm)))
    if lower == upper:
        lower -= 0.5
        upper += 0.5

    axes[1].plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.2,
        label="Ideal y=x",
    )

    fit = correlation_and_fit(gt_norm, pred_norm)
    if fit["slope"] is not None and fit["intercept"] is not None:
        fit_x = np.array([lower, upper], dtype=np.float64)
        fit_y = float(fit["slope"]) * fit_x + float(fit["intercept"])
        axes[1].plot(fit_x, fit_y, linewidth=1.2, label="Linear fit")

    axes[1].set_xlabel("GT translation norm")
    axes[1].set_ylabel("Predicted translation norm")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Translation magnitude audit", y=1.01)
    save_figure(fig, path, dpi)


def plot_correlation_matrix(
    audit: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot a correlation matrix across GT and prediction pose signals."""
    columns = []
    labels = []

    for component in COMPONENTS:
        prefix = "R" if component == "rotation" else "T"
        for role in ROLES:
            role_label = "GT" if role == "gt" else "Pred"
            for axis in AXES:
                columns.append(f"{component}_{role}_{axis}")
                labels.append(f"{prefix}-{role_label}-{axis.upper()}")

    matrix = audit[columns].corr(method="pearson").to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, aspect="auto")
    fig.colorbar(image, ax=ax, label="Pearson correlation")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "nan" if not np.isfinite(value) else "{:.2f}".format(value)
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title("Pose-signal correlation matrix")
    save_figure(fig, path, dpi)


# ---------------------------------------------------------------------------
# Single-experiment audit
# ---------------------------------------------------------------------------

def run_single_audit(
    csv_path: Path,
    output_dir: Path,
    label: str,
    version: int,
    overrides: Mapping[str, str],
    worst_count: int,
    dpi: int,
    histogram_bins: int,
    minimum_correlation: float,
    scale_tolerance: float,
    bias_fraction: float,
    collapse_std_ratio: float,
    outlier_z_threshold: float,
) -> Dict[str, object]:
    """Run the complete audit for one experiment."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Input CSV does not exist: {}".format(csv_path)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(str(csv_path))
    if raw.empty:
        raise ValueError("Input CSV is empty: {}".format(csv_path))

    mapping = discover_pose_columns(raw, overrides=overrides)
    audit, frame_column, dropped_rows = prepare_audit_frame(raw, mapping)

    statistics = build_statistics(audit)
    statistics.to_csv(output_dir / "statistics.csv", index=False)

    frame_errors = build_frame_errors_table(audit)
    frame_errors.to_csv(output_dir / "frame_errors.csv", index=False)

    plot_timeseries(
        audit,
        "rotation",
        plots_dir / "rotation_timeseries.png",
        dpi,
    )
    plot_timeseries(
        audit,
        "translation",
        plots_dir / "translation_timeseries.png",
        dpi,
    )
    plot_scatter(
        audit,
        "rotation",
        plots_dir / "rotation_scatter.png",
        dpi,
    )
    plot_scatter(
        audit,
        "translation",
        plots_dir / "translation_scatter.png",
        dpi,
    )

    correlation_table = build_correlation_table(audit)
    findings: List[Dict[str, object]] = []

    if version >= 2:
        correlation_table.to_csv(
            output_dir / "correlation.csv", index=False
        )

        worst_rotation = build_worst_frames(
            audit, "rotation", worst_count
        )
        worst_translation = build_worst_frames(
            audit, "translation", worst_count
        )
        worst_rotation.to_csv(
            output_dir / "worst_rotation_frames.csv", index=False
        )
        worst_translation.to_csv(
            output_dir / "worst_translation_frames.csv", index=False
        )

        plot_error_histograms(
            audit,
            "rotation",
            plots_dir / "rotation_error_histograms.png",
            dpi,
            histogram_bins,
        )
        plot_error_histograms(
            audit,
            "translation",
            plots_dir / "translation_error_histograms.png",
            dpi,
            histogram_bins,
        )
        plot_translation_norms(
            audit,
            plots_dir / "translation_norms.png",
            dpi,
        )
        plot_correlation_matrix(
            audit,
            plots_dir / "correlation_matrix.png",
            dpi,
        )

        findings = detect_failure_modes(
            audit=audit,
            correlations=correlation_table,
            minimum_correlation=minimum_correlation,
            scale_tolerance=scale_tolerance,
            bias_fraction=bias_fraction,
            collapse_std_ratio=collapse_std_ratio,
            outlier_z_threshold=outlier_z_threshold,
        )

    summary: Dict[str, object] = {
        "audit_version": version,
        "experiment": label,
        "input_csv": str(csv_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "rows": {
            "input": int(len(raw)),
            "analyzed": int(len(audit)),
            "dropped_non_finite": dropped_rows,
        },
        "frame_identifier_column": frame_column,
        "resolved_columns": mapping,
        "rotation": aggregate_component_metrics(audit, "rotation"),
        "translation": aggregate_component_metrics(
            audit, "translation"
        ),
        "automatic_failure_mode_detection": {
            "enabled": version >= 2,
            "configuration": {
                "minimum_absolute_correlation": minimum_correlation,
                "scale_slope_tolerance_from_one": scale_tolerance,
                "bias_fraction_of_gt_std": bias_fraction,
                "collapse_std_ratio": collapse_std_ratio,
                "outlier_robust_z_threshold": outlier_z_threshold,
            },
            "finding_count": len(findings),
            "findings": findings,
        },
        "outputs": {
            "summary": "summary.json",
            "statistics": "statistics.csv",
            "frame_errors": "frame_errors.csv",
            "correlation": (
                "correlation.csv" if version >= 2 else None
            ),
            "worst_rotation_frames": (
                "worst_rotation_frames.csv" if version >= 2 else None
            ),
            "worst_translation_frames": (
                "worst_translation_frames.csv" if version >= 2 else None
            ),
            "plots": {
                "rotation_timeseries": "plots/rotation_timeseries.png",
                "translation_timeseries": (
                    "plots/translation_timeseries.png"
                ),
                "rotation_scatter": "plots/rotation_scatter.png",
                "translation_scatter": "plots/translation_scatter.png",
                "rotation_error_histograms": (
                    "plots/rotation_error_histograms.png"
                    if version >= 2
                    else None
                ),
                "translation_error_histograms": (
                    "plots/translation_error_histograms.png"
                    if version >= 2
                    else None
                ),
                "translation_norms": (
                    "plots/translation_norms.png"
                    if version >= 2
                    else None
                ),
                "correlation_matrix": (
                    "plots/correlation_matrix.png"
                    if version >= 2
                    else None
                ),
            },
        },
    }

    write_json(output_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Multi-experiment comparison
# ---------------------------------------------------------------------------

def flatten_comparison_metrics(summary: Mapping[str, object]) -> Dict[str, object]:
    """Extract the principal metrics used in multi-experiment comparison."""
    row: Dict[str, object] = {
        "experiment": summary["experiment"],
        "input_csv": summary["input_csv"],
        "analyzed_rows": summary["rows"]["analyzed"],  # type: ignore[index]
        "finding_count": summary[
            "automatic_failure_mode_detection"
        ]["finding_count"],  # type: ignore[index]
    }

    for component in COMPONENTS:
        component_summary = summary[component]  # type: ignore[index]
        overall = component_summary["overall"]  # type: ignore[index]
        row[f"{component}_axiswise_mae"] = overall["axiswise_mae"]
        row[f"{component}_axiswise_rmse"] = overall["axiswise_rmse"]
        row[f"{component}_mean_vector_error"] = overall[
            "mean_vector_error"
        ]
        row[f"{component}_vector_error_rmse"] = overall[
            "vector_error_rmse"
        ]
        row[f"{component}_maximum_vector_error"] = overall[
            "maximum_vector_error"
        ]

        for axis in AXES:
            axis_summary = component_summary["per_axis"][axis]  # type: ignore[index]
            fit = axis_summary["correlation_and_fit"]
            error = axis_summary["error"]
            row[f"{component}_{axis}_pearson_r"] = fit["pearson_r"]
            row[f"{component}_{axis}_slope"] = fit["slope"]
            row[f"{component}_{axis}_bias"] = error["mean"]
            row[f"{component}_{axis}_rmse"] = error["rmse"]

    translation_norm_fit = summary["translation"]["overall"][  # type: ignore[index]
        "norm_correlation_and_fit"
    ]
    row["translation_norm_pearson_r"] = translation_norm_fit[
        "pearson_r"
    ]
    row["translation_norm_slope"] = translation_norm_fit["slope"]
    return row


def plot_experiment_comparison(
    table: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Generate compact cross-experiment metric comparison plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels = table["experiment"].astype(str).tolist()
    positions = np.arange(len(labels))

    metrics = [
        (
            "translation_vector_error_rmse",
            "Translation vector-error RMSE",
            "comparison_translation_vector_rmse.png",
        ),
        (
            "rotation_vector_error_rmse",
            "Rotation vector-error RMSE",
            "comparison_rotation_vector_rmse.png",
        ),
        (
            "translation_norm_pearson_r",
            "Translation-norm correlation",
            "comparison_translation_norm_correlation.png",
        ),
        (
            "translation_norm_slope",
            "Translation-norm fit slope",
            "comparison_translation_norm_slope.png",
        ),
    ]

    for column, title, filename in metrics:
        if column not in table.columns:
            continue

        values = pd.to_numeric(table[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5.5))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(column)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

        if "slope" in column:
            ax.axhline(1.0, linestyle="--", linewidth=1.2)
        if "correlation" in column:
            ax.axhline(0.0, linestyle="--", linewidth=1.0)

        save_figure(fig, plots_dir / filename, dpi)


def parse_experiment_spec(value: str) -> Tuple[str, Path]:
    """Parse label=path or infer a label from an unlabelled path."""
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = sanitize_label(label)
        path = Path(raw_path).expanduser()
    else:
        path = Path(value).expanduser()
        label = sanitize_label(path.parent.name or path.stem)
    return label, path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frame-level relative rotation and translation predictions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Path to one frame_predictions.csv file.",
    )
    input_group.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help=(
            "Experiment specification. Repeat for multiple experiments. "
            "A single specification is also supported."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Audit output directory.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label used with --input; defaults to the CSV parent name.",
    )
    parser.add_argument(
        "--version",
        type=int,
        choices=(1, 2),
        default=2,
        help="Audit feature level.",
    )
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        metavar="KEY=CSV_COLUMN",
        help=(
            "Override automatic column discovery. Valid keys include "
            "rotation.gt.x, rotation.pred.x, translation.gt.z, etc. "
            "Repeat as needed."
        ),
    )
    parser.add_argument(
        "--worst-count",
        type=int,
        default=50,
        help="Number of worst frames to export per pose component.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Plot resolution.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=60,
        help="Number of bins in error histograms.",
    )

    # Failure-mode thresholds.
    parser.add_argument(
        "--minimum-correlation",
        type=float,
        default=0.50,
        help=(
            "Minimum absolute Pearson correlation before poor-correlation "
            "is reported."
        ),
    )
    parser.add_argument(
        "--scale-tolerance",
        type=float,
        default=0.25,
        help=(
            "Allowed absolute difference between fitted slope and 1.0."
        ),
    )
    parser.add_argument(
        "--bias-fraction",
        type=float,
        default=0.25,
        help=(
            "Bias threshold expressed as a fraction of GT standard deviation."
        ),
    )
    parser.add_argument(
        "--collapse-std-ratio",
        type=float,
        default=0.20,
        help=(
            "Prediction/GT standard-deviation ratio below which output "
            "collapse is reported."
        ),
    )
    parser.add_argument(
        "--outlier-robust-z-threshold",
        type=float,
        default=5.0,
        help="Robust z-score threshold for large vector-error outliers.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.worst_count <= 0:
        raise ValueError("--worst-count must be greater than zero.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")
    if args.histogram_bins <= 1:
        raise ValueError("--histogram-bins must be greater than one.")
    if not 0.0 <= args.minimum_correlation <= 1.0:
        raise ValueError("--minimum-correlation must be in [0, 1].")
    if args.scale_tolerance < 0.0:
        raise ValueError("--scale-tolerance must be non-negative.")
    if args.bias_fraction < 0.0:
        raise ValueError("--bias-fraction must be non-negative.")
    if args.collapse_std_ratio < 0.0:
        raise ValueError("--collapse-std-ratio must be non-negative.")
    if args.outlier_robust_z_threshold <= 0.0:
        raise ValueError(
            "--outlier-robust-z-threshold must be greater than zero."
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        overrides = parse_column_overrides(args.column)

        experiments: List[Tuple[str, Path]] = []
        if args.input is not None:
            input_path = args.input.expanduser()
            default_label = input_path.parent.name or input_path.stem
            label = sanitize_label(args.label or default_label)
            experiments.append((label, input_path))
        else:
            for specification in args.experiment:
                experiments.append(parse_experiment_spec(specification))

        labels = [label for label, _ in experiments]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "Experiment labels must be unique: {}".format(
                    ", ".join(labels)
                )
            )

        root = args.output_dir.expanduser()
        root.mkdir(parents=True, exist_ok=True)

        multi_experiment = len(experiments) > 1
        summaries: List[Dict[str, object]] = []

        for label, csv_path in experiments:
            audit_dir = (
                root / "experiments" / label
                if multi_experiment
                else root
            )

            print(
                "[audit] experiment={!r} input={} output={}".format(
                    label, csv_path, audit_dir
                )
            )

            summary = run_single_audit(
                csv_path=csv_path,
                output_dir=audit_dir,
                label=label,
                version=args.version,
                overrides=overrides,
                worst_count=args.worst_count,
                dpi=args.dpi,
                histogram_bins=args.histogram_bins,
                minimum_correlation=args.minimum_correlation,
                scale_tolerance=args.scale_tolerance,
                bias_fraction=args.bias_fraction,
                collapse_std_ratio=args.collapse_std_ratio,
                outlier_z_threshold=args.outlier_robust_z_threshold,
            )
            summaries.append(summary)

            translation = summary["translation"]["overall"]
            rotation = summary["rotation"]["overall"]
            print(
                "[audit] {}: translation vector RMSE={}, "
                "rotation vector RMSE={}, findings={}".format(
                    label,
                    translation["vector_error_rmse"],
                    rotation["vector_error_rmse"],
                    summary["automatic_failure_mode_detection"][
                        "finding_count"
                    ],
                )
            )

        if multi_experiment:
            comparison_rows = [
                flatten_comparison_metrics(summary)
                for summary in summaries
            ]
            comparison = pd.DataFrame.from_records(comparison_rows)
            comparison.to_csv(
                root / "experiment_comparison.csv", index=False
            )
            plot_experiment_comparison(comparison, root, args.dpi)

            ranking = comparison.sort_values(
                "translation_vector_error_rmse",
                ascending=True,
                na_position="last",
            )
            ranking_records = ranking.to_dict(orient="records")

            comparison_summary = {
                "audit_version": args.version,
                "experiment_count": len(summaries),
                "experiments": [
                    {
                        "label": summary["experiment"],
                        "input_csv": summary["input_csv"],
                        "audit_directory": str(
                            (
                                root
                                / "experiments"
                                / str(summary["experiment"])
                            ).resolve()
                        ),
                    }
                    for summary in summaries
                ],
                "ranking_criterion": (
                    "translation_vector_error_rmse ascending"
                ),
                "ranking": ranking_records,
                "outputs": {
                    "comparison_csv": "experiment_comparison.csv",
                    "experiment_audits": "experiments/<label>/",
                    "comparison_plots": "plots/comparison_*.png",
                },
            }
            write_json(
                root / "comparison_summary.json",
                comparison_summary,
            )

        print("[audit] complete")
        return 0

    except Exception as exc:
        print("[audit] ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
