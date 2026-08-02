from __future__ import annotations

from typing import Mapping, Union

import torch
from torch import Tensor


ModelOutput = Union[Tensor, Mapping[str, Tensor]]


def _find_output_tensor(
    outputs: ModelOutput,
    key: str,
) -> Tensor:
    if isinstance(outputs, Tensor):
        if key not in {"output", "prediction"}:
            raise KeyError(
                f"The model returned a Tensor, but target key '{key}' was requested."
            )
        return outputs

    if key in outputs:
        return outputs[key]

    aliases = {
        "rotation": (
            "rotation",
            "rotation_pred",
            "pred_rotation",
            "rotation_output",
            "predicted_rotation",
        ),
        "translation": (
            "translation",
            "directional_translation",
            "translation_pred",
            "pred_translation",
            "translation_output",
        ),
    }

    for alias in aliases.get(key, (key,)):
        if alias in outputs:
            return outputs[alias]

    raise KeyError(
        f"Could not find '{key}' in model output. "
        f"Available keys: {list(outputs.keys())}"
    )


def regression_target(
    outputs: ModelOutput,
    target: str,
    sample_index: int = 0,
) -> Tensor:
    """
    Convert a DeepDCT-VO output into one scalar for backpropagation.

    Supported target names:
        rotation_x
        rotation_y
        rotation_z
        rotation_norm
        translation_x
        translation_y
        translation_z
        translation_norm
    """
    component_names = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if target.startswith("rotation_"):
        vector = _find_output_tensor(outputs, "rotation")
        component = target[len("rotation_"):]
    elif target.startswith("translation_"):
        vector = _find_output_tensor(outputs, "translation")
        component = target[len("translation_"):]
    else:
        raise ValueError(
            f"Unsupported target '{target}'. Use rotation_x/y/z/norm or "
            "translation_x/y/z/norm."
        )

    if vector.ndim == 1:
        vector = vector.unsqueeze(0)

    selected = vector[sample_index]

    if component == "norm":
        return torch.linalg.vector_norm(selected)

    if component not in component_names:
        raise ValueError(f"Unsupported target component '{component}'.")

    component_index = component_names[component]

    if selected.numel() <= component_index:
        raise IndexError(
            f"Output has only {selected.numel()} components; "
            f"component index {component_index} was requested."
        )

    return selected[component_index]